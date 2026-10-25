"""CPU tests: real ComfyUI packing/H3 scaling, synthetic denoiser (no weights)."""
import copy
import importlib.util
import logging
import os
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

PLUGIN = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("COMFYUI_ROOT", str(PLUGIN.parents[1])))
sys.path.insert(0, str(ROOT))
import comfy.options
comfy.options.enable_args_parsing()
sys.argv = [sys.argv[0], '--cpu']
spec = importlib.util.spec_from_file_location('selflift_avatar_test', PLUGIN/'__init__.py', submodule_search_locations=[str(PLUGIN)])
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
spec.loader.exec_module(plugin)
from selflift_avatar_test import nodes, avatar_masks as masks, selflift, avatar_sampling, h3_tiling, h3_global_tiling, h3_output_chunking, h3_upscaler
import torch
import comfy.latent_formats
import comfy.model_base
import comfy.model_sampling
import comfy.samplers
import comfy.utils
from comfy.nested_tensor import NestedTensor

logging.getLogger().setLevel(logging.ERROR)

class AVSampling(comfy.model_sampling.ModelSamplingAV, comfy.model_sampling.CONST):
    pass

class TinyH3(comfy.model_base.MiniMaxH3):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.latent_format = comfy.latent_formats.MiniMaxH3AV()
        self.model_sampling = AVSampling()
        self.model_sampling.set_parameters(shift=3.0, audio_shift=1.0)
        self.latent_shapes = None
        self.diffusion_model = torch.nn.Module()
        self.diffusion_model.patch_size = (1, 2, 2)

class Patcher:
    def __init__(self):
        self.model = TinyH3()
        self.model_options = {}
        self.load_device = torch.device('cpu')
    def clone(self):
        result = copy.copy(self)
        result.model_options = copy.deepcopy(self.model_options)
        return result
    def add_wrapper_with_key(self, kind, key, wrapper):
        comfy.patcher_extension.add_wrapper_with_key(kind, key, wrapper, self.model_options, is_model_options=True)
    def get_model_object(self, key):
        return getattr(self.model, key)

class UpscalerTemporalPaddingTests(unittest.TestCase):
    def test_reflect_padding_does_not_repeat_first_frame(self):
        x = torch.arange(1, 41, dtype=torch.float32).reshape(1, 1, 40, 1, 1)
        padded = h3_upscaler._temporal_pad(x, 5)

        self.assertEqual(padded.shape[2], 50)
        torch.testing.assert_close(
            padded[:, :, :5], x[:, :, torch.arange(5, 0, -1)])
        torch.testing.assert_close(
            padded[:, :, -5:], x[:, :, torch.arange(38, 33, -1)])
        self.assertEqual((padded == x[:, :, :1]).sum().item(), 1)

    def test_oversized_overlap_preserves_replicate_fallback(self):
        x = torch.arange(1, 6, dtype=torch.float32).reshape(1, 1, 5, 1, 1)
        with self.assertLogs(level="WARNING"):
            padded = h3_upscaler._temporal_pad(x, 5)

        self.assertEqual(padded.shape[2], 15)
        torch.testing.assert_close(padded[:, :, :5], x[:, :, :1].expand_as(padded[:, :, :5]))

    def test_long_latent_forward_uses_reflect_halo(self):
        model = h3_upscaler.LatentResizer3D(
            in_channels=1, in_blocks=1, out_blocks=1, channels=32,
            dropout=0, attn=False, temporal_every=1, temporal_kernel=5)
        x = torch.zeros(1, 1, 33, 1, 1)

        with patch.object(h3_upscaler.F, "pad", wraps=h3_upscaler.F.pad) as pad:
            out = model(x, target_size=(33, 2, 2))

        self.assertEqual(out.shape, (1, 1, 33, 2, 2))
        self.assertTrue(torch.isfinite(out).all())
        pad.assert_called_once()
        self.assertEqual(pad.call_args.kwargs["mode"], "reflect")

    def test_short_latent_forward_does_not_pad(self):
        model = h3_upscaler.LatentResizer3D(
            in_channels=1, in_blocks=1, out_blocks=1, channels=32,
            dropout=0, attn=False, temporal_every=1, temporal_kernel=5)
        x = torch.zeros(1, 1, 32, 1, 1)

        with patch.object(h3_upscaler.F, "pad", wraps=h3_upscaler.F.pad) as pad:
            out = model(x, target_size=(32, 2, 2))

        self.assertEqual(out.shape, (1, 1, 32, 2, 2))
        pad.assert_not_called()

    def test_reflect_only_changes_long_latent_end_windows(self):
        torch.manual_seed(7)
        model = h3_upscaler.LatentResizer3D(
            in_channels=1, in_blocks=1, out_blocks=1, channels=32,
            dropout=0, attn=False, temporal_every=1, temporal_kernel=5)
        x = torch.randn(1, 1, 107, 1, 1)

        with torch.no_grad():
            reflect = model(x, target_size=(107, 2, 2))
            original_pad = h3_upscaler._temporal_pad
            h3_upscaler._temporal_pad = lambda z, overlap: h3_upscaler.F.pad(
                z, (0, 0, 0, 0, overlap, overlap), mode="replicate")
            try:
                replicate = model(x, target_size=(107, 2, 2))
            finally:
                h3_upscaler._temporal_pad = original_pad

        self.assertFalse(torch.equal(reflect[:, :, 0], replicate[:, :, 0]))
        torch.testing.assert_close(
            reflect[:, :, 37:91], replicate[:, :, 37:91],
            rtol=0, atol=0)

class MaskTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.v = torch.randn(2, 24, 3, 8, 10)
        self.a = torch.randn(2, 32, 2, 7)
        self.streams = [self.v, self.a]
    def test_no_mask(self):
        self.assertIsNone(masks.normalize_masks(None, self.streams))
    def test_video_layouts(self):
        for shape in [(1, 8, 10), (2, 1, 8, 10), (1, 1, 3, 8, 10), (2, 24, 3, 8, 10)]:
            with self.subTest(shape=shape):
                out = masks.normalize_masks(torch.ones(shape), self.streams)
                self.assertTrue(torch.equal(out[0].expand_as(self.v), torch.ones_like(self.v)))
                self.assertTrue(torch.equal(out[1].expand_as(self.a), torch.ones_like(self.a)))
    def test_nested_channels_are_not_dropped(self):
        mv = torch.rand_like(self.v)
        ma = torch.rand_like(self.a)
        out = masks.normalize_masks(NestedTensor([mv, ma]), self.streams)
        torch.testing.assert_close(out[0], mv)
        torch.testing.assert_close(out[1], ma)
    def test_compact_av_and_broadcast(self):
        mv = torch.tensor([0., .5, 1.]).reshape(1, 1, 3, 1, 1)
        ma = torch.zeros(1, 1, 1, 7)
        out = masks.normalize_masks(NestedTensor([mv, ma]), self.streams)
        torch.testing.assert_close(out[0].expand_as(self.v)[:, :, 1], torch.full_like(self.v[:, :, 1], .5))
        self.assertEqual(out[1].expand_as(self.a).count_nonzero(), 0)
    def test_audio_layouts(self):
        for shape in [(7,), (1, 7), (2, 2, 7), (2, 32, 2, 7)]:
            with self.subTest(shape=shape):
                out = masks.normalize_masks(NestedTensor([torch.ones(1, 8, 10), torch.zeros(shape)]), self.streams)
                self.assertEqual(out[1].expand_as(self.a).count_nonzero(), 0)
    def test_solid_image_mask_on_audio(self):
        for size in [(64, 64), (928, 1664)]:
            for value in [0., .5, 1.]:
                with self.subTest(size=size, value=value):
                    out = masks.normalize_masks(NestedTensor([
                        torch.ones(1, 8, 10), torch.full((1, 1, *size), value)
                    ]), self.streams)
                    self.assertEqual(out[1].shape, (2, 1, 1, 1))
                    torch.testing.assert_close(out[1].expand_as(self.a), torch.full_like(self.a, value))
    def test_solid_audio_preserves_batch_channel_values(self):
        values = torch.rand(2, 32, 1, 1)
        spatial = values.expand(2, 32, 16, 20)
        out = masks.normalize_masks(NestedTensor([torch.ones(1,8,10), spatial]), self.streams)
        torch.testing.assert_close(out[1], values)
    def test_nonuniform_audio_image_mask_is_rejected(self):
        spatial = torch.zeros(1, 1, 64, 64)
        spatial[..., -1, -1] = 1.
        with self.assertRaisesRegex(ValueError, 'non-uniform image grid'):
            masks.normalize_masks(NestedTensor([torch.ones(1,8,10), spatial]), self.streams)
    def test_reported_shapes_without_large_latent_allocation(self):
        video = torch.zeros(1).expand(1,24,97,58,104)
        audio = torch.zeros(1).expand(1,32,2,520)
        video_mask = torch.ones(1).expand_as(video)
        audio_mask = torch.zeros(1,1,928,1664)
        out = masks.normalize_masks(NestedTensor([video_mask,audio_mask]), [video,audio])
        self.assertEqual(out[0].shape, video.shape)
        self.assertEqual(out[1].shape, (1,1,1,1))
        self.assertEqual(out[1].expand_as(audio).count_nonzero(), 0)
    def test_packed_mask(self):
        mv, ma = torch.rand_like(self.v), torch.rand_like(self.a)
        packed, _ = comfy.utils.pack_latents([mv, ma])
        out = masks.normalize_masks(packed, self.streams)
        torch.testing.assert_close(out[0], mv)
        torch.testing.assert_close(out[1], ma)
    def test_spatial_resize_keeps_channel_time_order(self):
        m = torch.arange(2*24*3).reshape(2,24,3,1,1).float().expand(2,24,3,4,5)
        out = masks.resize_spatial(m, (8, 10))
        torch.testing.assert_close(out[..., 0, 0], m[..., 0, 0])
    def test_invalid_time_not_silently_resampled(self):
        with self.assertRaisesRegex(ValueError, 'received.*latent shapes'):
            masks.normalize_masks(torch.ones(2,1,4,8,10), self.streams)
        with self.assertRaisesRegex(ValueError, 'audio is not time-resampled'):
            masks.normalize_masks(NestedTensor([torch.ones(1,8,10),torch.ones(1,8)]), self.streams)
    def test_invalid_channels_and_stream_count(self):
        with self.assertRaisesRegex(ValueError, 'axis 1'):
            masks.normalize_masks(torch.ones(2,12,3,8,10), self.streams)
        with self.assertRaisesRegex(ValueError, 'stream count'):
            masks.normalize_masks(NestedTensor([torch.ones(1,8,10)]), self.streams)
    def test_nan_and_clamping(self):
        with self.assertRaisesRegex(ValueError, 'NaN/Inf'):
            masks.normalize_masks(torch.full((1,8,10), float('nan')), self.streams)
        out = masks.normalize_masks(torch.full((1,8,10), 2.), self.streams)
        self.assertEqual(out[0].max(), 1.)
    def test_real_h3_audio_scale_in_anchor_sampler(self):
        inner = TinyH3()
        anchor, inner.latent_shapes = comfy.utils.pack_latents(self.streams)
        model_k = SimpleNamespace(inner_model=SimpleNamespace(inner_model=inner),latent_image=torch.ones_like(anchor))
        captured = []
        def original(k, x, sigmas, **kwargs):
            captured.append((k.latent_image.clone(), x.clone()))
            return x
        sampler = comfy.samplers.KSAMPLER(original)
        wrapped = avatar_sampling.anchored_sampler(sampler,anchor)
        initial_state = torch.full_like(anchor,17.)
        out = wrapped.sampler_function(model_k,initial_state,torch.tensor([.5,0.]))
        v,a = comfy.utils.unpack_latents(captured[0][0],inner.latent_shapes)
        torch.testing.assert_close(v,self.v)
        torch.testing.assert_close(a,self.a*3.)
        torch.testing.assert_close(out,initial_state)
    def test_soft_mask_not_reblended_at_output(self):
        out = masks.restore_kept([torch.ones(3)], [torch.zeros(3)], [torch.tensor([0., .5, 1.])])[0]
        torch.testing.assert_close(out, torch.tensor([0., 1., 1.]))
    def test_pure_anchor_with_mask_does_not_subtract_none(self):
        pix = torch.rand_like(self.v)
        out = selflift.artifact_aware_consistency_lift(None,pix,1.,1.,1.,mask=torch.zeros_like(self.v))
        self.assertIs(out, pix)
    def test_multichannel_risk(self):
        out = selflift.artifact_aware_consistency_lift(torch.zeros_like(self.v),torch.ones_like(self.v),.5,.5,1.,mask=torch.rand_like(self.v))
        self.assertEqual(out.shape, self.v.shape)
        self.assertTrue(torch.isfinite(out).all())
    def test_unique_registration(self):
        self.assertEqual(set(plugin.NODE_CLASS_MAPPINGS), {'SelfLiftAvatarH3Sampler','SelfLiftAvatarImageSampler','SelfLiftAvatarH3TST'})
        for cls in plugin.NODE_CLASS_MAPPINGS.values():
            self.assertEqual(cls.CATEGORY, 'selflift-Avatar')
            model_input = 'model' if cls is nodes.SelfLiftAvatarH3TST else 'low_res_model'
            self.assertIn(model_input, cls.INPUT_TYPES()['required'])

class StageModelInputTests(unittest.TestCase):
    def test_only_two_model_sockets_and_keyword_execution(self):
        low, high = Patcher(), Patcher()
        output = {'samples': torch.zeros(1)}
        for cls in (nodes.SelfLiftAvatarH3Sampler, nodes.SelfLiftAvatarImageSampler):
            for high_model in (None, high):
                with self.subTest(node=cls.__name__, separate_high=high_model is not None):
                    schema = cls.INPUT_TYPES()
                    self.assertEqual(schema['required']['low_res_model'][0], 'MODEL')
                    self.assertEqual(schema['optional']['high_res_model'][0], 'MODEL')
                    inputs = {**schema['required'], **schema['optional']}
                    self.assertEqual({k for k, v in inputs.items() if v[0] == 'MODEL'},
                                     {'low_res_model', 'high_res_model'})
                    self.assertNotIn('model', inputs)
                    self.assertNotIn('model_hires', inputs)
                    args = dict(low_res_model=low, positive=[], negative=[], vae=None,
                                latent_image=output, sampler=None, sigmas=None, seed=1, cfg=1.,
                                transition_step=2, lowres_scale=.5, rho=.5, w_min=.5, w_max=1.)
                    if cls is nodes.SelfLiftAvatarH3Sampler:
                        args['upscaler_model'] = 'none'
                    else:
                        args['latent_upsample'] = 'nearest'
                    if high_model is not None:
                        args['high_res_model'] = high_model
                    with patch.object(nodes, 'progressive_sample', return_value=output) as sample:
                        result = cls().sample(**args)
                    self.assertIs(sample.call_args.args[0], low)
                    self.assertIs(sample.call_args.kwargs['high_res_model'], high_model)
                    self.assertIs((result['result'] if isinstance(result, dict) else result)[0], output)

class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.v = torch.randn(2,24,3,8,8)
        self.a = torch.randn(2,32,2,7)
        self.stages = []
        self.stage_models = []
        self.stage_sigmas = []
        self.stage_shapes = []
        self.stage_devices = []
        self.audio_inputs = []
        self.mask_conds = []
        self.tile_inputs = []
    def sample(self, model, noise, positive, negative, cfg, device, sampler, sigmas, options,
               latent_image, callback, disable_pbar, seed):
        self.stage_models.append(model)
        self.stage_sigmas.append(sigmas.clone())
        self.stage_shapes.append(tuple(latent_image.unbind()[0].shape))
        self.stage_devices.append(device)
        raw, shapes = comfy.utils.pack_latents(latent_image.unbind())
        eps, _ = comfy.utils.pack_latents(noise.unbind())
        prepare_wrappers = comfy.patcher_extension.get_all_wrappers(
            comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING, options, is_model_options=True)
        if prepare_wrappers:
            # Exercise the real planner, forcing two tiles without querying GPU memory.
            def budget(patcher, noise_shape, conds, latent_shapes, regions, axis):
                minimum = 200 if len(regions) == 1 else 50
                return noise_shape, latent_shapes[0], 0, minimum, minimum
            with patch.object(h3_tiling, '_available_workspace', return_value=100),                  patch.object(h3_tiling, '_budget', side_effect=budget):
                prep = comfy.patcher_extension.WrapperExecutor.new_class_executor(
                    lambda *a, **kw: None, None, prepare_wrappers)
                prep.execute(model, raw.shape, {}, model_options=options)
        owner = self
        class Denoiser:
            def __init__(self, inner):
                self.inner_model = inner
                self.model_patcher = model
                self.cfg = cfg
            def __call__(self, x, sigma, model_options, seed):
                av = comfy.utils.unpack_latents(x,self.inner_model.latent_shapes)
                sigma_a = comfy.ldm.minimax.model.time_shift_sigma(sigma,self.inner_model.model_sampling.shift,
                                                                   self.inner_model.model_sampling.audio_shift)
                audio_seen = av[1] * (sigma_a / sigma).reshape(-1,1,1,1)
                owner.audio_inputs.append(audio_seen.clone())
                wrappers = comfy.patcher_extension.get_all_wrappers(
                    comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, model_options, is_model_options=True)
                if wrappers:
                    def predict_tile(streams, timestep, context, transformer_options, minimax_payload=None, **kw):
                        owner.tile_inputs.append((streams[0].shape, streams[1].clone(), kw["audio_denoise_mask"].clone()))
                        return [torch.full_like(streams[0], .125), torch.full_like(streams[1], .125)]
                    execute = comfy.patcher_extension.WrapperExecutor.new_class_executor(predict_tile, self, wrappers)
                    predicted = execute.execute([av[0], audio_seen], sigma * 1000,
                                                torch.zeros(x.shape[0], 2, 4), {}, **owner.mask_conds[-1])
                    return comfy.utils.pack_latents(predicted)[0]
                return torch.full_like(x,.125)
        def outer(noise, latent_image, stage_sampler, sigmas, denoise_mask=None,
                  callback=None, disable_pbar=False, seed=None, latent_shapes=None):
            inner = model.model
            inner.latent_shapes = latent_shapes
            native_conds = {} if denoise_mask is None else inner._denoise_mask_values(denoise_mask,latent_shapes)
            owner.mask_conds.append(native_conds)
            count = 0
            def report(i, x0, x, total):
                nonlocal count
                count += 1
                callback(i,NestedTensor(comfy.utils.unpack_latents(x0,latent_shapes)),
                         NestedTensor(comfy.utils.unpack_latents(x,latent_shapes)),total)
            sampled = stage_sampler.sample(Denoiser(inner),sigmas,{'model_options':options,'seed':seed},
                                           report,noise,latent_image=inner.process_latent_in(latent_image),
                                           denoise_mask=denoise_mask,disable_pbar=True)
            owner.stages.append(count)
            return inner.process_latent_out(sampled)
        wrappers = comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
                                                           options,is_model_options=True)
        executor = comfy.patcher_extension.WrapperExecutor.new_class_executor(outer,SimpleNamespace(model_patcher=model),wrappers)
        out = executor.execute(eps,raw,sampler,sigmas,None,callback,disable_pbar,seed,latent_shapes=shapes)
        return NestedTensor(comfy.utils.unpack_latents(out,shapes))
    def run_pipeline(self, mask, rho=0., tiling=False, low_res_model=None, **tiling_options):
        low_res_model = low_res_model if low_res_model is not None else Patcher()
        latent = {'samples':NestedTensor([self.v,self.a]), 'tag':'preserved'}
        if mask is not None:
            latent['noise_mask'] = mask
        def lift(z, vae, size, mode, lifter, need_lat, need_pix):
            out = torch.nn.functional.interpolate(z,size=(z.shape[2],*size),mode='nearest')
            return (out if need_lat else None, out + .03 if need_pix else None)
        with patch.object(nodes.comfy.samplers,'sample',side_effect=self.sample), \
             patch.object(nodes.latent_preview,'prepare_callback',return_value=lambda *a:None), \
             patch.object(nodes,'log_memory'), \
             patch.object(nodes.selflift,'paired_lifts',side_effect=lift):
            return nodes.progressive_sample(low_res_model,[],[],None,latent,comfy.samplers.sampler_object('euler'),
                torch.tensor([1.,.75,.5,.25,0.]),42,1.,2,.5,rho,1.,1.,'nearest',highres_tiling=tiling, **tiling_options)
    def test_separate_stage_models(self):
        low, high = Patcher(), Patcher()
        high.load_device = torch.device('cpu:0')
        self.run_pipeline(None, low_res_model=low, high_res_model=high)
        self.assertIs(self.stage_models[0], low)
        self.assertIs(self.stage_models[1], high)
        self.assertEqual(self.stage_devices, [low.load_device, high.load_device])
        self.assertEqual(self.stage_shapes, [(2,24,3,4,4), (2,24,3,8,8)])
        self.assertEqual(self.stages, [2,2])
        torch.testing.assert_close(self.stage_sigmas[0], torch.tensor([1.,.75,.5]))
        torch.testing.assert_close(self.stage_sigmas[1], torch.tensor([.5,.25,0.]))
        self.assertEqual(low.model_options, {})
        self.assertEqual(high.model_options, {})
    def test_low_model_is_high_fallback(self):
        low = Patcher()
        self.run_pipeline(None, low_res_model=low)
        self.assertEqual(self.stage_models, [low, low])
    def test_high_model_does_not_replace_low(self):
        high = Patcher()
        self.run_pipeline(None, high_res_model=high)
        self.assertIsNot(self.stage_models[0], high)
        self.assertIs(self.stage_models[1], high)
    def test_single_model_path(self):
        self.run_pipeline(None)
        self.assertIs(self.stage_models[0], self.stage_models[1])
    def test_separate_models_with_masks_and_tiling(self):
        low, high = Patcher(), Patcher()
        mask = NestedTensor([torch.ones_like(self.v), torch.zeros_like(self.a)])
        out = self.run_pipeline(mask, tiling=True, low_res_model=low, high_res_model=high)
        self.assertIs(self.stage_models[0].model, low.model)
        self.assertIs(self.stage_models[1].model, high.model)
        self.assertEqual(self.stages, [2,2])
        torch.testing.assert_close(out['samples'].unbind()[1], self.a, rtol=0, atol=0)
        self.assertEqual(low.model_options, {})
        self.assertEqual(high.model_options, {})
    def test_all_zero_av_is_preserved(self):
        out = self.run_pipeline(NestedTensor([torch.zeros_like(self.v),torch.zeros_like(self.a)]))
        v,a = out['samples'].unbind()
        torch.testing.assert_close(v,self.v,rtol=0,atol=0)
        torch.testing.assert_close(a,self.a,rtol=0,atol=0)
        self.assertEqual(self.stages,[2,2])
        self.assertEqual(out['tag'],'preserved')
    def test_prefix_and_audio_kept(self):
        mv = torch.ones_like(self.v); mv[:,:,:1] = 0
        ma = torch.ones_like(self.a); ma[...,:3] = 0
        out = self.run_pipeline(NestedTensor([mv,ma]))['samples'].unbind()
        torch.testing.assert_close(out[0][:,:,:1],self.v[:,:,:1],rtol=0,atol=0)
        torch.testing.assert_close(out[1][...,:3],self.a[...,:3],rtol=0,atol=0)
        self.assertFalse(torch.equal(out[0][:,:,1:],self.v[:,:,1:]))
        self.assertFalse(torch.equal(out[1][...,3:],self.a[...,3:]))
    def test_all_one_matches_unmasked(self):
        plain = self.run_pipeline(None)['samples'].unbind()
        masked = self.run_pipeline(NestedTensor([torch.ones_like(self.v),torch.ones_like(self.a)]))['samples'].unbind()
        for a,b in zip(plain,masked): torch.testing.assert_close(a,b)
    def test_video_only_mask_does_not_pin_audio(self):
        out = self.run_pipeline(torch.zeros(1,8,8))['samples'].unbind()
        torch.testing.assert_close(out[0],self.v,rtol=0,atol=0)
        self.assertFalse(torch.equal(out[1],self.a))
    def test_solid_audio_mask_keeps_audio_and_generates_video(self):
        mv = torch.ones_like(self.v)
        ma = torch.zeros(1,1,928,1664)
        out = self.run_pipeline(NestedTensor([mv,ma]))['samples'].unbind()
        torch.testing.assert_close(out[1],self.a,rtol=0,atol=0)
        self.assertFalse(torch.equal(out[0],self.v))
        self.assertEqual(self.stages,[2,2])
    def test_native_audio_condition_reaches_both_stages(self):
        self.run_pipeline(NestedTensor([torch.ones_like(self.v),torch.zeros_like(self.a)]))
        self.assertEqual(len(self.mask_conds),2)
        for cond in self.mask_conds:
            self.assertIn('audio_denoise_mask',cond)
            self.assertEqual(cond['audio_denoise_mask'].count_nonzero(),0)
        self.assertEqual(len(self.audio_inputs),4)
        for audio in self.audio_inputs:
            torch.testing.assert_close(audio,self.a,rtol=1e-5,atol=1e-5)
    def test_native_partial_audio_condition(self):
        ma=torch.ones_like(self.a); ma[...,:3]=0
        self.run_pipeline(NestedTensor([torch.ones_like(self.v),ma]))
        for cond in self.mask_conds:
            self.assertEqual(cond['audio_denoise_mask'][...,:3].count_nonzero(),0)
        for audio in self.audio_inputs:
            torch.testing.assert_close(audio[...,:3],self.a[...,:3],rtol=1e-5,atol=1e-5)
    def test_pure_anchor_pipeline(self):
        out = self.run_pipeline(NestedTensor([torch.zeros_like(self.v),torch.zeros_like(self.a)]),rho=1.)['samples'].unbind()
        for a,b in zip(out,[self.v,self.a]): torch.testing.assert_close(a,b,rtol=0,atol=0)
    def test_tiling_full_audio_keep_reaches_every_tile(self):
        self.v = torch.randn(2,24,3,12,24)
        original_video, original_audio = self.v.clone(), self.a.clone()
        result = self.run_pipeline(NestedTensor([torch.ones_like(self.v), torch.zeros(1,1,64,64)]),tiling=True)
        v, a = result['samples'].unbind()
        torch.testing.assert_close(a, self.a, rtol=0, atol=0)
        self.assertFalse(torch.equal(v, self.v))
        self.assertEqual(self.stages, [2,2])
        self.assertEqual(len(self.tile_inputs), 4)  # 2 spatial tiles x 2 high-resolution steps
        for shape, audio, audio_mask in self.tile_inputs:
            self.assertLess(shape[-1], self.v.shape[-1])
            self.assertEqual(shape[2], self.v.shape[2])
            torch.testing.assert_close(audio, self.a, rtol=1e-5, atol=1e-5)
            self.assertEqual(audio_mask.count_nonzero(), 0)
            self.assertEqual(audio_mask.shape[-1], self.a.shape[-1])
        torch.testing.assert_close(self.v, original_video)
        torch.testing.assert_close(self.a, original_audio)
    def test_tiling_explicitly_rejected(self):
        for vm, am in [(1.,1.), (0.,0.), (.5,0.), (1.,.5)]:
            with self.subTest(video_mask=vm, audio_mask=am):
                with self.assertRaisesRegex(ValueError,'highres_tiling'):
                    self.run_pipeline(NestedTensor([torch.full((1,1,1,1,1),vm),
                                                    torch.full((1,1,1,1),am)]),tiling=True)
    def test_manual_height_pipeline(self):
        self.v = torch.randn(2,24,3,24,36)
        reports=[]
        result=self.run_pipeline(NestedTensor([torch.ones_like(self.v),torch.zeros_like(self.a)]),
                                 tiling=True,tiling_mode="manual",tiling_tiles=3,tiling_axis="height",
                                 on_tiling_plan=reports.append)
        self.assertEqual(reports[-1]['axis'],3)
        self.assertEqual(reports[-1]['tiles'],3)
        self.assertEqual(len(self.tile_inputs),6)
        for shape,audio,mask in self.tile_inputs:
            self.assertLess(shape[-2],24)
            self.assertEqual(shape[-1],36)
            torch.testing.assert_close(audio,self.a,rtol=1e-5,atol=1e-5)
        torch.testing.assert_close(result['samples'].unbind()[1],self.a,rtol=0,atol=0)
    def test_tiling_partial_audio_rejected(self):
        am = torch.zeros_like(self.a); am[...,-1] = 1
        with self.assertRaisesRegex(ValueError,'partial audio'):
            self.run_pipeline(NestedTensor([torch.ones_like(self.v),am]),tiling=True)
    def test_global_mode_accepts_partial_masks_and_reports_plan(self):
        # Spatial tiling rejects this; full-sequence global context must accept it.
        am = torch.zeros_like(self.a); am[...,-1] = 1
        reports=[]
        fallback_model = Patcher()
        fallback_model._selflift_global_fallback = "attention_override"
        with patch.object(nodes, '_global_tiled_model',
                          side_effect=lambda model, chunk, mode, cache: fallback_model) as global_model:
            result = self.run_pipeline(
                NestedTensor([torch.ones_like(self.v), am]), tiling=True,
                highres_tiling_mode="global_stream", highres_token_chunk=2048,
                highres_weight_cache_gb=2.0, on_tiling_plan=reports.append)
        self.assertEqual(global_model.call_args.args[1:], (2048, "global_stream", 2.0))
        self.assertEqual(reports[-1]["kind"], "global")
        self.assertEqual(reports[-1]["mode"], "global_stream")
        self.assertEqual(reports[-1]["token_chunk"], 2048)
        self.assertEqual(reports[-1]["fallback"], "attention_override")
        self.assertTrue(reports[-1]["full_sequence"])
        self.assertEqual(self.stages, [2,2])
        self.assertEqual(self.tile_inputs, [])
        self.assertIn("samples", result)
    def test_global_mode_accepts_video_preservation(self):
        mv = torch.ones_like(self.v); mv[:,:,:1] = 0
        with patch.object(nodes, '_global_tiled_model',
                          side_effect=lambda model, chunk, mode, cache: model):
            self.run_pipeline(NestedTensor([mv, torch.zeros_like(self.a)]), tiling=True,
                              highres_tiling_mode="global")
    def test_global_mode_validates_chunk_and_cache(self):
        for kwargs, pattern in [
            ({"highres_tiling_mode":"global", "highres_token_chunk":64}, "highres_token_chunk"),
            ({"highres_tiling_mode":"global", "highres_weight_cache_gb":-1}, "highres_weight_cache_gb"),
            ({"highres_tiling_mode":"unknown"}, "highres_tiling_mode"),
        ]:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, pattern):
                    self.run_pipeline(None, tiling=True, **kwargs)


class GlobalWrapperTests(unittest.TestCase):
    def _patcher(self):
        p = Patcher()
        p.model.diffusion_model.blocks = torch.nn.ModuleList([torch.nn.Module()])
        return p

    def test_global_dense_wrapper_builds_without_native_sparse(self):
        patched = h3_global_tiling.tiled_model(self._patcher(), 128, use_stream=False)
        self.assertIsInstance(patched, Patcher)
        self.assertTrue(patched.model_options)

    def test_global_stream_requires_native_sparse(self):
        with self.assertRaisesRegex(ValueError, 'native Model Sparse Attention'):
            h3_global_tiling.tiled_model(self._patcher(), 128, use_stream=True)

    def test_unknown_attention_override_falls_back_to_full_sequence(self):
        patcher = self._patcher()
        patcher.model_options.setdefault('transformer_options', {})['optimized_attention_override'] = object()
        self.assertTrue(h3_global_tiling._has_unknown_attention_override(
            patcher.model_options['transformer_options']))
        with self.assertLogs(level='WARNING') as logs:
            patched = h3_global_tiling.tiled_model(patcher, 128, use_stream=True)
        self.assertIsInstance(patched, Patcher)
        self.assertEqual(patched._selflift_global_fallback, 'attention_override')
        self.assertTrue(any('Falling back to the full-sequence global path' in line for line in logs.output))
        self.assertEqual(h3_global_tiling._sparse_specs(
            patcher.model.diffusion_model,
            patcher.model_options['transformer_options'],
            force_dense_on_unknown=True), {})

    def test_backend_override_is_removed_for_full_sequence_fallback(self):
        def backend_override(_func, *args, **kwargs):
            raise AssertionError('backend override must not run in fallback')
        backend_override.__qualname__ = (
            'ModelPatcher.set_model_optimized_attention.<locals>.optimized_attention_override'
        )
        self.assertTrue(h3_global_tiling._is_backend_attention_override(backend_override))
        patcher = self._patcher()
        patcher.model_options.setdefault('transformer_options', {})['optimized_attention_override'] = backend_override
        with self.assertLogs(level='WARNING') as logs:
            patched = h3_global_tiling.tiled_model(patcher, 128, use_stream=True)
        self.assertEqual(patched._selflift_global_fallback, 'attention_backend')
        self.assertTrue(any('dense attention backend override detected' in line for line in logs.output))

    def test_forward_removes_backend_override_only(self):
        def backend_override(_func, *args, **kwargs):
            return None
        backend_override.__qualname__ = (
            'ModelPatcher.set_model_optimized_attention.<locals>.optimized_attention_override'
        )
        video = torch.randn(1, 24, 3, 8, 8)
        audio = torch.randn(1, 32, 2, 7)
        seen = {}
        class Network:
            blocks = torch.nn.ModuleList([])
        class Executor:
            class_obj = Network()
            def __call__(self, streams, timestep, context, options, minimax_payload=None, **kwargs):
                seen['options'] = options
                return streams
        h3_global_tiling._forward(
            Executor(), [video, audio], torch.tensor([500.]), torch.zeros(1, 2, 4),
            {'optimized_attention_override': backend_override},
        )
        self.assertNotIn('optimized_attention_override', seen['options'])

    def test_chunked_block_uses_full_attention_override_path(self):
        class FakeAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.seen_tokens = None
                self.seen_options = None
            def forward(self, x, rope_freqs=None, transformer_options=None):
                self.seen_tokens = x.shape[0]
                self.seen_options = transformer_options
                return x.clone()

        class FakeBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = FakeAttention()
                self.norm1 = torch.nn.Identity()
                self.norm2 = torch.nn.Identity()
                self.mlp = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.GELU(), torch.nn.Linear(8, 8))
            def adaln_proj(self, _t_emb):
                zero = torch.zeros(1, dtype=torch.float32)
                one = torch.ones(1, dtype=torch.float32)
                return zero, one, one, zero, one, one

        block = FakeBlock()
        x = torch.randn(11, 8)
        override = object()
        with torch.inference_mode():
            out = h3_global_tiling.chunked_block(
                block, x, None, [(0, 11, 0)], None,
                {'optimized_attention_override': override}, 4)
        self.assertEqual(block.attn.seen_tokens, 11)
        self.assertIs(block.attn.seen_options['optimized_attention_override'], override)
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.isfinite(out).all())

    def test_mode_to_wrapper_flags(self):
        model = self._patcher()
        cases = [('global_stream', False, True), ('global', False, False), ('global_ram', True, False)]
        for mode, use_ram, use_stream in cases:
            with self.subTest(mode=mode):
                with patch.object(h3_global_tiling, 'tiled_model') as build:
                    nodes._global_tiled_model(model, 2048, mode, 2.5)
                build.assert_called_once_with(
                    model, 2048, use_ram=use_ram, use_stream=use_stream, weight_cache_gb=2.5
                )

    def test_global_forward_keeps_full_streams_and_masks(self):
        video = torch.randn(1, 24, 3, 8, 8)
        audio = torch.randn(1, 32, 2, 7)
        vmask = torch.ones_like(video)
        amask = torch.zeros_like(audio)
        seen = {}

        class Network:
            blocks = torch.nn.ModuleList([torch.nn.Module()])

        class Executor:
            class_obj = Network()

            def __call__(self, streams, timestep, context, options, minimax_payload=None, **kwargs):
                seen['video_shape'] = tuple(streams[0].shape)
                seen['audio_shape'] = tuple(streams[1].shape)
                seen['video_mask'] = kwargs['denoise_mask'].clone()
                seen['audio_mask'] = kwargs['audio_denoise_mask'].clone()
                seen['layout'] = minimax_payload.get('layout')
                return streams

        out = h3_global_tiling._forward(
            Executor(), [video, audio], torch.tensor([500.]), torch.zeros(1, 2, 4), {},
            minimax_payload={'layout': 'full-layout'},
            denoise_mask=vmask, audio_denoise_mask=amask)
        self.assertTrue(torch.equal(out[0], video))
        self.assertTrue(torch.equal(out[1], audio))
        self.assertEqual(seen['video_shape'], tuple(video.shape))
        self.assertEqual(seen['audio_shape'], tuple(audio.shape))
        self.assertEqual(seen['layout'], 'full-layout')
        torch.testing.assert_close(seen['video_mask'], vmask)
        torch.testing.assert_close(seen['audio_mask'], amask)


class GlobalChunkingTests(unittest.TestCase):
    class _Attn(torch.nn.Module):
        def __init__(self, dim, heads):
            super().__init__()
            self.heads = heads
            self.head_dim = dim // heads
            self.qkv_proj = torch.nn.Linear(dim, 3 * dim, bias=False)
            self.q_norm = torch.nn.RMSNorm(self.head_dim)
            self.k_norm = torch.nn.RMSNorm(self.head_dim)
            self.out_proj = torch.nn.Linear(dim, dim, bias=False)

    class _Block(torch.nn.Module):
        def __init__(self, dim, heads):
            super().__init__()
            self.attn = GlobalChunkingTests._Attn(dim, heads)
            self.norm1 = torch.nn.Identity()
            self.norm2 = torch.nn.Identity()
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(dim, dim), torch.nn.GELU(), torch.nn.Linear(dim, dim)
            )

        def adaln_proj(self, _t_emb):
            zero = torch.zeros(1, dtype=torch.float32)
            one = torch.ones(1, dtype=torch.float32)
            return zero, one, one, zero, one, one

    def test_chunked_dense_block_matches_full_sequence(self):
        torch.manual_seed(7)
        dim, heads, tokens = 8, 2, 17
        base = self._Block(dim, heads)
        x = torch.randn(tokens, dim)
        full = copy.deepcopy(base)
        chunked = copy.deepcopy(base)
        x_full, x_chunked = x.clone(), x.clone()
        with torch.inference_mode():
            h3_global_tiling.chunked_block(full, x_full, None, [(0, tokens, 0)], None, {}, tokens)
            h3_global_tiling.chunked_block(chunked, x_chunked, None, [(0, tokens, 0)], None, {}, 4)
        torch.testing.assert_close(x_chunked, x_full, rtol=1e-5, atol=1e-6)

    def test_per_token_segments_are_sliced_with_chunks(self):
        rows = torch.arange(8).reshape(8, 1)
        segments = h3_global_tiling._segments_slice([(0, 8, rows)], 2, 6)
        self.assertEqual(len(segments), 1)
        start, end, sliced = segments[0]
        self.assertEqual((start, end), (0, 4))
        torch.testing.assert_close(sliced.flatten(), torch.arange(2, 6))

    def test_input_projection_chunking_matches_full(self):
        torch.manual_seed(11)
        linear = torch.nn.Linear(8, 4)
        x = torch.randn(17, 8)
        full = linear(x)
        chunked = h3_output_chunking.chunked_projection(linear.forward, 4, x)
        torch.testing.assert_close(chunked, full, rtol=1e-6, atol=1e-7)

    def test_projection_chunking_installation_is_restored(self):
        linear = torch.nn.Linear(8, 4)
        network = SimpleNamespace(video_patch_proj=linear, audio_patch_proj=None)
        original_func = linear.forward.__func__
        installed = h3_output_chunking._install_projection_chunking(network, 4)
        self.assertTrue(installed)
        self.assertIn("forward", linear.__dict__)
        h3_output_chunking._restore_projection_chunking(installed)
        self.assertNotIn("forward", linear.__dict__)
        self.assertIs(linear.forward.__func__, original_func)

    def test_output_chunking_without_native_embed_method(self):
        class Final:
            def forward(self, *args, **kwargs):
                return args[0]
        class Network:
            video_patch_proj = torch.nn.Linear(8, 8)
            audio_patch_proj = torch.nn.Linear(8, 8)
            def __init__(self):
                self.final_layer = Final()
        network = Network()
        with h3_output_chunking.output_chunking(network, 4):
            self.assertIn("forward", network.final_layer.__dict__)
        self.assertNotIn("forward", network.final_layer.__dict__)


class TilingTests(unittest.TestCase):
    def test_mask_gate(self):
        masks.validate_tiling_masks(None)
        masks.validate_tiling_masks([torch.ones(1,1,1,1,1),torch.zeros(1,1,1,1)])
        with self.assertRaises(ValueError):
            masks.validate_tiling_masks([torch.ones(1,1,1,1,1)])
    def test_overlap_stitch_full_audio_and_positions(self):
        for height, width in [(12,33),(33,12)]:
            for tiles in [1,2,4]:
                with self.subTest(height=height,width=width,tiles=tiles):
                    v = torch.randn(1,24,3,height,width)
                    a = torch.randn(1,32,2,7)
                    am = torch.zeros(1,1,2,7)
                    vm = torch.ones(1,1,3,height,width)
                    calls=[]
                    def predict(streams,timestep,context,transformer_options,minimax_payload=None,**kwargs):
                        calls.append(streams[0].shape)
                        self.assertIs(streams[1],a)
                        self.assertIs(kwargs['audio_denoise_mask'],am)
                        self.assertIsNone(kwargs.get('denoise_mask'))
                        if tiles > 1:
                            shape = streams[0].shape
                            signature = minimax_payload['layout'].signature
                            self.assertEqual(tuple(signature[1:]),(3,(shape[-2]+1)//2*2,(shape[-1]+1)//2*2,7))
                        return [streams[0]*2,streams[1]*3]
                    out = h3_tiling._tiled_forward(predict,[v,a],torch.tensor([500.]),torch.zeros(1,2,4),{},
                                                  plan={'tiles':tiles},denoise_mask=vm,audio_denoise_mask=am)
                    torch.testing.assert_close(out[0],v*2)
                    torch.testing.assert_close(out[1],a*3)
                    self.assertEqual(len(calls),tiles)
                    self.assertEqual(vm.min(),1.)
    def test_forward_rejects_video_and_soft_audio_masks(self):
        v=torch.zeros(1,24,3,12,24);a=torch.zeros(1,32,2,7)
        for kwargs in [{'denoise_mask':torch.zeros(1,1,3,12,24)},
                       {'audio_denoise_mask':torch.full((1,1,2,7),.5)}]:
            with self.assertRaises(ValueError):
                h3_tiling._tiled_forward(None,[v,a],torch.tensor([500.]),torch.zeros(1,2,4),{},**kwargs)
    def test_unmasked_forward_still_works(self):
        v=torch.ones(1,24,3,12,24);a=torch.ones(1,32,2,7)
        out=h3_tiling._tiled_forward(lambda streams,*args,**kwargs:streams,[v,a],torch.tensor([500.]),
                                    torch.zeros(1,2,4),{},plan={'tiles':2})
        torch.testing.assert_close(out[0],v)
        torch.testing.assert_close(out[1],a)

class TilingWorkspaceTests(unittest.TestCase):
    def workspace(self, loaded, free=100, total=1000):
        model = SimpleNamespace(load_device='cuda:0', model_size=lambda: 50)
        manager = h3_tiling.comfy.model_management
        with patch.object(manager, 'loaded_models', return_value=loaded), \
             patch.object(manager, 'get_free_memory', return_value=free), \
             patch.object(manager, 'get_total_memory', return_value=total), \
             patch.object(manager, 'minimum_inference_memory', return_value=10), \
             patch.object(manager, 'MIN_WEIGHT_MEMORY_RATIO', 0.5):
            return h3_tiling._available_workspace(model)

    def test_none_entries_are_skipped(self):
        self.assertEqual(self.workspace([None, None]), 40)

    def test_expired_loaded_model_is_skipped(self):
        patcher = Patcher()
        patcher.parent = None
        loaded = h3_tiling.comfy.model_management.LoadedModel(patcher)
        del patcher
        self.assertIsNone(loaded.model)
        self.assertEqual(self.workspace([loaded]), 40)

    def test_live_models_deduplicated_and_other_devices_excluded(self):
        patcher = SimpleNamespace(model=object(), load_device='cuda:0', loaded_size=lambda: 200)
        clone = SimpleNamespace(model=patcher.model, load_device='cuda:0', loaded_size=lambda: 200)
        other = SimpleNamespace(model=object(), load_device='cuda:1', loaded_size=lambda: 500)
        for entries in ([None, patcher, clone, other, None],
                        [SimpleNamespace(model=None), SimpleNamespace(model=patcher), clone, other]):
            with self.subTest(entries=entries):
                self.assertEqual(self.workspace(entries), 240)

    def test_pool_capped_and_workspace_nonnegative(self):
        patcher = SimpleNamespace(model=object(), load_device='cuda:0', loaded_size=lambda: 2000)
        self.assertEqual(self.workspace([None, patcher]), 940)
        self.assertEqual(self.workspace([None], free=0), 0)


class TilingControlTests(unittest.TestCase):
    def plan(self, mode="auto", tiles=2, axis="auto", shape=(1,24,3,12,24), available=100):
        audio=(1,32,2,7)
        config={'mode':mode,'requested_tiles':tiles,'axis_choice':axis}
        reported=[]
        def budget(model,noise_shape,conds,shapes,regions,dimension):
            tile=list(shapes[0]); tile[dimension]=max(b-a for a,b in regions)
            minimum=240/len(regions)
            return noise_shape,tuple(tile),0,minimum*2,minimum
        with patch.object(h3_tiling,'_available_workspace',return_value=available),              patch.object(h3_tiling,'_budget',side_effect=budget):
            h3_tiling._prepare_tiled_sampling(lambda *a,**kw:None,Patcher(),
                (shape[0],1,torch.tensor(shape[1:]).prod().item()+32*2*7),{},
                latent_shapes=[shape,audio],plan=config,on_plan=reported.append)
        self.assertEqual(reported[-1],config)
        return config
    def test_auto_uses_first_fitting_count(self):
        plan=self.plan()
        self.assertEqual(plan['tiles'],3)
        self.assertEqual(plan['axis'],4)
        self.assertTrue(plan['estimate_fits'])
    def test_manual_count_not_increased_when_insufficient(self):
        plan=self.plan(mode="manual",tiles=2,axis="height",available=1)
        self.assertEqual(plan['tiles'],2)
        self.assertEqual(plan['axis'],3)
        self.assertFalse(plan['estimate_fits'])
    def test_manual_one_is_full_frame(self):
        plan=self.plan(mode="manual",tiles=1)
        self.assertEqual(plan['regions'],[(0,24)])
    def test_tiny_size_reports_actual_count(self):
        plan=self.plan(mode="manual",tiles=8,axis="height",shape=(1,24,3,4,40))
        self.assertEqual(plan['requested_tiles'],8)
        self.assertEqual(plan['tiles'],1)
    def test_auto_axis_can_be_overridden(self):
        self.assertEqual(self.plan(axis="height")['axis'],3)
    def test_no_fit_reports_warning_at_max(self):
        plan=self.plan(shape=(1,24,3,68,120),available=1)
        self.assertEqual(plan['tiles'],8)
        self.assertFalse(plan['estimate_fits'])
    def test_invalid_options(self):
        for mode,n,axis in [('other',2,'auto'),('manual',9,'auto'),('manual',0,'auto'),('manual',2,'time')]:
            with self.assertRaises(ValueError):
                h3_tiling.validate_options(mode,n,axis)
    def test_report_displays_estimates_and_overlap(self):
        text=nodes.format_plan(self.plan(mode="manual",tiles=4,shape=(1,24,107,68,120)))
        self.assertIn('[22,68)',text)
        self.assertIn('[16, 16, 16]',text)
        self.assertIn('非实测峰值',text)
    def test_schema_backward_defaults(self):
        schema=nodes.SelfLiftAvatarH3Sampler.INPUT_TYPES()
        self.assertEqual(schema['optional']['tiling_mode'][1]['default'],'auto')
        self.assertEqual(schema['optional']['tiling_tiles'][1]['default'],2)
        self.assertEqual(schema['optional']['tiling_tiles'][0],[2,4,6,8])
        self.assertTrue(all(type(n) is int for n in schema['optional']['tiling_tiles'][0]))
        self.assertEqual(schema['optional']['tiling_axis'][1]['default'],'auto')
        self.assertEqual(schema['optional']['highres_tiling_mode'][1]['default'],'spatial')
        self.assertEqual(schema['optional']['highres_tiling_mode'][0],
                         ['spatial','global_stream','global','global_ram'])
        self.assertEqual(schema['optional']['highres_token_chunk'][1]['default'],4096)
        self.assertEqual(schema['optional']['highres_weight_cache_gb'][1]['default'],4.0)
        self.assertEqual(schema['hidden']['unique_id'],'UNIQUE_ID')
    def test_global_plan_text(self):
        for mode in ("global_stream","global","global_ram"):
            with self.subTest(mode=mode):
                text=nodes.format_plan({"kind":"global","mode":mode,"token_chunk":4096,"weight_cache_gb":4.0})
                self.assertIn('Full sequence',text)
                self.assertIn(mode,text)
                self.assertIn('full-sequence video/audio masks',text)
        fallback=nodes.format_plan({"kind":"global","mode":"global_stream","token_chunk":4096,
                                    "weight_cache_gb":4.0,"fallback":"attention_override"})
        self.assertIn('Fallback',fallback)
        backend=nodes.format_plan({"kind":"global","mode":"global_stream","token_chunk":4096,
                                   "weight_cache_gb":4.0,"fallback":"attention_backend"})
        self.assertIn('backend override removed',backend)
    def test_node_global_options_passed(self):
        output={'samples':torch.zeros(1)}
        def sample(*a,**kw):
            kw['on_tiling_plan']({'kind':'global','mode':'global_stream','token_chunk':2048,
                                  'weight_cache_gb':2.0,'full_sequence':True})
            return output
        with patch.object(nodes,'progressive_sample',side_effect=sample) as sample_call:
            result=nodes.SelfLiftAvatarH3Sampler().sample(
                None,[],[],None,output,None,None,1,1.,2,.5,.5,.5,1.,'none',
                highres_tiling=True,highres_tiling_mode='global_stream',
                highres_token_chunk=2048,highres_weight_cache_gb=2.0)
        kwargs=sample_call.call_args.kwargs
        self.assertEqual(kwargs['highres_tiling_mode'],'global_stream')
        self.assertEqual(kwargs['highres_token_chunk'],2048)
        self.assertEqual(kwargs['highres_weight_cache_gb'],2.0)
        self.assertIn('Full sequence',result['ui']['selflift_tiling'][0])
        self.assertIn('full-sequence video/audio masks',result['ui']['selflift_tiling'][0])
    def test_node_returns_latent_and_ui_with_default_options(self):
        output={'samples':torch.zeros(1)}
        with patch.object(nodes,'progressive_sample',return_value=output) as sample:
            result=nodes.SelfLiftAvatarH3Sampler().sample(None,[],[],None,output,None,None,1,1.,2,.5,.5,.5,1.,'none')
        self.assertIs(result['result'][0],output)
        self.assertIn('OFF',result['ui']['selflift_tiling'][0])
        self.assertEqual(sample.call_args.kwargs['tiling_mode'],'auto')
    def test_node_plan_event_and_final_ui(self):
        plan=self.plan(mode="manual",tiles=2)
        events=[]
        server=SimpleNamespace(client_id='test-client',send_sync=lambda *a:events.append(a))
        def sample(*a,**kw):
            kw['on_tiling_plan'](plan)
            return {'samples':torch.zeros(1)}
        with patch.object(nodes.PromptServer,'instance',server,create=True),              patch.object(nodes,'progressive_sample',side_effect=sample):
            result=nodes.SelfLiftAvatarH3Sampler().sample(None,[],[],None,{},None,None,1,1.,2,.5,.5,.5,1.,'none',
                highres_tiling=True,tiling_mode='manual',tiling_tiles=2,unique_id='78')
        self.assertIn('Effective',result['ui']['selflift_tiling'][0])
        self.assertEqual(events[-1][0],'selflift-avatar-tiling')
        self.assertEqual(events[-1][1]['node_id'],'78')
        self.assertEqual(events[-1][2],'test-client')
        self.assertEqual(events[-1][1]['plan']['tiles'],2)
        self.assertEqual(result['ui']['selflift_tiling_plan'][0]['tiles'],2)
    def test_node_failure_event_not_success(self):
        events=[]
        server=SimpleNamespace(client_id='test-client',send_sync=lambda *a:events.append(a))
        with patch.object(nodes.PromptServer,'instance',server,create=True),              patch.object(nodes,'progressive_sample',side_effect=ValueError('test failure')):
            with self.assertRaisesRegex(ValueError,'test failure'):
                nodes.SelfLiftAvatarH3Sampler().sample(None,[],[],None,{},None,None,1,1.,2,.5,.5,.5,1.,'none',unique_id='78')
        self.assertIn('FAILED',events[-1][1]['text'])

if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]], verbosity=2)
