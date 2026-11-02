"""SelfLift progressive-resolution sampler nodes.

Run the first sampling steps on a spatially downscaled latent, lift the clean
endpoint to the target resolution with the training-free Artifact-Aware
Consistency Lift (arXiv:2609.02036), re-noise it at the transition sigma, and
finish the schedule at full resolution.

Two front-ends share the same engine:
- SelfLiftAvatarH3Sampler: MiniMax H3 audio-video (nested AV latents; the audio stream
  has no spatial dimensions and continues through the reused Euler boundary step).
  This is an experimental extension beyond the paper's image-model evaluation.
- SelfLiftAvatarImageSampler: compatible rectified-flow image models.
"""

import copy
import logging
import os
import time

import torch

import comfy.k_diffusion.sampling
import comfy.model_management
import comfy.model_patcher
import comfy.model_sampling
import comfy.nested_tensor
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
from server import PromptServer

from . import selflift
from . import avatar_masks
from .avatar_sampling import native_mask_model
from . import h3_upscaler
from . import h3_tiling
from . import spatial_anchor as spatial_anchor_utils
from .tiling_ui import format_plan
from . import h3_tst
from .diagnostics import log_memory

_HIGHRES_TILING_MODES = ("spatial", "global_stream", "global", "global_ram")
_GLOBAL_TILING_MODES = frozenset(("global_stream", "global", "global_ram"))


def _global_tiled_model(model, chunk_size, mode, weight_cache_gb):
    """Build the full-sequence H3 memory-saving wrapper.

    Imported lazily so the existing spatial path still loads on installations
    without the optional comfy_kitchen/native-sparse dependencies.
    """
    try:
        from . import h3_global_tiling
    except Exception as exc:
        raise RuntimeError(
            "selflift-Avatar: global high-resolution mode is unavailable; "
            "check ComfyUI/native sparse dependencies"
        ) from exc
    return h3_global_tiling.tiled_model(
        model, chunk_size,
        use_ram=mode == "global_ram",
        use_stream=mode == "global_stream",
        weight_cache_gb=weight_cache_gb,
    )


class _StageTimer:
    def __init__(self, stage, device, resolution=None, pixel_scale=1):
        self.stage = stage
        self.resolution = resolution
        self.pixel_scale = pixel_scale
        self.device = torch.device(device)
        self.synchronize = os.environ.get("SELFLIFT_AVATAR_TIMING_SYNC", "0") == "1" and self.device.type == "cuda"
        self.started = self.previous = self._now()
        timing_mode = "CUDA-synchronized wall time" if self.synchronize else "wall time; no forced CUDA sync"
        suffix = ""
        if resolution is not None:
            pixels = tuple(int(value * self.pixel_scale) for value in resolution)
            suffix = f" latent={resolution} pixels={pixels}"
        logging.info("[selflift-Avatar timing] %s start (%s)%s", stage, timing_mode, suffix)

    def _now(self):
        if self.synchronize:
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def mark(self, label):
        current = self._now()
        logging.info("[selflift-Avatar timing] %s %s: %.3fs", self.stage, label, current - self.previous)
        self.previous = current

    def finish(self):
        current = self._now()
        logging.info("[selflift-Avatar timing] %s total: %.3fs; after last mark: %.3fs",
                     self.stage, current - self.started, current - self.previous)


def _upscaler_input():
    models = ["none"] + h3_upscaler.list_upscaler_models()
    h3_models = [name for name in models[1:] if "h3" in name.lower()]
    default = h3_models[0] if h3_models else "none"
    return (models, {"default": default,
                     "tooltip": "External H3 latent upscaler (models/latent_upscale_models). The first detected H3 model is selected by default; 'none' uses nearest-neighbor lifting."})


def _streams(samples):
    if samples.is_nested:
        return list(samples.unbind()), True
    return [samples], False


def _pack(streams, nested):
    if nested:
        return comfy.nested_tensor.NestedTensor(streams)
    return streams[0]


def _validate_sampling(model_sampling, sampler):
    if not isinstance(model_sampling, comfy.model_sampling.CONST):
        raise ValueError("SelfLift requires a rectified-flow model")
    if not isinstance(sampler, comfy.samplers.KSAMPLER) or sampler.sampler_function is not comfy.k_diffusion.sampling.sample_euler:
        raise ValueError("SelfLift requires the standard Euler sampler")
    if sampler.extra_options.get("s_churn", 0.0) != 0.0:
        raise ValueError("SelfLift requires Euler with s_churn=0")


def _validate_schedule(sigmas, transition_step):
    if sigmas.ndim != 1 or not sigmas.is_floating_point():
        raise ValueError("selflift-Avatar: sigmas must be a one-dimensional floating-point tensor")
    if not torch.isfinite(sigmas).all() or (sigmas < 0).any():
        raise ValueError("selflift-Avatar: sigmas must be finite and nonnegative")
    if sigmas.numel() < 2:
        return
    if not isinstance(transition_step, int) or not 1 <= transition_step <= sigmas.numel() - 2:
        raise ValueError(f"selflift-Avatar: transition_step {transition_step} out of range for {sigmas.numel() - 1} steps")
    if (sigmas[1:] > sigmas[:-1]).any():
        raise ValueError("selflift-Avatar: sigmas must be non-increasing")
    if (sigmas[:-1] <= 0).any():
        raise ValueError("selflift-Avatar: only the final sigma may be zero")
    if sigmas[transition_step] >= 1:
        raise ValueError("selflift-Avatar: the high-resolution starting sigma must be less than 1")


def _validate_latent_input(latent_image):
    streams, _ = _streams(latent_image["samples"])
    if not streams or streams[0].ndim not in (4, 5):
        raise ValueError("selflift-Avatar: expected an image or video latent")
    for stream in streams:
        if stream.ndim == 0 or any(size == 0 for size in stream.shape) or stream.shape[0] != streams[0].shape[0]:
            raise ValueError("selflift-Avatar: latent streams must be nonempty with matching batches")
    return avatar_masks.normalize_masks(latent_image.get("noise_mask"), streams)


def _euler_step(state, denoised, sigma, sigma_next):
    step = ((sigma_next - sigma) / sigma).to(device=state.device, dtype=state.dtype)
    return state + (state - denoised.to(state)) * step


def _resize_keyframes(cond, h, w):
    """Keyframe cond latents share the generation grid; resize them to the low-res one."""
    out = []
    for tensor, d in cond:
        kfs = d.get("minimax_keyframes")
        if kfs is None:
            out.append((tensor, d))
            continue
        d = d.copy()
        resized = []
        for kf in kfs:
            kf = dict(kf)
            lat = kf.get("latent")
            if lat is not None and (lat.shape[-2] != h or lat.shape[-1] != w):
                if lat.ndim == 5:
                    # spatial-only resize per frame; never interpolates across time
                    batch, channels, frames = lat.shape[:3]
                    resized_latent = torch.nn.functional.interpolate(
                        lat.float().permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, lat.shape[-2], lat.shape[-1]),
                        size=(h, w), mode="bilinear", align_corners=False
                    )
                    resized_latent = resized_latent.reshape(batch, frames, channels, h, w).permute(0, 2, 1, 3, 4)
                else:
                    resized_latent = torch.nn.functional.interpolate(
                        lat.float(), size=(h, w), mode="bilinear", align_corners=False
                    )
                # per-channel, per-frame mean match: fixes bilinear's color drift without
                # restoring variance the low-res grid cannot represent
                source_mean = lat.float().mean(dim=(-2, -1), keepdim=True)
                resized_mean = resized_latent.mean(dim=(-2, -1), keepdim=True)
                kf["latent"] = (resized_latent + (source_mean - resized_mean)).to(lat)
            resized.append(kf)
        d["minimax_keyframes"] = resized
        out.append((tensor, d))
    return out


def _debug_dump(vae, latents):
    """Decode transition intermediates to PNGs when SELFLIFT_AVATAR_DEBUG=1."""
    if os.environ.get("SELFLIFT_AVATAR_DEBUG", "0") != "1":
        return
    out_dir = os.path.join(os.path.dirname(__file__), "debug")
    os.makedirs(out_dir, exist_ok=True)
    logging.warning("selflift-Avatar: debug decoding enabled; intermediate video decodes can substantially increase time and memory")
    from PIL import Image
    for name, lat in latents.items():
        if lat is None:
            continue
        img = vae.decode(lat)
        if img.ndim == 5:
            img = img.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1])
        frame = (img[0].float().cpu().numpy().clip(0.0, 1.0) * 255).round().astype("uint8")
        Image.fromarray(frame).save(os.path.join(out_dir, name + ".png"))


def progressive_sample(low_res_model, positive, negative, vae, latent_image, sampler, sigmas, seed, cfg,
                       transition_step, lowres_scale, rho, w_min, w_max, latent_upsample, latent_lifter=None,
                       highres_tiling=False, high_res_model=None, tiling_mode="auto", tiling_tiles=2,
                       tiling_axis="auto", on_tiling_plan=None,
                       highres_tiling_mode="spatial", highres_token_chunk=4096,
                       highres_weight_cache_gb=4.0, spatial_anchor_mode="off",
                       tiling_overlap=None, diagnostics_out=None):
    _validate_schedule(sigmas, transition_step)
    if sigmas.numel() < 2:
        return latent_image
    if not 0.25 <= lowres_scale <= 1.0:
        raise ValueError("selflift-Avatar: lowres_scale must be between 0.25 and 1")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("selflift-Avatar: rho must be between 0 and 1")
    if not 0.0 <= w_min <= w_max <= 1.0:
        raise ValueError("selflift-Avatar: weights must satisfy 0 <= w_min <= w_max <= 1")
    noise_masks = _validate_latent_input(latent_image)
    global_tiling = (highres_tiling and isinstance(highres_tiling_mode, str)
                     and highres_tiling_mode in _GLOBAL_TILING_MODES)
    if spatial_anchor_mode not in ("off", "lowres_global"):
        raise ValueError(
            "selflift-Avatar: spatial_anchor_mode must be 'off' or 'lowres_global'"
        )
    use_lowres_anchor = (highres_tiling and highres_tiling_mode == "spatial"
                         and spatial_anchor_mode == "lowres_global")
    content_state = h3_tiling.SpatialContentState() if use_lowres_anchor else None
    if spatial_anchor_mode != "off" and not use_lowres_anchor:
        logging.warning(
            "selflift-Avatar: spatial_anchor_mode is only applied in spatial tiling; "
            "ignoring it for highres_tiling_mode=%s", highres_tiling_mode
        )
    if highres_tiling:
        if highres_tiling_mode not in _HIGHRES_TILING_MODES:
            raise ValueError(
                f"selflift-Avatar: unknown highres_tiling_mode {highres_tiling_mode!r}; "
                f"expected one of {_HIGHRES_TILING_MODES}"
            )
        if highres_tiling_mode == "spatial":
            h3_tiling.validate_options(tiling_mode, tiling_tiles, tiling_axis)
            avatar_masks.validate_tiling_masks(noise_masks)
            if noise_masks is not None:
                logging.info("[selflift-Avatar tiling] video=generate, audio=fully preserved; full audio conditioning on every spatial tile")
        else:
            if not isinstance(highres_token_chunk, int) or not 128 <= highres_token_chunk <= 16384:
                raise ValueError("selflift-Avatar: highres_token_chunk must be an integer from 128 to 16384")
            if not 0.0 <= float(highres_weight_cache_gb) <= 64.0:
                raise ValueError("selflift-Avatar: highres_weight_cache_gb must be between 0 and 64")
            # Global modes keep the packed sequence intact, so arbitrary video/audio
            # masks remain valid: no spatial crop or tile-local mask transform is needed.
            logging.info("[selflift-Avatar global] mode=%s token_chunk=%d weight_cache=%.2f GiB; "
                         "full-sequence video/audio masks enabled",
                         highres_tiling_mode, highres_token_chunk, highres_weight_cache_gb)

    model = low_res_model
    model_sampling = model.get_model_object("model_sampling")
    _validate_sampling(model_sampling, sampler)

    streams, nested = _streams(comfy.sample.fix_empty_latent_channels(
        model, latent_image["samples"], latent_image.get("downscale_ratio_spacial", None),
        latent_image.get("downscale_ratio_temporal", None)))
    hires_base = high_res_model if high_res_model is not None else model
    _validate_sampling(hires_base.get_model_object("model_sampling"), sampler)
    if highres_tiling and highres_tiling_mode == "spatial":
        high_model = h3_tiling.tiled_model(hires_base, [tuple(stream.shape) for stream in streams],
                                           mode=tiling_mode, tiles=tiling_tiles, axis=tiling_axis,
                                           on_plan=on_tiling_plan, overlap=tiling_overlap,
                                           content_state=content_state)
    elif global_tiling:
        high_model = _global_tiled_model(
            hires_base, highres_token_chunk, highres_tiling_mode, highres_weight_cache_gb
        )
        if on_tiling_plan is not None:
            plan = {
                "kind": "global",
                "mode": highres_tiling_mode,
                "token_chunk": int(highres_token_chunk),
                "weight_cache_gb": float(highres_weight_cache_gb),
                "full_sequence": True,
            }
            fallback = getattr(high_model, "_selflift_global_fallback", None)
            if fallback:
                plan["fallback"] = str(fallback)
            on_tiling_plan(plan)
    else:
        high_model = hires_base
    video = streams[0].ndim == 5
    if video:
        b, c, t, H, W = streams[0].shape
    else:
        b, c, H, W = streams[0].shape
        t = None
    h = max(2, round(H * lowres_scale / 2) * 2)
    w = max(2, round(W * lowres_scale / 2) * 2)
    low_shape = (b, c, t, h, w) if video else (b, c, h, w)
    logging.info("[selflift-Avatar plan] low_latent=%s target_latent=%s spatial_lift=(%.4f, %.4f) "
                 "low_nfe=%d high_nfe=%d sigma_prediction=%.8g sigma_resume=%.8g "
                 "rho=%.4f weights=(%.4f, %.4f) direct_lift=%s pixel_anchor=%s cfg=%.4f mask=%s hires_model=%s",
                 low_shape, tuple(streams[0].shape), H / h, W / w,
                 transition_step, sigmas.numel() - 1 - transition_step,
                 sigmas[transition_step - 1].item(), sigmas[transition_step].item(),
                 rho, w_min, w_max,
                 "skipped" if rho == 1.0 and w_min == 1.0 else "external" if latent_lifter is not None else latent_upsample,
                 rho > 0.0 and w_max > 0.0, cfg,
                 "none" if noise_masks is None else str([tuple(m.shape) for m in noise_masks]),
                 "custom" if hires_base is not model else "same")

    device = comfy.model_management.intermediate_device()
    source_video = streams[0].to(device)
    audio_streams = [s.to(device) for s in streams[1:]]

    if video:
        low_video = torch.nn.functional.interpolate(
            source_video.float(), size=(t, h, w), mode="trilinear", align_corners=False
        ).to(dtype=source_video.dtype)
    else:
        low_video = torch.nn.functional.interpolate(
            source_video.float(), size=(h, w), mode="bilinear", align_corners=False
        ).to(dtype=source_video.dtype)
    m_full = m_low = None
    low_masks = None
    if noise_masks is not None:
        m_full = noise_masks[0]
        m_low = avatar_masks.resize_spatial(m_full, (h, w))
        low_masks = [m_low] + noise_masks[1:]
    video_anchor = source_video
    low_latent = _pack([low_video] + audio_streams, nested)
    low_model = native_mask_model(model, [low_video] + audio_streams, low_masks, nested)
    high_model = native_mask_model(high_model, [video_anchor] + audio_streams, noise_masks, nested)
    del source_video, low_video
    del streams
    noise_low = comfy.sample.prepare_noise(low_latent, seed, latent_image.get("batch_index", None))

    total_steps = sigmas.shape[-1] - 1
    callback = latent_preview.prepare_callback(model, total_steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    if video:  # keyframe cond latents share the generation grid on MiniMax H3
        positive_low = _resize_keyframes(positive, h, w)
        negative_low = _resize_keyframes(negative, h, w)
    else:
        positive_low, negative_low = positive, negative

    transition = {}
    low_evaluations = 0

    def callback_low(step, x0, x, total):
        nonlocal low_evaluations
        step = low_evaluations
        low_evaluations += 1
        if low_evaluations > transition_step:
            raise RuntimeError("selflift-Avatar: too many low-resolution callbacks for the Euler schedule")
        if step == transition_step - 1:
            transition["state"] = x
            transition["x0"] = x0
        # Preview decoders expect the target grid.  Decode a temporary lifted
        # preview while keeping the sampler state at its true low resolution.
        preview_x0 = x0
        if video:
            preview_streams, preview_nested = _streams(x0)
            preview_video = torch.nn.functional.interpolate(
                preview_streams[0].float(), size=(t, H, W), mode="trilinear", align_corners=False
            ).to(preview_streams[0].dtype)
            preview_x0 = _pack([preview_video] + preview_streams[1:], preview_nested)
            del preview_video, preview_streams
        elif x0.ndim == 4:
            preview_x0 = torch.nn.functional.interpolate(
                x0.float(), size=(H, W), mode="bilinear", align_corners=False
            ).to(x0.dtype)
        result = callback(step, preview_x0, preview_x0, total_steps)
        del preview_x0
        low_timer.mark(f"step {step + 1}/{transition_step}" + (" (includes setup)" if step == 0 else ""))
        return result

    # The final low-resolution model evaluation is the Eq. 3 prediction. Its Euler
    # update is discarded and rebuilt after the resolution transition, preserving NFE.
    resolution_scale = (getattr(model.get_model_object("latent_format"), "spacial_downscale_ratio", 1)
                        if video else getattr(model.get_model_object("latent_format"), "spacial_downscale_ratio", 1))
    low_timer = _StageTimer("low_resolution", low_model.load_device, (h, w), resolution_scale)
    log_memory("low_resolution start", low_model.load_device)
    comfy.samplers.sample(low_model, noise_low, positive_low, negative_low, cfg, low_model.load_device,
                          sampler, sigmas[:transition_step + 1], low_model.model_options,
                          latent_image=low_latent, callback=callback_low,
                          disable_pbar=disable_pbar, seed=seed)
    if low_evaluations != transition_step:
        raise RuntimeError(f"selflift-Avatar: expected {transition_step} low-resolution callbacks, received {low_evaluations}; check sampler wrappers")
    low_timer.finish()
    log_memory("low_resolution end", model.load_device)
    transition_timer = _StageTimer("transition", model.load_device, (H, W), resolution_scale)
    low_streams, nested = _streams(transition.pop("state"))
    x0_streams, _ = _streams(transition.pop("x0"))
    sigma_k = sigmas[transition_step - 1]
    sigma_next = sigmas[transition_step]
    auxiliary_next = [_euler_step(state.to(device), denoised.to(device), sigma_k, sigma_next)
                      for state, denoised in zip(low_streams[1:], x0_streams[1:])]
    latent_format = model.get_model_object("latent_format")
    z0_low_vae = latent_format.process_out(x0_streams[0].float()).to(device)
    # Keep the aligned low-resolution endpoint as an optional global H3
    # reference. This is only consumed by the experimental spatial-anchor
    # node, so the original spatial path remains unchanged.
    lowres_anchor = z0_low_vae.detach().clone() if use_lowres_anchor else None
    if content_state is not None and lowres_anchor is not None:
        content_state.set_from_video(lowres_anchor)
        logging.info(
            "[selflift-Avatar content-aware split] scored low-resolution context "
            "%s; cuts will avoid high-gradient rows/columns",
            tuple(lowres_anchor.shape),
        )
    del low_latent, noise_low, low_streams, x0_streams, positive_low, negative_low
    transition_timer.mark("prepare_endpoint")
    log_memory("transition endpoint_ready", model.load_device)

    # Artifact-Aware Consistency Lift (Eqs. 4-9); skip branches the weights discard
    need_pix = rho > 0.0 and w_max > 0.0
    need_lat = not (rho >= 1.0 and w_min >= 1.0 and w_max >= 1.0)
    z_lat_vae, z_pix_vae = selflift.paired_lifts(
        z0_low_vae, vae, (H, W), latent_upsample, latent_lifter,
        need_lat=need_lat, need_pix=need_pix)
    transition_timer.mark("paired_lifts")
    log_memory("transition lifts_ready", model.load_device)
    z_lat = latent_format.process_in(z_lat_vae) if z_lat_vae is not None else None
    z_pix = latent_format.process_in(z_pix_vae) if z_pix_vae is not None else None
    z0_high = selflift.artifact_aware_consistency_lift(z_lat, z_pix, rho, w_min, w_max, mask=m_full)
    if m_full is not None:
        # restore the keep-region with the true original instead of any lifted estimate
        m_cast = m_full.to(device=z0_high.device, dtype=z0_high.dtype)
        clean_anchor = latent_format.process_in(video_anchor.to(z0_high))
        z0_high = z0_high * m_cast + clean_anchor * (1.0 - m_cast)
    if os.environ.get("SELFLIFT_AVATAR_DEBUG", "0") == "1":
        _debug_dump(vae, {
            "z0_low": z0_low_vae,
            "z_lat": z_lat_vae,
            "z_pix": z_pix_vae,
            "z0_high": latent_format.process_out(z0_high),
        })
    z0_high = z0_high.to(device)
    del z0_low_vae, z_lat_vae, z_pix_vae, z_lat, z_pix
    transition_timer.mark("correction_and_debug")
    log_memory("transition correction_ready", model.load_device)

    # Re-noise the corrected video/image endpoint at the sigma of the reused model
    # evaluation, then complete that Euler interval without another denoiser call.
    video_noise = comfy.sample.prepare_noise(z0_high, (seed + 1) % (1 << 64),
                                             latent_image.get("batch_index", None)).to(z0_high)
    video_state = model_sampling.noise_scaling(sigma_k, video_noise, z0_high)
    next_streams = [_euler_step(video_state, z0_high, sigma_k, sigma_next)] + auxiliary_next
    del z0_high, video_noise, video_state, auxiliary_next
    resume_streams = [model_sampling.inverse_noise_scaling(sigma_next, s) for s in next_streams]
    resume_latent = model.model.process_latent_out(_pack(resume_streams, nested))
    resume_noise = _pack([torch.zeros_like(s) for s in resume_streams], nested)
    del next_streams, resume_streams
    transition_timer.mark("renoise")
    transition_timer.finish()
    log_memory("transition end / high_resolution start", model.load_device)

    high_evaluations = 0

    def callback_high(step, x0, x, total):
        nonlocal high_evaluations
        step = high_evaluations
        high_evaluations += 1
        if high_evaluations > total_steps - transition_step:
            raise RuntimeError("selflift-Avatar: too many high-resolution callbacks for the Euler schedule")
        result = callback(step + transition_step, x0, x, total_steps)
        high_timer.mark(f"step {step + 1}/{total_steps - transition_step}" + (" (includes setup)" if step == 0 else ""))
        return result

    high_timer = _StageTimer("high_resolution", high_model.load_device, (H, W), resolution_scale)
    positive_high = positive
    if lowres_anchor is not None:
        positive_high = spatial_anchor_utils.append_global_video_reference(positive, lowres_anchor)
        logging.info(
            "[selflift-Avatar spatial anchor] appended aligned low-resolution video "
            "latent %s as an uncropped global reference for every spatial tile",
            tuple(lowres_anchor.shape),
        )
    if highres_tiling:
        if highres_tiling_mode == "spatial":
            logging.info("[selflift-Avatar plan] high-resolution tiling enabled; mode=%s axis=%s", tiling_mode, tiling_axis)
        else:
            logging.info("[selflift-Avatar plan] high-resolution global context enabled; mode=%s token_chunk=%d "
                         "weight_cache=%.2f GiB; masks remain full-sequence",
                         highres_tiling_mode, highres_token_chunk, highres_weight_cache_gb)
    out = comfy.samplers.sample(high_model, resume_noise, positive_high, negative, cfg, high_model.load_device,
                                sampler, sigmas[transition_step:], high_model.model_options,
                                latent_image=resume_latent, callback=callback_high,
                                disable_pbar=disable_pbar, seed=seed)
    del resume_latent, resume_noise, lowres_anchor, positive_high
    if high_evaluations != total_steps - transition_step:
        raise RuntimeError(f"selflift-Avatar: expected {total_steps - transition_step} high-resolution callbacks, received {high_evaluations}; check sampler wrappers")

    result = latent_image.copy()
    if noise_masks is not None:
        # Only exact keep-regions are restored here; do not blend soft masks twice.
        out_streams, _ = _streams(out)
        out = _pack(avatar_masks.restore_kept(
            out_streams, [video_anchor] + audio_streams, noise_masks), nested)
    result["samples"] = out.to(device=comfy.model_management.intermediate_device(),
                                dtype=comfy.model_management.intermediate_dtype())
    high_timer.finish()
    log_memory("high_resolution end", model.load_device)
    if diagnostics_out is not None and content_state is not None:
        diagnostics_out.update({
            "axis": "H" if content_state.axis == 3 else "W",
            "length": content_state.length,
            "image_shape": content_state.image_shape,
            "regions": content_state.regions,
            "seams": list(content_state.seam_metrics),
        })
    return result


class SelfLiftAvatarH3Sampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "low_res_model": ("MODEL", {"tooltip": "Low-resolution model used for the first transition_step evaluations. Also used for the high-resolution stage when high_res_model is disconnected."}),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "vae": ("VAE", {"tooltip": "Video VAE used for the pixel re-encode anchor at the resolution transition."}),
            "latent_image": ("LATENT", {"tooltip": "Target-resolution H3 AV latent defining size and duration. Experimental: accepts existing latents, Tensor masks and H3 NestedTensor video/audio masks. 0 keeps content; 1 generates. Spatial tiling requires video=1 everywhere and audio=0 everywhere; global modes keep full-sequence masks unchanged."}),
            "sampler": ("SAMPLER", {"tooltip": "Standard Euler only; SelfLift reuses its transition-step prediction to keep the original NFE count."}),
            "sigmas": ("SIGMAS",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
            "transition_step": ("INT", {"default": 6, "min": 1, "max": 10000, "tooltip": "Number of low-resolution denoiser evaluations. The paper uses 6 of 8 NFEs for its 8-step image model; H3 requires independent validation."}),
            "lowres_scale": ("FLOAT", {"default": 0.5, "min": 0.25, "max": 1.0, "step": 0.05, "tooltip": "Spatial scale of the low-resolution prefix (paper: 0.5)."}),
            "rho": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Fraction of highest-risk spatiotemporal locations corrected toward the pixel-VAE anchor. The H3 default 0 uses only the external latent upscaler and skips the VAE round trip. For SelfLift-zero with upscaler_model=none, start near 0.6."}),
            "w_min": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Correction-strength floor. H3's widespread nearest-lift error can require 1.0; 0.5 is the paper's image-model setting."}),
            "w_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Correction-strength ceiling. Keep at 1.0 for the H3 SelfLift-zero diagnostic."}),
            "upscaler_model": _upscaler_input(),
        }, "optional": {
            "high_res_model": ("MODEL", {"tooltip": "高分辨率阶段模型 / High-resolution model. Used after the resolution lift. If disconnected, uses low_res_model. Both stages must use compatible architecture, latent format, sampling parameterization, VAE and conditioning."}),
            "highres_tiling": ("BOOLEAN", {"default": False, "label_on": "高分辨率省显存：开启", "label_off": "高分辨率省显存：关闭", "tooltip": "Experimental high-resolution memory saving. Choose Spatial for crop tiles, or Global stream/Global for full-sequence chunking that preserves cross-tile attention context. Spatial supports only video=1 everywhere / audio=0 everywhere; global modes keep arbitrary full-sequence masks."}),
            "tiling_mode": (["auto", "manual"], {"default": "auto", "tooltip": "Spatial mode only. auto 自动: choose 1–8 tiles using memory estimates. manual 手动: use tiling_tiles, no automatic increase."}),
            "tiling_tiles": ([2, 4, 6, 8], {"default": 2, "tooltip": "Spatial mode only. Manual tiles: 2, 4, 6, 8. Ignored in auto mode. Small dimensions can reduce the effective count."}),
            "tiling_axis": (["auto", "width", "height"], {"default": "auto", "tooltip": "Spatial mode only. auto = longer patch-grid side; width = left/right strips; height = top/bottom strips; audio/time are not split."}),
            "highres_tiling_mode": (["spatial", "global_stream", "global", "global_ram"], {"default": "spatial", "tooltip": "Memory-saving strategy. spatial: crop tiles with an automatic full-frame low-resolution anchor, content-aware cuts and seam diagnostics. global_stream: full-sequence context, GPU head groups, requires native SLA/sol-attn (VSA not adapted). global: full-sequence dense-compatible path; higher VRAM. global_ram: full-sequence host-RAM staging; slower. Global modes preserve the complete video/audio mask on the packed sequence."}),
            "highres_token_chunk": ("INT", {"default": 4096, "min": 128, "max": 16384, "step": 128, "tooltip": "Global modes only. Tokens per projection/feed-forward batch. Smaller batches usually save intermediate VRAM but add overhead. This is not a spatial tile or attention window."}),
            "highres_weight_cache_gb": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 64.0, "step": 0.5, "tooltip": "Global stream only. Resident DynamicVRAM weight-cache budget in GiB. Not total VRAM or system RAM. 0 keeps the automatic policy; it does not disable caching."}),
        }, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "selflift-Avatar"

    def sample(self, low_res_model, positive, negative, vae, latent_image, sampler, sigmas, seed, cfg,
               transition_step, lowres_scale, rho, w_min, w_max, upscaler_model, high_res_model=None, highres_tiling=False,
               tiling_mode="auto", tiling_tiles=2, tiling_axis="auto",
               highres_tiling_mode="spatial", highres_token_chunk=4096,
               highres_weight_cache_gb=4.0, unique_id=None):
        if rho == 0.0 and upscaler_model == "none":
            raise ValueError(
                "SelfLift H3: rho=0 with upscaler_model=none disables both SelfLift-zero correction "
                "and external latent upscaling; choose rho>0 or select an external H3 upscaler."
            )
        lifter = None
        if upscaler_model != "none":
            if rho > 0.0 and w_max > 0.0:
                logging.warning("SelfLift H3: rho > 0 with an external upscaler is a hybrid experiment; select upscaler_model=none to test the paper's SelfLift-zero direct route")
            lifter = lambda z, hw: h3_upscaler.learned_latent_lift(z, hw, upscaler_model)
        if not highres_tiling:
            summary = "OFF / 高分辨率省显存已关闭"
        elif highres_tiling_mode == "spatial":
            summary = "Preparing / 准备中: spatial + full-frame low-res anchor | {} | axis={} | tiles={}\nActual plan appears at high-resolution preparation / 高清阶段准备时显示实际方案".format(
                tiling_mode, tiling_axis, tiling_tiles if tiling_mode == "manual" else "auto")
        else:
            cache = (f" | weight_cache={float(highres_weight_cache_gb):.1f} GiB"
                     if highres_tiling_mode == "global_stream" else "")
            summary = "Preparing / 准备中: {} | token_chunk={}{}\nFull-sequence masks / 全序列遮罩: enabled".format(
                highres_tiling_mode, highres_token_chunk, cache)
        last_plan = None

        def send_status(text):
            server = getattr(PromptServer, "instance", None)
            if unique_id is not None and server is not None and server.client_id is not None:
                server.send_sync("selflift-avatar-tiling", {"node_id": str(unique_id), "text": text, "plan": last_plan}, server.client_id)

        def report(plan):
            nonlocal summary, last_plan
            last_plan = plan
            summary = format_plan(plan)
            send_status("RUNNING / 高清采样中\n" + summary)

        send_status(summary)
        diagnostics = {}
        spatial_anchor_mode = ("lowres_global" if highres_tiling and highres_tiling_mode == "spatial"
                               else "off")
        spatial_overlap = 6 if spatial_anchor_mode == "lowres_global" else None
        try:
            output = progressive_sample(low_res_model, positive, negative, vae, latent_image, sampler, sigmas, seed, cfg,
                                        transition_step, lowres_scale, rho, w_min, w_max, "nearest",
                                        latent_lifter=lifter, highres_tiling=highres_tiling, high_res_model=high_res_model,
                                        tiling_mode=tiling_mode, tiling_tiles=tiling_tiles, tiling_axis=tiling_axis,
                                        on_tiling_plan=report, highres_tiling_mode=highres_tiling_mode,
                                        highres_token_chunk=highres_token_chunk,
                                        highres_weight_cache_gb=highres_weight_cache_gb,
                                        spatial_anchor_mode=spatial_anchor_mode,
                                        tiling_overlap=spatial_overlap,
                                        diagnostics_out=diagnostics)
        except Exception:
            send_status("FAILED / 运行失败：以下不是成功结果；详见错误报告\n" + summary)
            raise
        if highres_tiling and last_plan is None:
            summary = "NOT RUN / 本次未执行高清分块规划（例如空采样日程）"
        seams = diagnostics.get("seams") or []
        if seams:
            values = [float(item.get("relative_diff", 0.0)) for item in seams]
            summary += (f"\nSeam diag / 接缝差异: {len(seams)} boundaries | "
                        f"mean={sum(values) / len(values):.4f} | max={max(values):.4f}")
        summary = "DONE / 上次运行完成\n" + summary
        ui = {"selflift_tiling": [summary], "selflift_tiling_plan": [last_plan]}
        if diagnostics:
            ui["selflift_seam_diag"] = [diagnostics]
        return {"ui": ui, "result": (output,)}


class SelfLiftAvatarH3SpatialAnchoredSampler(SelfLiftAvatarH3Sampler):
    """Experimental spatial-only sampler with a full-frame low-resolution anchor.

    This node is deliberately separate from ``SelfLiftAvatarH3Sampler`` so the
    original spatial/global behavior can be kept as an A/B control. It adds the
    aligned low-resolution SelfLift endpoint as an uncropped H3 video reference
    on the high-resolution stage, and exposes a wider explicit stripe overlap.
    """

    @classmethod
    def INPUT_TYPES(cls):
        schema = copy.deepcopy(SelfLiftAvatarH3Sampler.INPUT_TYPES())
        optional = schema["optional"]
        optional["highres_tiling"] = ("BOOLEAN", {
            "default": True,
            "label_on": "空间分块：开启",
            "label_off": "空间分块：关闭",
            "tooltip": "TEST node: spatial tiling only. Disable for a direct full-frame control run.",
        })
        optional["tiling_mode"] = (["manual", "auto"], {
            "default": "manual",
            "tooltip": "manual is recommended for A/B tests: 2 or 3 strips first; increase only if OOM.",
        })
        optional["tiling_tiles"] = ([2, 3, 4, 6, 8], {
            "default": 2,
            "tooltip": "Fewer strips preserve more context. 3 can keep a centered subject in one middle strip.",
        })
        optional["tiling_overlap"] = ("INT", {
            "default": 6, "min": 1, "max": 16, "step": 1,
            "tooltip": "Explicit overlap in H3 patch units. Larger values give each strip more neighbour context but use more VRAM.",
        })
        optional["spatial_anchor"] = (["lowres_global", "off"], {
            "default": "lowres_global",
            "tooltip": "lowres_global appends the aligned full-frame low-resolution endpoint as an uncropped H3 video reference. off is the control path.",
        })
        optional["highres_tiling_mode"] = (["spatial"], {
            "default": "spatial",
            "tooltip": "This test node intentionally supports spatial tiling only. Use the original sampler for global modes.",
        })
        optional.pop("highres_token_chunk", None)
        optional.pop("highres_weight_cache_gb", None)
        return schema

    def sample(self, low_res_model, positive, negative, vae, latent_image, sampler, sigmas, seed, cfg,
               transition_step, lowres_scale, rho, w_min, w_max, upscaler_model, high_res_model=None,
               highres_tiling=True, tiling_mode="manual", tiling_tiles=2, tiling_axis="auto",
               spatial_anchor="lowres_global", tiling_overlap=6, highres_tiling_mode="spatial",
               unique_id=None):
        if rho == 0.0 and upscaler_model == "none":
            raise ValueError(
                "SelfLift H3: rho=0 with upscaler_model=none disables both SelfLift-zero correction "
                "and external latent upscaling; choose rho>0 or select an external H3 upscaler."
            )
        if spatial_anchor not in ("lowres_global", "off"):
            raise ValueError("SelfLift H3 spatial anchor must be 'lowres_global' or 'off'")

        lifter = None
        if upscaler_model != "none":
            if rho > 0.0 and w_max > 0.0:
                logging.warning(
                    "SelfLift H3: rho > 0 with an external upscaler is a hybrid experiment; "
                    "select upscaler_model=none to test the paper's SelfLift-zero direct route"
                )
            lifter = lambda z, hw: h3_upscaler.learned_latent_lift(z, hw, upscaler_model)

        if not highres_tiling:
            summary = "OFF / 空间分块已关闭（全帧对照）"
        elif spatial_anchor == "lowres_global":
            summary = ("Preparing / 准备中: spatial + full-frame low-res anchor | "
                       "{} | axis={} | tiles={} | overlap={}\n"
                       "Actual plan appears at high-resolution preparation / 高清阶段准备时显示实际方案").format(
                           tiling_mode, tiling_axis,
                           tiling_tiles if tiling_mode == "manual" else "auto",
                           tiling_overlap)
        else:
            summary = ("Preparing / 准备中: spatial control (anchor off) | "
                       "{} | axis={} | tiles={} | overlap={}").format(
                           tiling_mode, tiling_axis,
                           tiling_tiles if tiling_mode == "manual" else "auto",
                           tiling_overlap)

        last_plan = None

        def send_status(text):
            server = getattr(PromptServer, "instance", None)
            if unique_id is not None and server is not None and server.client_id is not None:
                server.send_sync(
                    "selflift-avatar-tiling",
                    {"node_id": str(unique_id), "text": text, "plan": last_plan},
                    server.client_id,
                )

        def report(plan):
            nonlocal summary, last_plan
            last_plan = plan
            summary = format_plan(plan)
            send_status("RUNNING / 高清采样中\n" + summary)

        send_status(summary)
        diagnostics = {}
        try:
            output = progressive_sample(
                low_res_model, positive, negative, vae, latent_image, sampler, sigmas, seed, cfg,
                transition_step, lowres_scale, rho, w_min, w_max, "nearest",
                latent_lifter=lifter, highres_tiling=highres_tiling, high_res_model=high_res_model,
                tiling_mode=tiling_mode, tiling_tiles=tiling_tiles, tiling_axis=tiling_axis,
                on_tiling_plan=report, highres_tiling_mode="spatial",
                spatial_anchor_mode=spatial_anchor, tiling_overlap=tiling_overlap,
                diagnostics_out=diagnostics,
            )
        except Exception:
            send_status("FAILED / 运行失败：以下不是成功结果；详见错误报告\n" + summary)
            raise
        if highres_tiling and last_plan is None:
            summary = "NOT RUN / 本次未执行高清分块规划（例如空采样日程）"
        seams = diagnostics.get("seams") or []
        if seams:
            values = [float(item.get("relative_diff", 0.0)) for item in seams]
            summary += (f"\nSeam diag / 接缝差异: {len(seams)} boundaries | "
                        f"mean={sum(values) / len(values):.4f} | max={max(values):.4f}")
        summary = "DONE / 上次运行完成\n" + summary
        ui = {"selflift_tiling": [summary], "selflift_tiling_plan": [last_plan]}
        if diagnostics:
            ui["selflift_seam_diag"] = [diagnostics]
        return {"ui": ui, "result": (output,)}


class SelfLiftAvatarImageSampler:
    """SelfLift-zero for compatible rectified-flow image backbones."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "low_res_model": ("MODEL", {"tooltip": "Low-resolution model used for the first transition_step evaluations. Also used for the high-resolution stage when high_res_model is disconnected."}),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "vae": ("VAE", {"tooltip": "VAE used for the pixel re-encode anchor at the resolution transition."}),
            "latent_image": ("LATENT", {"tooltip": "Target-resolution image latent. Experimental: accepts existing content and BHW/BCHW Tensor masks. 0 keeps content; 1 generates."}),
            "sampler": ("SAMPLER", {"tooltip": "Standard Euler only; SelfLift reuses its transition-step prediction to keep the original NFE count."}),
            "sigmas": ("SIGMAS",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
            "transition_step": ("INT", {"default": 6, "min": 1, "max": 10000, "tooltip": "Number of low-resolution denoiser evaluations. Paper: 3 of 4 for FLUX.2-Klein, 6 of 8 for Z-Image-Turbo."}),
            "lowres_scale": ("FLOAT", {"default": 0.5, "min": 0.25, "max": 1.0, "step": 0.05, "tooltip": "Spatial scale of the low-resolution prefix (paper: 0.5)."}),
            "rho": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Fraction of locations corrected toward the pixel-VAE anchor. Paper: 0.4 for FLUX.2-Klein, 0.3 for Z-Image-Turbo."}),
            "w_min": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            "w_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "latent_upsample": (["nearest", "bilinear"], {"default": "nearest", "tooltip": "Interpolation for the direct latent lift (paper: nearest)."}),
        }, "optional": {
            "high_res_model": ("MODEL", {"tooltip": "高分辨率阶段模型 / High-resolution model. Used after the resolution lift. If disconnected, uses low_res_model. Both stages must use compatible architecture, latent format, sampling parameterization, VAE and conditioning."}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "selflift-Avatar"

    def sample(self, low_res_model, positive, negative, vae, latent_image, sampler, sigmas, seed, cfg,
               transition_step, lowres_scale, rho, w_min, w_max, latent_upsample, high_res_model=None):
        return (progressive_sample(low_res_model, positive, negative, vae, latent_image, sampler, sigmas, seed, cfg,
                                   transition_step, lowres_scale, rho, w_min, w_max, latent_upsample,
                                   high_res_model=high_res_model),)


class SelfLiftAvatarH3TST:
    """Training-free Temporal State Transport correction (arXiv:2609.08505) for MiniMax H3."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "tau": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05,
                              "tooltip": "Homeostatic correction strength; 0.2 is the paper setting. 0 disables correction while keeping the diagnostic active."}),
            "log_diagnostics": ("BOOLEAN", {"default": True, "label_on": "诊断日志：开启", "label_off": "诊断日志：关闭",
                                            "tooltip": "Log per-forward Spectral Tension and correction statistics to the console. Note: TST is skipped with a warning when highres_tiling is enabled on the SelfLift sampler."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "selflift-Avatar"

    def patch(self, model, tau, log_diagnostics):
        return (h3_tst.patch_model(model, tau, log_diagnostics),)


NODE_CLASS_MAPPINGS = {
    "SelfLiftAvatarH3Sampler": SelfLiftAvatarH3Sampler,
    "SelfLiftAvatarH3SpatialAnchoredSampler": SelfLiftAvatarH3SpatialAnchoredSampler,
    "SelfLiftAvatarImageSampler": SelfLiftAvatarImageSampler,
    "SelfLiftAvatarH3TST": SelfLiftAvatarH3TST,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SelfLiftAvatarH3Sampler": "selflift-Avatar Sampler (MiniMax H3)",
    "SelfLiftAvatarH3SpatialAnchoredSampler": "selflift-Avatar Spatial Anchor (TEST)",
    "SelfLiftAvatarImageSampler": "selflift-Avatar Sampler (Image)",
    "SelfLiftAvatarH3TST": "selflift-Avatar H3 TST",
}
