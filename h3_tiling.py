"""Experimental spatial tiling of H3 high-resolution model evaluations."""

from functools import partial
import inspect
import logging
import math

import torch

import comfy.ldm.common_dit
from comfy.ldm.minimax.model import PackedLayout
import comfy.model_base
import comfy.patcher_extension
import comfy.sampler_helpers


class SpatialContentState:
    """Mutable content score shared through model-option deep copies.

    The low-resolution SelfLift endpoint only exists after the low-resolution
    stage, while the high-resolution tiling wrappers are created earlier.  A
    normal dict would be deep-copied by ModelPatcher.clone(); this small
    object deliberately stays shared so the planner sees the scores.
    """

    def __init__(self):
        self.axis_scores = None
        self.axis = None
        self.length = None
        self.image_shape = None
        self.regions = None
        self.seam_metrics = []

    def __deepcopy__(self, memo):
        return self

    def set_from_video(self, video):
        self.axis_scores = content_scores(video)
        self.seam_metrics = []


def content_scores(video):
    """Return per-axis spatial gradient energy for one H3 video latent.

    Scores are computed on the aligned low-resolution endpoint.  A high value
    means the corresponding row/column contains more visual change, so a
    content-aware seam should avoid it.  The result is intentionally tiny and
    CPU-side; no extra UI or model call is involved.
    """
    if not torch.is_tensor(video) or video.ndim != 5:
        return None
    if video.shape[2] < 1 or video.shape[3] < 2 or video.shape[4] < 2:
        return None

    x = video.detach().float()
    # [B,C,T,H,W] -> one gradient curve per spatial axis.
    gx = (x[..., :, 1:] - x[..., :, :-1]).abs().mean(dim=(0, 1, 2, 3))
    gy = (x[..., 1:, :] - x[..., :-1, :]).abs().mean(dim=(0, 1, 2, 4))
    wx, hx = int(video.shape[-1]), int(video.shape[-2])
    sx = torch.zeros(wx, dtype=torch.float32)
    sy = torch.zeros(hx, dtype=torch.float32)
    sx[1:] = gx.cpu()
    sy[1:] = gy.cpu()

    def normalize(curve):
        curve = torch.nan_to_num(curve, nan=0.0, posinf=0.0, neginf=0.0)
        floor = curve.median().clamp_min(1e-6)
        return (curve / floor).clamp(0.0, 4.0)

    return {"H": normalize(sy), "W": normalize(sx)}


def _score_at(scores, patch_index, length):
    # Regions are built in patch units; a patch is two latent cells in H3's
    # target layout.  A small local mean avoids choosing a single noisy cell.
    idx = max(0, min(length - 1, int(patch_index) * 2))
    lo, hi = max(0, idx - 1), min(length, idx + 2)
    return float(scores[lo:hi].mean()) if hi > lo else 0.0


def _content_aware_regions(length, tile_count=2, overlap=None, scores=None):
    """Build overlapping strips with cuts moved toward low-content lines."""
    if scores is None:
        return _regions(length, tile_count, overlap=overlap)
    patches = (length + 1) // 2
    tile_count = max(1, min(tile_count, max(1, patches // 2)))
    if patches < 4 or tile_count <= 1:
        return [(0, length)]
    if overlap is None:
        overlap = min(4, max(1, patches // 8))
    else:
        overlap = max(1, min(int(overlap), max(1, patches // 2)))
    scores = torch.as_tensor(scores, dtype=torch.float32).flatten()
    if scores.numel() != length:
        return _regions(length, tile_count, overlap=overlap)

    radius = max(2, min(12, patches // (tile_count * 4)))
    cuts = []
    previous = 0
    for index in range(1, tile_count):
        base = int(round(index * patches / tile_count))
        remaining = tile_count - index
        lo = max(previous + 1, base - radius)
        hi = min(patches - remaining, base + radius)
        if hi < lo:
            lo = max(previous + 1, 0)
            hi = min(patches - remaining, max(lo, base))
        best = min(range(lo, hi + 1), key=lambda p: (
            _score_at(scores, p, length), abs(p - base)))
        cuts.append(best)
        previous = best

    regions = []
    for index in range(tile_count):
        start_p = 0 if index == 0 else cuts[index - 1]
        end_p = patches if index == tile_count - 1 else cuts[index]
        start = 0 if index == 0 else max(0, (start_p - overlap) * 2)
        end = length if index == tile_count - 1 else min(length, (end_p + overlap) * 2)
        if end <= start:
            return _regions(length, tile_count, overlap=overlap)
        regions.append((start, end))
    return regions


def _plan_for_ui(plan):
    """Drop internal shared objects before a plan reaches JSON/WebSocket UI."""
    if plan is None:
        return None
    return {
        key: value for key, value in plan.items()
        if key != "content_state" and not str(key).startswith("_")
    }


def _regions_for_plan(plan, length, tile_count):
    if not plan:
        return _regions(length, tile_count)
    state = plan.get("content_state")
    scores = None
    axis = plan.get("axis")
    if state is not None and axis in (3, 4):
        scores = (state.axis_scores or {}).get("H" if axis == 3 else "W")
    return _content_aware_regions(
        length, tile_count, overlap=plan.get("overlap"), scores=scores)


def _regions(length, tile_count=2, overlap=None):
    """Split one video axis into overlapping strips.

    ``overlap`` is in patch units and is optional.  ``None`` keeps the legacy
    automatic halo (1-4 patches), so existing callers are byte-for-byte
    behaviour-compatible.  The experimental spatial-anchor node passes a
    larger explicit halo to give each strip more neighbour context.
    """
    patches = (length + 1) // 2
    tile_count = max(1, min(tile_count, max(1, patches // 2)))
    if patches < 4 or tile_count <= 1:
        return [(0, length)]
    if overlap is None:
        overlap = min(4, max(1, patches // 8))
    else:
        overlap = max(1, min(int(overlap), max(1, patches // 2)))
    regions = []
    for index in range(tile_count):
        start = max(0, index * patches // tile_count - overlap) * 2
        end = min(patches, (index + 1) * patches // tile_count + overlap) * 2
        end = min(end, length)
        regions.append((start, end))
    regions[-1] = (regions[-1][0], length)
    return regions


def validate_options(mode, tiles, axis):
    if mode not in ("auto", "manual") or axis not in ("auto", "width", "height"):
        raise ValueError("selflift-Avatar: tiling mode must be auto/manual and axis auto/width/height")
    if mode == "manual" and (type(tiles) is not int or not 1 <= tiles <= 8):
        raise ValueError("selflift-Avatar: manual tiling_tiles must be an integer from 1 to 8")


def _select_axis(video_shape, choice="auto"):
    if choice == "width":
        return 4
    if choice == "height":
        return 3
    return 3 if (video_shape[3] + 1) // 2 >= (video_shape[4] + 1) // 2 else 4


def _packed_layout(signature, payload):
    options = {"keyframes": payload.get("keyframes"), "refs": payload.get("refs")}
    if "frame_count" in inspect.signature(PackedLayout).parameters:
        options["frame_count"] = payload.get("frame_count")
    return PackedLayout(*signature, **options)


def _tile_payload(payload, context, video, audio, axis, start, end):
    height, width = video.shape[-2:]
    padded_height, padded_width = (height + 1) // 2 * 2, (width + 1) // 2 * 2
    full_layout = payload.get("layout")
    signature = (context.shape[1], video.shape[2], padded_height, padded_width, audio.shape[-1])
    if full_layout is None or full_layout.signature != signature:
        full_layout = _packed_layout(signature, payload)
    tiled = payload.copy()
    if payload.get("keyframes"):
        keyframes = []
        for keyframe in payload["keyframes"]:
            latent = keyframe.get("latent")
            if latent is None:
                # audio-only guide (MiniMaxH3AddGuide with no image): no video
                # rows to crop, pass the anchor through untouched
                keyframes.append(dict(keyframe))
                continue
            if latent.shape[-2:] != (height, width):
                raise ValueError("SelfLift: tiled H3 keyframes must match the target latent height and width")
            region = latent.narrow(axis, start, end - start)
            keyframes.append({**keyframe, "latent": comfy.ldm.common_dit.pad_to_patch_size(
                region, (1, 2, 2)).contiguous()})
        tiled["keyframes"] = keyframes
        if not payload.get("refs"):
            tiled["cond_video_latents"] = [keyframe["latent"] for keyframe in keyframes
                                           if keyframe.get("latent") is not None]
    tile_height = end - start if axis == 3 else height
    tile_width = end - start if axis == 4 else width
    layout = _packed_layout((context.shape[1], video.shape[2], (tile_height + 1) // 2 * 2,
                             (tile_width + 1) // 2 * 2, audio.shape[-1]), tiled)
    for (source_start, source_end, kind), (target_start, target_end, _) in zip(full_layout.segments, layout.segments):
        positions = full_layout.position_ids[source_start:source_end]
        if kind in ("cond", "video"):
            positions = positions.reshape(-1, padded_height // 2, padded_width // 2, 3)
            positions = positions.narrow(axis - 2, start // 2, (end - start + 1) // 2).reshape(-1, 3)
        layout.position_ids[target_start:target_end].copy_(positions)
    tiled["layout"] = layout
    return tiled


def _tiled_forward(executor, streams, timestep, context, transformer_options, minimax_payload=None,
                   n_tiles=2, plan=None, **kwargs):
    video, audio = streams
    # Only the fully generated video / fully preserved audio combination is supported.
    # A redundant full-frame all-one video condition must not reach a smaller tile.
    video_mask = kwargs.get("denoise_mask")
    if video_mask is not None:
        if not bool((video_mask == 1).all()):
            raise ValueError("selflift-Avatar: tiled video preservation is not supported")
        kwargs = {**kwargs, "denoise_mask": None}
    audio_mask = kwargs.get("audio_denoise_mask")
    if audio_mask is not None and not bool((audio_mask == 0).all()):
        raise ValueError("selflift-Avatar: tiled audio conditioning requires a fully preserved audio mask")
    axis = plan.get("axis", _select_axis(video.shape)) if plan is not None else _select_axis(video.shape)
    length = video.shape[axis]
    regions = _regions_for_plan(
        plan, length, plan['tiles'] if plan is not None else n_tiles)
    if len(regions) == 1:
        return executor(streams, timestep, context, transformer_options, minimax_payload=minimax_payload, **kwargs)
    if kwargs.get("control") is not None:
        raise ValueError("SelfLift: high-resolution H3 tiling does not support ControlNet")
    # Keep the stitched accumulator on CPU so previous tiles do not remain on
    # the accelerator while the next tile is evaluated.
    content_state = plan.get("content_state") if plan is not None else None
    if content_state is not None and not plan.get("_regions_logged"):
        content_state.axis = axis
        content_state.length = length
        content_state.image_shape = tuple(video.shape[-2:])
        content_state.regions = list(regions)
        logging.info("[selflift-Avatar content-aware split] axis=%s regions=%s",
                     "H" if axis == 3 else "W", regions)
        plan["_regions_logged"] = True
    video_output = torch.zeros(video.shape, dtype=torch.float32, device="cpu")
    audio_output = None
    weights = torch.zeros(length, dtype=torch.float32, device="cpu")
    prev_tail = None
    diagnose_seams = (plan is not None and plan.get("content_state") is not None
                      and not plan.get("_seam_diag_logged"))
    for index, (start, end) in enumerate(regions):
        tile = video.narrow(axis, start, end - start).contiguous()
        payload = _tile_payload(minimax_payload or {}, context, video, audio, axis, start, end)
        predicted_video, predicted_audio = executor(
            [tile, audio], timestep, context, transformer_options.copy(), minimax_payload=payload, **kwargs)
        window = torch.ones(end - start, device=video.device, dtype=torch.float32)
        if index > 0:
            overlap = min(end, regions[index - 1][1]) - start
            window[:overlap] *= (torch.arange(overlap, device=video.device, dtype=torch.float32) + 0.5) / overlap
        if index + 1 < len(regions):
            overlap = end - regions[index + 1][0]
            window[-overlap:] *= 1.0 - (torch.arange(overlap, device=video.device, dtype=torch.float32) + 0.5) / overlap
        predicted_video = predicted_video.float().cpu()
        if diagnose_seams and index > 0 and prev_tail is not None:
            overlap = min(end, regions[index - 1][1]) - start
            if overlap > 0:
                current_tail = predicted_video.narrow(axis, 0, overlap)
                if current_tail.shape == prev_tail.shape:
                    diff = float((current_tail - prev_tail).abs().mean())
                    scale = max(0.5 * float(current_tail.abs().mean()
                                            + prev_tail.abs().mean()), 1e-6)
                    relative_diff = diff / scale
                    if content_state is not None:
                        content_state.seam_metrics.append({
                            "boundary": index,
                            "axis": "H" if axis == 3 else "W",
                            "overlap": overlap,
                            "cut": (regions[index - 1][1] + start) // 2,
                            "relative_diff": relative_diff,
                        })
                    logging.info(
                        "[selflift-Avatar seam diag] boundary=%d axis=%s "
                        "overlap=%d relative_diff=%.4f",
                        index, "H" if axis == 3 else "W", overlap, relative_diff)
        window = window.cpu()
        weights[start:end].add_(window)
        window_shape = [1] * video.ndim
        window_shape[axis] = end - start
        video_output.narrow(axis, start, end - start).addcmul_(predicted_video.float(), window.view(window_shape))
        if audio_output is None:
            audio_output = predicted_audio.float().clone()
        if index + 1 < len(regions):
            next_start = regions[index + 1][0]
            next_overlap = end - next_start
            if next_overlap > 0:
                prev_tail = predicted_video.narrow(
                    axis, predicted_video.shape[axis] - next_overlap, next_overlap).clone()
            else:
                prev_tail = None
        if diagnose_seams and index == len(regions) - 1:
            plan["_seam_diag_logged"] = True
        del tile, payload, predicted_video, predicted_audio, window
    window_shape[axis] = length
    video_output.div_(weights.view(window_shape))
    return [video_output.to(device=video.device, dtype=video.dtype), audio_output.to(audio.dtype)]


def _condition_elements(condition, tile_height, tile_width, channels):
    text = condition.get("cross_attn")
    elements = text.shape[-2] * channels * 4 if text is not None else 0
    for keyframe in condition.get("minimax_keyframes") or []:
        latent = keyframe.get("latent")
        if latent is not None:
            shape = latent.shape
            elements += shape[1] * shape[2] * tile_height * tile_width
        audio = keyframe.get("audio_latent")
        if audio is not None:
            elements += math.prod(audio.shape[1:])
    for reference in condition.get("minimax_refs") or []:
        latent = reference.get("latent")
        if latent is not None:
            shape = latent.shape
            elements += shape[1] * shape[2] * ((shape[3] + 1) // 2 * 2) * ((shape[4] + 1) // 2 * 2)
        audio = reference.get("audio_latent")
        if audio is not None:
            elements += math.prod(audio.shape[1:])
    return elements


def _budget(model, noise_shape, conds, latent_shapes, regions, axis):
    video_shape, audio_shape = latent_shapes
    full_elements = math.prod(video_shape[1:]) + math.prod(audio_shape[1:])
    tile_shape = list(video_shape)
    tile_shape[axis] = max(end - start for start, end in regions)
    tile_shape[3] = (tile_shape[3] + 1) // 2 * 2
    tile_shape[4] = (tile_shape[4] + 1) // 2 * 2
    tile_elements = math.prod(tile_shape[1:]) + math.prod(audio_shape[1:])
    condition_elements = max((_condition_elements(condition, tile_shape[3], tile_shape[4], video_shape[1])
                              for group in conds.values() for condition in (group or [])), default=0)
    buffer_bytes = full_elements * 4 * 8
    bytes_per_element = model.model.memory_required((1, 1, 1))
    budget_elements = tile_elements + condition_elements + math.ceil(buffer_bytes / bytes_per_element)
    budget_shape = (noise_shape[0], 1, budget_elements)
    preferred, minimum = comfy.sampler_helpers.estimate_memory(model, budget_shape, conds)
    return budget_shape, tuple(tile_shape), buffer_bytes, preferred, minimum


def _prepare_tiled_sampling(executor, model, noise_shape, conds, model_options=None,
                            force_full_load=False, force_offload=False, *, latent_shapes, plan=None,
                            on_plan=None):
    if force_offload:
        return executor(model, noise_shape, conds, model_options=model_options,
                        force_full_load=force_full_load, force_offload=True)
    video_shape, audio_shape = latent_shapes
    config = plan if plan is not None else {}
    mode = config.get("mode", "auto")
    requested = config.get("requested_tiles", 2)
    choice = config.get("axis_choice", "auto")
    validate_options(mode, requested, choice)
    axis = _select_axis(video_shape, choice)
    full_elements = math.prod(video_shape[1:]) + math.prod(audio_shape[1:])
    if tuple(noise_shape) != (video_shape[0], 1, full_elements):
        raise ValueError("SelfLift: tiled memory planning received a different latent shape than the sampling input")
    available = _available_workspace(model)
    candidates = range(1, 9) if mode == "auto" else (requested,)
    for count in candidates:
        regions = _regions_for_plan(config, video_shape[axis], count)
        budget_shape, tile_shape, buffer_bytes, preferred, minimum = _budget(
            model, noise_shape, conds, latent_shapes, regions, axis)
        if minimum <= available:
            break
    count = len(regions)
    config.update(axis=axis, tiles=count, regions=regions, largest_tile=tuple(tile_shape),
                  video_shape=tuple(video_shape), audio_shape=tuple(audio_shape),
                  available_mib=available / 2**20, minimum_mib=minimum / 2**20,
                  preferred_mib=preferred / 2**20, estimate_fits=minimum <= available)
    logging.info("[SelfLift tiling plan] mode=%s axis=%s tiles=%d requested=%s target=%.2f MiB estimate_fits=%s",
                 mode, 'H' if axis == 3 else 'W', count, requested if mode == "manual" else "auto",
                 available / 2**20, minimum <= available)
    if minimum > available:
        logging.warning("selflift-Avatar: tiling estimate exceeds available workspace; OOM is possible. "
                        "No automatic OOM retry; increase tiles or reduce resolution/length.")
    if on_plan is not None:
        on_plan(_plan_for_ui(config))
    if count == 1:
        return executor(model, noise_shape, conds, model_options=model_options,
                        force_full_load=force_full_load, force_offload=force_offload)
    logging.info("[SelfLift tiling memory] largest_tile=%s full_audio=%s full_state_buffers=%.2f MiB "
                 "minimum=%.2f MiB preferred=%.2f MiB (estimates, not measured peaks)",
                 tuple(tile_shape), audio_shape, buffer_bytes * noise_shape[0] / 2**20,
                 minimum / 2**20, preferred / 2**20)
    return executor(model, budget_shape, conds, model_options=model_options,
                    force_full_load=force_full_load, force_offload=force_offload)


def _available_workspace(model):
    manager = comfy.model_management
    free = manager.get_free_memory(model.load_device)
    reclaimable = 0
    seen = set()
    for loaded in manager.loaded_models():
        # Model-manager weak references can expire between sampling stages.
        if loaded is None:
            continue
        patcher = loaded if callable(getattr(loaded, "loaded_size", None)) else loaded.model
        if patcher is None:
            continue
        load_device = getattr(patcher, "load_device", None)
        identity = id(patcher.model)
        if load_device == model.load_device and identity not in seen:
            seen.add(identity)
            size_fn = getattr(patcher, "loaded_size", None)
            if callable(size_fn):
                reclaimable += size_fn()
    pool = min(manager.get_total_memory(model.load_device), free + reclaimable)
    weights = min(model.model_size(), pool * manager.MIN_WEIGHT_MEMORY_RATIO)
    available = max(0, pool - weights - manager.minimum_inference_memory())
    logging.info("[SelfLift tiling capacity] free=%.2f MiB reclaimable_weights=%.2f MiB "
                 "weight_allowance=%.2f MiB workspace=%.2f MiB",
                 free / 2**20, reclaimable / 2**20, weights / 2**20, available / 2**20)
    return available


def tiled_model(model, latent_shapes, mode="auto", tiles=2, axis="auto", on_plan=None,
               overlap=None, content_state=None):
    validate_options(mode, tiles, axis)
    if not isinstance(model.model, comfy.model_base.MiniMaxH3):
        raise ValueError("SelfLift: high-resolution tiling requires a MiniMax H3 model")
    if len(latent_shapes) != 2 or len(latent_shapes[0]) != 5 or len(latent_shapes[1]) != 4:
        raise ValueError("SelfLift: high-resolution tiling requires H3 video and audio latent streams")
    if overlap is not None and (not isinstance(overlap, int) or overlap < 1):
        raise ValueError("SelfLift: tiling overlap must be a positive integer or None")
    plan = {"mode": mode, "requested_tiles": tiles, "axis_choice": axis,
            "overlap": overlap, "content_state": content_state}
    patched = model.clone()
    patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                                 "selflift_high_resolution_tiling", partial(_tiled_forward, plan=plan))
    patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING,
                                 "selflift_high_resolution_tiling",
                                 partial(_prepare_tiled_sampling, latent_shapes=tuple(tuple(shape) for shape in latent_shapes), plan=plan, on_plan=on_plan))
    return patched
