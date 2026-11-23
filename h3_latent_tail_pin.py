import logging

import torch

import comfy.nested_tensor
import comfy.utils

LOG = logging.getLogger("H3LatentTailPin")

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FPS = 24
AUDIO_HZ = 40.0


def _streams(latent):
    samples = latent.get("samples")
    if samples is None:
        raise ValueError("H3 Latent Tail Pin: latent has no samples")
    if getattr(samples, "is_nested", False):
        parts = list(samples.unbind())
    elif isinstance(samples, (tuple, list)):
        parts = list(samples)
    else:
        raise ValueError(
            "H3 Latent Tail Pin: expected a MiniMax H3 AV latent with video and audio streams"
        )
    if len(parts) != 2:
        raise ValueError(
            "H3 Latent Tail Pin: expected exactly two streams (video, audio), got %d"
            % len(parts)
        )
    return parts


def _existing_masks(latent, streams):
    mask = latent.get("noise_mask")
    if mask is None:
        return [torch.ones_like(stream, dtype=torch.float32) for stream in streams]
    if getattr(mask, "is_nested", False):
        masks = list(mask.unbind())
    elif isinstance(mask, (tuple, list)):
        masks = list(mask)
    else:
        masks = [mask]
    if len(masks) == 1:
        masks = masks + [None] * (len(streams) - 1)
    if len(masks) != len(streams):
        raise ValueError(
            "H3 Latent Tail Pin: existing noise_mask stream count does not match AV streams"
        )
    return [_fit_mask(m, stream) for m, stream in zip(masks, streams)]


def _fit_mask(mask, stream):
    if mask is None:
        return torch.ones_like(stream, dtype=torch.float32)
    mask = mask.to(device=stream.device, dtype=torch.float32).clamp(0.0, 1.0)
    if tuple(mask.shape) == tuple(stream.shape):
        return mask
    try:
        return mask.expand_as(stream)
    except RuntimeError:
        pass
    # SetLatentNoiseMask stores audio constraints as an image-sized constant
    # mask. Collapse that constant to one broadcastable cell.
    if mask.ndim >= 3 and mask.numel() > 1:
        spatial = mask.reshape(mask.shape[0], mask.shape[1], -1)
        if torch.all(spatial == spatial[..., :1]):
            collapsed = spatial[..., :1].reshape(mask.shape[0], mask.shape[1], *((1,) * (mask.ndim - 2)))
            try:
                return collapsed.expand_as(stream)
            except RuntimeError:
                pass
    return comfy.utils.reshape_mask(mask, stream.shape).to(stream)

def _resize_video_spatial(tensor, height, width, mode="bilinear"):
    if tuple(tensor.shape[-2:]) == (height, width):
        return tensor
    b, c, t, h, w = tensor.shape
    frames = tensor.float().permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    frames = torch.nn.functional.interpolate(frames, size=(height, width), mode=mode)
    return frames.reshape(b, t, c, height, width).permute(0, 2, 1, 3, 4).to(tensor.dtype)


def _steps_for_frames(frames):
    frames = int(frames)
    if frames <= 0:
        raise ValueError("H3 Latent Tail Pin: pin_frames must be positive")
    steps = 0
    covered = 0
    while covered < frames:
        covered += FRAME_PER_TOKEN[steps % len(FRAME_PER_TOKEN)]
        steps += 1
    if covered != frames:
        raise ValueError(
            "H3 Latent Tail Pin: %d frames is not a whole H3 latent-step window. "
            "Use 5, 22, 39 or 56 frames." % frames
        )
    return steps


class H3LatentTailPin:
    """Hard-pin the previous H3 AV tail into the current chunk.

    The output latent contains the copied previous tail at the head. Its
    noise_mask keeps those video/audio latent steps at 0 (preserve) and sets
    the remaining sequence to 1 (generate). SelfLift-Avatar consumes this
    mask through latent_image['noise_mask'] and restores the kept region.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "current_latent": ("LATENT", {
                    "tooltip": "Target H3 AV latent for the new chunk."
                }),
                "previous_latent": ("LATENT", {
                    "tooltip": "Previous chunk's sampler output AV latent."
                }),
                "pin_frames": (["5", "22", "39", "56"], {
                    "default": "22",
                    "tooltip": "Pixel-frame overlap to hard-pin. Must be 17k+5; 22 = 7 video latent steps."
                }),
                "pin_audio": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Also copy and preserve the matching audio tail."
                }),
                "bypass": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Return the current latent unchanged. Connect StartLoop.is_first for the first chunk."
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT", "INT")
    RETURN_NAMES = ("latent", "pin_frames", "video_steps", "audio_steps")
    FUNCTION = "apply"
    CATEGORY = "selflift-Avatar/latent"
    DESCRIPTION = (
        "Copy the previous H3 AV latent tail into the current chunk head and "
        "create a temporal noise mask: 0=keep pinned tail, 1=generate. "
        "Use global/global_stream/global_ram in SelfLift; spatial tiling rejects partial masks."
    )

    def apply(self, current_latent, previous_latent, pin_frames="22", pin_audio=True, bypass=False):
        pin_frames = int(pin_frames)
        video_steps = _steps_for_frames(pin_frames)
        audio_steps = int(round(pin_frames / float(FPS) * AUDIO_HZ))
        if bypass:
            LOG.info("H3 Latent Tail Pin: bypassed; current latent returned unchanged")
            return (current_latent.copy(), 0, 0, 0)

        current_video, current_audio = _streams(current_latent)
        previous_video, previous_audio = _streams(previous_latent)
        existing_video_mask, existing_audio_mask = _existing_masks(
            current_latent, (current_video, current_audio)
        )

        current_video = current_video.clone()
        current_audio = current_audio.clone()

        if current_video.ndim != 5:
            raise ValueError(
                "H3 Latent Tail Pin: video latent must be [B,C,T,H,W], got %s"
                % (tuple(current_video.shape),)
            )
        if current_audio.ndim != 4:
            raise ValueError(
                "H3 Latent Tail Pin: audio latent must be [B,C,2,T], got %s"
                % (tuple(current_audio.shape),)
            )
        if previous_video.ndim != 5 or previous_audio.ndim != 4:
            raise ValueError("H3 Latent Tail Pin: previous latent is not an H3 AV latent")
        if int(current_video.shape[1]) != int(previous_video.shape[1]):
            raise ValueError(
                "H3 Latent Tail Pin: video channel mismatch: %s vs %s"
                % (tuple(current_video.shape), tuple(previous_video.shape))
            )
        if tuple(current_video.shape[-2:]) != tuple(previous_video.shape[-2:]):
            target_h, target_w = map(int, previous_video.shape[-2:])
            source_shape = tuple(current_video.shape)
            current_video = _resize_video_spatial(current_video, target_h, target_w, mode="bilinear")
            existing_video_mask = _resize_video_spatial(existing_video_mask, target_h, target_w, mode="nearest")
            LOG.info(
                "H3 Latent Tail Pin: aligned current latent spatial %s -> %s "
                "to match the previous sampler tail",
                source_shape, tuple(current_video.shape),
            )
        if int(current_video.shape[2]) < video_steps:
            raise ValueError(
                "H3 Latent Tail Pin: current video has %d steps, need %d"
                % (int(current_video.shape[2]), video_steps)
            )
        if int(previous_video.shape[2]) < video_steps:
            raise ValueError(
                "H3 Latent Tail Pin: previous video has %d steps, need %d"
                % (int(previous_video.shape[2]), video_steps)
            )

        prev_v_tail = previous_video[:, :, -video_steps:].to(
            device=current_video.device, dtype=current_video.dtype
        )
        current_video[:, :, :video_steps].copy_(prev_v_tail)

        mask_video = torch.ones_like(
            current_video[:, :1, :, :1, :1], dtype=torch.float32
        )
        mask_video[:, :, :video_steps] = 0.0
        mask_video = torch.minimum(existing_video_mask, mask_video.expand_as(current_video))

        if pin_audio:
            if int(current_audio.shape[-1]) < audio_steps:
                raise ValueError(
                    "H3 Latent Tail Pin: current audio has %d steps, need %d"
                    % (int(current_audio.shape[-1]), audio_steps)
                )
            if int(previous_audio.shape[-1]) < audio_steps:
                raise ValueError(
                    "H3 Latent Tail Pin: previous audio has %d steps, need %d"
                    % (int(previous_audio.shape[-1]), audio_steps)
                )
            prev_a_tail = previous_audio[..., -audio_steps:].to(
                device=current_audio.device, dtype=current_audio.dtype
            )
            current_audio[..., :audio_steps].copy_(prev_a_tail)
            mask_audio = torch.ones_like(
                current_audio[:, :1, :1, :], dtype=torch.float32
            )
            mask_audio[..., :audio_steps] = 0.0
            mask_audio = torch.minimum(existing_audio_mask, mask_audio.expand_as(current_audio))
        else:
            mask_audio = existing_audio_mask

        result = current_latent.copy()
        result["samples"] = comfy.nested_tensor.NestedTensor((current_video, current_audio))
        result["noise_mask"] = comfy.nested_tensor.NestedTensor((mask_video, mask_audio))

        LOG.info(
            "H3 Latent Tail Pin: pin_frames=%d video_steps=%d audio_steps=%d "
            "video_mask=%s audio_mask=%s pin_audio=%s",
            pin_frames, video_steps, audio_steps,
            tuple(mask_video.shape), tuple(mask_audio.shape), bool(pin_audio),
        )
        return (result, pin_frames, video_steps, audio_steps)


__all__ = ["H3LatentTailPin"]
