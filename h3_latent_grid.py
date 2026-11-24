import logging

import torch
import comfy.nested_tensor

from .h3_latent_tail_pin import _existing_masks, _streams

LOG = logging.getLogger("H3LatentGridSnap")

H3_SPATIAL_LATENT_MULTIPLE = 4


def _resize_video_spatial(tensor, height, width, mode="bilinear"):
    if tuple(tensor.shape[-2:]) == (height, width):
        return tensor
    b, c, t, h, w = tensor.shape
    frames = tensor.float().permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    frames = torch.nn.functional.interpolate(frames, size=(height, width), mode=mode)
    return frames.reshape(b, t, c, height, width).permute(0, 2, 1, 3, 4).to(tensor.dtype)


def snap_h3_latent_spatial(height, width, multiple=H3_SPATIAL_LATENT_MULTIPLE):
    """Choose the nearest H3 latent size aligned to the VAE/DiT grid."""
    height = int(height)
    width = int(width)
    multiple = int(multiple)
    if height % multiple == 0 and width % multiple == 0:
        return height, width

    target_ratio = width / float(height)
    target_area = width * height

    def candidates(value):
        low = max(multiple, ((value - 2 * multiple) // multiple) * multiple)
        high = max(low, value + 2 * multiple)
        return range(low, high + 1, multiple)

    best = (height, width)
    best_score = float("inf")
    for candidate_h in candidates(height):
        for candidate_w in candidates(width):
            ratio_error = (
                abs(candidate_w / float(candidate_h) - target_ratio)
                / max(target_ratio, 1e-9)
            )
            area_error = (
                abs(candidate_w * candidate_h - target_area)
                / max(target_area, 1)
            )
            score = ratio_error * 10.0 + area_error
            if score < best_score:
                best_score = score
                best = (candidate_h, candidate_w)
    return best


class H3LatentGridSnap:
    """Snap the video stream of an H3 AV latent onto the sampler grid.

    The audio stream and its temporal axis are left untouched. This node is
    intended immediately after the H3 conditioning/latent node and before
    Motion Context or SelfLift. Doing the snap at the source means every
    consumer in a chained workflow sees the same spatial shape; the sampler
    no longer has to resize a latent after Motion Context has already compared
    the previous and current resolutions.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {
                    "tooltip": "H3 audio-video latent to align to the nearest valid spatial grid."
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "apply"
    CATEGORY = "selflift-Avatar/latent"
    DESCRIPTION = (
        "Resize only the spatial axes of an H3 video latent to the nearest "
        "4-latent-pixel grid (64 output pixels). Audio and all temporal "
        "dimensions are unchanged. Put this before Motion Context in chained "
        "graphs so previous and current chunks have identical dimensions."
    )

    def apply(self, latent):
        video, audio = _streams(latent)
        if video.ndim != 5:
            raise ValueError(
                "H3 Latent Grid Snap: video latent must be [B,C,T,H,W], got %s"
                % (tuple(video.shape),)
            )
        if audio.ndim != 4:
            raise ValueError(
                "H3 Latent Grid Snap: audio latent must be [B,C,2,T], got %s"
                % (tuple(audio.shape),)
            )

        source_h, source_w = map(int, video.shape[-2:])
        target_h, target_w = snap_h3_latent_spatial(source_h, source_w)
        if (target_h, target_w) == (source_h, source_w):
            LOG.info(
                "H3 Latent Grid Snap: spatial %s already aligned (%dx%d px)",
                (source_h, source_w), target_w * 16, target_h * 16,
            )
            return (latent.copy(),)

        masks = _existing_masks(latent, (video, audio))
        video = _resize_video_spatial(video, target_h, target_w, mode="bilinear")
        video_mask = _resize_video_spatial(
            masks[0], target_h, target_w, mode="nearest"
        )

        result = latent.copy()
        result["samples"] = comfy.nested_tensor.NestedTensor((video, audio))
        if latent.get("noise_mask") is not None:
            result["noise_mask"] = comfy.nested_tensor.NestedTensor(
                (video_mask, masks[1])
            )

        LOG.info(
            "H3 Latent Grid Snap: spatial %s -> %s (pixels %dx%d -> %dx%d); "
            "audio and temporal axes unchanged",
            (source_h, source_w), (target_h, target_w),
            source_w * 16, source_h * 16, target_w * 16, target_h * 16,
        )
        return (result,)