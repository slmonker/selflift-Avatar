import math

import torch
import torch.nn.functional as F


class H3SeamColorAlign:
    """Smoothly align the color/brightness of a chunk start to the previous tail."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "current_frames": ("IMAGE", {
                    "tooltip": "Current chunk images after ColorMatchV2."
                }),
                "previous_frames": ("IMAGE", {
                    "tooltip": "Last frames from the previous already-aligned chunk."
                }),
                "align_frames": ("INT", {
                    "default": 36, "min": 1, "max": 240, "step": 1,
                    "tooltip": "Number of frames at the start of the current chunk that receive the color transition."
                }),
                "analysis_frames": ("INT", {
                    "default": 12, "min": 1, "max": 120, "step": 1,
                    "tooltip": "Frame count used to estimate the previous/current color statistics."
                }),
                "strength": ("FLOAT", {
                    "default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Maximum color correction strength at the seam."
                }),
                "scale_limit": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "Maximum contrast gain change around 1.0. 0.15 means 0.85-1.15."
                }),
                "bypass": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Return the current frames unchanged. Connect StartLoop.is_first."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("aligned_frames", "last_frames")
    FUNCTION = "align"
    CATEGORY = "selflift-Avatar/image"
    DESCRIPTION = (
        "Align the first N frames of a new chunk to the previous chunk tail using "
        "per-channel mean/std matching with a smooth temporal falloff. Color and "
        "brightness only; no geometry or motion changes."
    )

    @staticmethod
    def _resize_like(frames, target):
        if tuple(frames.shape[1:3]) == tuple(target.shape[1:3]):
            return frames
        x = frames[..., :3].permute(0, 3, 1, 2).float()
        x = F.interpolate(x, size=tuple(target.shape[1:3]), mode="bilinear", align_corners=False)
        return x.permute(0, 2, 3, 1).to(device=target.device, dtype=target.dtype)

    @staticmethod
    def _stats(frames):
        x = frames[..., :3].permute(0, 3, 1, 2).float()
        # A small analysis image is enough for color statistics and avoids a
        # large temporary allocation on 1080p clips.
        x = F.interpolate(x, size=(144, 256), mode="area")
        mean = x.mean(dim=(0, 2, 3))
        std = x.std(dim=(0, 2, 3), unbiased=False).clamp_min(1e-4)
        return mean, std

    def align(self, current_frames, previous_frames, align_frames=36,
              analysis_frames=12, strength=0.9, scale_limit=0.15, bypass=False):
        if current_frames.ndim != 4 or current_frames.shape[-1] < 3:
            raise ValueError("H3 Seam Color Align: current_frames must be [B,H,W,C] RGB")
        if previous_frames.ndim != 4 or previous_frames.shape[-1] < 3:
            raise ValueError("H3 Seam Color Align: previous_frames must be [B,H,W,C] RGB")

        current_frames = current_frames.float()
        previous_frames = previous_frames.float()
        total = int(current_frames.shape[0])
        analysis_n = min(int(analysis_frames), int(previous_frames.shape[0]), total)
        out = current_frames.clone()

        if bypass or strength <= 0.0 or analysis_n <= 0:
            return (out, out[-analysis_n:].clone() if analysis_n else out)

        previous_frames = self._resize_like(previous_frames, current_frames)
        prev_mean, prev_std = self._stats(previous_frames[-analysis_n:])
        cur_mean, cur_std = self._stats(current_frames[:analysis_n])

        limit = max(0.0, float(scale_limit))
        scale = (prev_std / cur_std).clamp(1.0 - limit, 1.0 + limit)
        offset = prev_mean - scale * cur_mean

        align_n = min(max(1, int(align_frames)), total)
        progress = torch.linspace(0.0, 1.0, align_n, device=current_frames.device, dtype=torch.float32)
        weights = 0.5 * (1.0 + torch.cos(math.pi * progress))
        weights = (weights * float(strength)).view(align_n, 1, 1, 1)

        rgb = current_frames[:align_n, ..., :3]
        matched = (rgb * scale.view(1, 1, 1, 3) + offset.view(1, 1, 1, 3)).clamp(0.0, 1.0)
        out[:align_n, ..., :3] = rgb * (1.0 - weights) + matched * weights
        if current_frames.shape[-1] > 3:
            out[:align_n, ..., 3:] = current_frames[:align_n, ..., 3:]

        return (out, out[-analysis_n:].clone())


class H3SeamFrameCover:
    """Cover the first frame(s) of a new chunk with the previous chunk's last frame."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "current_frames": ("IMAGE", {
                    "tooltip": "Current chunk images after ColorMatchV2."
                }),
                "previous_frames": ("IMAGE", {
                    "tooltip": "Last visible frames from the previous chunk after ColorMatchV2."
                }),
                "cover_frames": ("INT", {
                    "default": 1, "min": 0, "max": 8, "step": 1,
                    "tooltip": "Replace this many leading frames with the previous chunk's last frame."
                }),
                "blend_frames": ("INT", {
                    "default": 1, "min": 0, "max": 8, "step": 1,
                    "tooltip": "After the covered frames, blend this many frames with the previous last frame."
                }),
                "bypass": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Return the current frames unchanged. Connect StartLoop.is_first."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "cover"
    CATEGORY = "selflift-Avatar/image"
    DESCRIPTION = (
        "Replace the leading frame(s) of a new chunk with the previous chunk's last frame, "
        "then optionally blend following frames at 50%, 25%, ... strength. "
        "Frame count and audio timing stay unchanged."
    )

    @staticmethod
    def _fit(previous, current):
        if tuple(previous.shape[1:3]) == tuple(current.shape[1:3]):
            return previous.to(device=current.device, dtype=current.dtype)
        x = previous[..., :3].permute(0, 3, 1, 2).float()
        x = F.interpolate(x, size=tuple(current.shape[1:3]), mode="bilinear", align_corners=False)
        return x.permute(0, 2, 3, 1).to(device=current.device, dtype=current.dtype)

    def cover(self, current_frames, previous_frames, cover_frames=1,
              blend_frames=1, bypass=False):
        if current_frames.ndim != 4 or current_frames.shape[-1] < 3:
            raise ValueError("H3 Seam Frame Cover: current_frames must be [B,H,W,C] RGB")
        if previous_frames.ndim != 4 or previous_frames.shape[-1] < 3:
            raise ValueError("H3 Seam Frame Cover: previous_frames must be [B,H,W,C] RGB")

        out = current_frames.float().clone()
        if bypass or previous_frames.shape[0] < 1 or out.shape[0] < 1:
            return (out,)

        previous_last = self._fit(previous_frames[-1:], out)
        total = int(out.shape[0])
        cover_n = min(max(0, int(cover_frames)), total)
        if cover_n:
            out[:cover_n, ..., :3] = previous_last.expand(cover_n, -1, -1, -1)

        blend_n = min(max(0, int(blend_frames)), total - cover_n)
        for k in range(blend_n):
            index = cover_n + k
            weight_previous = 0.5 ** (k + 1)
            current_rgb = current_frames[index, ..., :3].float()
            mixed = previous_last[0, ..., :3] * weight_previous + current_rgb * (1.0 - weight_previous)
            out[index, ..., :3] = mixed.clamp(0.0, 1.0)

        return (out,)
__all__ = ["H3SeamColorAlign", "H3SeamFrameCover"]
