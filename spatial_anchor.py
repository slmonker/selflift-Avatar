"""Experimental global low-resolution anchoring for classic spatial tiling.

The low-resolution SelfLift stage already produces an aligned full-frame
video latent.  This module reuses that latent as an H3 video reference on
the high-resolution stage.  The original sampler does not call this module,
so existing workflows and A/B comparisons remain unchanged.
"""

from __future__ import annotations

import torch


def append_global_video_reference(conditioning, latent):
    """Return conditioning with one uncropped full-frame video reference.

    H3 packs visual references separately from the target video.  The classic
    spatial tiler already leaves ``minimax_refs`` untouched for every tile, so
    adding the low-resolution stage result here gives each strip the same
    global composition context without changing the target latent or masks.

    The input conditioning is not mutated.
    """
    if conditioning is None:
        return None
    if not torch.is_tensor(latent) or latent.ndim != 5:
        raise ValueError(
            "selflift-Avatar spatial anchor requires a 5D H3 video latent "
            f"[B,C,T,H,W]; got {type(latent).__name__} shape={getattr(latent, 'shape', None)}"
        )

    if any(int(v) <= 0 for v in latent.shape[2:]):
        raise ValueError("selflift-Avatar spatial anchor latent has an empty dimension")

    ref = {
        "kind": "video",
        "latent_t": int(latent.shape[2]),
        "latent_h": int(latent.shape[3]),
        "latent_w": int(latent.shape[4]),
        # H3's PackedLayout requires this field for every video reference,
        # even when the reference has no soundtrack.
        "ref_audio_t": 0,
        "latent": latent,
    }
    out = []
    for item in conditioning:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            raise ValueError("selflift-Avatar received malformed conditioning for spatial anchoring")
        embedding, values = item[0], item[1]
        values = dict(values)
        refs = list(values.get("minimax_refs", []))
        refs.append(ref)
        values["minimax_refs"] = refs
        out.append([embedding, values])
    return out
