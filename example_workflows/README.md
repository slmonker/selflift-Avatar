# Example workflows

## `H3+selflift-avatar-fixed.json`

This workflow demonstrates `SelfLiftAvatarH3Sampler` with the full-sequence
global-context path:

```text
highres_tiling          = true
highres_tiling_mode     = global_stream
highres_token_chunk     = 4096
highres_weight_cache_gb = 4
```

It also includes `Model Attention Backend`, `MiniMax Low VRAM Attention`,
`MiniMax Chunk Feed Forward`, and `Block Sparse Attention` nodes used by the
maintainer's H3 setup.

### Before running

- Install the custom nodes required by the node types in the workflow.
- Download the model files listed in the workflow's Markdown notes. Model
  weights are not included in this repository.
- Replace the local reference image in the `LoadImage` node.
- Restart the ComfyUI backend and hard-refresh the browser after updating this
  plugin.

### Notes

- `global_stream` needs native SLA/sol-attn on every H3 block for the fastest
  memory-saving path. Without it, the sampler logs a fallback and uses the
  full-sequence global path.
- If TST or another attention override is active, the sampler preserves it and
  falls back to full-sequence dense attention with chunked MLP.
- A ComfyUI `Model Attention Backend` selector is removed only for the
  full-sequence fallback; the fallback uses default PyTorch attention to avoid
  long-sequence backend instability.
- `spatial` requires `video mask = 1` everywhere and `audio mask = 0`
  everywhere. Global modes support arbitrary full-sequence video/audio masks.
- The workflow uses a `global_stream` configuration, but the actual path is
  shown in the node status panel and the console. Do not judge the result from
  the first high-resolution preview alone; let all high-resolution steps finish.

## `Selflift-avatar参考生视频_多提示词版_硬Latent续接版1124.json`

Reference-to-video long-form example using four editable segment prompts, automatic loop allocation, Motion Context, H3 Context Windows, Add Guide, and the bundled seam nodes. `H3LatentGridSnap` sits directly after the base H3 AV latent source and before every latent consumer.

- Resolution: 1.5 MP, 16:9
- Chunk frames: 209
- Overlap: 22 frames
- Extra settle trim: 10 frames
- SelfLift: `global_stream`, token chunk 4096, 4 GiB weight cache
- Optional seam nodes: `H3LatentTailPin`, `H3SeamColorAlign`, `H3SeamFrameCover`

## `Selflift-avatar数字人长视频自循环1124.json`

Audio-driven digital-human long-form example with automatic loop allocation from the selected audio duration and the same seam-preservation nodes. `H3LatentGridSnap` sits directly after the H3 AV latent source and before Motion Context and the initial loop-state latent.

- Resolution: 1.0 MP, 16:9
- Chunk frames: 209
- Overlap: 22 frames
- Extra settle trim: 10 frames
- SelfLift: `global_stream`, token chunk 4096, 4 GiB weight cache

The 1124 workflows preserve their original megapixel selectors. `H3LatentGridSnap` converts a request such as `1376x768` to the sampler's valid `1408x768` grid before Motion Context sees it, so the current and previous chunks always have the same latent shape.

### Before running the 1124 examples

- Install all custom nodes used by the workflow, including the H3 Context Windows, Motion Context, KJNodes `ColorMatchV2`, and any attention/optimization nodes shown in the graph.
- Model files are not included. Replace local reference images and audio paths.
- Run these workflows only after restarting ComfyUI so the bundled nodes are loaded from `selflift-Avatar`.
- The 1124 examples use `ref_image_size=match` so reference images are encoded at the generation canvas instead of LynnReal's official `max` path, which upsizes references to a 2048 px short edge and can cost roughly 7.4-7.7 GiB per request.
- The 1.5 MP Ref2VA example can still use nearly all 32 GB of VRAM on a 5090-class GPU. Lower `megapixels`, `highres_token_chunk`, or `highres_weight_cache_gb` when memory is tight.

## Credits

Special thanks to [Fish4AI](https://github.com/Fish4AI), author of
[comfyui-SelfLift-Fish](https://github.com/Fish4AI/comfyui-SelfLift-Fish).
The full-sequence global-context implementation used by this workflow is
adapted from that project and integrated with selflift-Avatar's masks and node
wiring. Thanks also to facok for the original comfyui-SelfLift project.
