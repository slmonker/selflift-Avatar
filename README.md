# selflift-Avatar

**Development: `main` (model input naming and layout)** · **Packaged release: `v0.1.4-experimental`** · [中文说明](README_CN.md) · [Changelog](CHANGELOG.md) · [v0.1.4 release](https://github.com/slmonker/selflift-Avatar/releases/tag/v0.1.4-experimental)

An independent experimental SelfLift branch for ComfyUI / MiniMax H3. It focuses on H3 audio-video latent masks, preserving source audio during sampling, and a narrowly supported high-resolution tiling path. It registers separate node IDs, so it can coexist with the original SelfLift plugin.

> **Experimental, unofficial project.** The current code has passed 53 CPU tests and two frontend suites with stubs. Earlier versions were exercised in the maintainer's H3 workflow; the new controls still need broader real-model and frontend testing. This is not a systematic benchmark of lip-sync accuracy, speed, or every model/plugin combination. Results can vary by character, audio, prompt, sampler, latent upscaler, and seed.

## Main-branch update: model input naming and layout (2026-10-05)

**Dual-model sampling was already supported through `model` and `model_hires`; this update does not introduce that capability.** Both the MiniMax H3 and Image samplers now use clearer names: `model` → `low_res_model`, and `model_hires` → `high_res_model`. The optional high-resolution input retains its existing fallback to the low-resolution model.

This update groups the two sockets together, corrects the high-resolution sampling call to use the selected model's load device, validates that model's sampler compatibility, and adds regression tests. It does not add forced model unloading or an OOM-prevention guarantee.

| Input | Required | Used for |
| --- | --- | --- |
| `low_res_model` | Yes | The first `transition_step` low-resolution denoiser evaluations. |
| `high_res_model` | No | Remaining evaluations after the resolution lift. If disconnected, reuses `low_res_model`. |

The two sockets appear together, with `high_res_model` directly below `low_res_model`. No extra switch or sampler node is needed.

```text
Model A ── low_res_model  → low-resolution prefix
Model B ── high_res_model → high-resolution continuation (optional)
```

For a single-model workflow, connect only `low_res_model`. High-resolution tiling and masking use the selected high-resolution model. Model loading/offloading remains managed by ComfyUI; this does not guarantee that both models reside in VRAM simultaneously.

### Upgrade / reconnect existing workflows

**Breaking input rename:** `model` is now `low_res_model`, and `model_hires` is now `high_res_model`; the old input names are no longer accepted. Back up your workflow before updating, restart the ComfyUI backend, and hard-refresh the browser. Re-add the sampler if stale sockets remain, then reconnect:

- Previous `model` connection → `low_res_model`.
- Previous `model_hires` connection → `high_res_model` (if used).
- Reconnect the remaining inputs and restore your sampler settings before running.

The separate H3 TST patch node still uses its original `model` socket. Socket ordering does not migrate obsolete workflow inputs automatically.

### Compatibility and validation

Both stages share the VAE, positive/negative conditioning, CFG, sampler and sigma schedule. Use compatible architectures, latent formats and sampling parameterizations, such as different checkpoints or LoRA stacks for the same backbone—not arbitrary model families. The switch follows the **resolution transition**, not an independent high-noise/low-noise threshold.

**53 CPU tests** and both Node.js frontend suites pass. Coverage includes model selection, single-model fallback, stage boundaries, masks/tiling, socket ordering and preserving link targets when reordering. These are synthetic-denoiser and stub-frontend tests; real dual-model output quality, peak VRAM and all frontend/subgraph combinations remain unverified.

This update is available from the repository's `main` branch. The existing `v0.1.4-experimental` release archive does **not** contain it; no new release archive is created with this update.

## Previous upgrade: v0.1.4

**Automatic mode now shows the actual tile count on the first line of the node status panel, not just `auto`.**

- Before high-resolution preparation: **Actual tiles: pending**. Auto planning is not available during the low-resolution prefix.
- Once planned: **Actual tiles: N**, including odd counts selected by auto mode. **1 tile** is explicitly labeled **full frame / no split**.
- The completed result stays visible; changing tiling controls or starting a new run clears the old count.
- Actual results are delivered as structured plan data for live events and completed/cached node UI results. Older cached text is still recognized.
- Add a numeric-node-ID lookup fallback for live updates. This is not a general subgraph-routing fix.
- The manual **2 / 4 / 6 / 8** dropdown is never overwritten by auto results.

**No sampling algorithm change:** tile selection, spatial overlap, audio-mask constraints, temporal upscaler windows and VRAM estimation remain unchanged. **47 CPU tests** plus extended frontend callback tests with a stub DOM passed. Restart the ComfyUI backend and hard-refresh the browser after updating. These tests are not a full browser or real-model benchmark.

[Download v0.1.4](https://github.com/slmonker/selflift-Avatar/releases/tag/v0.1.4-experimental) · [Full changelog](CHANGELOG.md)

## Previous upgrade: v0.1.3

**This upgrade makes spatial tiling selectable and visible directly on the H3 sampler node.**

| Area | v0.1.2 | v0.1.3 |
| --- | --- | --- |
| Tile count | Automatic memory-based planning only | Choose **auto** or **manual**; manual dropdown offers **2 / 4 / 6 / 8**, default **2** |
| Direction | Automatically use the longer patch-grid side | Choose **auto / width / height**, in either count mode |
| Plan display | Read console logs | Read-only **in-node status panel** with actual count, direction, ranges, overlap and largest tile |
| Memory feedback | Console estimates | Estimated available/minimum workspace and insufficient-memory warnings on the node; **not measured VRAM peaks** |
| Tests | 34 CPU tests | **47 CPU tests** plus frontend callback tests with a stub DOM |

**Unchanged:** auto mode still searches **1–8** tiles, including odd counts; overlap is automatic; temporal upscaler windows are unaffected. Masked tiling still requires **video mask all 1 / audio mask all 0**. Manual mode does not silently increase the count and has no automatic OOM retry. This release does not claim a measured speedup, VRAM-saving percentage or improved lip-sync accuracy.

**After upgrading:** restart the backend and hard-refresh the browser. If an existing sampler does not show the new controls, recreate that node. If an older experimental workflow stored 1 or an odd manual count, re-select **2 / 4 / 6 / 8**. Existing published v0.1.2 workflows default to auto.

[Full changelog](CHANGELOG.md) · [v0.1.3 release and installation ZIP](https://github.com/slmonker/selflift-Avatar/releases/tag/v0.1.3-experimental)

## Selectable spatial tiling — v0.1.3

The H3 sampler now exposes optional controls. Existing workflows keep the old defaults.

| Control | Values | Meaning |
| --- | --- | --- |
| `highres_tiling` | false / true | Enable spatial tiling; off ignores the controls below. |
| `tiling_mode` | auto / manual | Auto selects the first estimated fitting count from 1–8. Manual uses the requested count without silently increasing it. |
| `tiling_tiles` | 2 / 4 / 6 / 8 (default 2) | Manual dropdown. Turn off highres_tiling for full-frame processing. Tiny dimensions may reduce the effective count. |
| `tiling_axis` | auto / width / height | Auto uses the longer patch-grid side. Width creates left/right strips; height creates top/bottom strips. Applies in both modes. |

The node includes a read-only status panel: current settings, last actual plan, effective tile count, spatial ranges, overlaps, maximum latent tile, and estimated available/minimum workspace. It updates at high-resolution preparation; before that the automatic result is unknown. Root-graph nodes receive a live event; completion/cached execution uses the ordinary node UI result. Subgraph live-event routing is not verified. Changing controls invalidates the displayed previous plan. Values are estimates, not measured VRAM peaks; manual mode does not guarantee fit or retry on OOM.

Overlap remains automatic; these controls do not change the latent upscaler's temporal windows. The video-all-1/audio-all-0 mask restriction and existing ControlNet/TST limitations remain. Restart the backend and hard-refresh the frontend after updating. If an old node does not show new controls, recreate that H3 sampler node. The 47 CPU tests include the old workflow defaults, manual direction/count, effective-count reporting, insufficient-memory warnings, and node UI results. Frontend callbacks are tested with a stub DOM; this is not a full browser or real-model benchmark.

## Inherited audio-only tiling support (v0.1.2)

- Allows high-resolution tiling when the video is fully generated (`video mask = 1` everywhere) and the source audio is fully preserved (`audio mask = 0` everywhere).
- Automatically chooses the spatial tile count and direction from available workspace; the current planner can select 1–8 tiles.
- Every spatial tile receives the complete audio latent and complete H3 audio conditioning. Audio is not split by the visual tile boundary.
- Keeps rejecting partial video preservation, partial/soft audio masks, and other mask combinations that have not been validated with tiling.
- Adds tests for tile stitching, full-audio preservation, H3 audio mask forwarding, invalid mask combinations, and the no-mask path.

## Demo

[![H3 audio-driven demo preview](docs/assets/demo-preview.gif)](https://github.com/slmonker/selflift-Avatar/releases/download/v0.1.2-experimental/selfliftAvatar_00011-audio.mp4)

**H3 audio-driven demo:** [Download / watch the full 13.67-second MP4](https://github.com/slmonker/selflift-Avatar/releases/download/v0.1.2-experimental/selfliftAvatar_00011-audio.mp4) · [open the v0.1.2 release](https://github.com/slmonker/selflift-Avatar/releases/tag/v0.1.2-experimental)

The preview above is a muted excerpt; click it for the complete video with audio. The demo was provided by the maintainer and is published as a public Release asset.

[![SelfLift Avatar workflow demo preview](docs/assets/demo-selflift-avatar-preview.gif)](https://github.com/slmonker/selflift-Avatar/releases/download/v0.1.2-experimental/selflift-Avatar-demo-selflift-avatar.mp4)

**SelfLift Avatar workflow demo:** [Download / watch the full 10.05-second MP4](https://github.com/slmonker/selflift-Avatar/releases/download/v0.1.2-experimental/selflift-Avatar-demo-selflift-avatar.mp4)

This second preview is also muted; click it for the complete video with audio. Both videos are examples of the maintainer's workflows, not universal lip-sync benchmarks. The repository does not include model weights.

## Installation

You need a working MiniMax H3 ComfyUI setup and the models, VAE, LoRA, and optional latent upscaler required by your workflow. This project adds no dependencies and does not include model weights.

### Git installation

```bash
git clone https://github.com/slmonker/selflift-Avatar.git ComfyUI/custom_nodes/selflift-Avatar
```

To update an existing Git installation:

```bash
cd ComfyUI/custom_nodes/selflift-Avatar
git pull --ff-only
```

Restart the ComfyUI backend after updating. A browser refresh alone does not reload Python nodes.

### ZIP installation

Download the latest ZIP from the [Releases page](https://github.com/slmonker/selflift-Avatar/releases), back up the old `selflift-Avatar` folder outside `custom_nodes`, replace it with the new folder, and restart ComfyUI. Do not keep two copies of this plugin in `custom_nodes`.

The original `comfyui-SelfLift` folder can remain installed; this project uses different node IDs.

## Nodes

| Node ID | Display name |
| --- | --- |
| `SelfLiftAvatarH3Sampler` | selflift-Avatar Sampler (MiniMax H3) |
| `SelfLiftAvatarImageSampler` | selflift-Avatar Sampler (Image) |
| `SelfLiftAvatarH3TST` | selflift-Avatar H3 TST |

The sampler uses the standard `sampler` and `sigmas` inputs. Use the standard Euler sampler with `s_churn=0`.

## Audio-driven H3 workflow

A typical source-audio path is:

```text
source audio → trim to target clip → H3 audio VAE encode
                                             ↓
SolidMask(value=0) → SetLatentNoiseMask
                                             ↓
H3 video latent ───────────────→ concatenate AV latent
                                             ↓
                         selflift-Avatar H3 sampler
```

- Audio mask `0` preserves the source audio; audio mask `1` allows audio generation. Values between 0 and 1 are soft constraints.
- The encoded audio, video frame range, and final output FPS must refer to the same clip.
- If the final video node uses the original audio directly, hearing the correct soundtrack does not prove that H3 used it to drive the mouth during generation. Evaluate the generated frames.
- The sampler is not a standalone post-production lip-sync tool.

## High-resolution tiling

Set `highres_tiling=true` to enable high-resolution spatial tiling. Choose `tiling_mode=auto` for estimated-memory planning, or `tiling_mode=manual` for a **2 / 4 / 6 / 8** tile dropdown. Auto mode still considers counts from 1–8 (including odd counts) and may choose one tile, meaning no split. `tiling_axis` can override the direction in either mode. Disable `highres_tiling` for full-frame processing without the tiling planner.

Example: `highres_tiling=true`, `tiling_mode=manual`, `tiling_tiles=4`, `tiling_axis=width` requests four left/right spatial strips. Overlap remains automatic. Manual mode warns if the estimate exceeds available workspace, but does not silently increase the count and does not retry on OOM.

### VRAM estimates are not fixed GPU requirements

There is no reliable mapping such as “2 tiles = 32 GB” or “4 tiles = 24 GB”. The planner combines current free VRAM and estimated reclaimable weights, subtracts reservations, then compares that estimated workspace against the model's estimated minimum needs. Resolution, duration, batch, references, audio, model offloading, and overlapping tiles all affect the result. The node panel shows these **estimates**, not measured peaks. Planning happens before high-resolution sampling, not continuously at every step. No percentage-saving or speed guarantee is claimed.

### Supported masked tiling mode

The currently validated masked mode is deliberately narrow:

```text
video mask: all 1 → generate the complete video
 audio mask: all 0 → preserve the complete source audio
```

Every video tile receives the complete audio latent and complete H3 audio conditioning. The tiles are stitched in the video spatial dimension; audio is not spatially tiled.

### Still rejected with tiling

- Any video mask containing 0 or soft values.
- Partial or soft audio masks.
- ControlNet with the H3 tiling path.

Static masks only: upstream dynamic `denoise_mask_function` schedules are ignored with a warning in both modes. Disabling tiling does not restore dynamic-mask support. Disable `highres_tiling` for the unsupported spatial-mask/ControlNet combinations above. Removing the validation would not make those combinations correct because their masks would also need to be cropped, transformed, and aligned with each tile.

Look for these console messages to confirm the planner:

```text
[selflift-Avatar plan] automatic high-resolution tiling enabled
[SelfLift tiling plan] mode=manual axis=W tiles=4 requested=4 ...
```

`axis=H` or `axis=W` is the actual spatial direction. `tiles=N` is the effective tile count; `requested` is the manual request or `auto`. Tiny dimensions can reduce the effective count. The node status panel reports the same actual plan once high-resolution preparation begins.

## Mask support without tiling

- Regular image/video tensors: BHW, BCHW, and BCTHW; one channel or the actual latent channel count.
- H3 `NestedTensor(video_mask, audio_mask)` with separate video/audio handling.
- Packed masks whose flattened size matches the AV latent streams.
- Audio masks in T, BT, BST, and BCST layouts; dimensions of 1 broadcast.
- Spatial video masks are resized to the latent grid. Audio time is not guessed or automatically resampled.
- A constant image-sized `SolidMask` attached to audio, such as `[1,1,928,1664]`, is recognized as a constant audio constraint and broadcast to the audio latent.

Mask semantics: **0 = preserve, 1 = generate, 0–1 = soft constraint**. A meaningful preserve result requires an initialized latent containing the content to preserve.

## Native H3 audio conditioning

The sampler passes the packed mask into ComfyUI's native H3 conditioning path before sampling. It uses H3's `audio_denoise_mask` and input-side `scale_latent_inpaint` behavior instead of only restoring audio after the model prediction. The high-resolution resume state and the clean audio anchor are kept separate so residual sampling noise is not treated as source audio.

This fixes mask/conditioning transport. It does not guarantee perfect lip sync. The external H3 latent upscaler has temporal convolutions and does not directly receive the target audio; the low-resolution prefix and the number of high-resolution correction steps can also affect mouth detail.

For a controlled comparison, keep the same model, audio clip, prompt, frame count, FPS, and seed. Compare the native baseline, the current SelfLift route, different `transition_step` values, and the `upscaler_model=none` pixel-anchor route one variable at a time.

## Testing

Run with the ComfyUI Python environment:

```bash
python tests/test_avatar.py
```

Optional frontend callback checks (Node.js, no npm packages):

```bash
node tests/test_tiling_ui.mjs
node tests/test_model_input_order.mjs
```

The current suite contains 53 CPU tests covering AV packing, H3 audio scaling, full/partial masks, constant audio SolidMask conversion, native audio conditioning, two-stage sampling, automatic tile planning, tile stitching, complete-audio forwarding to every tile, invalid masked-tiling combinations, and the unmasked tiling path. The tests do not load a large H3 checkpoint or measure final video lip-sync quality.

## Source and licensing

This is based on a local snapshot of [facok/comfyui-SelfLift](https://github.com/facok/comfyui-SelfLift). The upstream documentation is preserved under [`docs/upstream/`](docs/upstream/), and source hashes are recorded in [`PROVENANCE.json`](PROVENANCE.json).

The upstream repository did not declare a license at the time this project was prepared. This repository does not add a license on behalf of the upstream author and does not imply permission to redistribute upstream code or model weights beyond the applicable rights. Research papers, third-party code, and model checkpoints remain the responsibility of their respective authors.

## Rollback

[v0.1.2-experimental](https://github.com/slmonker/selflift-Avatar/releases/tag/v0.1.2-experimental) remains available for comparison. You can also reconnect the original SelfLift node in the workflow. To uninstall, stop ComfyUI and move this folder out of `custom_nodes`.
