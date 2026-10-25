# Changelog

## Main branch — 2026-10-25 (not yet packaged)

- Fix the MiniMax H3 latent upscaler inventing frame-0 content in long clips (`T > 32` latent frames): temporal chunk halos now use `reflect` padding instead of repeating the causal first frame.
- Preserve the previous `replicate` behavior as a fallback for unusual custom models where the temporal overlap is not smaller than the latent length.
- Add five regression tests. Short clips remain on the unpadded path, and long-clip interior frames outside the first/last windows remain byte-for-byte unchanged.
- 80 CPU tests and both frontend suites pass. A real `minimax_h3_latent_upscaler_3d_conv_v1_bf16` A/B reduced synthetic frame-0 residual max from 13.46 (`replicate`) to 1.01 (`reflect`), versus 0.92 for the full-sequence reference.

## Main branch — 2026-10-20 (not yet packaged)

- Add `highres_tiling_mode` with `spatial`, `global_stream`, `global`, and `global_ram`.
- Add `highres_token_chunk` for global projection/MLP batching and `highres_weight_cache_gb` for the `global_stream` DynamicVRAM budget.
- Keep `spatial` as the backward-compatible default; existing workflows are unchanged.
- Add full-sequence H3 block chunking adapted from Fish4AI/comfyui-SelfLift-Fish: Q/K/V projections and MLP are batched, while K/V context and layout remain complete.
- Chunk video/audio input projections and final-head rows without changing attention math.
- Preserve arbitrary video/audio masks in global modes; `spatial` keeps its existing video-all-1/audio-all-0 restriction.
- If TST or another attention override is active, automatically fall back to full-sequence dense attention with chunked MLP instead of raising.
- Remove only the ComfyUI dense attention backend selector for the fallback path; TST/Motion Context overrides are preserved.
- Fix compatibility with ComfyUI H3 versions that do not expose `MiniMaxH3Model._embed_and_pack`.
- Add `example_workflows/H3+selflift-avatar-fixed.json`, a working `global_stream` example with the new controls and sanitized local paths.
- 75 CPU tests pass, including the existing workspace-regression tests; both stub-frontend suites pass. Real H3 GPU quality/VRAM verification remains required.
- Special thanks to [Fish4AI](https://github.com/Fish4AI) and the [comfyui-SelfLift-Fish](https://github.com/Fish4AI/comfyui-SelfLift-Fish) project for the global-context implementation this work builds on.

## Main branch — 2026-10-09 (not yet packaged)

- Fix high-resolution tiling workspace estimation crashing on `None` entries or expired wrapped model references in ComfyUI's loaded-model list.
- Preserve live-model memory accounting, deduplication, device filtering and sampling/tiling behavior; do not force model unloading or garbage collection.
- Document the trigger, interruption/retry observation, backend restart requirement and verification limits in both READMEs.
- 57 CPU tests (four new regression tests) and both stub-frontend suites pass. Full GPU generation after this fix remains unverified; the v0.1.4 release archive is unchanged.

## Documentation correction — 2026-10-06

- Clarify that dual-model sampling already existed through `model` / `model_hires`. The October 5 update renames and reorders those inputs, corrects high-resolution device selection, and adds validation/tests; it does not introduce dual-model sampling or forced unloading.
- Correct both READMEs and the update description below. No runtime code changes.

## Main branch — 2026-10-05 (not yet packaged)

- Rename existing dual-model inputs on both H3 and Image samplers: required `model` → `low_res_model`, optional `model_hires` → `high_res_model`. Preserve the existing high-resolution fallback to the low-resolution model; dual-model sampling is not new in this update.
- Breaking workflow change: the old sampler input names are no longer accepted; existing workflows must reconnect to the renamed inputs. The H3 TST patch node is unchanged.
- Display the high-resolution socket directly below the low-resolution socket and preserve link targets when reordering restored root-graph inputs.
- Use the selected high-resolution model's load device for its sampling stage and validate its sampler compatibility.
- Document wiring, migration, shared-conditioning constraints, and verification limits in both READMEs.
- 53 CPU tests and both frontend test suites pass. Real dual-model quality and peak VRAM remain unverified; the v0.1.4 release archive is unchanged.

## v0.1.4-experimental — 2026-09-18

- Display actual automatic tile count on the first line of the node panel; show pending before planning and no-split for one tile.
- Send structured plan data through live events and completed/cached UI results; preserve old cached-text support.
- Reset stale counts for settings changes and new runs; do not overwrite manual dropdown values.
- Add numeric-node-ID lookup fallback for live updates.
- Sampling and tiling algorithms unchanged. 47 CPU tests and expanded stub-DOM frontend callback tests pass.

## v0.1.3-experimental — 2026-09-18

- Add backward-compatible auto/manual tile count and direction controls to the H3 sampler.
- Manual dropdown offers 2, 4, 6, 8 (default 2); auto retains the original 1–8 search.
- Keep English as the homepage and update the separate Chinese README, including VRAM estimation caveats.
- Show actual spatial plan, estimated memory and errors in a read-only node panel.
- Keep overlaps automatic and preserve the full-audio mask restriction.
- 47 CPU tests and frontend callback tests with a stub DOM passed; real model and live frontend verification pending.

## v0.1.2-experimental — 2026-09-16

- Allow high-resolution tiling for the validated audio-only mask case: video mask all 1, audio mask all 0.
- Automatically select spatial direction and tile count from available workspace; 1–8 tiles are possible.
- Forward the complete audio latent and H3 audio conditioning to every video tile.
- Keep rejecting partial/soft video masks, partial/soft audio masks, dynamic mask schedules, and ControlNet with the H3 tiling path.
- Add tile-stitching and masked-tiling regression tests; 34 CPU tests pass.
- Add separate English and Chinese documentation; English README is the repository homepage.
- Add a clickable GIF preview and publish the maintainer-provided `_00011-audio.mp4` as a public Release asset.
- Publish the additional maintainer-provided `Selflift Avatar demo.mp4` as a public demo Release asset.

## v0.1.1-experimental

- Connect H3 native audio conditioning and model-input audio injection instead of only restoring audio after prediction.
- Separate the high-resolution noisy resume state from the clean inpaint anchor.
- Preserve dual-stream masks, multi-channel masks, and constant image-sized SolidMask audio compatibility.
- 28 CPU tests passed; no systematic lip-sync benchmark.

## v0.1.0-experimental

- Independent Avatar H3, Image, and TST node registrations without overriding the original plugin.
- H3 dual-stream masks, multi-channel video masks, packed masks, and static audio constraints.
- H3 audio-scale correction and the masked pure pixel-anchor fix.
