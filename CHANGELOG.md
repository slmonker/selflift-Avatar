# Changelog

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
