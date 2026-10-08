# selflift-Avatar

**开发版：`main`（分块失效引用修复）** · **已打包版本：`v0.1.4-experimental`** · [English README](README.md) · [更新记录](CHANGELOG.md) · [v0.1.4 发布页](https://github.com/slmonker/selflift-Avatar/releases/tag/v0.1.4-experimental)

这是一个面向 ComfyUI / MiniMax H3 的 SelfLift 独立实验分支，重点处理 H3 音视频 latent 遮罩、采样过程中的原始音频保留，以及一个有限制的高分辨率自动分块路径。它使用独立的节点 ID，可以和原版 SelfLift 共存。

> **实验版本，非官方项目。** 当前代码已通过 57 项 CPU 测试和两套模拟前端测试。此前版本已在维护者的 H3 工作流中试用；新控件仍需更多真实模型与前端测试。尚未进行系统性的口型准确率、速度和所有模型/插件组合测试。不同角色、音频、提示词、采样器、latent upscaler 和 seed 可能产生不同结果。

## 主分支修复：高分辨率分块遇到失效模型引用时报错（2026-10-09）

修复 `SelfLiftAvatarH3Sampler` 在准备高分辨率空间分块时出现的 `AttributeError: 'NoneType' object has no attribute 'model'`。

- **触发条件：**开启 `highres_tiling` 后，显存估算器遍历 ComfyUI 的已加载模型列表，遇到失效的模型引用。自动与手动分块均使用该估算器。此次报错前出现了中断上一轮、重新运行和垃圾回收，但不是每次中断都会触发，日志也无法确定具体哪个对象失效。
- **修复方式：**统计可回收显存时，跳过空条目和包装对象中已失效的模型引用。正常模型的统计、去重、设备筛选，以及采样、遮罩、分块选择逻辑保持不变。这是失效引用导致的崩溃，不代表模型文件损坏，也不是显存不足报错。
- **更新方法：**更新到当前 `main` 分支后，重启 ComfyUI 后端；只刷新网页不会重新加载 Python 代码。本修复无需调整参数。从已打包版本升级的用户，还需遵循下方已有的模型接口迁移说明。
- **验证范围：**57 项 CPU 测试通过，其中新增 4 项回归测试，覆盖空引用、已回收对象、混合有效条目、显存统计与边界；两套模拟前端测试通过。修复后的完整 GPU 生成尚未验证，也不保证避免其他报错或显存溢出。

本修复仅在 `main` 分支提供。现有 `v0.1.4-experimental` 发布压缩包保持不变，**不包含此修复**。

## 主分支更新：模型接口命名与布局优化（2026-10-05）

**双模型采样原本就通过 `model` 和 `model_hires` 支持，并非本次新增能力。** 本次将 MiniMax H3 和 Image 采样器的接口改为更明确的名称：`model` → `low_res_model`，`model_hires` → `high_res_model`。高分辨率接口仍为可选，不接时沿用低分辨率模型的原有逻辑不变。

本次还将两个接口相邻排列，修正高清采样调用以使用所选模型的加载设备，校验该模型的采样器兼容性，并补充回归测试。本次没有加入强制卸载模型的功能，也不保证不会显存溢出。

| 接口 | 是否必填 | 作用 |
| --- | --- | --- |
| `low_res_model` | 必填 | 前 `transition_step` 次低分辨率去噪使用的模型。 |
| `high_res_model` | 可选 | 分辨率提升后剩余采样使用的模型；不接时沿用 `low_res_model`。 |

两个接口相邻显示，`high_res_model` 位于 `low_res_model` 正下方，无需额外开关或第二个采样节点。

```text
模型 A ── low_res_model  → 低分辨率阶段
模型 B ── high_res_model → 高分辨率阶段（可选）
```

单模型工作流只接 `low_res_model` 即可。高分辨率分块和遮罩使用选定的高分辨率模型。模型加载、卸载仍由 ComfyUI 管理，不保证两个模型同时驻留显存。

### 升级与旧工作流迁移

**接口重命名导致的兼容性变更：**`model` 已改名为 `low_res_model`，`model_hires` 已改名为 `high_res_model`，旧输入名不再接受。升级前先备份工作流，更新后重启 ComfyUI 后端并强制刷新网页；如果旧接口仍残留，请重新添加采样节点，再按下列方式接线：

- 原 `model` 连线 → `low_res_model`。
- 原 `model_hires` 连线 → `high_res_model`（如有使用）。
- 重新连接其他输入并恢复采样参数，然后再运行。

独立的 H3 TST 模型补丁节点仍保留其原有 `model` 接口。接口排序不会自动迁移已删除的旧工作流输入。

### 兼容性与验证范围

两个阶段共享 VAE、正负条件、CFG、采样器和 sigma 日程。模型必须具有兼容的架构、latent 格式和采样参数化，适合同架构的不同权重或 LoRA 组合，不支持任意模型家族混用。模型切换跟随**分辨率提升边界**，不是独立设置的高噪 / 低噪阈值。

**53 项 CPU 测试**和两套 Node.js 前端测试通过，覆盖模型选择、单模型回退、阶段边界、遮罩与分块、接口排序及重排后的连线目标保持。这些是合成去噪器和模拟前端测试；双真实模型的输出质量、显存峰值及全部前端 / 子图组合仍待验证。

本次更新位于仓库 `main` 分支；已有的 `v0.1.4-experimental` 发布压缩包**不包含本次修改**，本次没有创建新的 Release 压缩包。

## 历史升级：v0.1.4

**自动模式也会在节点状态栏第一行显示实际分块数量，不再只看到 `auto`。**

- 高清准备前：**实际分块数量：待估算**。低清阶段尚未确定自动方案。
- 方案确定后：显示 **实际 N 块**，包括自动模式选出的奇数；**1 块**明确标注为**整图、不拆分**。
- 完成后保留结果；修改分块选项或开始新运行时清除旧数量。
- 实时事件和完成/缓存的节点 UI 结果使用结构化方案数据，并兼容旧缓存文本。
- 为实时更新增加数字节点 ID 的查找回退；不代表已修复所有子图事件路由。
- 自动结果不会覆盖手动 **2 / 4 / 6 / 8** 下拉值。

**本次不改采样算法：**块数选择、空间重叠、音频遮罩限制、upscaler 时间窗口和显存估算保持不变。**47 项 CPU 测试**及扩展的模拟 DOM 前端回调测试通过。升级后请重启后端并强制刷新网页。测试不等于完整浏览器或真实大模型效果验证。

[下载 v0.1.4](https://github.com/slmonker/selflift-Avatar/releases/tag/v0.1.4-experimental) · [完整更新记录](CHANGELOG.md)

## 上一版升级：v0.1.3

**本次把空间分块改成了用户可选，并把实际方案直接显示在 H3 采样器节点上。**

| 项目 | v0.1.2 | v0.1.3 |
| --- | --- | --- |
| 块数 | 只能按显存估算自动规划 | 新增 **auto / manual**；手动下拉菜单为 **2 / 4 / 6 / 8**，默认 **2** |
| 方向 | 自动选择 patch 网格长边 | 可选 **auto / width / height**，自动和手动块数模式均可使用 |
| 方案显示 | 主要看控制台日志 | **节点内只读状态栏**显示实际块数、方向、范围、重叠和最大块尺寸 |
| 显存提示 | 控制台估算 | 节点显示预计可用/最低工作区和不足警告；**不是实测显存峰值** |
| 测试 | 34 项 CPU 测试 | **47 项 CPU 测试**，另有模拟 DOM 前端回调测试 |

**保持不变：**自动模式仍尝试 **1～8 块，包括奇数**；重叠自动处理，不改变 upscaler 的时间窗口。带遮罩分块仍要求 **视频全 1、音频全 0**。手动模式不会悄悄增加块数，没有 OOM 自动重试。本版没有宣称实测速度提升、固定显存节省比例或口型准确率提升。

**升级后操作：**重启后端并强制刷新网页；旧节点若没出现新控件，重新添加该采样器。之前本地实验工作流若保存了 1 或奇数手动块数，请重新选择 **2 / 4 / 6 / 8**。原公开 v0.1.2 工作流默认继续使用 auto。

[完整更新记录](CHANGELOG.md) · [v0.1.3 发布页与安装包](https://github.com/slmonker/selflift-Avatar/releases/tag/v0.1.3-experimental)

## v0.1.3：可选空间分块策略

H3 采样器新增可选参数，旧工作流保持原有默认行为。

| 控件 | 选项 | 作用 |
| --- | --- | --- |
| `highres_tiling` | false / true | 总开关，关闭时忽略下方分块设置。 |
| `tiling_mode` | auto / manual | 自动：从 1～8 块中选择首个估计能放下的方案；手动：按请求块数，不悄悄增加块数。 |
| `tiling_tiles` | 2 / 4 / 6 / 8，默认 2 | 仅手动有效。整图处理请关闭 highres_tiling；空间尺寸很小时，实际块数可能减少。 |
| `tiling_axis` | auto / width / height | 自动沿 patch 网格长边；width 左右分条；height 上下分条。自动和手动模式均有效。 |

节点新增只读状态栏：当前设置、上次实际方案、有效块数、空间范围、重叠区域、最大 latent 块，以及可用/最低工作区估算。高清准备阶段才产生实际方案；之前不会伪造自动块数。普通根图节点会收到运行中事件，完成或缓存执行时通过标准节点 UI 结果显示；子图中的实时事件显示尚未验证。修改选项会清除旧方案，提示重新运行。

显存数字不是实测峰值，手动模式不能保证不爆显存，也没有 OOM 自动重试。重叠范围仍然自动处理，这些参数不改变 latent upscaler 的时间窗口。视频全 1、音频全 0 的遮罩限制，以及 ControlNet/TST 限制不变。

安装后重启后端并强制刷新网页；若旧节点不显示新字段，可重新添加 H3 采样器节点。47 项 CPU 测试覆盖旧工作流默认值、手动方向和块数、实际块数显示、显存估算不足警告及节点 UI 返回。前端回调使用模拟 DOM 测试，不等于完整浏览器或真实大模型效果验证。

## 沿用 v0.1.2 的音频遮罩分块支持

- 当视频全量生成（`video mask` 全为 1）、输入音频全量保留（`audio mask` 全为 0）时，允许开启高分辨率分块。
- 根据可用工作区自动选择分块方向和数量，当前规划器可选择 1～8 块。
- 每个视频块都会收到完整音频 latent 和完整 H3 音频条件；音频不会按画面分块。
- 仍然拒绝视频局部保留、音频局部/软遮罩，以及尚未验证的其他遮罩组合。
- 新增分块拼接、完整音频保留、H3 音频遮罩传递、非法组合和无遮罩路径测试。

## Demo

[![H3 音频驱动 Demo 预览](docs/assets/demo-preview.gif)](https://github.com/slmonker/selflift-Avatar/releases/download/v0.1.2-experimental/selfliftAvatar_00011-audio.mp4)

**H3 音频驱动 Demo：** [下载 / 查看完整 13.67 秒 MP4](https://github.com/slmonker/selflift-Avatar/releases/download/v0.1.2-experimental/selfliftAvatar_00011-audio.mp4) · [打开 v0.1.2 发布页](https://github.com/slmonker/selflift-Avatar/releases/tag/v0.1.2-experimental)

上方是静音 GIF 片段，点击即可获取带声音的完整视频。该视频由维护者提供，作为公开 Release 附件发布。

[![SelfLift Avatar 工作流 Demo 预览](docs/assets/demo-selflift-avatar-preview.gif)](https://github.com/slmonker/selflift-Avatar/releases/download/v0.1.2-experimental/selflift-Avatar-demo-selflift-avatar.mp4)

**SelfLift Avatar 工作流 Demo：** [下载 / 查看完整 10.05 秒 MP4](https://github.com/slmonker/selflift-Avatar/releases/download/v0.1.2-experimental/selflift-Avatar-demo-selflift-avatar.mp4)

第二个预览同样是静音 GIF，点击即可获取带声音的完整视频。两个视频都只是维护者工作流的示例，不代表所有角色和音频都能达到相同的口型效果。仓库不包含模型权重。

## 安装

你需要已有可运行的 MiniMax H3 ComfyUI 环境，以及工作流本身需要的模型、VAE、LoRA 和可选 latent upscaler。本项目不新增依赖，也不包含模型权重。

### Git 安装

```bash
git clone https://github.com/slmonker/selflift-Avatar.git ComfyUI/custom_nodes/selflift-Avatar
```

更新已有 Git 安装：

```bash
cd ComfyUI/custom_nodes/selflift-Avatar
git pull --ff-only
```

更新后必须重启 ComfyUI 后端；仅刷新浏览器不会重新加载 Python 节点。

### ZIP 安装

从 [Releases 页面](https://github.com/slmonker/selflift-Avatar/releases) 下载最新 ZIP，将旧的 `selflift-Avatar` 文件夹备份到 `custom_nodes` 之外，用新版文件夹替换后重启 ComfyUI。不要在 `custom_nodes` 中同时放置两个本插件副本。

原版 `comfyui-SelfLift` 可以保留；本项目使用不同的节点 ID。

## 节点

| 节点 ID | 显示名称 |
| --- | --- |
| `SelfLiftAvatarH3Sampler` | selflift-Avatar Sampler (MiniMax H3) |
| `SelfLiftAvatarImageSampler` | selflift-Avatar Sampler (Image) |
| `SelfLiftAvatarH3TST` | selflift-Avatar H3 TST |

采样器使用标准的 `sampler` 和 `sigmas` 输入。请使用标准 Euler，并保持 `s_churn=0`。

## 音频驱动 H3 工作流

典型的输入音频连接方式：

```text
输入音频 → 裁剪到目标片段 → H3 音频 VAE 编码
                                  ↓
SolidMask(value=0) → SetLatentNoiseMask
                                  ↓
H3 视频 latent ───────────────→ 合并 AV latent
                                  ↓
                       selflift-Avatar H3 采样器
```

- 音频遮罩为 `0` 表示保留输入音频，为 `1` 表示允许生成音频，0～1 表示软约束。
- 编码音频、视频帧范围和最终输出 FPS 必须对应同一段素材。
- 如果最终视频节点直接使用原始音频，听到正确的声音并不能证明 H3 在生成画面时真正用它驱动了嘴部；需要检查生成的视频帧。
- 本采样器不是独立的后期口型同步工具。

## 高分辨率分块

打开 `highres_tiling=true` 后，用 `tiling_mode=auto` 自动估算块数，或者用 `tiling_mode=manual` 从 **2 / 4 / 6 / 8** 下拉菜单手动选择。自动模式仍会尝试 1～8 块（包括奇数），可能选择 1 块，即不实际拆分。两种模式都可用 `tiling_axis` 指定方向。需要整图处理时，直接关闭 `highres_tiling`。

例如：`highres_tiling=true`、`tiling_mode=manual`、`tiling_tiles=4`、`tiling_axis=width` 表示请求左右分成 4 条。重叠区域仍自动处理。手动模式在估算显存不足时会警告，但不会偷偷增加块数，也不会在 OOM 后自动重试。

### 不把块数绑定到固定显卡容量

不存在可靠的“2 块需要 32GB、4 块需要 24GB”对应表。规划器综合当前空闲显存和预计可回收的权重显存，扣除预留量，再与模型估计的最低需求比较。分辨率、时长、批次、参考图、音频、权重卸载和重叠范围都会影响结果。节点显示的是**估算工作区，不是实测总显存峰值**。方案在高清准备阶段计算，不会每一步动态重算。本项目不承诺固定的显存节省比例或速度提升。

### 当前支持的带遮罩分块模式

目前经过验证的带遮罩模式被有意限制为：

```text
视频遮罩：全 1 → 生成完整视频
音频遮罩：全 0 → 保留完整输入音频
```

每个视频块都会获得完整音频 latent 和完整 H3 音频条件。分块只发生在视频空间方向，音频不会按画面空间切分。

### 开启分块时仍会拒绝

- 视频遮罩中存在 0 或软值；
- 音频局部遮罩或软遮罩；
- H3 分块路径与 ControlNet 组合。

仅支持静态遮罩：上游动态 `denoise_mask_function` 在两种模式下都会被忽略并输出警告，关闭分块不会恢复动态遮罩支持。上述空间遮罩和 ControlNet 组合请关闭 `highres_tiling`。直接删除校验并不能让它们正确运行，因为对应遮罩还需要按 tile 裁剪、变换并对齐。

在控制台中看到以下日志即可确认规划器：

```text
[selflift-Avatar plan] automatic high-resolution tiling enabled
[SelfLift tiling plan] mode=manual axis=W tiles=4 requested=4 ...
```

`axis=H` 或 `axis=W` 表示实际方向，`tiles=N` 表示有效块数，`requested` 表示手动请求值或 auto。空间尺寸过小时，实际块数可能减少。节点只读状态栏在高清准备阶段显示同一实际方案。

## 不使用分块时的遮罩支持

- 普通图像/视频 Tensor：BHW、BCHW、BCTHW；支持单通道或实际 latent 通道数。
- H3 `NestedTensor(video_mask, audio_mask)`：分别处理视频和音频遮罩。
- 与音视频 latent 展平尺寸严格匹配的 packed mask。
- 音频 T、BT、BST、BCST 布局；维度为 1 时可广播。
- 视频空间遮罩会缩放到 latent 网格；不会猜测音频时间含义，也不会自动重采样音频时间轴。
- 附加到音频上的恒定图像尺寸 `SolidMask`，例如 `[1,1,928,1664]`，会被识别为恒定音频约束，并广播到音频 latent。

遮罩语义：**0=保留，1=生成，0～1=软约束**。要产生有意义的保留效果，输入 latent 中必须确实包含要保留的内容。

## H3 原生音频条件

采样器会在采样前把打包后的遮罩传入 ComfyUI 原生 H3 条件路径，使用 H3 的 `audio_denoise_mask` 和输入侧 `scale_latent_inpaint`，而不只是模型预测后再恢复音频。高清阶段的带噪恢复状态和干净音频 anchor 分开处理，避免把残余采样噪声当成原始音频。

这修复的是遮罩和条件的传递问题，并不保证口型绝对准确。外部 H3 latent upscaler 包含时间卷积，而且不会直接接收目标音频；低分辨率前缀和高清修正步数也可能影响嘴部细节。

做对比时，请固定模型、音频片段、提示词、帧数、FPS 和 seed。一次只改变一个因素，对比原生基准、当前 SelfLift 路径、不同 `transition_step`，以及 `upscaler_model=none` 的 pixel-anchor 路径。

## 测试

使用 ComfyUI 对应的 Python 运行：

```bash
python tests/test_avatar.py
```

可选前端回调测试（需要 Node.js，无需安装 npm 包）：

```bash
node tests/test_tiling_ui.mjs
node tests/test_model_input_order.mjs
```

当前共有 53 项 CPU 测试，覆盖音视频打包、H3 音频尺度、全量/局部遮罩、恒定音频 SolidMask 转换、原生音频条件、两阶段采样、自动分块规划、分块拼接、每个分块获得完整音频、非法遮罩组合和无遮罩分块路径。测试不会加载大型 H3 权重，也不衡量最终视频的口型质量。

## 来源与许可

本项目基于 [facok/comfyui-SelfLift](https://github.com/facok/comfyui-SelfLift) 的本地快照修改。上游说明保存在 [`docs/upstream/`](docs/upstream/)，来源哈希记录在 [`PROVENANCE.json`](PROVENANCE.json)。

准备本项目时，上游仓库没有声明许可证。本项目没有代上游作者添加许可证，也不代表获得重新分发上游代码或模型权重的授权。论文、第三方代码和模型权重仍由各自作者负责。

## 回退

[v0.1.2-experimental](https://github.com/slmonker/selflift-Avatar/releases/tag/v0.1.2-experimental) 仍可用于对比。也可以在工作流中重新连接原版 SelfLift 节点。卸载时先停止 ComfyUI，再将本文件夹移出 `custom_nodes`。
