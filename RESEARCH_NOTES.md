# ACE-Step 1.5 研究调研报告

> 调研日期:2026-09-15 · 基于 commit `8eaf50c`(Linux x86_64 + CUDA 精简版)
> 结论均经代码验证,引用处标注 `文件:行号`。文档可能滞后于代码,以代码为准。

---

## 1. 项目速览

ACE-Step 1.5 是 ACE Studio / StepFun 的开源音乐生成基础模型(MIT),三段式架构:

```
用户输入 ──> LM 规划器(0.6B/1.7B/4B, Qwen3 微调)
              │  CoT 生成:元数据(bpm/key/duration)、caption 润色、歌词、5Hz 音频语义码
              ▼
            DiT 扩散变换器(2B turbo/sft/base 或 XL 4B)── flow-matching 去噪
              ▼
            VAE(scraggvae)解码 ──> 48kHz 音频
```

- LM 对齐采用内在强化学习(无外部奖励模型);turbo 系把 CFG 蒸馏进权重(8 步、无 CFG)
- 性能:A100 整曲 <2s,3090 <10s;10s~600s 时长,一次最多 8 首

### 本仓库现状

| 项 | 状态 |
|----|------|
| 远程 | `origin` = `zllmma/ACE-Step-1.5`(本人 fork,推送目标);`upstream` = 官方(不打算同步) |
| 环境 | 已精简为 **Linux x86_64 + CUDA 专用**(多平台启动脚本/依赖 marker/mlx 已删;`gpu_config.py` 的 mps/xpu/rocm 代码分支原样保留,永不命中) |
| 模型 | `checkpoints/` 已有:turbo、xl-base、xl-turbo、lm-1.7B、lm-4B、vae、scragvae、Qwen3-Embedding-0.6B |
| 入口 | `uv run acestep`(Gradio :7860)、`uv run acestep-api`(:8001)、`train.py {vanilla\|fixed\|estimate}`(Side-Step 训练) |
| Python | 3.11-3.12,uv 管理;LM 后端 vllm(vendored `acestep/third_parts/nano-vllm`) |

---

## 2. 配置体系

### 2.1 配置变量表

**模型与推理**

| 变量 | 值 | 含义 |
|------|-----|------|
| `ACESTEP_CONFIG_PATH` | `acestep-v15-turbo` / `sft` / `base` / `xl-*` | DiT 模型 |
| `ACESTEP_LM_MODEL_PATH` | `acestep-5Hz-lm-0.6B` / `1.7B` / `4B` | LM 规划器 |
| `ACESTEP_LM_BACKEND` | `vllm`(快) / `pt`(兼容) | LM 推理后端 |
| `ACESTEP_INIT_LLM` | `auto` / `true` / `false` | 是否加载 LM。决策链:**GPU 显存检测 → 此变量覆盖 → 加载**。`false` 连懒加载都禁止(`acestep/api/llm_readiness.py:50`),CoT/思考模式等随之失效 |
| `ACESTEP_DEVICE` | `auto` / `cuda` / `cpu` | 设备覆盖 |
| `ACESTEP_CHECKPOINTS_DIR` | 路径 | 多安装共享模型目录 |
| `ACESTEP_DOWNLOAD_SOURCE` | `auto` / `huggingface` / `modelscope` | 模型下载源 |
| `ACESTEP_BATCH_SIZE` | 1~GPU 上限 | 默认批量(缺省 `min(2, GPU_max)`) |

**服务与 UI**

| 变量 | 含义 |
|------|------|
| `PORT` / `SERVER_NAME`(API 用 `HOST`) | 端口(7860/8001)与绑定地址 |
| `LANGUAGE` | UI 语言 `en`/`zh`/`he`/`ja` |
| `ACESTEP_API_KEY` | API 鉴权 |
| `ACESTEP_NO_INIT` | `true`(默认)懒加载,`false` 启动即加载 |
| `SERVICE_MODE_{DIT_MODEL,LM_MODEL,BACKEND}` | 仅 `--service_mode` 时生效的预设 |

### 2.2 优先级链(代码验证)

```
请求参数/命令行参数  >  真实环境变量(export)  >  .env  >  .env.example  >  代码内置默认/GPU 自动检测
```

关键机制:

1. **进程环境变量压过 `.env`**:`load_dotenv` 默认不覆盖已有变量,`api_server.py:175` 显式传 `override=False`。`export ACESTEP_INIT_LLM=false` 后再改 `.env` 不会生效,直到重开 shell
2. **`.env` 回退 `.env.example`**:根目录无 `.env` 时直接用 example(`acestep/acestep_v15_pipeline.py:20-25`)——删掉 `.env` 也能跑,配置即 example 值
3. **每进程只加载一次**(模块级 `_env_loaded` 标志),改 `.env` 必须重启;启动时 proxy 变量被主动清除(`acestep_v15_pipeline.py:32-39`)
4. **命令行 > 环境变量**的实现是"参数为 None 才回退 env"(`acestep_v15_pipeline.py:397-402`),且该回退仅在 `--enable-api` / `--service_mode` 下走;普通 Gradio 模式 `config_path=None` 时初始化阶段自动选 `acestep-v15-turbo`

**两条特殊通路**

- 启动脚本 `start_gradio_ui.sh:10-66` 自己解析 `.env` 并把 `ACESTEP_*` 翻译成**命令行参数**再传给 `uv run acestep`,优先级:export 变量 > `.env` > 脚本默认(`: "${PORT:=7860}"` 风格)。两个坑:脚本内置默认 LM 是 **0.6B**(`:97`)而 `.env.example` 默认 **1.7B**(有 `.env` 时以 `.env` 为准);`ACESTEP_INIT_LLM=auto` 时脚本不传参,交给 GPU 检测(`:38-41`)
- `LANGUAGE` 特例:env 直接内嵌进 argparse default(`--language`),不走"参数为 None"那套(`acestep_v15_pipeline.py:231`)

**实用结论**:持久配置用 `.env`,临时实验用 `export` 覆盖(权重最高且不动文件);排查配置问题先 `echo $ACESTEP_INIT_LLM` 确认没有残留 export。

---

## 3. 推理 Python API

### 3.1 初始化

```python
from acestep.handler import AceStepHandler      # 实现在 acestep/core/generation/handler/init_service_orchestrator.py:48
from acestep.llm_inference import LLMHandler    # acestep/llm_inference.py:499

dit = AceStepHandler()
dit.initialize_service(
    project_root=".",                 # 项目根(checkpoints 所在)
    config_path="acestep-v15-turbo",  # DiT 模型;None 默认 turbo
    device="auto",                    # auto/cuda/cpu
    use_flash_attention=False,        # flash-attn 已装,可开
    compile_model=False,              # torch.compile
    offload_to_cpu=False,             # <20GB 建议 True
    offload_dit_to_cpu=False,
    quantization=None,                # "int8_weight_only"(<20GB 建议)
    prefer_source=None,
    use_mlx_dit=True,                 # 仅 macOS 生效,Linux 忽略
    vae_checkpoint=None,              # 自定义 VAE(见 docs/en/ALT_VAE.md)
)
llm = LLMHandler()
llm.initialize(
    checkpoint_dir="./checkpoints",
    lm_model_path="acestep-5Hz-lm-1.7B",
    backend="vllm",                   # vllm/pt
    device="auto", offload_to_cpu=False, dtype=None,
)
# initialize_service 支持重复调用换模型重初始化,不会短路
```

### 3.2 核心生成:`generate_music`(`acestep/inference.py:514`)

```python
result = generate_music(dit, llm, params, config, save_dir="out/")
```

两阶段:**Phase 1** LM(`thinking=True` 时 CoT 生成元数据 + 音频语义码)→ **Phase 2** DiT 扩散 → VAE 解码。

结果对象 `GenerationResult`:

- `result.audios`:每项 `{path, tensor(CPU float32), key, sample_rate=48000, params}`;**实际 seed 从 `audios[i]["params"]["seed"]` 读回**
- `result.extra_outputs`:`lm_metadata`(LM 生成的元数据)、`time_costs`(含 `lm_phase1_time` / `lm_phase2_time` / `dit_total_time_cost` / `pipeline_total_time`,性能实验直接用)、`latents`、`masks`
- `result.success` / `result.error` / `result.status_message`

### 3.3 GenerationParams 全参数

签名位置:`acestep/inference.py:56`。文档(`docs/en/INFERENCE.md`)为 v1.5.2 版,**代码新增了文档未收录的参数**,分述如下。

**文本与元数据(常规)**

| 参数 | 默认 | 说明 |
|------|------|------|
| `caption` | `""` | 风格描述,<512 字符 |
| `lyrics` | `""` | 歌词,`"[Instrumental]"` 表纯音乐,<4096 字符,支持结构标签 `[Verse]` `[Chorus]` |
| `instrumental` | `False` | 强制纯音乐 |
| `bpm` | `None` | 30~300,None=LM 自动估计 |
| `keyscale` / `timesignature` | `""` | 如 `"C Major"` / `"4/4"`,空=自动 |
| `vocal_language` | `"unknown"` | ISO 639-1,合法值见 `acestep/constants.py:VALID_LANGUAGES` |
| `duration` | `-1.0` | 秒,10~600;<0 自动按歌词长度定 |

**DiT 采样**

| 参数 | 默认 | 说明 |
|------|------|------|
| `inference_steps` | `8` | turbo 1-20(推荐 8);base 推荐 32-64 |
| `guidance_scale` | `7.0` | 仅非 turbo 生效;**turbo 会被自动纠正为 1.0**(`generate_music.py:312-318`,打日志) |
| `seed` | `-1` | -1 随机 |
| `use_adg` | `False` | Adaptive Dual Guidance,仅 base |
| `cfg_interval_start/end` | `0.0`/`1.0` | CFG 应用区间 |
| `shift` | `1.0` | 时间步偏移 `t = shift*t/(1+(shift-1)*t)`;**turbo 实验必设 `shift=3.0`** |
| `infer_method` | `"ode"` | `ode`(Euler,确定性快)/ `sde`(随机) |
| `timesteps` | `None` | 自定义时间步列表(1.0→0.0),给出时覆盖 steps+shift |

**代码新增(文档未收录,研究价值高)**

| 参数 | 默认 | 说明 |
|------|------|------|
| `dcw_enabled` | `None` | 小波域 SNR-t 偏差校正(CVPR 2026,training-free,`docs/en/DCW.md`);None=跟随模型默认 |
| `retake_seed` / `retake_variance` | `None` / `0.0` | Retake 局部重生成:固定种子 + 方差控制变化度 |
| `flow_edit_morph` 及 `flow_edit_source_{caption,lyrics}`、`flow_edit_n_{min,max,avg}` | `False` | Flow-Edit:源/目标 caption+lyrics 在 flow 轨迹上变形插值(text2music + `src_audio` 时生效;源码注释强调源音频走 VAE 编码而非 codes 解码,否则 OOD 崩溃) |
| `latent_shift` / `latent_rescale` | `0` / `1.0` | VAE 解码前对 DiT 潜变量加性/乘性操控 |
| `enable_normalization` / `normalization_db` | `True` / `-1.0` | 输出响度归一化 |
| `chunk_mask_mode` | `"auto"` | `"explicit"` 时用 repaint 区间构造 0/1 掩码(Gradio Repaint 自动用它) |

**任务相关**

| 参数 | 默认 | 说明 |
|------|------|------|
| `task_type` | `"text2music"` | 见 3.5 任务类型 |
| `instruction` | 固定串 | lego/extract/complete 必须指明轨道 |
| `reference_audio` / `src_audio` | `None` | 参考音频 / 源音频(cover/repaint 等) |
| `audio_codes` | `""` | 预提取 5Hz 语义码字符串(高级) |
| `repainting_start` / `repainting_end` | `0.0` / `-1` | repaint/lego 区间(秒,-1 到结尾) |
| `audio_cover_strength` | `1.0` | cover 结构保持度(0~1,风格迁移取 ~0.2) |

**LM 参数**:`thinking=True`、`lm_temperature=0.85`、`lm_cfg_scale=2.0`、`lm_top_k=0`、`lm_top_p=0.9`、`lm_negative_prompt="NO USER INPUT"`、`use_cot_metas=True`、`use_cot_caption=True`、`use_cot_language=True`、`use_constrained_decoding=True`。`cot_*` 字段为 LM 回填结果,只读。

### 3.4 GenerationConfig(`acestep/inference.py:236`)

`batch_size=2`(1-8)、`allow_lm_batch=False`(`thinking=True` 且 batch≥2 时开可加速)、`use_random_seed=True`、`seeds=None`(int 或 List,不足自动补随机)、`lm_batch_chunk_size=8`、`constrained_decoding_debug=False`、`audio_format="flac"`(另支持 mp3/wav/wav32/opus/aac)、`mp3_bitrate="128k"`、`mp3_sample_rate=48000`。

### 3.5 任务类型(代码 7 种,文档只写 6 种)

`acestep/constants.py:76`:

```
text2music | repaint | cover | cover-nofsq(文档未收录) | extract | lego | complete
```

- **turbo 只支持前 4 种**(`TASK_TYPES_TURBO`,`constants.py:82`);extract/lego/complete 需 base 系模型
- **LM 自动跳过**:cover/repaint/extract(无论 `thinking` 为何);仅 text2music/lego/complete 走 LM

### 3.6 LM 纯辅助函数(无需 DiT)

| 函数 | 用途 | 结果对象 |
|------|------|----------|
| `understand_music(llm, audio_codes, temperature=0.85, ...)` | 音频语义码 → caption/lyrics/bpm/key/duration/language | `UnderstandResult` |
| `create_sample(llm, query, instrumental=False, vocal_language=None, ...)` | 自然语言描述 → 完整样本(Simple Mode) | `CreateSampleResult` |
| `format_sample(llm, caption, lyrics, user_metadata=None, ...)` | 润色输入 + 结构化元数据(`user_metadata` 可锁字段) | `FormatSampleResult` |

三者均支持 `top_k`/`top_p`/`repetition_penalty`/`use_constrained_decoding`;成功结果可直接灌入 `GenerationParams` 生成。

### 3.7 实验关键行为备忘

1. **可复现三件套**:`config.use_random_seed=False` + `params.seed`/`config.seeds` + `lm_temperature` 调低(0.7-0.85)
2. **VRAM 自动兜底**:批量自动缩减、VAE 三级解码回退(GPU 分块 → GPU+offload → 全 CPU)、超限 duration/batch 钳制——发现 batch 被悄悄改小看日志
3. **turbo 实验分布参数只调 `shift`**;CFG 会被吞掉
4. **LM 加速**:`allow_lm_batch=True` + `lm_batch_chunk_size` 按 VRAM 调
5. **性能计时**直接取 `extra_outputs["time_costs"]`

---

## 4. 工作流备忘(测试陷阱)

- unittest 风格,`*_test.py` / `test_*.py` 约 190 个
- **不要全量 discover**:`discover -s .` / `-s acestep` 会 import 带副作用的模块而卡住(已验证);用单模块 `uv run python -m unittest acestep.cli_args_test`,或确认干净的小目录(如 `acestep/training`)
- 模块 LOC 目标 ≤150、硬上限 200;日志 loguru;`.env`/`checkpoints/`/`gradio_outputs/` 均 gitignored
- 详细规范见 `AGENTS.md`、`CONTRIBUTING.md`

## 5. 关键代码位置索引

| 关注点 | 位置 |
|--------|------|
| 推理 API 与数据类 | `acestep/inference.py`(GenerationParams:56, Config:236, generate_music:514) |
| DiT 生成主流程 | `acestep/core/generation/handler/generate_music.py`(turbo CFG 纠正:312) |
| handler 初始化 | `acestep/core/generation/handler/init_service_orchestrator.py:48` |
| LM 初始化 | `acestep/llm_inference.py:499` |
| 任务类型常量 | `acestep/constants.py:76-89` |
| Gradio 入口与 env 处理 | `acestep/acestep_v15_pipeline.py` |
| API server 与 env 优先级 | `acestep/api_server.py:163-176`、`acestep/api/llm_readiness.py:49-64` |
| 硬件检测 | `acestep/gpu_config.py` |
| 启动脚本 env 解析 | `start_gradio_ui.sh:10-66` |

## 6. 命令行推理入口

仓库有两条现成推理路径,另有本 fork 新增的轻量脚本:

### 6.1 `cli.py`(官方全功能 CLI,2051 行)

```bash
uv run python cli.py                      # 交互式向导
uv run python cli.py --configure          # 只保存 TOML 不生成
uv run python cli.py -c my_config.toml    # 非交互,TOML 直接生成
```

比 Python API 直接调用多出的能力:
- DCW 全套参数(`dcw_enabled/mode/scaler/high_scaler/wavelet`);**`dcw_enabled=None` 时跟随模型:turbo 默认开、base 默认关**(`cli.py:1658-1663`,issue #1273)——对比实验需显式设值
- `sample_mode`(一句话 → create_sample 全套参数)、`use_format`(format_sample 润色)、`lyrics = "generate"`(create_sample 自动写词)
- 缺模型自动下载、按任务自动选模型(lego/extract/complete 强制 base)
- 智能 seed(`cli.py:1221-1240`):TOML 给了 `seed` 未显式设 `use_random_seed` 时自动关随机(#1259)

**非交互 + thinking 的坑**(`cli.py:1636-1656`):`-c config.toml` + thinking 时,项目根目录若无 `instruction.txt`,LM 提示词生成后会阻塞在 `input()` 等待编辑;有则直接采用。无人值守跑法:`thinking = false`,或预置有内容的 `instruction.txt`。

### 6.2 `generate.py`(本 fork 新增,已实测)

轻量直通,全部参数走命令行,适合批量实验脚本内嵌;输出实际 seed 与 lm1/lm2/dit 分相耗时。`--no-lm` 跳过 vllm 走纯 DiT。用法示例见脚本 docstring。

### 6.3 其他

- `profile_inference.py`:基准/性能分析(`--mode benchmark` 跑配置矩阵)
- `scripts/flow_edit_overlay_smoke.py`:flow-edit 冒烟
- `run_generate_test.py` 已删除(硬编码 mlx + 0.6B,Linux CUDA 不可用,由 `generate.py` 取代)

## 7. Repaint 编辑能力评测 v2(2026-09-15)

**背景**:ACE-Step 1.5 官方论文与第三方(仅 YuE2 团队,2026-09-12 榜单,测过文生歌与 cover)均未评测 repaint 等区域编辑能力;文献双轴协议(AUDIT/MusicMagus/SteerMusic/Melodia)移植至此。工具:`repaint_bench.py`(协议)+ `edit_metrics.py`(指标)。

**协议**:MUSDB18 test 20 首(ffmpeg 抽 mixture,截 40s@48kHz)× 每首 2 个随机区间(8-15s 抖动,避开前 5s)× 每区间 2 条 MusicMagus 式模板 caption(mood/genre/timbre 随机池)= 80 条;turbo 8 步,`chunk_mask_mode="explicit"`,seed=42,**关响度归一化**(否则区间外 L1 虚高 200 倍)。

**结果(80 条)**:

| 轴 | 指标 | 均值 | 中位数 | 解读 |
|----|------|------|--------|------|
| 一致性 | 区间外 mel L1(↓) | **0.0020** | 0.0019 | 未动区域近乎完美保留 |
| 一致性 | 全曲 Chroma 余弦(↑) | 0.9976 | 0.9986 | 音乐内容整体保持 |
| 遵循 | 区间内 CLAP(↑) | 0.2163 | 0.2269 | 与文献同级(SteerMusic 表中 AudioLDM2 系方法 0.218~0.264) |
| 综合 | ASB(调和平均,↑) | 0.6120 | 0.6345 | 双轴平衡 |

**位置效应**:front 段重绘 CLAP 0.257 > mid 0.200 > back 0.125(区间外保真各段一致 0.0019-0.0022)——前奏类器乐段的重绘遵循度最高;back 样本仅 4 条,需扩样验证。

**基础设施备忘**:torchcodec 必须 `==0.10` 且走 pytorch cu128 index(已固化);laion-clap 需用 music checkpoint `music_audioset_epoch_15_esc_90.14.pt`(HF 的 `630k-audioset-best.pt` 是旧版投影 [512,768],与 laion_clap 1.1.4 的 HTSAT-base 配置 [512,1024] 不匹配;本地副本在 `/data/local/tmp/ceval/`)。

### 7.1 模型对照:遵循度由 SFT 决定,strength 控制改动量(2026-09-15)

同一条件(AM Contra 首,区间 [5.51,17.99]s,caption "peaceful jazz + acoustic guitar",seed=42)四配置对照:

| 模型(SFT/RL) | 步数/CFG | strength | 区间内改动量 | 区间外 L1 | 区间内 CLAP |
|----------------|----------|----------|--------------|-----------|-------------|
| **turbo 2B**(✅/❌) | 8 / 1.0 | 0.5 | **4.62** | 0.0020 | **0.234** |
| **xl-turbo 4B**(✅/❌) | 8 / 1.0 | 0.5 | 1.90 | 0.0018 | 0.111 |
| xl-base 4B(❌/❌) | 120 / 7.0 | 0.5 | **0.00**(恒等) | 0.0018 | -0.010 |
| xl-base 4B(❌/❌) | 120 / 7.0 | 1.0 | 2.32 | 0.0013 | -0.027 |

结论:
1. **repaint_strength 是"改动量"旋钮**(balanced 模式下 injection_ratio=1-strength;0.5=半量源注入,1.0=纯扩散重绘),xl-base 在 0.5 时甚至恒等
2. **遵循度(CLAP)由 SFT 决定**:无 SFT 的 xl-base 改了也不对(CLAP≈0/-0.03);有 SFT 的 turbo 系 0.11-0.23。步数与 CFG(xl-base 120 步 + CFG 7.0)救不了遵循度
3. 区间外保真所有配置均 0.002 级——repaint 的结构保持是机制级强项
4. 同条件两次生成的 CLAP 波动 ±0.07(CUDA 非确定性)→ 遵循度结论必须多条件平均;80 条均值的标准误 ≈0.007
5. 失败模式实证:低 CLAP 样本(如 Arise -0.087)听感为"区间近似未改、仅人声被抑制"——保守重绘而非错误重绘,与 AudioLDM2 系编辑方法的头号失败模式一致

## 8. 歌词编辑(flow-edit)评测(2026-09-15)

**协议**:100 首中文语料中取 10 首(官方+手写素材),人工撰写 PSub(部分替换)方案——每首替换 Verse 1 的 2 个短语(唯一性经断言校验,新短语字数接近、语义可辨);flow-edit(xl-turbo,probe 三模型自动选型 score 满分)生成;**对照组** = 新歌词直接重生成;Whisper-small 转录(长格式,强制中文)。

**结果(10 条,xl-turbo)**:

| 轴 | 指标 | flow-edit | 重生成对照 | 解读 |
|----|------|-----------|-----------|------|
| 旋律保持 | CQT1-PCC(↑) | **0.966** | 0.167 | flow-edit 保住 97% 旋律相关性;重生成完全重作曲 |
| 和声保持 | Chroma(↑) | 1.000 | 0.938 | 同上 |
| 歌词遵循 | 新词命中(↑) | 自动 0.15 / **人工容错 ≈0.65** | — | 自动值被 ASR 误差低估(见下) |
| 歌词遵循 | 旧词残留(↓) | **0.00** | — | 旧词全部消失 |

**人工逐条复核**(对照 Whisper 转录与新歌词):
- 7/10 有明确新词痕迹:3 条完美(#000"黎明前…窗前"、#085"公園/微風"、#111"阿翔"≈"阿强"),3 条近音变体(#104 車隊都醒了、#107 每一封、#124 乘光初心),1 条半命中(#115 月光)
- **真实失败 3 条**:#073 转录全"啊啊啊"、#122 全"迷"——**人声退化成无词哼鸣**(过激型失败);#100 Verse 1 语义混乱
- 旧词残留 0/10:所有被替换的词都确实消失了

**关键结论**:
1. flow-edit 的**结构保持是卓越且真实的**:旋律相关性 0.966 vs 重生成 0.167,差距 5.8 倍——它确实是"编辑"而非"重新作曲"
2. **歌词遵循约 2/3 成功率**,失败分两类:人声退化成哼鸣(2/10)与近音误唱(转录被放大)
3. **ASR 是测量瓶颈**:whisper-small 对唱歌中文转录有繁体倾向+同音误听(如"阿强"→"阿翔"),自动指标 0.15 是下界;LyricEditBench 用唱歌特训的 Qwen3-ASR 正是为此。升级 ASR 或人工校对可提升测量精度
4. probe 还揭示:turbo 2B 的 flow-edit 词全部唱糊(new_hit 0),xl-turbo 才清晰——与 repaint 的 SFT 结论一致
5. **空白确认**:这是 ACE-Step 1.5 歌词编辑能力的首个公开评测;LyricEditBench(SVS 干声范式)未覆盖全曲 flow-edit

**ASR 复测(2026-09-15,Qwen3-ASR-1.7B,`rescore_asr.py`)**:换用 LyricEditBench 同款唱歌特训 ASR(transformers>=5.13 隔离环境,主 venv 未动)重转录 20 个音频:

| 测量 | 新词命中(↑) | 旧词残留(↓) | 综合遵循 |
|------|--------------|--------------|----------|
| whisper-small 自动 | 0.15 | 0.00 | 0.15 |
| **Qwen3-ASR 自动** | **0.55** | **0.05** | **0.55** |
| 人工容错复核 | ≈0.65-0.70 | 0.00 | ≈0.65 |

- ASR 升级使遵循度测量提升 3.7 倍,证实"Whisper 误差低估"假设;Qwen3-ASR 与人工复核已接近(剩余差距为唱歌转写的固有噪声)
- 重生成对照的 new_hit 也是 0.55:flow-edit 的歌词遵循与"直接重生成"相当,**同时**保住了旋律(CQT 0.966 vs 0.167)——净收益明确
- 个例:#000 flow-edit 新词逐字完美("黎明前的雨落满窗前…梦不再远");#122 Verse 1 改动未落在目标关键词上(半编辑)

**基础设施备忘**:whisper 长音频需 pipeline(return_timestamps=True)(顶层参数,放 generate_kwargs 无效);whisper-small 输出繁体倾向,关键词匹配需容错
