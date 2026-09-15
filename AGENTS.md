# AGENTS.md

ACE-Step 1.5:开源音乐生成模型(LM 规划器 + DiT + VAE 架构)。支持 CUDA / ROCm / Intel XPU / MPS / MLX / CPU 多平台,任何代码改动都要考虑不破坏其他平台路径。

## 常用命令

```bash
uv sync                # 安装依赖(Python 3.11-3.12,uv 管理)
uv run acestep         # 启动 Gradio UI(:7860),首次运行自动下载模型
uv run acestep-api     # 启动 REST API(:8001)
uv run acestep-download  # 仅下载模型
python train.py {vanilla|fixed|estimate}  # 训练 CLI(Training V2 / Side-Step 集成)
```

- 根目录 `cli.py` 是交互式 CLI 向导(不是库代码);`profile_inference.py` 是基准/性能分析工具。
- 文档在 `docs/en|zh|ja|ko/`(INSTALL、API、Tutorial、GPU_COMPATIBILITY、LoRA_Training_Tutorial 等)。

## 测试(重要陷阱)

- unittest 风格,文件命名 `*_test.py` 或 `test_*.py`(仓库约 190 个)。
- **不要全量 discover**:`uv run python -m unittest discover -s . -p "*_test.py"`(以及 `-s acestep`)会 import 带副作用的模块而卡住,已验证不可用。
- 正确跑法:
  ```bash
  # 单模块(最可靠)
  uv run python -m unittest acestep.cli_args_test
  # 单类 / 单方法
  uv run python -m unittest acestep.training.test_lora_utils.TestUnwrapDecoder
  # 小子目录 discover 可用(仅限确认干净的目录,如 training)
  uv run python -m unittest discover -s acestep/training -p "*_test.py"
  ```
- 测试必须 CPU 可跑、确定性、快速;用 `unittest.mock.MagicMock`/`patch` 隔离 GPU、文件系统、网络、外部服务。
- 每个行为变更/修复至少附:1 个成功路径测试 + 1 个回归/边界测试。

## 配置与环境

- `.env`(已存在、gitignored)优先于 `.env.example`;Python 脚本与启动脚本都会加载它。关键变量:`ACESTEP_CONFIG_PATH`(DiT 模型)、`ACESTEP_LM_MODEL_PATH`、`ACESTEP_INIT_LLM`(auto/true/false,控制是否加载 LM)、`ACESTEP_LM_BACKEND`(vllm/pt)。
- `checkpoints/` 存模型权重(gitignored),`gradio_outputs/`、`lora_output/` 同为生成产物。
- 模型默认 lazy-load(服务快速启动);改启动/加载逻辑时注意 `ACESTEP_NO_INIT` 语义。

## 架构速览

- 入口:`acestep/acestep_v15_pipeline.py`(Gradio `main`)、`acestep/api_server.py`(FastAPI,job 运行时在 `acestep/api/`)。
- `acestep/core/`:generation(handler)、llm、lora、audio、scoring、system。
- `acestep/models/`:按模型变体分目录(turbo / sft / base / xl_* / mlx)。
- `acestep/ui/gradio/`:interfaces、events、i18n(UI 文案多语言)。
- `acestep/training/`(LoRA V1)与 `acestep/training_v2/`(Side-Step:presets、cli、ui)是两套训练实现。
- `acestep/text_tasks/`:外部 LM(OpenRouter 等)captioning/标注集成,含密钥安全存储。
- `acestep/third_parts/nano-vllm` 是 vendored 本地包(`pyproject.toml` 的 `[tool.uv.sources]` 指向它),不是 PyPI 依赖;macOS arm64 不安装。

## 多平台纪律

- 本 fork 已精简为 **Linux x86_64 + CUDA 专用研究环境**(pyproject 仅保留该目标环境;Windows/macOS/ROCm/XPU 启动脚本与对应依赖已删除)。上游仍维护多平台,故 `acestep/gpu_config.py` 的 mps/xpu/rocm 分支与 `models/mlx/` 代码原样保留(永不命中),不要顺手清理它们。
- 仍然只在 `acestep/gpu_config.py` 里做硬件检测,不要在业务代码里自行判断设备。
- 改 `pyproject.toml` 时保持 `required-environments` 为 linux x86_64,勿恢复多平台 marker。

## 变更纪律(强制,源自 CONTRIBUTING.md 与 Discussion #408/#365)

- 一个 PR 只解决一个问题;最小 diff;禁止顺手重构、格式化清扫、无关清理。
- 保留现有公共接口,除非任务明确要求改接口。
- 模块规模:目标 ≤150 LOC,硬上限 200;超限需按职责拆分,或在 PR 说明理由并附拆分计划。拆分大文件时保留稳定 facade import,避免破坏外部调用者。
- docstring 强制(模块/类/公共函数,含 Args/Returns);日志用 `loguru.logger`,禁止 `print`(CLI 输出除外);禁止裸 `except:`。
- PR 描述需注明:范围、非目标平台未变(或论证)、已跑的回归检查。评审流程模板见 `CONTRIBUTING.md`。

## Git / PR 注意

- `.githooks/pre-push`(防止分支交叉污染)不会自动生效,需手动启用:`git config core.hooksPath .githooks`。
- 远程:`origin` = 本人 fork(`zllmma/ACE-Step-1.5`,推送目标),`upstream` = 官方 `ace-step/ACE-Step-1.5`(同步官方更新用 `git fetch upstream`);PR 分支必须基于 upstream/main 新建,不要复用其他分支的提交。
