# Whisper Small 视频转录与分析

将中文视频转成可阅读的校订稿和思维分析。FFmpeg 提取音轨后，faster-whisper 使用本地 Whisper Small 模型转录，优先使用 NVIDIA GPU，失败时自动切换到 CPU。大模型收到一次完整转录，先勘误并连接断句，再基于校订稿分析论证结构和思维模型。

音频提取和语音转录在本机运行。勘误和分析会把转录文本发送给你配置的 API 服务。已有音频也可以通过 [本地音频转录](#本地音频转录) 直接生成 TXT。

## 快速开始

以下命令适用于 Windows PowerShell。克隆仓库后，在项目根目录运行。需要 Python 3.11 或更高版本。

### 1. 准备 Whisper Small 模型

创建虚拟环境并下载 CTranslate2 格式的模型。已有模型时，只需创建虚拟环境并在 `config.toml` 中设置 `model_path`，相对路径以项目根目录为准。 [模型文件](https://huggingface.co/Systran/faster-whisper-small) · [Hugging Face CLI 文档](https://huggingface.co/docs/huggingface_hub/en/guides/cli#download-to-a-local-folder)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install huggingface_hub
.\.venv\Scripts\hf.exe download Systran/faster-whisper-small --local-dir models/faster-whisper-small
```

### 2. 配置大模型 API

首次使用时复制配置模板，再编辑 `api_config.toml`。

```powershell
if (-not (Test-Path .\api_config.toml)) {
    Copy-Item .\api_config.example.toml .\api_config.toml
}
```

| 字段 | 填写方式 |
| --- | --- |
| `base_url` | 服务商提供的 OpenAI Chat Completions 基础地址，保留版本路径并去掉末尾的 `/chat/completions` |
| `model` | 服务商提供的模型 ID |
| `api_key` | 服务商签发的 API Key |
| `stream` | 可选，设置为 `true` 时随正文到达显示接收进度 |
| `max_completion_tokens` 或 `max_tokens` | 可选，按服务商支持情况选填其中一个，用于设置输出上限 |

真实密钥保存在本机的 `api_config.toml`，该文件已由 `.gitignore` 排除。提交到 GitHub 的配置模板使用空值。 [配置模板](api_config.example.toml)

<details>
<summary>火山方舟 Agent Plan 示例</summary>

使用 Agent Plan 专属 API Key。如果你已在 Pi 中使用同一套餐，可沿用对应的基础地址、模型 ID 和专属密钥。示例的输出上限与该模型在 Pi 中的配置一致。 [火山方舟配置文档](https://docs.volcengine.com/docs/ark/agent-plan-personal-pi?lang=zh)

```toml
base_url = "https://ark.cn-beijing.volces.com/api/plan/v3"
model = "deepseek-v4.1-flash"
api_key = "<Agent Plan 专属 API Key>"
stream = true
max_completion_tokens = 393216
```

</details>

### 3. 转录视频

将示例路径换成要处理的视频文件。

```powershell
.\.venv\Scripts\python.exe .\workflow.py 'E:\视频\示例.mp4'
```

工作流会检查 `faster-whisper` 和 `OpenCC`，并安装缺失的 Python 依赖。Windows 上缺少 FFmpeg 时会尝试通过 WinGet 安装。使用其他系统或没有 WinGet 时，请先安装 FFmpeg 并将其加入 `PATH`。 [FFmpeg 下载](https://ffmpeg.org/download.html) · [WinGet 文档](https://learn.microsoft.com/windows/package-manager/winget/install)

每次运行在 `txt/<视频名>-<随机后缀>/` 下保存三个文件。

| 文件 | 内容 |
| --- | --- |
| `原始转录.txt` | 从视频第一条音轨得到的 Whisper 转录 |
| `勘误后.txt` | 大模型校订后的全文，包括错字修正、断句连接和语义分段 |
| `思维分析.md` | 基于校订稿整理的核心结论、论证链、思维模型、证据及适用边界 |

API 请求结束后，终端会显示模型请求耗时。请求失败或响应不完整时，工作流会保留原始转录。

## 本地音频转录

这两个交互式入口可以直接读取音频文件。按提示输入音频路径，输出写入 `txt/<音频名>.txt`，编码为 UTF-8 BOM。转录使用中文识别、静音过滤和繁体转简体。

```powershell
.\.venv\Scripts\python.exe -m pip install faster-whisper OpenCC

# CPU
.\.venv\Scripts\python.exe .\test_whisper.py

# NVIDIA CUDA
.\.venv\Scripts\python.exe .\test_whisper_gpu.py
```

## 常见问题

- **GPU 加载失败**。检查 NVIDIA 驱动、cuBLAS 和 cuDNN 是否与已安装的 CTranslate2 版本匹配。视频工作流会自动回退到 CPU。 [faster-whisper GPU 环境说明](https://github.com/SYSTRAN/faster-whisper#gpu)
- **找不到模型**。检查 `config.toml` 的 `model_path`，确认目标目录下有 `model.bin`、`config.json` 和 `tokenizer.json`。
- **模型返回不完整**。一次调用需要容纳完整转录和两份输出。检查所用模型的上下文窗口和服务商支持的输出上限配置。
- **FFmpeg 提取失败**。确认 `ffmpeg -version` 能运行，且视频文件包含音轨。

## 项目文件

| 文件 | 用途 |
| --- | --- |
| `workflow.py` | 视频工作流入口，负责提取音轨、设备回退和 API 调用 |
| `transcribe.py` | 视频工作流使用的本地转录脚本 |
| `test_whisper.py`、`test_whisper_gpu.py` | CPU 和 CUDA 音频转录入口 |
| `config.toml` | 本地 Whisper 模型路径 |
| `api_config.example.toml` | 大模型 API 配置模板 |
| `test_workflow.py` | API 配置、流式响应和输出校验的单元测试 |

运行工作流测试。

```powershell
python -m unittest -v test_workflow
```

模型权重、虚拟环境、`input/` 素材和 `txt/` 输出均保留在本机。 [Git 忽略规则](.gitignore)
