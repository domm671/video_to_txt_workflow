# Whisper Small 本地语音识别

使用 [faster-whisper](https://github.com/SYSTRAN/faster-whisper) 加载本地 **Whisper Small** 模型，对音频进行离线语音转文字（默认中文）。音频转录支持 CPU 与 CUDA（GPU）。视频工作流可将转录文本交给指定的大模型 API 勘误和分析。

## 视频工作流

在项目目录下使用 Python 3.11+。如果目录下还没有 `api_config.toml`，先将 `api_config.example.toml` 复制为 `api_config.toml`。按服务商文档填写 `base_url`、`model` 和 `api_key`。所选服务需支持 `POST /chat/completions`，基础地址保留服务所需的版本路径（如 `/v1`）。存放真实密钥的 `api_config.toml` 已加入 `.gitignore`。

保存配置后，在 PowerShell 中运行。

```powershell
python .\workflow.py 'E:\视频\示例.mp4'
```

1. 检查并安装缺失的 `faster-whisper` 和 `OpenCC`。如果当前运行在虚拟环境内，则使用该环境，否则使用项目 `.venv`，缺失时自动创建。FFmpeg 缺失时，Windows 使用 [WinGet](https://learn.microsoft.com/windows/package-manager/winget/install) 安装 `Gyan.FFmpeg`。没有 WinGet 的环境需要先安装 FFmpeg 并加入 `PATH`。
2. 提取第一条音轨到临时 WAV 文件，转换为单声道、16 kHz PCM。
3. 调用 `transcribe.py`，以 `config.toml` 指定的本地模型优先用 GPU 转录。GPU 转录失败时在独立进程中用 CPU 完整重跑。
4. 一次请求发送完整转录。模型先校订全文、连接断句，再基于校订稿分析论证链、思维模型、证据与边界。终端显示请求耗时。

每次运行在 `txt/<视频名称>-<随机后缀>/` 下生成 `原始转录.txt`、`勘误后.txt` 和 `思维分析.md`。API 请求失败或返回不完整时保留原始转录。音频提取和 Whisper 转录在本地完成，勘误和分析会将文本发往配置的 API。模型须有足够的上下文和输出额度来处理全文并生成两个结果。若响应达到输出上限，可按服务商文档在 `api_config.toml` 中选填 `max_completion_tokens` 或 `max_tokens`。

### 火山方舟 Agent Plan

使用和 Pi 相同的 [Agent Plan 专属 API Key](https://docs.volcengine.com/docs/ark/agent-plan-personal-pi?lang=zh)，在 `api_config.toml` 中填入以下参数。`stream` 开启后仍然只发一次请求，终端会随正文到达显示接收进度。`max_completion_tokens` 对应 Pi 配置里的 `maxTokens`，表示上限，生成提前结束时按实际输出计量。

```toml
base_url = "https://ark.cn-beijing.volces.com/api/plan/v3"
model = "deepseek-v4.1-flash"
api_key = "<Agent Plan 专属 API Key>"
stream = true
max_completion_tokens = 393216
```

其他 OpenAI 兼容服务可按支持情况设置 `stream = true`，省略 `stream` 时按整包 JSON 响应读取。

## 功能特点

- **完全离线**：模型与识别过程均在本地完成，`local_files_only=True`
- **CPU / GPU 双版本**：`int8` 量化，CPU 可跑，GPU 更快
- **VAD 过滤**：自动跳过静音片段，减少空转与幻听
- **逐段输出**：识别结果按句打印，并保存为同名 `.txt` 文件（UTF-8 BOM，Windows 记事本可直接打开）

## 目录结构

```
whisper-small/
├── models/
│   └── faster-whisper-small/     # 本地模型（需自行下载，未纳入版本控制）
│       ├── config.json
│       ├── model.bin
│       ├── tokenizer.json
│       └── vocabulary.txt
├── txt/                          # 识别输出目录（自动创建）
├── config.toml                   # 模型路径配置
├── api_config.example.toml       # 大模型 API 配置模板
├── api_config.toml               # 本机大模型 API 配置（不提交）
├── test_whisper.py               # CPU 版本
├── test_whisper_gpu.py           # GPU（CUDA）版本
├── transcribe.py                 # 视频工作流的转录入口
├── workflow.py                   # 视频转录、勘误和分析入口
├── .gitignore
└── README.md
```

## 环境准备

要求 Python 3.11+（使用内置 `tomllib` 读取配置）。

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install faster-whisper OpenCC
```

GPU 版本还需要安装与 CUDA 版本匹配的 PyTorch / cuBLAS / cuDNN 运行库；若缺少依赖，faster-whisper 会提示对应的 `.dll` 缺失，请按提示安装对应版本的 `nvidia-cublas-cu12`、`nvidia-cudnn-cu12` 等包。

## 模型下载

项目默认从本地目录读取模型：

```
models/faster-whisper-small
```

可使用 Hugging Face CLI 下载（CTranslate2 格式）：

```bash
pip install huggingface_hub
huggingface-cli download Systran/faster-whisper-small --local-dir models/faster-whisper-small
```

也可以直接使用 `WhisperModel("small")`，让 faster-whisper 自动下载到缓存目录。

## 使用方法

1. 在 `config.toml` 中设置模型路径（相对路径以脚本所在目录为准）：

```toml
model_path = "models/faster-whisper-small"
```

2. 运行脚本：

```bash
# CPU
python test_whisper.py

# GPU
python test_whisper_gpu.py
```

3. 按提示输入音频路径（支持拖拽，自动去掉首尾引号）：

```text
请输入音频文件路径：E:\audio\meeting.m4a
```

识别结果会打印到终端，同时写入 `txt/` 目录下与输入文件同名的 `.txt`（如 `meeting.m4a` → `txt/meeting.txt`，UTF-8 BOM 编码）。

## 主要参数

| 参数 | 说明 |
| --- | --- |
| `model_path` | 本地模型目录（在 `config.toml` 中配置） |
| `device` | `"cpu"` 或 `"cuda"` |
| `compute_type` | 计算精度，`"int8"`（省显存/内存）、`"float16"`（GPU 更快）、`"float32"` |
| `language` | 识别语言，固定为 `"zh"` |
| `beam_size` | 束搜索宽度，越大越准但越慢 |
| `vad_filter` | 是否启用语音活动检测过滤静音 |
| `local_files_only` | 仅使用本地文件，禁止联网下载 |

## 可能遇到的问题

- **`模型加载成功` 后长时间无输出**：首次运行需初始化计算图，属正常现象；若音频较长请耐心等待。
- **GPU 报 DLL 缺失**：CUDA / cuDNN 版本不匹配，建议改用 CPU 版本，或安装匹配版本的 NVIDIA pip 包。
- **中文识别乱码**：输出文件已使用 `utf-8-sig` 编码，如需在其他程序中使用可改为 `utf-8`。
- **识别结果不理想**：可尝试增大 `beam_size`，或换用更大的模型（`medium`、`large-v3`）。

## 说明

`models/`、`.venv/`、音频文件与识别结果均已在 `.gitignore` 中忽略，不会提交到仓库。
