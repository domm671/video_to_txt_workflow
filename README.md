# Whisper Small 本地语音识别

使用 [faster-whisper](https://github.com/SYSTRAN/faster-whisper) 加载本地 **Whisper Small** 模型，对音频进行离线语音转文字（默认中文）。支持 CPU 与 CUDA（GPU）两种运行方式，无需联网、无需调用云端 API。

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
├── txt/                          # 识别输出目录
├── test_whisper.py               # CPU 版本
├── test_whisper_gpu.py           # GPU（CUDA）版本
├── .gitignore
└── README.md
```

## 环境准备

要求 Python 3.9+。

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install faster-whisper
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

1. 将待识别音频放到项目根目录（示例为 `test.m4a`）。
2. 修改脚本中的两个路径：

```python
model_path = r"E:\AI\whisper-small\models\faster-whisper-small"
audio_path = r"E:\AI\whisper-small\test.m4a"
```

3. 运行：

```bash
# CPU
python test_whisper.py

# GPU
python test_whisper_gpu.py
```

识别结果会打印到终端，同时写入与音频同名的 `.txt` 文件。

## 主要参数

| 参数 | 说明 |
| --- | --- |
| `model_path` | 本地模型目录，或模型名（如 `"small"`） |
| `device` | `"cpu"` 或 `"cuda"` |
| `compute_type` | 计算精度，`"int8"`（省显存/内存）、`"float16"`（GPU 更快）、`"float32"` |
| `language` | 识别语言，如 `"zh"`、`"en"`；传 `None` 可自动检测 |
| `beam_size` | 束搜索宽度，越大越准但越慢 |
| `vad_filter` | 是否启用语音活动检测过滤静音 |
| `local_files_only` | 仅使用本地文件，禁止联网下载 |

## 可能遇到的问题

- **`模型加载成功` 后长时间无输出**：首次运行需初始化计算图，属正常现象；若音频较长请耐心等待。
- **GPU 报 DLL 缺失**：CUDA / cuDNN 版本不匹配，建议改用 CPU 版本，或安装匹配版本的 NVIDIA pip 包。
- **中文识别乱码**：输出文件已使用 `utf-8-sig` 编码，如需在其他程序中使用可改为 `utf-8`。
- **识别结果不理想**：可尝试增大 `beam_size`、明确指定 `language`，或换用更大的模型（`medium`、`large-v3`）。

## 说明

`models/`、`.venv/`、音频文件与识别结果均已在 `.gitignore` 中忽略，不会提交到仓库。
