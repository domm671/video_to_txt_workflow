
import tomllib
from pathlib import Path

from faster_whisper import WhisperModel
from opencc import OpenCC

base_dir = Path(__file__).resolve().parent

# 从 config.toml 读取模型路径
with open(base_dir / "config.toml", "rb") as f:
    model_path = Path(tomllib.load(f)["model_path"])

if not model_path.is_absolute():
    model_path = base_dir / model_path

# 音频路径由用户输入
audio_path = Path(
    input("请输入音频文件路径：").strip().strip('"')
).expanduser()

if not audio_path.is_absolute():
    audio_path = Path.cwd() / audio_path

# 输出到 txt 文件夹下与输入文件同名的 txt
output_path = base_dir / "txt" / f"{audio_path.stem}.txt"
output_path.parent.mkdir(parents=True, exist_ok=True)

# 初始化繁体转简体转换器
cc = OpenCC("t2s")

print("正在加载本地 Whisper Small 模型……")

model = WhisperModel(
    str(model_path),
    device="cuda",
    compute_type="int8",
    local_files_only=True,
)

print("模型加载成功，开始识别……")

segments, info = model.transcribe(
    str(audio_path),
    language="zh",
    beam_size=3,
    vad_filter=True,
)

with output_path.open("w", encoding="utf-8-sig") as f:
    for segment in segments:
        text = segment.text.strip()

        if text:
            # 繁体中文转换为简体中文
            text = cc.convert(text)

            print(text)
            f.write(text + "\n")

print(f"\n识别完成！输出文件：{output_path}")
