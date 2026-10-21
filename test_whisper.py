
from faster_whisper import WhisperModel

# 本地模型目录
model_path = r"E:\AI\whisper-small\models\faster-whisper-small"

# M4A 音频路径
audio_path = r"E:\AI\whisper-small\test.m4a"

print("正在加载本地 Whisper Small 模型……")

model = WhisperModel(
    model_path,
    device="cpu",
    compute_type="int8",
    local_files_only=True,
)

print("模型加载成功，开始识别……")

segments, info = model.transcribe(
    audio_path,
    language="zh",
    beam_size=3,
    vad_filter=True,
)

output_path = audio_path.rsplit(".", 1)[0] + ".txt"

with open(output_path, "w", encoding="utf-8-sig") as f:
    for segment in segments:
        text = segment.text.strip()
        if text:
            print(text)
            f.write(text + "\n")

print(f"\n识别完成！输出文件：{output_path}")
