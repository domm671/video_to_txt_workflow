import argparse
import tomllib
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="为视频工作流转录音频")
    parser.add_argument("audio", type=Path, help="音频文件路径")
    parser.add_argument("output", type=Path, help="输出 TXT 路径")
    parser.add_argument("device", choices=("cuda", "cpu"), help="转录设备")
    args = parser.parse_args()

    from faster_whisper import WhisperModel
    from opencc import OpenCC

    with (BASE_DIR / "config.toml").open("rb") as config:
        model_path = BASE_DIR / Path(tomllib.load(config)["model_path"])

    print(f"正在加载本地 Whisper Small 模型（{args.device}）……", flush=True)
    model = WhisperModel(str(model_path), device=args.device, compute_type="int8", local_files_only=True)
    print("模型加载成功，开始识别……", flush=True)
    segments, _ = model.transcribe(str(args.audio), language="zh", beam_size=3, vad_filter=True)

    cc = OpenCC("t2s")
    lines = []
    for segment in segments:
        if text := segment.text.strip():
            text = cc.convert(text)
            print(text, flush=True)
            lines.append(text + "\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(lines), encoding="utf-8-sig")
    print(f"\n识别完成！输出文件：{args.output}", flush=True)


if __name__ == "__main__":
    main()
