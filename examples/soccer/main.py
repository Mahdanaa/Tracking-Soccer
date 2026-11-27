import argparse
import sys
from pathlib import Path

# Support the original direct script command as well as python -m examples.soccer.main.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import Mode
from src.processor import process_video


def main():
    parser = argparse.ArgumentParser(description="Process a match with the Streamlit analysis pipeline.")
    parser.add_argument("--source_video_path", type=Path, required=True)
    parser.add_argument("--target_video_path", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "auto"], default="cpu")
    parser.add_argument("--mode", type=Mode, choices=list(Mode), default=Mode.RADAR)
    parser.add_argument("--no_preview", action="store_true")
    args = parser.parse_args()
    result = process_video(args.source_video_path, args.target_video_path, args.mode, device="cpu")
    print(f"{result.frames_processed} frames in {result.total_seconds:.2f}s: {result.output_path}")
    for warning in result.warnings:
        print(warning)
    if not args.no_preview:
        import cv2
        from contextlib import closing
        from src.video import iter_frames, read_video_info

        delay = max(1, round(1000 / read_video_info(result.output_path).fps))
        try:
            with closing(iter_frames(result.output_path)) as frames:
                for frame in frames:
                    cv2.imshow("Soccer Analytics", frame)
                    if cv2.waitKey(delay) & 0xFF == ord("q"):
                        break
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
