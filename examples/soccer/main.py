import argparse

import cv2
import supervision as sv
from tqdm import tqdm

from config import Mode
from pipeline import analyze_video


def main(
    source_video_path: str,
    target_video_path: str,
    device: str,
    mode: Mode,
    no_preview: bool,
) -> None:
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    frames = analyze_video(source_video_path=source_video_path, mode=mode, device=device)

    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in tqdm(frames, total=video_info.total_frames, desc=f'Processing ({mode.value})'):
            sink.write_frame(frame)
            if not no_preview:
                cv2.imshow('Soccer Analytics', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        if not no_preview:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Soccer Analytics ONNX Runner')
    parser.add_argument('--source_video_path', type=str, required=True)
    parser.add_argument('--target_video_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--mode', type=Mode, default=Mode.RADAR)
    parser.add_argument('--no_preview', action='store_true')
    args = parser.parse_args()

    main(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        device=args.device,
        mode=args.mode,
        no_preview=args.no_preview,
    )
