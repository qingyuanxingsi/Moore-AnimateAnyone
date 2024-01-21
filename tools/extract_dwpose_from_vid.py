import concurrent.futures
import os
import random
from pathlib import Path

import numpy as np
import sys
from tqdm import tqdm
import shutil
from PIL import Image
import cv2
sys.path.append('/mnt/chongqinggeminiceph1fs/geminicephfs/security-others-common/doodleliang/Moore-AnimateAnyone')
from src.dwpose import DWposeDetector
from src.utils.util import get_fps, read_frames, save_videos_from_pil

# Extract dwpose mp4 videos from raw videos
# /path/to/video_dataset/*/*.mp4 -> /path/to/video_dataset_dwpose/*/*.mp4


def process_single_video(video_path, detector, root_dir, save_dir):
    relative_path = os.path.relpath(video_path, root_dir)
    print(relative_path, video_path, root_dir)
    out_path = os.path.join(save_dir, relative_path)
    if os.path.exists(out_path):
        return
    output_dir = Path(os.path.dirname(os.path.join(save_dir, relative_path)))
    frame_dir = os.path.join(os.path.dirname(save_dir), 'frames', relative_path)
    if os.path.exists(frame_dir):
        shutil.rmtree(frame_dir)
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir, exist_ok=True)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    fps = get_fps(video_path)
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # kps_results = []
    bar = tqdm(total=frame_count)
    frame_idx = 0
    while cap.isOpened():
        bar.update(1)
        ret, frame = cap.read()
        if not ret:
            break
        result, score = detector(frame)
        score = np.mean(score, axis=-1)
        print(score)
        # kps_results.append(result)
        result.save(os.path.join(frame_dir, f"{frame_idx:06d}.jpg"))
        frame_idx += 1
    bar.close()
    cap.release()

    save_videos_from_pil(frame_dir, out_path, fps=fps)


def process_batch_videos(video_list, detector, root_dir, save_dir):
    for i, video_path in enumerate(video_list):
        print(f"Process {i}/{len(video_list)} video")
        process_single_video(video_path, detector, root_dir, save_dir)


if __name__ == "__main__":
    # -----
    # NOTE:
    # python tools/extract_dwpose_from_vid.py --video_root /path/to/video_dir
    # -----
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", type=str)
    parser.add_argument(
        "--save_dir", type=str, help="Path to save extracted pose videos"
    )
    parser.add_argument("-j", type=int, default=1, help="Num workers")
    args = parser.parse_args()
    num_workers = args.j
    args.video_root = os.path.abspath(args.video_root)
    if args.save_dir is None:
        save_dir = args.video_root + "_dwpose"
    else:
        save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    gpu_ids = [int(id) for id in range(len(cuda_visible_devices.split(",")))]
    print(f"avaliable gpu ids: {gpu_ids}")

    # collect all video_folder paths
    video_mp4_paths = set()
    for root, dirs, files in os.walk(args.video_root):
        for name in files:
            if name.endswith(".mp4"):
                video_mp4_paths.add(os.path.join(root, name))
    video_mp4_paths = list(video_mp4_paths)
    random.shuffle(video_mp4_paths)

    # split into chunks,
    batch_size = (len(video_mp4_paths) + num_workers - 1) // num_workers
    print(f"Num videos: {len(video_mp4_paths)} {batch_size = }")
    video_chunks = [
        video_mp4_paths[i : i + batch_size]
        for i in range(0, len(video_mp4_paths), batch_size)
    ]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i, chunk in enumerate(video_chunks):
            # init detector
            gpu_id = gpu_ids[i % len(gpu_ids)]
            detector = DWposeDetector()
            # torch.cuda.set_device(gpu_id)
            detector = detector.to(f"cuda:{gpu_id}")

            futures.append(
                executor.submit(
                    process_batch_videos, chunk, detector, args.video_root, save_dir
                )
            )
        for future in concurrent.futures.as_completed(futures):
            future.result()
