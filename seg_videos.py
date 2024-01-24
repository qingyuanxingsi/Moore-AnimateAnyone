# -*- coding: utf-8 -*-

import os
import pickle
import time
from tqdm import tqdm
from scenedetect import open_video, SceneManager, split_video_ffmpeg
from scenedetect.detectors import ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg

def split_video_into_scenes(video_path, threshold=27.0):
    # Open our video, create a scene manager, and add a detector.
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video, show_progress=True)
    scene_list = scene_manager.get_scene_list()
    max_len = 10.0
    final_segs = []
    for start, end in scene_list:
        start_time = start.get_seconds()
        end_time = end.get_seconds()
        duration = end_time - start_time
        num_segments = int(duration / max_len) + 1
        for j in range(num_segments):
            seg_start = start + j * max_len
            if seg_start >= end:
                break
            seg_end = seg_start + max_len
            if seg_end > end:
                seg_end = end
            if seg_start + 2.0 <= seg_end:
                final_segs.append((seg_start, seg_end))
    split_video_ffmpeg(video_path, final_segs, show_progress=True)


if __name__ == '__main__':
    pickle_path = 'seg_info.pkl'
    if not os.path.exists(pickle_path):
        seg_dict = {}
    else:
        seg_dict = pickle.load(open(pickle_path, 'rb'))
    local_dir = 'datasets/talk/videos'
    local_dir = os.path.abspath(local_dir)
    out_dir = os.path.join(os.path.dirname(local_dir), 'seg')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    os.chdir(out_dir)

    pre_ts = 0
    for file in tqdm(sorted(os.listdir(local_dir))):
        if not file.endswith('.mp4'):
            continue
        input_path = os.path.join(local_dir, file)
        split_video_into_scenes(input_path)
        cur_ts = int(time.time())
        if cur_ts - pre_ts >= 600:
            pickle.dump(seg_dict, open(pickle_path, 'wb'))
            pre_ts = cur_ts