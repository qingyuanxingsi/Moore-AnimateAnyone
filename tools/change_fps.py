# -*- coding: utf-8 -*-

import os
import subprocess

local_dir = 'datasets/talk/videos'
local_dir = os.path.abspath(local_dir)
out_dir = os.path.join(os.path.dirname(local_dir), 'raw')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for file in os.listdir(local_dir):
    if not file.endswith('.mp4'):
        continue
    input_path = os.path.join(local_dir, file)
    output_path = os.path.join(out_dir, file)
    ffmpeg_cmd = f"ffmpeg -i {input_path} -r 8 {output_path}"
    subprocess.call(ffmpeg_cmd, shell=True)
