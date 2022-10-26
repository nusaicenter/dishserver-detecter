import argparse
import os

from tqdm import tqdm
from pathlib import Path
import sys


parser = argparse.ArgumentParser()
parser.add_argument('-f','--folder')
args = parser.parse_args()
if args.folder is None:
    print('Please use -f or --folder to pass the video folder path')
    exit(0)
if not os.path.isdir(args.folder):
    print('Please pass a valid folder path')
    exit(0)

sys.path.append(str(Path(__file__).parent.parent))
from video_sampler import VideoSampler
video_folder = args.folder

# video_folder = '/Volumes/20220831'
video_paths = [
    os.path.join(video_folder, name) for name in os.listdir(video_folder)
    if name.endswith('.mp4')
]
basename = os.path.basename(video_folder)
save_path = f'./tmp/{basename}'

for path in tqdm(video_paths):
    vs = VideoSampler()
    video_name = os.path.splitext(os.path.basename(path))[0]
    vs.capture(src=path, fps=5,assume_stable=1000, diff_area_base=0.2, video_name=video_name)
    vs.save_to_folder(root_path=save_path)

print(f"All videos in folder {video_folder} have been processed")