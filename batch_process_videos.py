from video_sampler import VideoSampler
import os
from tqdm import tqdm
# video_folder = '/Users/jiahua/Downloads/dishvideo_220725'
# video_folder = '/Users/jiahua/Downloads/moving_det_cv/videos/20220817'
video_folder = '/Volumes/JIAHUA_32/20220831'
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
