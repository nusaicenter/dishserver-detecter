import os
from video_sampler import VideoSampler


if __name__ == '__main__':
    vs = VideoSampler()


    # src_folder_path = '/Users/jiahua/Downloads/moving_det_cv/testdata'
    # for path in os.listdir(src_folder_path):
    #     if '.mp4' in path:

    #         video_path = os.path.join(src_folder_path, path)
    #         video_name = os.path.splitext(os.path.basename(video_path))[0]
    #         vs.capture(src=video_path, fps=5,assume_stable=900, diff_area_base=0.2, video_name=video_name)

    #         save_path = os.path.join('capture', video_name)
    #         os.makedirs(save_path, exist_ok=True)
    #         vs.save_to_folder(root_path=save_path)

    video_path = '/Users/jiahua/Downloads/moving_det_cv/testdata/test2.mp4'
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    vs.capture(src=video_path, fps=5,assume_stable=900, diff_area_base=0.2, video_name=video_name)

    save_path = 'capture'
    os.makedirs(save_path, exist_ok=True)

    vs.save_to_folder(root_path=save_path)
