# This is the API for sampling stable frames from one video
import cv2
import numpy as np
import os
import time
from math import ceil

# MOVE -> MOVE: do nothing
# MOVE -> STAY: counter + 1
# STAY -> MOVE: reset counter
# STAY -> STAY: counter + 1, if trigger limit then save frame. Not repeat to save
STAY, MOVE = 0, 1


def check_status(prev: np.ndarray,
                 cur: np.ndarray,
                 diff_thresh: float = 255 * 0.4,
                 diff_area: float = 0.3):
    # First check the diff value: [0~255] diff_thresh
    # then check the diff percent over all frame: [0~1] diff_area

    # Args:
    # prev: np.ndarray, the previous frame
    # cur: np.ndarray, the current frame
    # # The frames should be one channel
    # diff_area: float, 0~1, the ratio of moving pixel number over total
    #   judge STAY if diff pixel ratio > diff_area
    # diff_thresh: float, 0~255, the absolute threshold to split diff frames

    diff = cv2.absdiff(cur, prev)
    diff = cv2.threshold(diff,
                         thresh=diff_thresh,
                         maxval=255,
                         type=cv2.THRESH_BINARY)[1]

    # count percent of diff pixels in image
    area = diff.shape[0] * diff.shape[1]
    diff_percent = (diff == 255).sum() / area
    status = MOVE if diff_percent > diff_area else STAY

    return status, diff, diff_percent


def gen_filename(video_name):
    # each frame to save has the relative path "video_name_time.time.jpg"
    return f'{video_name}_{str(time.time())}.jpg'


def gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def if_background(img_path, bg_path, fps=5, diff_area_base=0.2):
    img = gray(cv2.imread(img_path))
    bg = gray(cv2.imread(bg_path))

    diff_area = diff_area_base / fps
    notBackground, _, _ = check_status(prev=bg,
                                       cur=img,
                                       diff_thresh=255 * 0.4,
                                       diff_area=diff_area)
    return notBackground

class VideoSampler(object):
    def __init__(self) -> None:
        # save frames to memory temporary and save to dish together
        self.save_buffer = dict()

    def reset(self):
        self.save_buffer = dict()

    def capture(self,
                src: str,
                fps: int = 5,
                assume_stable: int = 900,
                diff_thresh: float = 255 * 0.4,
                diff_area_base: float = 0.2,
                background: np.ndarray = None,
                video_name: str = 'video'):
        # The main function of the class. Capturing all stable frames from one
        # video, and save them into a dict whose keys is
        # {video_name}/{timestamp}. This is the relative path format and all
        # images??will be saved in {video_name} folder.

        # Args:
        # src: the path of the video
        # fps: sampling fps, but will be up-limited by video fps, lower fps makes
        #   runing faster
        # assume_stable: unit is 'ms', if no frames move after this time, assume
        #   stable and capture this frame
        # diff_thresh: how much the difference between previous/after pixel value
        #   will be assumed as different(move)
        # diff_area_base: how many moving pixels(per) in one frame will be
        #   assumed as the moving frame
        # background: if the moving frame is similar to background, it won't be
        #   captured
        # video_name: the name of src video

        # initialization
        assert os.path.exists(src), 'File not exists'

        prev_status = MOVE
        stay_counter = 0
        save_flag = False
        skip_counter = 0

        cap = cv2.VideoCapture(src)  # load file
        ret, frame = cap.read()  # try to load one frame
        prev_frame = gray(frame)
        if background is None:
            background = gray(frame)

        # 'skip' when reading video, to reduce computation
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if not video_fps:
            video_fps = 30
        skip_frames = ceil(video_fps / fps)

        # if the frame stay more than some time(ms), it is assumed as stable
        frame_to_stay = ceil(assume_stable * fps / 1000)

        # the ratio of moving pixel number over total, diff more than the ratio is stay
        diff_area = diff_area_base / fps

        # start capture stabel frames
        # t0 = time.time()
        while True:
            ret, ori_frame = cap.read()
            if not ret:
                break

            skip_counter += 1
            if skip_counter % skip_frames != 0:
                continue

            frame = gray(ori_frame)
            status, diff, percent = check_status(prev=prev_frame,
                                                 cur=frame,
                                                 diff_thresh=diff_thresh,
                                                 diff_area=diff_area)

            # if prev_status == MOVE and status == MOVE:
            #     pass

            if prev_status == MOVE and status == STAY:
                stay_counter += 1

            if prev_status == STAY and status == MOVE:
                stay_counter = 0
                save_flag = False

            if prev_status == STAY and status == STAY:
                stay_counter += 1
                if stay_counter > frame_to_stay and not save_flag:
                    # not save backgroud images
                    notBackground, _, _ = check_status(prev=background,
                                                       cur=frame,
                                                       diff_thresh=255 * 0.4,
                                                       diff_area=diff_area)
                    if notBackground:
                        self.save_buffer[gen_filename(video_name)] = ori_frame
                        print('static: save frame')
                        # print('backgroud, not save')

                    save_flag = True

            prev_status = status
            prev_frame = frame.copy()

            # print(f"{'MOVE' if status else 'STAY'}: {stay_counter}, diff_percent: {percent*100:.2f}%", end='\r')
        # print(f'process finished, time: {(time.time()-t0):.4f}s')
        cap.release()

    def save_to_folder(self, root_path: str):
        os.makedirs(root_path, exist_ok=True)
        for path, frame in self.save_buffer.items():
            save_path = os.path.join(root_path, path)
            cv2.imwrite(save_path, frame)



if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--path', help='path of the video to process')
    parser.add_argument('--folder', help='path of the video folder to process')
    parser.add_argument('--fps', type=int, default=5, help='sample frequency of the program')
    parser.add_argument('--stable', type=int, default=900, help='over how many milliseconds, assume the object is stable')
    parser.add_argument('--area', type=float, default=0.2, help='over how much frame percent, assume the object is moving')
    parser.add_argument('--save', default='capture', help='path of save folder for results')

    args = parser.parse_args()

    vs = VideoSampler()
    # video_path = '/Users/jiahua/Downloads/moving_det_cv/testdata/test2.mp4'
    # src_folder_path = '/Users/jiahua/Downloads/moving_det_cv/testdata'

    if args.folder is not None:
        paths = []
        for path in os.listdir(args.folder):
            if '.mp4' in path:
                video_path = os.path.join(args.folder, path)
                paths.append(video_path)
    else:
        paths = [args.path]

    for video_path in paths:
            
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        vs.capture(src=video_path, fps=args.fps,assume_stable=args.stable, diff_area_base=args.area, video_name=video_name)

        save_path = os.path.join(args.save, video_name)

        os.makedirs(save_path, exist_ok=True)

        vs.save_to_folder(root_path=save_path)
