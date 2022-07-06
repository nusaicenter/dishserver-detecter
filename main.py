# 提取视频中的静态画面，并保存下来
from math import ceil
import cv2
from matplotlib import pyplot as plt
import time
import os

# MOVE -> MOVE: do nothing
# MOVE -> STAY: counter + 1
# STAY -> MOVE: reset counter
# STAY -> STAY: counter + 1, if trigger limit then save frame. Not repeat to save
STAY, MOVE = 0, 1


def gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


# threshold and the callback function of slider to adjust it
def changeThres(val, count_th):
    global diff_th
    diff_th = val
    print(val)


def check_status(prev, cur, stay_ratio, diff_thresh=255 * 0.4):
    # First check the diff value: [0~255] diff_thresh
    # then check the diff percent over all frame: [0~1] stay_ratio

    # Args
    # prev: np.ndarray, the previous frame
    # cur: np.ndarray, the current frame
    # stay_ratio: float, 0~1, the ratio of moving pixel number over total
    #   judge STAY if diff pixel ratio > stay_ratio
    # diff_thresh: float, 0~255, the absolute threshold to split diff frames
    diff = cv2.absdiff(cur, prev)
    diff = cv2.threshold(diff, thresh=diff_thresh, maxval=255, type=cv2.THRESH_BINARY)[1]

    # count percent of diff pixels in image
    diff_percent = (diff == 255).sum() / (diff.shape[0] * diff.shape[1])
    status = MOVE if diff_percent > stay_ratio else STAY

    return status, diff, diff_percent


def gen_filename(dir):
    return f'{os.path.join(dir, str(time.time()))}.jpg'


# initialization
# src = 'data/test1.mp4' #'0'

def capture(src, ):
    # initialization
    cap = cv2.VideoCapture(src) # open camera
    ret, frame = cap.read() # try to load one frame
    prev_frame = gray(frame)
    background = gray(frame)
    prev_status = MOVE
    stay_counter = 0
    save_flag = False
    save_dir = os.path.join('capture', os.path.splitext(os.path.basename(src))[0])
    os.makedirs(save_dir, exist_ok=True)

    # parameters
    # unit of delay is 'ms'
    delay = 200
    fps = 1000/delay
    # 'skip' for reading video
    skip_frames = ceil(cap.get(cv2.CAP_PROP_FPS)/fps)
    skip_counter = 0

    # if the frame stay more than some time(ms), it is assumed as stable
    assume_stable = 900
    frame_to_stay = ceil(assume_stable / delay)
    # frame_to_stay = ceil(stay_time / delay)

    # the ratio of moving pixel number over total, diff more than the ratio is stay
    stay_ratio_base = 0.2 # move percent per second
    stay_ratio = stay_ratio_base / fps

    # save frames to memory temporary and save to dish together
    save_buffer = dict()

    # start capture stabel frames
    t0 = time.time()
    while True:
        ret, ori_frame = cap.read()
        if not ret:
            break

        skip_counter += 1
        if skip_counter % skip_frames != 0:
            continue

        frame = gray(ori_frame)
        status, diff, percent = check_status(prev=prev_frame, cur=frame, stay_ratio=stay_ratio)
        # cv2.putText(diff, f'{percent:.5f}', (50,150), cv2.FONT_HERSHEY_PLAIN, 5, 255, 4)

        # # draw a square box to show detect area
        # h, w = diff.shape
        # side = int((stay_ratio * h * w)**0.5)
        # x1, y1 = int(w / 2 - side / 2), int(h / 2 - side / 2)
        # x2, y2 = int(w / 2 + side / 2), int(h / 2 + side / 2)
        # cv2.rectangle(diff, (x1, y1), (x2, y2), 127, 4)

        # cv2.imshow('diff', diff)

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
                notBackground, _, _ = check_status(prev=background, cur=frame, stay_ratio=stay_ratio)
                if notBackground:
                    print('static: save frame')
                    save_buffer[gen_filename(save_dir)] = ori_frame
                    # print('backgroud, not save')

                save_flag = True

        prev_status = status
        prev_frame = frame.copy()

        print(
            f"{'MOVE' if status else 'STAY'}: {stay_counter}, diff_percent: {percent*100:.2f}%",
            end='\r')

        # cv2.imshow('cam', frame)
        # cv2.createTrackbar('slider', 'diff', 0, 255, changeThres)
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord('q'):
        #     break
        # if key == ord('b'):
        #     background = frame.copy()
        #     print('reset background')

    print(f'process finished, time: {(time.time()-t0):.4f}s')

    for name, frame in save_buffer.items():
        cv2.imwrite(name, frame)


    cap.release()
# cv2.destroyAllWindows()
if __name__ == '__main__':
    folder_path = '/Volumes/data/菜品数据2022'
    for path in os.listdir(folder_path):
        if '.mp4' in path:
            capture(src=os.path.join(folder_path, path))