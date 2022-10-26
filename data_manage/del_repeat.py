import os
import shutil

import cv2

from main import gray

repeat_img_path = [
    '/Users/jiahua/Downloads/moving_det_cv/capture/IPS_2022-03-26.16.21.14.3479/1649007352.68297.jpg',
    '/Users/jiahua/Downloads/moving_det_cv/capture/IPS_2022-03-26.16.21.14.3479/1649007353.582127.jpg'
]
folder_to_del = 'capture/IPS_2022-03-26.16.21.14.3479'
save_dir = os.path.join(folder_to_del, 'delete')
os.makedirs(save_dir, exist_ok=True)

repeat_imgs = [cv2.imread(p) for p in repeat_img_path]
repeat_imgs = [gray(img) for img in repeat_imgs]

diff_thresh = 255*0.25
stay_ratio = 0.2

for path in os.listdir(folder_to_del):
    if '.jpg' in path:
        src = os.path.join(folder_to_del, path)
        img = gray(cv2.imread(src))

        for rep_im in repeat_imgs:
                
            diff = cv2.absdiff(img, rep_im)
            diff = cv2.threshold(diff, thresh=diff_thresh, maxval=255, type=cv2.THRESH_BINARY)[1]

            # count percent of diff pixels in image
            diff_percent = (diff == 255).sum() / (diff.shape[0] * diff.shape[1])
            if diff_percent < stay_ratio: # same image
                # # del this image
                # move this image to another place
                dst = os.path.join(save_dir, path)
                shutil.move(src, dst)
                break
