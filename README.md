## Introduction
This is the AI algorithm and programs used for dish detection and classification. There are three parts:
1. A two-stage, online learning, object detection model with pretrained weights. It can be launched independently as flask service, or imported in other projects for higher inference speed.
2. A data management system for processing videos, labelling, cleaning photos. It separates frontend and backend. For frontend, it is a webpage tool to help the user clean and label captured images. This program is based on JS/Vue3.
3. For backend, the program specifies the folder structures, rules of loading, modifying and saving data. The plate photo is named `mainImage` and dish photo is `subImage`. The program is file-based: the relationship between `mainImage` and `subImage` only depends on folder structures and matching of names.

## Install Requirements
For webpage tool's backend and detect API, the basic python version is 3.6. Each `requirements.txt` file in `detect_api` and `data_manage` records the python packages to install.

The webpage tool's frontend is developed on NPM 8.11.0. The versions of Vue3, vite, element-plus and other packages are shown in `package.json` in `image_clean` folder.

## Run this program
- Run the detect API as a separate service 
``` bash
nohup python detect_api/app.py > detectlogs${time}.txt 2>&1 &
```
There are two arguments can be adjusted:
```
  --host host of flask service, default: 127.0.0.1
  --port port of flask service, default: 36188
```

-  Run the webpage tool
``` bash
nohup python data_manage/app.py > managelogs${time}.txt 2>&1 &
cd image_clean
nphup npm run preview > weblogs${time}.txt 2>&1 &
```
For the backend, the service's host address and port can be modified in `data_manage/config.txt`. The corresponding docking address config of frontend is `image_clean/src/utils/request.js`. And the webpage is address can be changed in `image_clean/vite.config.js`

## APIs
- For the detect API, the APIs are shown below.

|api|method|request|response|note|
|-|-|-|-|-|
|/detect|POST|{"img": Image file}|{"explanations":{"box_info":List[{"class_name":str, "box_info":[x1,y1,x2,y2]}]}}|
|/detect/bbox|POST|{"img": Image file}|List[List[x1,y1,x2,y2]]|
|/detect/class|POST|{"img": Image file}|[str]
|/clearcache|GET|-|status|
|/train/class|POST|{"img": Image file, "label":str}|status|deprecated
|/train/notdet|POST|{"img": Image file, "label":str}|status|deprecated
|/train|POST|Dict[label: img_path]|List[List[x1,y1,x2,y2]]|deprecated
|/train/upload|POST|{"labels": List[str], "url":List[str], "img": List[Image file]}|List[List[x1,y1,x2,y2]]|
|/label|GET|-|List[str]|

- For the web tool, the APIs are shown below.

|api|method|request|response|note|
|-|-|-|-|-|
|/get_video_paths|GET|-|List[video_path]|
|/process_video|GET|{"path":video_path, "fps":15, "stable":900, "diff_area":0.2}|status|
|/save_result|GET|{"path":folder_path}|status|deprecated|
|/get_directories|GET|-|List[folder_path]|
|/get_mainimg|POST|{"path":folder_path}|{"mainimg":List[{"key":mainimg_id, "path":img_path}]}|
|/get_subimg|POST|{"path":folder_path, "labels": List[str]}|{"subimg":List[{"key": subimg_id, "path": img_path, "class": str}], "confirmed":List[{"key": subimg_id, "path": img_path, "class": str}]}|
|/mainimg/add_mainimg|POST|{"img": Image file}|status|
|/delete_images|POST|{"path":List[img_path]}|status|
|/find_similar|POST|{"path":List[img_path]}|List[{"key":mainimg_id, "path":img_path}]|
|/detect_box|POST|{"path":List[img_path]}|status|
|/get_cls_list|GET|-|List[name_str]|
|/add_cls_name|GET|{"cls_name":str}|status|
|/set_cls_name|POST|{"key": List[subimg_id], "cls_name":str}|status|
|/confirm_subimgs|POST|{"key": List[subimg_id]}|status|
|/auto_label_subimg|GET|-|status|
|/delet_subimgs|POST|{"key": List[subimg_id]}|status|
|/analysis/counting_class|GET|-|Dict[cls_name: sample_num]|



## File structure of web tool backend

When loading a image folder, the file-based backend scans through the directory and find all images, labels and classes. For `MainImage`(end with .jpg), if it is labelled, the label path is the same as image, with .txt suffix and yolo-format(each line is [id x y w h(percent)]). Also, the corresponding `SubImage` will be stored in sub_images folder. `SubImage`s of the same class are placed in the same sub folder. A json file with the same name in root folder records the `SubImage` order and confirmation status. The images with boxes are stored in preview folder. If the `MainImage` is not labelled, these files won't be generated.

The directory structure is:

```
--- root directory
    |-- id_name_map.json # convert class id to class name
    |-- sub_images
    |   |-- [cls_name] # class name
    |   |    |--[subimg_id].jpg # sub image (classified)
    |   |    |--[subimg_id].pkl # CNN feature of sub image (classified)
    |   |-- [subimg_id].jpg # sub image (unclassified)
    |   |-- [subimg_id].pkl # sub image feature (unclassified)

    |-- preview
    |   |--[image_preview].jpg # visualized detection result
    |-- [mainimg_id].jpg # main image. The necessary file, other files are all based on it.
    |-- [mainimg_id].txt # label of main image (yolo format)
    |-- [mainimg_id].json # the subimg_ids of mainimg and their comfirming status

```
The most basic files are main images, and then the label files.
Preview/info/sub-images are all depended on label files.

## Detect API models
The dish detector has three parts. YOLO is used to detect plates. It has been quantified and optimized for CPU inference by OpenVINO. The related files are:
```
utils
models
det_model.py
detect_api/weights/yolov5_best.xml
detect_api/weights/yolov5_best.bin
```
Then `SubImage`s are cropped from the detected bounding boxes, and passed to MobileNetV3, which is chosen as feature extractor. This model is also converted by OpenVINO. The related files are:
```
detect_api/feat_extractor.py
detect_api/weights/distil_mobv3_run1_best.xml
detect_api/weights/distil_mobv3_run1_best.bin
```

Last, the encoded features of `SubImage`s are passed to classifier, in this program, K-NearestCentroid from SciPy is chosen. It can handle few-shot learning well. The related files are:
```
detect_api/clf_model_simple.py
```