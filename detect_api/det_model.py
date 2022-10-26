import sys
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device
import torch
import cv2
import numpy as np

class yolo(object):
    def __init__(
            self,
            weights,  # model.pt path(s)
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            imgsz=640,  # inference size (pixels)
    ):
        # global MODEL, NAMES, DEVICE, SIZE, CLS_MAP
        # with open(class_map) as f:
        #     CLS_MAP = json.load(f)

        # Load model
        self.device = select_device(device)
        self.model = attempt_load(weights,
                             map_location=self.device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        self.size = check_img_size(imgsz, s=stride)  # check image size
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        self.model.eval()
        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.size, self.size).to(self.device).type_as(next(self.model.parameters())))  # run once

        print('detection model initialization finished')

    @torch.no_grad()
    def predict(
            self,
            source,  # file/dir/URL/glob, 0 for webcam
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
    ):

        # Dataloader
        stride = int(self.model.stride.max())  # model stride
        dataset = LoadImages(source, img_size=self.size, stride=stride)

        results = []
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.device).float()
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = self.model(img)[0]
            # Apply NMS
            pred = non_max_suppression(pred,
                                       conf_thres,
                                       iou_thres,
                                       classes,
                                       agnostic_nms,
                                       max_det=max_det)

            # Process detections
            res = []
            for i, det in enumerate(pred):  # detections per image
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                    # only keep position, the first 4 valuse
                    det = det[:,:4]
                    res.append(det.cpu().int().tolist())
                else:
                    res.append([])
            results.extend(res)
        # return self.postProcessJSON(results, self.names)
        # return self.postProcess_1_cls(results)
        return results

    def postProcessJSON(self, results, names=None):
        post_results = []
        for res in results:
            box_info = []
            if len(res) > 0:
                for row in res[0]:
                    box_info.append({
                        'class_name': 'dish',  #CLS_MAP[names[row[-1]]],
                        'coord': row[:4]
                    })
            post_results.append({'explanations': {'box_info': box_info}})
        return post_results

    def postProcess_1_cls(self, results):
        return [[bbox[:4] for bbox in img_res] for img_res in results]
            
                

class yolo_openvino(yolo):
    def __init__(
            self,
            bin_path,  # path to openvino bin file
            xml_path,  # path to openvino xml file
    ):

        from openvino.inference_engine import IENetwork, IECore
        ie = IECore()
        net = ie.read_network(model=xml_path, weights=bin_path)
        self.model = ie.load_network(network=net, device_name='CPU')

        input_info = net.input_info['images']
        self.n, self.c, self.h, self.w = input_info.input_data.shape

        print('detection model initialization finished')

    def letterbox(self, img, size=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
        # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
        shape = img.shape[:2]  # current shape [height, width]
        w, h = size

        # Scale ratio (new / old)
        r = min(h / shape[0], w / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = w - new_unpad[0], h - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (w, h)
            ratio = w / shape[1], h / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

        top2, bottom2, left2, right2 = 0, 0, 0, 0
        if img.shape[0] != h:
            top2 = (h - img.shape[0])//2
            bottom2 = top2
            img = cv2.copyMakeBorder(img, top2, bottom2, left2, right2, cv2.BORDER_CONSTANT, value=color)  # add border
        elif img.shape[1] != w:
            left2 = (w - img.shape[1])//2
            right2 = left2
            img = cv2.copyMakeBorder(img, top2, bottom2, left2, right2, cv2.BORDER_CONSTANT, value=color)  # add border
        return img




    def predict(
            self,
            source,  # file/dir/URL/glob, 0 for webcam
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
    ):

        results = []
        if isinstance(source, list):
            files = source
        else:
            files = [source]

        for path in files:
            im0s = cv2.imread(path)
            im0s = cv2.cvtColor(im0s, cv2.COLOR_BGR2RGB)

            img = self.letterbox(im0s, (self.w, self.h))
            img = img.transpose((2, 0, 1))
            img = img.reshape((self.n, self.c, self.h, self.w))
            img = img.astype(np.float32)/255

            pred = self.model.infer(inputs={'images': img})
            pred = torch.FloatTensor(pred['output'])
            pred = non_max_suppression(pred,
                                        conf_thres,
                                        iou_thres,
                                        classes,
                                        agnostic_nms,
                                        max_det)

            # Process detections
            res = []
            for i, det in enumerate(pred):  # detections per image
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                    # det[:, :4] = scale_coords((736,736), det[:, :4], (800,800,3)).round()
                    # only keep position, the first 4 valuse
                    det = det[:,:4]
                    res.append(det.cpu().int().tolist())
                else:
                    res.append([])
            results.extend(res)

        # return self.postProcessJSON(results, self.names)
        # return self.postProcess_1_cls(results)
        return results