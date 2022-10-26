from typing import List, Union
from PIL import Image
import numpy as np
from PIL.Image import Image as PILImage
import torch
from torch import nn
from torchvision.models.resnet import ResNet, Bottleneck

from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

class ImageDataset(Dataset):
    def __init__(self, paths, transform) -> None:
        super(ImageDataset, self).__init__()
        self.paths = paths
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert("RGB")
        return self.transform(img)

    def __len__(self):
        return len(self.paths)

class resnext(object):
    def __init__(self, weights, imgsz, device, num_classes=None, base_model=None, half=False) -> None:
        # TODO: modify to a generalized class
        self.device = device if isinstance(device, torch.device) else torch.device('cpu')
        self.half = half
        self.imgsz = imgsz
        self.preTransform = transforms.Compose([
            transforms.Resize((imgsz,imgsz), interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        ckpt = torch.load(weights, map_location='cpu')
        ckpt = OrderedDict({k.replace('module.', '').replace('backbone.',''): v for k, v in ckpt.items()})

        assert bool(num_classes) ^ bool(base_model), 'only choose one arg from `num_classes` and `base_model`'
        if num_classes:
            model = ResNet(block=Bottleneck, layers = [3,4,23,3], groups=32, width_per_group=16)
            model.fc = nn.Linear(2048, num_classes)

            model.load_state_dict(ckpt)
            # Remove linear and pool layers (since we're not doing classification)
            modules = list(model.children())[:-2]
            modules.append(nn.AdaptiveAvgPool2d((1,1)))
            self.model = nn.Sequential(*modules).to(self.device)
        else:
            base_model.load_state_dict(ckpt)
            # Remove linear and pool layers (since we're not doing classification)
            modules = list(base_model.children())[:-1]
            self.model = nn.Sequential(*modules).to(self.device)

        if self.half:
            self.model.half()
        self.model.eval()
        self.test_inference()
        print('detection model initialization finished')

    @torch.no_grad()
    def test_inference(self):
        # Run inference
        imgs = torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device)
        outs = self.model(imgs.type_as(next(self.model.parameters())))
        self.feat_length = outs.squeeze(-1).squeeze(-1).shape[-1]


    def encode(self,
               imgs: List[Union[PILImage, str]],
               batch_size=16,
               num_worker=0,
               progress=False) -> List[np.ndarray]:

        # make dataloader to ensure memory stable
        loader_config = {
            'batch_size': batch_size,
            'num_workers': num_worker,
            # 'prefetch_factor': 2
        }
        if len(imgs)==0:
            return []
        elif isinstance(imgs[0], PILImage):
            imgs = [self.preTransform(img) for img in imgs]
            loader = DataLoader(imgs, **loader_config)
        elif isinstance(imgs[0], str):
            dataset = ImageDataset(imgs, transform=self.preTransform)
            loader = DataLoader(dataset, **loader_config)
        else:
            raise f"not support input of this type: {type(imgs[0])}"

        # imgs = torch.stack(imgs).to(self.device)
        # batch inference
        outputs = []
        with torch.no_grad():
            if progress:
                loader = tqdm(loader)
            for inputs in loader:
                if self.half:
                    inputs = inputs.type(torch.HalfTensor)
                inputs = inputs.to(self.device)
                outs = self.model(inputs).squeeze(-1).squeeze(-1)
                outputs.extend(outs.cpu().numpy())
        return outputs

    def det_encode(self, images:List[PILImage], bboxes:List[List[int]]) -> List[List[np.ndarray]]:
        # image: list of PIL.Image, image -> detect -> feat_extract -> feat
        # booxes: location of all objects in each image, n-bboxes/image

        # crop out all bboxes, flatten to 1-d list
        indexes = []
        imgs = []
        for i, (img, img_bboxes) in enumerate(zip(images, bboxes)):
            for bbox in img_bboxes:
                imgs.append(img.crop(bbox))
                indexes.append(i)

        # batch inference
        outs = self.encode(imgs) if len(imgs) > 0 else []

        # revert to nested list
        outputs = [[] for _ in range(len(images))]
        for idx, out in zip(indexes, outs):
            outputs[idx].append(out)

        return outputs



class resnext_pretrain(resnext):
    def __init__(self, weights, base_model, imgsz, device=None, half=False) -> None:

        super().__init__(weights=weights,
                         imgsz=imgsz,
                         device=device,
                         base_model=base_model,
                         half=half)


# TODO: add to master branch
class mobnet_openvino(resnext):
    def __init__(
            self,
            bin_path,  # path to openvino bin file
            xml_path,  # path to openvino xml file
    ):
        from openvino.inference_engine import IENetwork, IECore

        ie = IECore()
        net = ie.read_network(model=xml_path, weights=bin_path)
        self.model = ie.load_network(network=net, device_name='CPU')
        input_info = self.model.input_info['images']
        self.n, self.c, self.h, self.w = input_info.input_data.shape

        self.preTransform = transforms.Compose([
            transforms.Resize((self.w, self.h), interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.feat_length = self.model.outputs['output'].shape[-1]
        print('detection model initialization finished')

    def encode(self, imgs: List[Union[PILImage, str]]) -> List[np.ndarray]:

        # type check
        if len(imgs) == 0:
            return []
        elif isinstance(imgs[0], PILImage):
            imgs = [self.preTransform(img) for img in imgs]
        elif isinstance(imgs[0], str):
            imgs = [Image.open(p).convert('RGB') for p in imgs]
            imgs = [self.preTransform(im) for im in imgs]
        else:
            raise f"not support input of this type: {type(imgs[0])}"

        outputs = []
        for im in imgs:
            res = self.model.infer(inputs={'images': im})
            outputs.append(res['output'].squeeze())
        
        return outputs
    
    def test_inference(self):
        return