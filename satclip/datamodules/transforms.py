import torchvision.transforms as T
import torch
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform  
from albumentations.pytorch import ToTensorV2
import numpy as np


def get_train_transform(resize_crop_size = 256,
                  mean = [0.4139, 0.4341, 0.3482, 0.5263],
                  std = [0.0010, 0.0010, 0.0013, 0.0013]
                  ):

    augmentation = A.Compose(
        [
            A.RandomResizedCrop(height=resize_crop_size, width=resize_crop_size),
            A.RandomBrightnessContrast(),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.GaussianBlur(),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    def transform(sample):
        image = sample["image"].numpy().transpose(1,2,0)
        point = sample["point"]

        image = augmentation(image=image)["image"]
        point = coordinate_jitter(point)

        return dict(image=image, point=point)

    return transform

def get_s2_train_transform(resize_crop_size = 256):
    augmentation = T.Compose([
        T.RandomCrop(resize_crop_size),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.GaussianBlur(3),
    ])

    def transform(sample):
        image = sample["image"].astype(np.float32) / 10000.0
        point = sample["point"]
        image = torch.tensor(image)
        image = augmentation(image)
        point = coordinate_jitter(point)
        return dict(image=image, point=point)

    return transform

def get_pretrained_s2_train_transform(resize_crop_size = 256):
    augmentation = T.Compose([
        T.RandomCrop(resize_crop_size),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.GaussianBlur(3),
    ])

    def transform(sample):
        # S2Geo now hands back raw uint16; cast here to preserve prior behaviour.
        image = sample["image"].astype(np.float32) / 10000.0
        point = sample["point"]

        B10 = np.zeros((1, *image.shape[1:]), dtype=image.dtype)
        image = np.concatenate([image[:10], B10, image[10:]], axis=0)
        image = torch.tensor(image)

        image = augmentation(image)

        point = coordinate_jitter(point)

        return dict(image=image, point=point)

    return transform

def get_uint16_train_transform():
    """Lean transform for the GPU-augmentation path.

    Returns the raw uint16, 12-band image untouched (only ~1.5 MB/sample crosses
    the worker->main boundary, vs ~3.4 MB for float32 13-band). The float cast,
    /10000 scaling, B10 zero-band insertion, random flips and GaussianBlur are all
    done on the GPU in SatCLIPLightningModule.on_after_batch_transfer, which keeps
    the augmentation math identical (verified) while ~halving PCIe traffic.

    Assumes patches are already at the model resolution (256), so RandomCrop(256)
    is a no-op and is dropped.
    """
    def transform(sample):
        image = np.ascontiguousarray(sample["image"])  # uint16 [12, H, W]
        point = coordinate_jitter(sample["point"])
        return dict(image=torch.from_numpy(image), point=point)

    return transform

def coordinate_jitter(
        point,
        radius=0.01 # approximately 1 km
    ):
    return point + torch.rand(point.shape) * radius