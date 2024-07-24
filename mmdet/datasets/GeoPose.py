import json

import cv2
import numpy as np
import torch
from mmcv.parallel import DataContainer
from osgeo import gdal
from torch.utils.data import Dataset
from pathlib import Path

from .geopose_utils import segmentation_models_pytorch as smp
from .geopose_utils.augmentation_vflow import augment_vflow
from .builder import DATASETS

RNG = np.random.RandomState(4321)

@DATASETS.register_module()
class GeoPose(Dataset):
    def __init__(
        self,
        sub_dir,
        args,
        rng=RNG,
        crop_size=1024,
        collect=['gt_angle']
    ):

        self.is_test = sub_dir == args.test_sub_dir
        self.rng = rng
        self.crop_size=crop_size
        self.collect=collect


        # create all paths with respect to RGB path ordering to maintain alignment of samples
        dataset_dir = Path(args.dataset_dir) / sub_dir
        rgb_paths = list(dataset_dir.glob(f"*_RGB.{args.rgb_suffix}"))####
        if rgb_paths == []: rgb_paths = list(dataset_dir.glob(f"*_RGB*.{args.rgb_suffix}")) # original file names
        agl_paths = list(
            pth.with_name(pth.name.replace("_RGB", "_AGL")).with_suffix(".tif")
            for pth in rgb_paths
        )
        vflow_paths = list(
            pth.with_name(pth.name.replace("_RGB", "_VFLOW")).with_suffix(".json")
            for pth in rgb_paths
        )

        if self.is_test:
            self.paths_list = rgb_paths
        else:
            self.paths_list = [
                (rgb_paths[i], vflow_paths[i], agl_paths[i])
                for i in range(len(rgb_paths))
            ]

            self.paths_list = [
                self.paths_list[ind]
                for ind in self.rng.permutation(len(self.paths_list))
            ]
            if args.sample_size is not None:
                self.paths_list = self.paths_list[: args.sample_size]
        self.flag = np.zeros(len(self), dtype=np.uint8)# 也不知道做啥的，ConcatDataset要有
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(
            args.backbone, "imagenet"
        )

        self.args = args
        self.sub_dir = sub_dir

    def __getitem__(self, i):

        if self.is_test:
            rgb_path = self.paths_list[i]
            image = load_image(rgb_path, self.args)
        else:
            rgb_path, vflow_path, agl_path = self.paths_list[i]
            image = load_image(rgb_path, self.args)
            agl = load_image(agl_path, self.args)
            mag, xdir, ydir, vflow_data = load_vflow(vflow_path, agl, self.args)

            if self.args.random_crop:
                crop_size = self.crop_size
                x0 = np.random.randint(2048 - crop_size)
                y0 = np.random.randint(2048 - crop_size)
                # print(image.shape, agl.shape, mag.shape, flush=True)
                image = image[x0 : x0 + crop_size, y0 : y0 + crop_size]
                agl = agl[x0 : x0 + crop_size, y0 : y0 + crop_size]
                mag = mag[x0 : x0 + crop_size, y0 : y0 + crop_size]


            scale = vflow_data["scale"]
            if self.args.augmentation:
                image, mag, xdir, ydir, agl, scale = augment_vflow(
                    image,
                    mag,
                    xdir,
                    ydir,
                    vflow_data["angle"],
                    vflow_data["scale"],
                    agl=agl,
                )
            xdir = np.float32(xdir)
            ydir = np.float32(ydir)
            mag = mag.astype("float32")
            agl = agl.astype("float32")
            scale = np.float32(scale)

            xydir = np.array([xdir, ydir])

        if self.is_test and self.args.downsample > 1:
            image = cv2.resize(
                image,
                (
                    int(image.shape[0] / self.args.downsample),
                    int(image.shape[1] / self.args.downsample),
                ),
                interpolation=cv2.INTER_NEAREST,
            )

        image = self.preprocessing_fn(image).astype("float32")
        image = np.transpose(image, (2, 0, 1))

        angle = np.angle(-xdir - ydir * 1j).astype(np.float32)# geopose的方向是屋顶朝向footprint的，要与bonai相一致，就要反向
        results = {'img': image}
        if 'gt_angle' in self.collect:
            results['gt_angle']=angle
        # if self.is_test:
        #     return image, str(rgb_path)
        # else:
        #     return image, xydir, agl, mag, scale

        # todo:用pipline来处理图像，包括datacontainer等
        #       用pipline处理增广，会不会有区别？
        #        image_meta在后续的训练是否完全没用了?能否让angle模型去除这个？
        image_meta= {}
        data = {}
        data['img_metas'] = DataContainer(image_meta, cpu_only=True)
        data['img'] = DataContainer(torch.tensor(results['img']), stack=True)
        for key in self.collect:
            data[key] = DataContainer(torch.tensor(results[key]))
            # data[key] = results[key]

        return data

    def __len__(self):
        return len(self.paths_list)


UNITS_PER_METER_CONVERSION_FACTORS = {"cm": 100.0, "m": 1.0}

def load_image(
    image_path,
    args,
    dtype_out="float32",
    units_per_meter_conversion_factors=UNITS_PER_METER_CONVERSION_FACTORS,
):
    image_path = Path(image_path)
    if not image_path.exists():
        return None
    image = gdal.Open(str(image_path))
    image = image.ReadAsArray()

    # convert AGL units and fill nan placeholder with nan
    if "AGL" in image_path.name:
        image = image.astype(dtype_out)
        np.putmask(image, image == args.nan_placeholder, np.nan)
        # e.g., (cm) / (cm / m) = m
        units_per_meter = units_per_meter_conversion_factors[args.unit]
        image = (image / units_per_meter).astype(dtype_out)

    # transpose if RGB
    if len(image.shape) == 3:
        image = np.transpose(image, [1, 2, 0])

    return image

def load_vflow(
        vflow_path,
        agl,
        args,
        dtype_out="float32",
        units_per_meter_conversion_factors=UNITS_PER_METER_CONVERSION_FACTORS,
        return_vflow_pred_mat=False,
):
    vflow_path = Path(vflow_path)
    vflow_data = json.load(vflow_path.open("r"))

    if np.isnan(vflow_data["scale"]): return None
    if np.isnan(vflow_data["angle"]): return None

    # e.g., (pixels / cm) * (cm / m) = (pixels / m)
    units_per_meter = units_per_meter_conversion_factors[args.unit]
    vflow_data["scale"] = vflow_data["scale"] * units_per_meter

    xdir, ydir = np.sin(vflow_data["angle"]), np.cos(vflow_data["angle"])
    mag = agl * vflow_data["scale"]

    vflow_items = [mag.astype(dtype_out), xdir.astype(dtype_out), ydir.astype(dtype_out), vflow_data]

    if return_vflow_pred_mat:
        vflow = np.zeros((agl.shape[0], agl.shape[1], 2))
        vflow[:, :, 0] = mag * xdir
        vflow[:, :, 1] = mag * ydir
        vflow_items.insert(0, vflow)

    return vflow_items