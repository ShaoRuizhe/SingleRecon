import os
import pathlib
import warnings

import skimage.io
from multiprocess import Pool
from functools import partial
import _pickle

import numpy as np
from pycocotools.coco import COCO
import shapely.geometry

from tqdm import tqdm

import torch

from lydorn_utils import print_utils
from lydorn_utils import python_utils

from torch_lydorn.torch.utils.data import Dataset as LydornDataset, makedirs, files_exist, __repr__

from torch_lydorn.torchvision.datasets import utils
from torch_lydorn.torchvision.datasets import MappingChallenge

class BONAIRotated(MappingChallenge):
    def __init__(self, root, transform=None, pre_transform=None, fold="train", small=False, pool_size=1):
        assert fold in ["train", "val", "test_images"], "Input fold={} should be in [\"train\", \"val\", \"test_images\"]".format(fold)
        if fold == "test_images":
            print_utils.print_error("ERROR: fold {} not yet implemented!".format(fold))
            exit()
        self.root = root
        self.fold = fold
        makedirs(self.processed_dir)
        self.small = small
        if self.small:
            print_utils.print_info("INFO: Using small version of the Mapping challenge dataset.")
        self.pool_size = pool_size

        self.coco = None
        self.image_file_list = self.load_image_file_names()
        self.stats_filepath = os.path.join(self.processed_dir, "stats.pt")
        self.stats = None
        if os.path.exists(self.stats_filepath):
            self.stats = torch.load(self.stats_filepath)

        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.__indices__ = None

    def load_image_file_names(self):
        image_file_list=[file for file in os.listdir(self.processed_dir) if file.endswith('.pt')]
        image_file_list.sort()
        # if 'image_id_list.json' in image_file_list:
        #     image_file_list.remove('image_id_list.json')
        # if 'image_file_list' in image_file_list:
        #     image_file_list.remove('image_file_list.json')
        image_name_list_filepath = os.path.join(self.processed_dir, "image_file_list.json" )
        python_utils.save_json(image_name_list_filepath, image_file_list)
        return image_file_list

    def get_coco(self):
        if self.coco is None:
            annotation_filename = "annotation-small.json" if self.small else "annotation.json"
            annotations_filepath = os.path.join(self.root, self.fold, annotation_filename)
            self.coco = COCO(annotations_filepath)
        return self.coco

    def __len__(self):
        return len(self.image_file_list)

    def get(self, idx):
        img_file_name = self.image_file_list[idx]
        data = torch.load(os.path.join(self.processed_dir, img_file_name))
        if 'gt_polygon_image' in data:
            data['gt_polygons_image']=data.pop('gt_polygon_image')
        data["image_mean"] = np.array([0.5,0.5,0.5])
        data["image_std"] = np.array([0.15,0.15,0.15])
        data["class_freq"] = np.array([0.225,0.055,0.01])
        return data


def main():
    # Test using transforms from the frame_field_learning project:
    from frame_field_learning import data_transforms

    config = {
        "data_dir_candidates": [
                "/data/titane/user/nigirard/data",
                "~/data",
                "/data"
        ],
        "dataset_params": {
            "small": True,
            "root_dirname": "mapping_challenge_dataset",
            "seed": 0,
            "train_fraction": 0.75
        },
        "num_workers": 8,
        "data_aug_params": {
            "enable": False,
            "vflip": True,
            "affine": True,
            "color_jitter": True,
            "device": "cuda"
        }
    }

    # Find data_dir
    data_dir = python_utils.choose_first_existing_path(config["data_dir_candidates"])
    if data_dir is None:
        print_utils.print_error("ERROR: Data directory not found!")
        exit()
    else:
        print_utils.print_info("Using data from {}".format(data_dir))
    root_dir = os.path.join(data_dir, config["dataset_params"]["root_dirname"])

    # --- Transforms: --- #
    # --- pre-processing transform (done once then saved on disk):
    # --- Online transform done on the host (CPU):
    train_online_cpu_transform = data_transforms.get_online_cpu_transform(config,
                                                                          augmentations=config["data_aug_params"][
                                                                              "enable"])
    test_online_cpu_transform = data_transforms.get_eval_online_cpu_transform()

    train_online_cuda_transform = data_transforms.get_online_cuda_transform(config,
                                                                            augmentations=config["data_aug_params"][
                                                                                "enable"])
    # --- --- #

    dataset = MappingChallenge(root_dir,
                               transform=test_online_cpu_transform,
                               pre_transform=data_transforms.get_offline_transform_patch(),
                               fold="train",
                               small=config["dataset_params"]["small"],
                               pool_size=config["num_workers"])

    print("# --- Sample 0 --- #")
    sample = dataset[0]
    print(sample.keys())

    for key, item in sample.items():
        print("{}: {}".format(key, type(item)))

    print(sample["image"].shape)
    print(len(sample["gt_polygons_image"]))
    print("# --- Samples --- #")
    # for data in tqdm(dataset):
    #     pass

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=config["num_workers"])
    print("# --- Batches --- #")
    for batch in tqdm(data_loader):
        print("Images:")
        print(batch["image_relative_filepath"])
        print(batch["image"].shape)
        print(batch["gt_polygons_image"].shape)

        print("Apply online tranform:")
        batch = utils.batch_to_cuda(batch)
        batch = train_online_cuda_transform(batch)
        batch = utils.batch_to_cpu(batch)

        print(batch["image"].shape)
        print(batch["gt_polygons_image"].shape)

        # Save output to visualize
        seg = np.array(batch["gt_polygons_image"][0])
        seg = np.moveaxis(seg, 0, -1)
        seg_display = utils.get_seg_display(seg)
        seg_display = (seg_display * 255).astype(np.uint8)
        skimage.io.imsave("gt_seg.png", seg_display)
        skimage.io.imsave("gt_seg_edge.png", seg[:, :, 1])

        im = np.array(batch["image"][0])
        im = np.moveaxis(im, 0, -1)
        skimage.io.imsave('im.png', im)

        gt_crossfield_angle = np.array(batch["gt_crossfield_angle"][0])
        gt_crossfield_angle = np.moveaxis(gt_crossfield_angle, 0, -1)
        skimage.io.imsave('gt_crossfield_angle.png', gt_crossfield_angle)

        distances = np.array(batch["distances"][0])
        distances = np.moveaxis(distances, 0, -1)
        skimage.io.imsave('distances.png', distances)

        sizes = np.array(batch["sizes"][0])
        sizes = np.moveaxis(sizes, 0, -1)
        skimage.io.imsave('sizes.png', sizes)

        # valid_mask = np.array(batch["valid_mask"][0])
        # valid_mask = np.moveaxis(valid_mask, 0, -1)
        # skimage.io.imsave('valid_mask.png', valid_mask)

        input("Press enter to continue...")


if __name__ == '__main__':
    main()
