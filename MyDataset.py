import numpy as np
import h5py
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, root, topk_labels=None, transform=None, normalize=True, depth_norm=10):
        """
            NYU Depth Dataset
            root - path to NYU dataset .mat file
            topk_labels - (list) top k classes for segmentation
            transform - transforms for segmentation labels and depth maps
            normalize - determines whether to apply ImageNet normalization to RGB images
            depth_norm - Normalization for depth map based on training data
            extra_augs - determines whether to apply extra agumentations to RGB images
        """
        # open .mat file as an h5 object
        self.h5_obj = h5py.File(root, mode='r')

        self.transform = transform
        # obtain desired groups
        self.images = self.h5_obj['images']  # rgb images
        self.depths = self.h5_obj['depths']  # depths
        self.labels = self.h5_obj['labels']  # sematic class mask for each image
        self.names = self.h5_obj['names']  # sematic class labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.rotate(self.images[idx].transpose(1, 2, 0), cv2.ROTATE_90_CLOCKWISE)  # rgb image
        depth = cv2.rotate(self.depths[idx], cv2.ROTATE_90_CLOCKWISE)  # depth map
        label = cv2.rotate(self.labels[idx], cv2.ROTATE_90_CLOCKWISE).astype(np.float32)  # semantic segmentation label

        return self.transform({'image': image, 'depth': depth, 'mask': label})

    def str_label(self, idx):
        """
            Obtains string label for a class index. Names/Labels are indexed from 1,
            this function is able to take this into account by subtracting 1.
            In the NYU depth dataset, labels equal 0 are considered unlabeled.
        """
        if idx - 1 < 0:
            return 'unlabeled'
        return ''.join(chr(i[0]) for i in self.h5_obj[self.names[0, idx - 1]])

    def close(self):
        self.h5_obj.close()

    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    root = 'nyu_depth_v2_labeled.mat'
    transform = transforms.Compose([transforms.ToTensor()])
    nyu_dataset = MyDataset(root, transform=transform)

    idx = 100  # 125
    sample = nyu_dataset[idx]
    image, depth, label = sample['image'], sample['depth'], sample['label']
    print(f"Shape of tensors: {image.shape}, {depth.shape}, {label.shape}")



