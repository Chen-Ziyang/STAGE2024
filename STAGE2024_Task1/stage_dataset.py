from torch.utils import data
import numpy as np
import pandas as pd
import os

stage = {'normal': 0, 'early': 1, 'intermediate': 2, 'advanced': 3}
gender = {'male': 0, 'female': 1}
eye = {'OD': 0, 'OS': 1}


class STAGE_sub1_dataset(data.Dataset):
    """
    getitem() output:
    	fundus_img: RGB uint8 image with shape (3, image_size, image_size)
        oct_img:    Uint8 image with shape (256, oct_img_size[0], oct_img_size[1])
    """

    def __init__(self,
                 oct_transforms,
                 fundus_transforms,
                 dataset_root,
                 label_file='',
                 info_file='',
                 filelists=None,
                 num_classes=3,
                 mode='train',
                 TTA=False, oct_TTA_transforms=None, fundus_TTA_transforms=None):

        self.dataset_root = dataset_root
        self.oct_transforms = oct_transforms
        self.fundus_transforms = fundus_transforms
        self.mode = mode.lower()
        self.num_classes = num_classes
        self.TTA = TTA
        self.oct_TTA_transforms = oct_TTA_transforms
        self.fundus_TTA_transforms = fundus_TTA_transforms

        if self.mode == 'train':
            label = {int(row['ID']): row[1]
                        for _, row in pd.read_csv(label_file).iterrows()}
            info = {int(row['ID']): [row[1], row[2], row[3], row[4]]
                        for _, row in pd.read_excel(info_file).iterrows()}
            self.file_list = [[f, label[int(f)], info[int(f)]] for f in os.listdir(dataset_root)]
        elif self.mode == "test":
            info = {int(row['ID']): [row[1], row[2], row[3], row[4]]
                        for _, row in pd.read_excel(info_file).iterrows()}
            self.file_list = [[f, None, info[int(f)]] for f in os.listdir(dataset_root)]
            self.file_list.sort(key=lambda x: int(x[0]))

        if filelists is not None:
            self.file_list = [item for item in self.file_list if item[0] in filelists]

    def __getitem__(self, idx):
        real_index, label, info = self.file_list[idx]
        if self.mode == 'train':
            oct_img = np.load(os.path.join(self.dataset_root, real_index, 'img.npy'))[..., np.newaxis]
            fundus_img = np.load(os.path.join(self.dataset_root, real_index, 'fundus_crop.npy'))
        elif self.mode == "test":
            oct_img = np.load(os.path.join(self.dataset_root, real_index, 'img.npy'))[..., np.newaxis]
            fundus_img = np.load(os.path.join(self.dataset_root, real_index, 'fundus_crop.npy'))

        if self.mode == 'test' and self.TTA:
            oct_imgs, fundus_imgs = [], []
            oct_imgs.append(self.oct_transforms(oct_img).squeeze(-1).copy())
            fundus_imgs.append(self.fundus_transforms(fundus_img).transpose(2, 0, 1).copy())
            for _ in range(31):
                oct_imgs.append(self.oct_TTA_transforms(oct_img).squeeze(-1).copy())
                fundus_imgs.append(self.fundus_TTA_transforms(fundus_img).transpose(2, 0, 1).copy())
            oct_imgs = np.stack(oct_imgs, 0)
            fundus_imgs = np.stack(fundus_imgs, 0)
        else:
            oct_imgs = self.oct_transforms(oct_img).squeeze(-1).copy()
            fundus_imgs = self.fundus_transforms(fundus_img).transpose(2, 0, 1).copy()

        if self.mode == 'test':
            return oct_imgs, fundus_imgs, real_index, np.array([gender[info[0]], info[1]/100., eye[info[2]], stage[info[3]]])
        if self.mode == "train":
            return oct_imgs, fundus_imgs, label, np.array([gender[info[0]], info[1]/100., eye[info[2]], stage[info[3]]])

    def __len__(self):
        return len(self.file_list)

