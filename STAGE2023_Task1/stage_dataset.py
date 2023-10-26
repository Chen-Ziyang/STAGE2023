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
                 transforms,
                 dataset_root,
                 label_file='',
                 info_file='',
                 filelists=None,
                 num_classes=3,
                 mode='train',
                 TTA=False, TTA_transforms=None):

        self.dataset_root = dataset_root
        self.oct_transforms = transforms
        self.mode = mode.lower()
        self.num_classes = num_classes
        self.TTA = TTA
        self.TTA_transforms = TTA_transforms

        if self.mode == 'train':
            label = {row['ID']: row[1]
                        for _, row in pd.read_excel(label_file).iterrows()}
            info = {row['ID']: [row[1], row[2], row[3], row[4]]
                        for _, row in pd.read_excel(info_file).iterrows()}
            self.file_list = [[f, label[int(f)], info[int(f)]] for f in os.listdir(dataset_root)]
        elif self.mode == "test":
            info = {row['ID']: [row[1], row[2], row[3], row[4]]
                        for _, row in pd.read_excel(info_file).iterrows()}
            self.file_list = [[f, None, info[int(f)]] for f in os.listdir(dataset_root)]
            self.file_list.sort(key=lambda x: int(x[0]))

        if filelists is not None:
            self.file_list = [item for item in self.file_list if item[0] in filelists]

    def __getitem__(self, idx):
        real_index, label, info = self.file_list[idx]
        if self.mode == 'train':
            oct_img = np.load(os.path.join(self.dataset_root, real_index, 'img.npy'))[..., np.newaxis]
        elif self.mode == "test":
            oct_img = np.load(os.path.join(self.dataset_root, real_index, 'img.npy'))[..., np.newaxis]

        if self.TTA_transforms is not None and self.mode == 'test' and self.TTA:
            oct_imgs = []
            if self.oct_transforms is not None:
                oct_imgs.append(self.oct_transforms(oct_img).squeeze(-1).copy())
            for _ in range(31):
                oct_imgs.append(self.TTA_transforms(oct_img).squeeze(-1).copy())
            oct_imgs = np.stack(oct_imgs, 0)
        else:
            if self.oct_transforms is not None:
                oct_imgs = self.oct_transforms(oct_img)
            oct_imgs = oct_imgs.squeeze(-1) # D, H, W, 1 -> D, H, W
            oct_imgs = oct_imgs.copy()

        if self.mode == 'test':
            return oct_imgs, real_index, np.array([gender[info[0]], info[1]/100., eye[info[2]], stage[info[3]]])
        if self.mode == "train":
            return oct_imgs, label, np.array([gender[info[0]], info[1]/100., eye[info[2]], stage[info[3]]])

    def __len__(self):
        return len(self.file_list)

