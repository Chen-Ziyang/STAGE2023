import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from stage_dataset import STAGE_sub1_dataset
from cnn_model import *

import transforms as trans

import warnings
warnings.filterwarnings('ignore')

torch.set_num_threads(1)


def test(loader, model):
    model.eval()
    cache = []
    idxs = []
    with torch.no_grad():
        for batch, (oct_img, idx, info) in enumerate(loader):
            print(idx)
            if len(oct_img.size()) == 5:    # TTA delete the first dimension
                oct_img = oct_img.squeeze(0)
            oct_img = (oct_img / 255.).to(dtype=torch.float32).to(device)
            info = info.repeat(oct_img.shape[0], 1).to(dtype=torch.float32).to(device)

            logits = model(oct_img, info)
            if logits.shape[0] == 1:
                cache.append(logits.detach().cpu().numpy()[0])        # w/o TTA
            else:
                cache.append(logits.detach().cpu().numpy().mean(0))   # w TTA
            idxs.append(idx[0])
    return np.stack(idxs, 0), np.stack(cache, 0)


save_root = "./results/MD_Results.csv"
test_root = "/czy_SSD/1T/zychen/STAGE2023/STAGE_validation/validation_images"
oct_img_size = [512, 512]
device = 'cuda:0'

oct_test_transforms = trans.Compose([
    trans.CenterCrop([256] + oct_img_size)
])

oct_tta_transforms = trans.Compose([
    trans.CenterRandomCrop([256] + oct_img_size),
    trans.RandomHorizontalFlip()
])

test_dataset = STAGE_sub1_dataset(dataset_root=test_root,
                                  transforms=oct_test_transforms,
                                  info_file='/czy_SSD/1T/zychen/STAGE2023/STAGE_validation/data_info_validation.xlsx',
                                  mode='test', TTA=True, TTA_transforms=oct_tta_transforms)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    drop_last=False,
    num_workers=2,
    pin_memory=True
)

best_model_path = ["./model_checkpoints/best_model_5.3361.pth"]

caches = []
for i in range(len(best_model_path)):
    print('Model {}: '.format(i+1), best_model_path[i])
    model = Model().to(device)
    para_state_dict = torch.load(best_model_path[i])
    model.load_state_dict(para_state_dict)

    ids, cache = test(test_loader, model)
    caches.append(cache)

caches = np.stack(caches, 0)
print(caches.shape)
caches = np.mean(caches, 0)     # calculate the average value along the first dimension
caches = np.concatenate((ids[:, np.newaxis], caches[:, np.newaxis]), axis=-1)
submission_result = pd.DataFrame(list(caches), columns=['ID', 'pred_MD'])
submission_result.to_csv(save_root, index=False)
print('Finish: ', save_root)
