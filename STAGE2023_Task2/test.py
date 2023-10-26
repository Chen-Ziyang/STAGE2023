import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from stage_dataset import STAGE_sub2_dataset
from cnn_model import *

import transforms as trans

import warnings
warnings.filterwarnings('ignore')

torch.set_num_threads(1)

# 230809-2nd-best_model_4.6266+center crop+128 info_embedding nums+256 latent nums
# 230811-1st-best_model_4.6266+center crop+128 info_embedding nums+256 latent nums + baseline 0 pred
# 230816-1st-0-->-15-weight=5.0:
# "./model_checkpoints/20230815_012530_600991/best_model_4.0035.pth",
# "./model_checkpoints/20230815_040205_414030/best_model_3.4613.pth",
# "./model_checkpoints/20230815_063235_108394/best_model_3.8295.pth",
# "./model_checkpoints/20230815_090301_656103/best_model_4.2512.pth",
# "./model_checkpoints/20230815_113515_194440/best_model_4.0099.pth",

# 230819-1st-0-->-20-weight=5/10.0:
# "./model_checkpoints/20230819_012012_112835/best_model_3.7182.pth",
# "./model_checkpoints/20230819_045434_307097/best_model_3.4856.pth",
# "./model_checkpoints/20230819_113006_562095/best_model_3.6287.pth",
# "./model_checkpoints/20230818_203512_244334/best_model_4.1823.pth",
# "./model_checkpoints/20230819_155348_818569/best_model_3.8266.pth",

# 230820-1st-0-->-40-weight=5.0:
# "./model_checkpoints/20230820_015801_624258/best_model_4.0173.pth",
# "./model_checkpoints/20230820_041348_977368/best_model_4.8764.pth",
# "./model_checkpoints/20230820_065822_452511/best_model_4.1459.pth",
# "./model_checkpoints/20230820_093613_812095/best_model_4.0484.pth",
# "./model_checkpoints/20230820_120634_461864/best_model_4.1515.pth",


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

            logits = model(oct_img, info) * 40.

            if logits.shape[0] == 1:
                cache.append(list(logits.detach().cpu().numpy()[0]))        # w/o TTA
            else:
                cache.append(list(logits.detach().cpu().numpy().mean(0)))   # w TTA
            idxs.append(idx[0])
    return np.stack(idxs, 0), np.stack(cache, 0)


save_root = "./results/Sensitivity_map_Results.csv"
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

test_dataset = STAGE_sub2_dataset(dataset_root=test_root,
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

best_model_path = [
    "./model_checkpoints/20230819_012012_112835/best_model_3.7182.pth",
    "./model_checkpoints/20230819_045434_307097/best_model_3.4856.pth",
    "./model_checkpoints/20230819_113006_562095/best_model_3.6287.pth",
    "./model_checkpoints/20230818_203512_244334/best_model_4.1823.pth",
    "./model_checkpoints/20230819_155348_818569/best_model_3.8266.pth",

    "./model_checkpoints/20230820_015801_624258/best_model_4.0173.pth",
    "./model_checkpoints/20230820_041348_977368/best_model_4.8764.pth",
    "./model_checkpoints/20230820_065822_452511/best_model_4.1459.pth",
    "./model_checkpoints/20230820_093613_812095/best_model_4.0484.pth",
    "./model_checkpoints/20230820_120634_461864/best_model_4.1515.pth",
]

columns = ['ID'] + ['point'+str(i) for i in range(1, 52+1)]
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
caches = np.mean(caches, 0)  # calculate the average value along the first dimension
caches[caches < 1] = 0
caches = np.concatenate((ids[:, np.newaxis], caches), axis=-1)
submission_result = pd.DataFrame(list(caches), columns=columns)
submission_result.to_csv(save_root, index=False)
print('Finish: ', save_root)
