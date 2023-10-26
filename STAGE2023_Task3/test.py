import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import numpy as np
from cnn_model import OrdinalRegressionLoss
import pandas as pd
import torch
from torch.utils.data import DataLoader
from stage_dataset import STAGE_sub3_dataset
from cnn_model import *

import transforms as trans

import warnings
warnings.filterwarnings('ignore')


torch.set_num_threads(1)


def test(loader, model):
    cache = []
    ids = []
    model.eval()
    with torch.no_grad():
        for batch, (oct_img, idx, info) in enumerate(loader):
            print(idx)
            if len(oct_img.size()) == 5:  # TTA delete the first dimension
                oct_img = oct_img.squeeze(0)

            oct_img = (oct_img / 255.).to(dtype=torch.float32).to(device)
            info = info.repeat(oct_img.shape[0], 1).to(dtype=torch.float32).to(device)

            logits = model(oct_img, info)

            # Save Results
            if logits.shape[0] == 1:
                _, output = OrdinalRegressionLoss(num_class=5)(logits.reshape(-1, 1), None)
                output = output.detach().cpu().numpy()
            else:
                outputs = []
                for i in range(logits.shape[0]):
                    _, task3_output = OrdinalRegressionLoss(num_class=5)(logits[i].reshape(-1, 1), None)
                    outputs.append(task3_output.detach().cpu().numpy())     # B, 52, 5
                output = np.mean(outputs, 0)    # 52, 5

            ids.append(idx[0])
            cache.append(output)
    return np.stack(ids, 0), np.stack(cache, 0)


test_root = "/czy_SSD/1T/zychen/STAGE2023/STAGE_validation/validation_images"
save_root = "./results/PD_Results.csv"
oct_img_size = [512, 512]
device = 'cuda:0'

oct_test_transforms = trans.Compose([
    trans.CenterCrop([256] + oct_img_size)
])

oct_tta_transforms = trans.Compose([
    trans.CenterRandomCrop([256] + oct_img_size),
    trans.RandomHorizontalFlip()
])

test_dataset = STAGE_sub3_dataset(dataset_root=test_root,
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
    "./model_checkpoints/20230810_002356_254709/best_model_5.6714.pth",
    "./model_checkpoints/20230810_002356_254709/best_model_5.2380.pth",
    "./model_checkpoints/20230810_002356_254709/best_model_5.6319.pth",
    "./model_checkpoints/20230810_002356_254709/best_model_5.5494.pth",
    "./model_checkpoints/20230810_002356_254709/best_model_5.4792.pth"
]

columns = ['ID'] + ['point'+str(i) for i in range(1, 52+1)]
caches = []
for i in range(len(best_model_path)):
    print('Model {}: '.format(i+1), best_model_path[i])
    model = Model().to(device)
    para_state_dict = torch.load(best_model_path[i])
    model.load_state_dict(para_state_dict)

    ids, cache = test(test_loader, model)   # ids: [N], cache: [N, 52, 5]
    caches.append(cache)    # [N, 52, 5] --> [Models, N, 52, 5]

caches = np.stack(caches, 0)
print(caches.shape)
caches = np.mean(caches, 0).argmax(-1)  # [Models, N, 52, 5] --> [N, 52, 5] --> [N, 52]
caches = np.concatenate((ids[:, np.newaxis], caches), axis=-1)   # [N, 1] + [N, 52] --> [N, 53]
submission_result = pd.DataFrame(caches, columns=columns)
submission_result.to_csv(save_root, index=False)
print('Finish: ', save_root)
