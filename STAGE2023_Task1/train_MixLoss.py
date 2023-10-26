import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import numpy as np
from sklearn.model_selection import train_test_split
import datetime
from sklearn.metrics import r2_score
import torch
from torch.utils.data import DataLoader
from stage_dataset import STAGE_sub1_dataset
from cnn_model import *
from tqdm import tqdm
import transforms as trans

import warnings
warnings.filterwarnings('ignore')

val_ratio = 0.2 # Train/Val Data Split, 80:20
random_state = 42 # DataSplit Random Seed
num_workers = 2

oct_img_size = [512, 512]
image_size = 256
batchsize = 24
init_lr = 1e-2
optimizer_type = "sgd"
scheduler = None
iters = 300 # Run Epochs

trainset_root = "/czy_SSD/1T/zychen/STAGE2023/STAGE_training/training_images"
model_root = "./model_checkpoints"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(model, iters, train_dataloader, val_dataloader, optimizer, criterion, log_interval, eval_interval):
    time_now = datetime.datetime.now().__format__("%Y%m%d_%H%M%S_%f")
    model_save_path = os.path.join(model_root, time_now)
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    print(time_now, "Total Nums: {}, train: {}, val: {}".format(len(filelists), len(train_filelists), len(val_filelists)))

    scheduler = EpochLR(optimizer, epochs=iters, gamma=0.9)
    model.train()
    avg_loss_list = []
    avg_score_list = []
    avg_smape_list = []
    avg_r2_list = []
    best_score = 0
    best_epoch = 0
    for iter in tqdm(range(1, iters+1)):
        for batch, data in enumerate(train_dataloader):
            oct_imgs = (data[0] / 255.).to(dtype=torch.float32).to(device)
            labels = (data[1]).to(dtype=torch.float32).to(device)
            info = (data[2]).to(dtype=torch.float32).to(device)

            logits = model(oct_imgs, info)

            loss = criterion(labels, logits)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            smape = Smape_(labels.cpu(), logits.detach().cpu()).numpy()
            r2 = r2_score(labels.cpu(), logits.detach().cpu())

            avg_loss_list.append(loss.item())
            avg_score_list.append(Score(smape, r2))
            avg_smape_list.append(smape)
            avg_r2_list.append(r2)

        if scheduler is not None:
            scheduler.step()
            
        if iter % log_interval == 0:
            avg_loss = np.array(avg_loss_list).mean()
            avg_score = np.array(avg_score_list).mean()
            smape_ = np.array(avg_smape_list).mean()
            r2_ = np.array(avg_r2_list).mean()
            avg_loss_list = []
            avg_score_list = []
            avg_smape_list = []
            avg_r2_list = []
            print("[TRAIN] iter={}/{} avg_loss={:.4f} avg_score={:.4f} SMAPE={:.4f} R2={:.4f}".format(iter, iters, avg_loss, avg_score, smape_, r2_))

        if iter % eval_interval == 0:
            avg_loss, avg_score, smape_, r2_ = val(model, val_dataloader, criterion)
            print("[EVAL] iter={}/{} avg_loss={:.4f} score={:.4f} SMAPE={:.4f} R2={:.4f}".format(iter, iters, avg_loss, avg_score, smape_, r2_))
            if avg_score >= best_score:
                best_score = avg_score
                best_epoch = iter
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(),
                               os.path.join(model_save_path, "best_model_{:.4f}".format(best_score) + '.pth'))
                else:
                    torch.save(model.state_dict(),
                               os.path.join(model_save_path, "best_model_{:.4f}".format(best_score) + '.pth'))
            model.train()

    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), os.path.join(model_save_path, "last_model.pth"))
    else:
        torch.save(model.state_dict(), os.path.join(model_save_path, "last_model.pth"))
    print('Best Epoch:{} Best Score:{:.4f}'.format(best_epoch, best_score))


def val(model, val_dataloader, criterion):
    model.eval()
    avg_loss_list = []
    avg_score_list = []
    avg_smape_list = []
    avg_r2_list = []
    with torch.no_grad():
        for batch, data in enumerate(val_dataloader):
            oct_imgs = (data[0] / 255.).to(dtype=torch.float32).to(device)
            labels = data[1].to(dtype=torch.float32).to(device)
            info = data[2].to(dtype=torch.float32).to(device)

            logits = model(oct_imgs, info)

            loss = criterion(labels, logits)

            smape = Smape_(labels.cpu(), logits.detach().cpu()).numpy()
            r2 = r2_score(labels.cpu(), logits.detach().cpu())

            avg_loss_list.append(loss.item())
            avg_score_list.append(Score(smape, r2))
            avg_smape_list.append(smape)
            avg_r2_list.append(r2)

    avg_score = np.array(avg_score_list).mean()
    avg_loss = np.array(avg_loss_list).mean()
    smape_ = np.array(avg_smape_list).mean()
    r2_ = np.array(avg_r2_list).mean()
    return avg_loss, avg_score, smape_, r2_


def main(train_filelists, val_filelists):
    train_transforms = trans.Compose([
        trans.CenterRandomCrop([256] + oct_img_size),
        trans.RandomHorizontalFlip(),
        trans.RandomVerticalFlip()
    ])

    val_transforms = trans.Compose([
        trans.CenterCrop([256] + oct_img_size)
    ])

    train_dataset = STAGE_sub1_dataset(dataset_root=trainset_root,
                            transforms=train_transforms,
                            filelists=train_filelists,
                            label_file='/czy_SSD/1T/zychen/STAGE2023/STAGE_training/training_GT/task1_GT_training.xlsx',
                            info_file='/czy_SSD/1T/zychen/STAGE2023/STAGE_training/data_info_training.xlsx')

    val_dataset = STAGE_sub1_dataset(dataset_root=trainset_root,
                            transforms=val_transforms,
                            filelists=val_filelists,
                            label_file='/czy_SSD/1T/zychen/STAGE2023/STAGE_training/training_GT/task1_GT_training.xlsx',
                            info_file='/czy_SSD/1T/zychen/STAGE2023/STAGE_training/data_info_training.xlsx')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batchsize,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batchsize,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True
    )

    model = Model(mixstyle_layer=['layer1', 'layer2']).to(device)
    if torch.cuda.device_count() > 1:
        device_ids = list(range(0, torch.cuda.device_count()))
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, betas=(0.9, 0.99))
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.99, weight_decay=0.005, nesterov=True)

    criterion = MixLoss()

    train(model, iters, train_loader, val_loader, optimizer, criterion, log_interval=1, eval_interval=10)


filelists = os.listdir(trainset_root)
train_filelists, val_filelists = train_test_split(filelists, test_size=val_ratio, random_state=random_state)
main(train_filelists, val_filelists)
