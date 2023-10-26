import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
import datetime
from sklearn.model_selection import train_test_split
import traceback, sys
import torch
from torch.utils.data import DataLoader
from stage_dataset import STAGE_sub2_dataset
from cnn_model import *
from tqdm import tqdm
from my_transforms import collate_fn_w_transform
import transforms as trans

import warnings
warnings.filterwarnings('ignore')

file_dict = {'advance': ['0137', '0138', '0139', '0140', '0141', '0142', '0143', '0144', '0145', '0146', '0147', '0148', '0149', '0150', '0151', '0152', '0153', '0154', '0155', '0156', '0157', '0158', '0159', '0160', '0165', '0166', '0169', '0170', '0171', '0181', '0182'], 'inter': ['0077', '0078', '0079', '0080', '0081', '0082', '0083', '0084', '0085', '0086', '0087', '0088', '0089', '0090', '0091', '0092', '0093', '0094', '0095', '0096', '0097', '0161', '0167'], 'early': ['0002', '0003', '0004', '0006', '0007', '0008', '0010', '0016', '0019', '0021', '0022', '0025', '0026', '0030', '0031', '0032', '0035', '0036', '0037', '0038', '0043', '0044', '0045', '0046', '0047', '0048', '0053', '0054', '0055', '0057', '0059', '0060', '0061', '0062', '0063', '0064', '0065', '0066', '0067', '0068', '0069', '0070', '0071', '0072', '0073', '0074', '0075', '0076', '0098', '0099', '0100', '0101', '0102', '0103', '0104', '0105', '0107', '0108', '0109', '0110', '0111', '0112', '0113', '0114', '0115', '0116', '0117', '0118', '0119', '0120', '0121', '0122', '0123', '0124', '0125', '0126', '0127', '0128', '0129', '0130', '0131', '0132', '0133', '0134', '0135', '0136', '0162', '0164', '0168', '0172', '0173', '0174', '0175', '0176', '0177', '0178', '0179', '0180', '0183', '0184', '0185', '0186', '0187', '0188', '0191', '0192', '0193', '0194', '0195', '0197', '0198', '0199', '0200'], 'normal': ['0000', '0001', '0005', '0009', '0011', '0012', '0013', '0014', '0015', '0017', '0018', '0020', '0023', '0024', '0027', '0028', '0029', '0033', '0034', '0039', '0040', '0041', '0042', '0049', '0050', '0051', '0052', '0056', '0058', '0106', '0163', '0189', '0190', '0196']}


val_ratio = 0.2 # Train/Val Data Split, 80:20
random_state = 42 # DataSplit Random Seed
num_workers = 2

oct_img_size = [512, 512]
batchsize = 12 * torch.cuda.device_count()
init_lr = 1e-2
optimizer_type = "sgd"
scheduler = None
iters = 80 # Run Epochs

trainset_root = "/czy_SSD/1T/zychen/STAGE2023/STAGE_training/training_images"
model_root = "./model_checkpoints"
logs_root = "./logs"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.filename = filename
        self.log = open(filename, 'w')
        self.hook = sys.excepthook
        # sys.excepthook = self.kill

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def kill(self, ttype, tvalue, ttraceback):
        for trace in traceback.format_exception(ttype, tvalue, ttraceback):
            print(trace)
        os.remove(self.filename)

    def flush(self):
        pass


def train(model, iters, train_dataloader, val_dataloader, optimizer, criterion, log_interval, eval_interval):
    time_now = datetime.datetime.now().__format__("%Y%m%d_%H%M%S_%f")
    model_save_path = os.path.join(model_root, time_now)
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    print(time_now, "Total Nums: {}, train: {}, val: {}".format(len(filelists), len(train_filelists), len(val_filelists)))

    log_path = os.path.join(logs_root, time_now + '.log')
    sys.stdout = Logger(log_path, sys.stdout)

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

            loss = criterion(labels, logits, 5.)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            smape = Smape_(labels.cpu(), logits.detach().cpu()).numpy()
            r2 = R2_(labels.cpu(), logits.detach().cpu()).numpy()

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
                    torch.save(model.module.state_dict(), os.path.join(model_save_path, "best_model_{:.4f}".format(best_score) + '.pth'))
                else:
                    torch.save(model.state_dict(), os.path.join(model_save_path, "best_model_{:.4f}".format(best_score) + '.pth'))
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
            r2 = R2_(labels.cpu(), logits.detach().cpu()).numpy()

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
        trans.GaussianBlurTransform(),
        trans.GaussianNoiseTransform()
    ])

    val_transforms = trans.Compose([
        trans.CenterCrop([256] + oct_img_size)
    ])

    train_dataset = STAGE_sub2_dataset(dataset_root=trainset_root,
                            transforms=train_transforms,
                            filelists=train_filelists,
                            label_file='/czy_SSD/1T/zychen/STAGE2023/STAGE_training/training_GT/task2_GT_training.xlsx',
                            info_file='/czy_SSD/1T/zychen/STAGE2023/STAGE_training/data_info_training.xlsx')

    val_dataset = STAGE_sub2_dataset(dataset_root=trainset_root,
                            transforms=val_transforms,
                            filelists=val_filelists,
                            label_file='/czy_SSD/1T/zychen/STAGE2023/STAGE_training/training_GT/task2_GT_training.xlsx',
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

    model = Model().to(device)
    if torch.cuda.device_count() > 1:
        device_ids = list(range(0, torch.cuda.device_count()))
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    if optimizer_type == "adam":
        optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr, betas=(0.9, 0.99))
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.99, weight_decay=0.01, nesterov=True)

    criterion = MixLoss()

    train(model, iters, train_loader, val_loader, optimizer, criterion, log_interval=1, eval_interval=10)


filelists = os.listdir(trainset_root)
# train_filelists, val_filelists = train_test_split(filelists, test_size=val_ratio, random_state=random_state)
# val_size = int(len(filelists) * val_ratio)
train_filelists, val_filelists = [], []
for k in range(4, int(1 / val_ratio)):
    print('Fold-{}'.format(k))
    for key in file_dict.keys():
        val_size = int(len(file_dict[key]) * val_ratio)
        train_filelists += (file_dict[key][:k*val_size] + file_dict[key][(k+1)*val_size:])
        val_filelists += file_dict[key][k*val_size:(k+1)*val_size]
    print(len(train_filelists), len(val_filelists))
    main(train_filelists, val_filelists)

