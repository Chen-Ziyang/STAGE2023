import torch
import torch.nn as nn
from resnet import resnet34


class Model(nn.Module):
    """
    simply create a 2-branch network, and concat global pooled feature vector.
    each branch = single resnet34
    """
    def __init__(self, mixstyle_layer=[]):
        super(Model, self).__init__()
        self.backbone = resnet34(pretrained=True, mixstyle_layer=mixstyle_layer)
        self.embedding_layer = nn.Linear(9, 128)
        self.linear = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 + 128, 192),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(192, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        self.sigmoid = nn.Sigmoid()
        # 在oct_branch更改第一个卷积层通道数
        self.backbone.conv1 = nn.Conv2d(256, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, input, info):
        one_hot = torch.cat((
            info[:, 1].unsqueeze(-1),
            torch.nn.functional.one_hot(info[:, 0].long(), num_classes=2),
            torch.nn.functional.one_hot(info[:, 2].long(), num_classes=2),
            torch.nn.functional.one_hot(info[:, 3].long(), num_classes=4),
                             ), dim=1)
        info_emb = self.embedding_layer(one_hot).view(input.size(0), -1)
        fea = self.backbone(input)
        logit = self.linear(torch.cat((fea, info_emb), dim=-1))
        return process(self.sigmoid(logit), infos=info[:, -1])


class MixLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, labels, preds):
        r2loss = torch.sum((preds - labels) ** 2) / torch.sum((labels - labels.mean()) ** 2)
        smapeloss = 1 / len(preds) * torch.sum(2 * torch.abs(preds - labels) / (torch.abs(preds) + torch.abs(labels)))
        return smapeloss + r2loss


class R2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, labels, preds):
        return torch.sum((preds - labels) ** 2) / torch.sum((labels - labels.mean()) ** 2)


class SMAPELoss(nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, labels, preds):
        return 1 / len(preds) * torch.sum(2 * torch.abs(preds - labels) / (torch.abs(preds) + torch.abs(labels)))


def process(logits, infos):
    logits = logits.squeeze()
    label = [3 if i == 0 else i * (-11) for i in infos]
    return logits * torch.tensor(label).to(logits.device)

def Smape_(labels, preds):
    return 1 / len(preds) * torch.sum(2 * torch.abs(preds - labels) / (torch.abs(preds) + torch.abs(labels)))
  
def R2(labels, preds):
    return 1. - torch.sum((preds - labels) ** 2) / torch.sum((labels - labels.mean()) ** 2)

def Score(smape, R2):
    return 0.5 * (1. / (smape + 0.1)) + 0.5 * (R2 * 10.)

class EpochLR(torch.optim.lr_scheduler._LRScheduler):
    # lr_n = lr_0 * (1 - epoch / epoch_nums)^gamma
    def __init__(self, optimizer, epochs, gamma=0.9, last_epoch=-1):
        self.lr = optimizer.param_groups[0]['lr']
        self.epochs = epochs
        self.gamma = gamma
        super(EpochLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.lr * pow((1. - self.last_epoch / self.epochs), self.gamma)]