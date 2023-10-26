import torch
import torch.nn as nn
import torchvision.models as models
from resnet import resnet34, resnet50, resnext50_32x4d


class Model(nn.Module):
    """
    simply create a 2-branch network, and concat global pooled feature vector.
    each branch = single resnet34
    """
    def __init__(self):
        super(Model, self).__init__()
        self.backbone = resnext50_32x4d(pretrained=True)
        self.backbone.conv1 = nn.Conv2d(256, 64, kernel_size=7, stride=2, padding=3, bias=False)
        output_dim = 2048

        self.embedding_layer = nn.Linear(9, 256)

        self.task3_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(output_dim + 256, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 52),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, input, info):
        one_hot = torch.cat((
            info[:, 1].unsqueeze(-1),
            torch.nn.functional.one_hot(info[:, 0].long(), num_classes=2),
            torch.nn.functional.one_hot(info[:, 2].long(), num_classes=2),
            torch.nn.functional.one_hot(info[:, 3].long(), num_classes=4),
                             ), dim=1)
        info_emb = self.embedding_layer(one_hot).view(input.size(0), -1)
        fea = self.backbone(input)

        head_input = torch.cat((fea, info_emb), dim=1)

        task3_logit = self.task3_head(head_input)
        return task3_logit


def f1_loss(pred, label):
    eps = 1e-10
    label = torch.nn.functional.one_hot(label.long(), num_classes=5).squeeze(1)
    true_positives = pred * label
    false_positives = pred * (1. - label)
    false_negatives = (1. - pred) * label

    precision = true_positives / (true_positives + false_positives + eps)
    recall = true_positives / (true_positives + false_negatives + eps)

    f1 = 2. * (precision * recall) / (precision + recall + eps)
    loss = torch.mean(1. - f1)
    return loss


class OrdinalRegressionLoss(nn.Module):

    def __init__(self, num_class, train_cutpoints=False, scale=2.0):
        super().__init__()
        self.num_classes = num_class
        num_cutpoints = self.num_classes - 1
        self.cutpoints = torch.arange(num_cutpoints).float() * scale / (num_class - 2) - scale / 2
        self.cutpoints = nn.Parameter(self.cutpoints)
        if not train_cutpoints:
            self.cutpoints.requires_grad_(False)

    def forward(self, pred, label):
        sigmoids = torch.sigmoid(self.cutpoints.to(pred.device) - pred)  # [b, num_cutpoints]
        link_mat = sigmoids[:, 1:] - sigmoids[:, :-1]
        link_mat = torch.cat((
            sigmoids[:, [0]],
            link_mat,
            (1 - sigmoids[:, [-1]])
        ), dim=1)

        eps = 1e-15
        likelihoods = torch.clamp(link_mat, eps, 1 - eps)

        log_likelihood = torch.log(likelihoods)
        if label is None:
            loss = 0
        else:
            celoss = -torch.gather(log_likelihood, 1, label.long()).mean()
            f1loss = f1_loss(pred=likelihoods, label=label)
            loss = celoss + 0.5 * f1loss
        return loss, likelihoods


class EpochLR(torch.optim.lr_scheduler._LRScheduler):
    # lr_n = lr_0 * (1 - epoch / epoch_nums)^gamma
    def __init__(self, optimizer, epochs, gamma=0.9, last_epoch=-1):
        self.lr = optimizer.param_groups[0]['lr']
        self.epochs = epochs
        self.gamma = gamma
        super(EpochLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.lr * pow((1. - self.last_epoch / self.epochs), self.gamma)]

