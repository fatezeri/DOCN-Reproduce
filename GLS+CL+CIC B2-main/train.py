from model.unet_model import UNet
from utils.dataset import FundusSeg_Loader
from torch import optim
import torch.nn as nn
import random
import torch
import numpy as np
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import os
import torch.nn.functional as F
from model.gls import Gls
from model.contrastive_loss import PixelContrastLoss

# os.environ["CUDA_VISIBLE_DEVICES"]="0"

dataset_name = sys.argv[1]
run_num = int(sys.argv[2])

if dataset_name == "drive":
    train_data_path = "/home/zhangsh/dataset/drive/train/"
    valid_data_path = "/home/zhangsh/dataset/drive/test/"
    N_epochs = 2500
    lr_decay_step = [2400]
    lr_init = 0.001
    batch_size = 2
    test_epoch = 5
    dataset_mean = [0.4969, 0.2702, 0.1620]
    dataset_std = [0.3479, 0.1896, 0.1075]
    early_epoch = 400

if dataset_name == "stare":
    train_data_path = "/home/zhangsh/dataset/stare/train/"
    valid_data_path = "/home/zhangsh/dataset/stare/test/"
    N_epochs = 2500
    lr_decay_step = [2400]
    lr_init = 0.001
    batch_size = 2
    test_epoch = 5
    dataset_mean = [0.5889, 0.3272, 0.1074]
    dataset_std = [0.3458, 0.1844, 0.1104]
    early_epoch = 400

if dataset_name == "chase":
    train_data_path = "/home/zhangsh/dataset/chase_db1/train/"
    valid_data_path = "/home/zhangsh/dataset/chase_db1/test/"
    N_epochs = 2500
    lr_decay_step = [2400]
    lr_init = 0.001
    batch_size = 2
    test_epoch = 5
    dataset_mean = [0.4416, 0.1606, 0.0277]
    dataset_std = [0.3530, 0.1407, 0.0366]
    early_epoch = 400

if dataset_name == "rimone":
    train_data_path = "/home/zhangsh/dataset/oc/rimone/train/"
    valid_data_path = "/home/zhangsh/dataset/oc/rimone/test/"
    N_epochs = 2500
    lr_decay_step = [2400]
    lr_init = 0.0001
    batch_size = 8
    test_epoch = 2
    dataset_mean = [0.3383, 0.1164, 0.0465]  # In use
    dataset_std = [0.1849, 0.0913, 0.0441]
    early_epoch = 400

if dataset_name == "refuge":
    train_data_path = "/home/zhangsh/dataset/oc/refuge/train/"
    valid_data_path = "/home/zhangsh/dataset/oc/refuge/train_valid/"
    N_epochs = 2500
    lr_decay_step = [2400]
    lr_init = 0.0001
    batch_size = 8
    test_epoch = 2
    dataset_mean = [0.4237, 0.2414, 0.1182]  # In Use
    dataset_std = [0.1996, 0.1206, 0.0712]
    early_epoch = 400

if dataset_name == "refuge2":
    train_data_path = "/home/zhangsh/dataset/oc/refuge/valid_train/"
    valid_data_path = "/home/zhangsh/dataset/oc/refuge/valid_test/"
    N_epochs = 2500
    lr_decay_step = [2000]
    lr_init = 0.0001
    batch_size = 8
    test_epoch = 2
    dataset_mean = [0.5984, 0.4048, 0.3161]  # In Use
    dataset_std = [0.2416, 0.1871, 0.1442]
    early_epoch = 400


def train_net(net, device, run_num, epochs = N_epochs, batch_size = batch_size, lr = lr_init):
    train_dataset = FundusSeg_Loader(train_data_path, 1, dataset_name, dataset_mean, dataset_std)
    valid_dataset = FundusSeg_Loader(valid_data_path, 0, dataset_name, dataset_mean, dataset_std)
    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        num_workers = 6,
        batch_size = batch_size,
        shuffle = True,
        drop_last = True
    )
    valid_loader = torch.utils.data.DataLoader(dataset = valid_dataset, batch_size = 1, shuffle = False)
    print('Train images: %s' % len(train_loader.dataset))
    print('Valid images: %s' % len(valid_loader.dataset))

    criterion = nn.BCEWithLogitsLoss()

    glsfunc = Gls(
        alpha = 0.2,
        glsmix_f = 1,
        out_channel = 3,
        in_channel = 3,
        interm_channel = 2,
        n_layer = 4,
        out_norm = 'frob'
    ).to(device)

    # ===== 为了对齐原文 exp_trainer.py 中 x=256 的特征空间 =====
    reducefunc = nn.Sequential(
        nn.Conv2d(512, 256, kernel_size = 1, stride = 1, padding = 0, bias = False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace = True)
    ).to(device)

    # ===== 对应原文 models/exp_trainer.py -> __init__() 中 x=256 的 self.projfunc =====
    projfunc = nn.Sequential(
        nn.Conv2d(256, 256, kernel_size = 1, stride = 1, padding = 0, bias = False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace = True),
        nn.Conv2d(256, 256, kernel_size = 1, stride = 1, padding = 0, bias = True),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace = True)
    ).to(device)

    # ===== 对应原文 models/exp_trainer.py -> __init__() 里的 self.contrast_loss =====
    contrast_loss_func = PixelContrastLoss(
        temperature = 0.05,
        n_view = 10
    ).to(device)

    optimizer = optim.Adam(
        list(net.parameters()) +
        list(reducefunc.parameters()) +
        list(projfunc.parameters()),
        lr = lr,
        weight_decay = 1e-6
    )

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones = lr_decay_step,
        gamma = 0.1
    )

    # ===== 对应原文 models/exp_trainer.py -> forward_con() 的 object-focused consistency =====
    def consistency_loss(pred1, pred2, p1, p2, q1, q2, label):
        # 1) resize 到和 label 一样大小
        size_hw = [label.shape[2], label.shape[3]]

        p1 = F.interpolate(p1, size = size_hw, mode = "bilinear", align_corners = False)
        p2 = F.interpolate(p2, size = size_hw, mode = "bilinear", align_corners = False)
        q1_up = F.interpolate(q1, size = size_hw, mode = "bilinear", align_corners = False)
        q2_up = F.interpolate(q2, size = size_hw, mode = "bilinear", align_corners = False)

        # 2) prediction -> binary mask
        pred1_bin = (torch.sigmoid(pred1).detach() > 0.5).float()
        pred2_bin = (torch.sigmoid(pred2).detach() > 0.5).float()

        # 3) mask = GT ∨ pred
        mask1 = ((label > 0.5) | (pred1_bin > 0.5)).float().squeeze(1)
        mask2 = ((label > 0.5) | (pred2_bin > 0.5)).float().squeeze(1)

        # 4) cosine similarity over channel dim
        tmp1 = F.cosine_similarity(q1_up, p2, dim = 1)
        tmp2 = F.cosine_similarity(p1, q2_up, dim = 1)

        # 5) masked average，处理空 mask
        if mask1.sum() > 0 and mask2.sum() > 0:
            tmp1 = (tmp1 * mask1).sum() / mask1.sum()
            tmp2 = (tmp2 * mask2).sum() / mask2.sum()
            loss_cs = 1.0 - (tmp1 + tmp2) / 2.0
        elif mask1.sum() > 0:
            tmp1 = (tmp1 * mask1).sum() / mask1.sum()
            loss_cs = 1.0 - tmp1
        elif mask2.sum() > 0:
            tmp2 = (tmp2 * mask2).sum() / mask2.sum()
            loss_cs = 1.0 - tmp2
        else:
            loss_cs = torch.zeros(1, device = label.device, dtype = label.dtype).squeeze()

        return loss_cs

    best_loss = float('inf')
    best_epoch = 10
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        net.train()
        train_loss = 0
        for i, (img1, img2, label, filename, raw_height, raw_width) in enumerate(train_loader):
            img1 = img1.to(device = device, dtype = torch.float32)
            img2 = img2.to(device = device, dtype = torch.float32)
            label = label.to(device = device, dtype = torch.float32)

            optimizer.zero_grad()

            # 对应原文：img2 固定走 GLS
            img2 = glsfunc(img2)

            pred1, encf1 = net(img1)
            pred2, encf2 = net(img2)

            # ===== 对齐原文 exp_trainer.py 中 x=256 的特征空间 =====
            p1 = reducefunc(encf1)
            p2 = reducefunc(encf2)

            p1_detach = p1.detach()
            p2_detach = p2.detach()

            q1 = projfunc(p1)
            q2 = projfunc(p2)

            loss1 = criterion(pred1, label)
            loss2 = criterion(pred2, label)
            loss_seg = (loss1 + loss2) / 2.0

            # ===== 对应原文 models/exp_trainer.py -> forward_con() =====
            loss_cs = consistency_loss(pred1, pred2, p1_detach, p2_detach, q1, q2, label)

            # ===== 对应原文 models/exp_trainer.py -> forward_con() 的 contrast 分支 =====
            # 原文使用 projector 输出做 contrast
            enca_c = q1
            encb_c = q2

            size2_tmp = enca_c.shape[2] * 2
            size3_tmp = enca_c.shape[3] * 2

            enca_c = F.interpolate(enca_c, size = [size2_tmp, size3_tmp], mode = "bilinear", align_corners = False)
            encb_c = F.interpolate(encb_c, size = [size2_tmp, size3_tmp], mode = "bilinear", align_corners = False)

            embedding = torch.cat((enca_c, encb_c), dim = 0)
            embedding = F.normalize(embedding, dim = 1)

            pred1_bin = (torch.sigmoid(pred1).detach() > 0.5).float()
            pred2_bin = (torch.sigmoid(pred2).detach() > 0.5).float()

            predict_ct = torch.cat((pred1_bin, pred2_bin), dim = 0)
            label_ct = torch.cat((label, label), dim = 0)

            predict_ct = F.interpolate(predict_ct, size = [size2_tmp, size3_tmp], mode = "nearest")
            label_ct = F.interpolate(label_ct, size = [size2_tmp, size3_tmp], mode = "nearest")

            loss_ct = contrast_loss_func(embedding, label_ct, predict_ct)

            lambda_cs = 0.1
            lambda_ct = 0.05
            loss = loss_seg + lambda_cs * loss_cs + lambda_ct * loss_ct

            loss.backward()
            optimizer.step()

            if epoch == 0 and i == 0:
                print("pred1 shape:", pred1.shape)
                print("encf1 shape:", encf1.shape)
                print("p1 shape:", p1.shape)
                print("q1 shape:", q1.shape)

                print("pred2 shape:", pred2.shape)
                print("encf2 shape:", encf2.shape)
                print("p2 shape:", p2.shape)
                print("q2 shape:", q2.shape)

                print("loss1:", loss1.item())
                print("loss2:", loss2.item())
                print("loss_seg:", loss_seg.item())
                print("loss_cs:", loss_cs.item())
                print("loss_ct:", loss_ct.item())
                print("loss_total:", loss.item())



        # Validation
        # epoch != test_epoch
        if ((epoch + 1) % test_epoch == 0):
            net.eval()
            val_loss = 0
            for i, (image, label, filename, raw_height, raw_width) in enumerate(valid_loader):
                image = image.to(device = device, dtype = torch.float32)
                label = label.to(device = device, dtype = torch.float32)
                pred, _ = net(image)
                loss = criterion(pred, label)
                val_loss = val_loss + loss.item()
            if val_loss < best_loss:
                best_loss = val_loss

                save_dir = './snapshot'
                os.makedirs(save_dir, exist_ok = True)

                torch.save(net.state_dict(), './snapshot/' + dataset_name + '_b' + str(run_num) + '.pth')
                print('saving model............................................')
                best_epoch = epoch
            if (epoch - best_epoch) > early_epoch:
                print('Early Stopping ............................................')
                exit()

            print('Loss/valid', val_loss / i)
            sys.stdout.flush()

        scheduler.step()


if __name__ == "__main__":
    random.seed(run_num)
    np.random.seed(run_num)
    torch.manual_seed(run_num)
    torch.cuda.manual_seed(run_num)
    torch.cuda.manual_seed_all(run_num)
    device = torch.device('cuda')
    net = UNet(n_channels = 3, n_classes = 1)
    net.to(device = device)
    train_net(net, device, run_num)
