from os import mkdir
import os.path as osp
from datetime import datetime
from math import inf
from copy import deepcopy
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Lambda
import argparse
from tqdm import tqdm

from models import VanillaVAE, VanillaCNNVAE


def auto_name():
    return datetime.now().strftime("%b%d_%H-%M-%S")


def vae_kl(mu, logvar):
    return -0.5*(1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()


class Loss:

    def __init__(self):
        self.loss_sum = 0.
        self.count = 0

    def add(self, batch_loss, batch_size):
        self.loss_sum += batch_loss.cpu().detach().item() * batch_size
        self.count += batch_size

    def value(self):
        return self.loss_sum / self.count


def main():

    autoN = auto_name()

    parser = argparse.ArgumentParser()
    parser.add_argument("--exper", default="./results/%s" % autoN, type=str)
    parser.add_argument("--epoch", default=100, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--bs", default=256, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--log_dir", default="./runs", type=str)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--usefc", action="store_true")
    parser.add_argument("--latent_num", default=100, type=int)
    args = parser.parse_args()

    device = torch.device(args.device)

    # 读取数据集
    if args.usefc:
        transfer = Compose([
            ToTensor(), Lambda(lambda x: x.flatten())
        ])
    else:
        transfer = Compose([ToTensor()])
    train_data = MNIST(
        "~/Datasets", train=True, download=True, transform=transfer)
    test_data = MNIST(
        "~/Datasets", train=False, download=True, transform=transfer)
    train_dataloader = DataLoader(
        train_data, batch_size=args.bs, shuffle=True,
        num_workers=args.num_workers
    )
    test_dataloader = DataLoader(
        test_data, batch_size=args.bs, num_workers=args.num_workers
    )

    # 构建模型
    if args.usefc:
        model = VanillaVAE(
            28*28, [1000, 500, 500, 200],
            [200, 500, 500, 1000], args.latent_num
        ).to(device)
    else:
        model = VanillaCNNVAE(
            1, [32, 64, 128, 256, 512], args.latent_num, True
        ).to(device)
    criterion_rec = nn.BCEWithLogitsLoss()
    criterion_kl = vae_kl
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练
    writer = SummaryWriter(osp.join(args.log_dir, osp.basename(args.exper)))
    history = {
        "train": {"rec": [], "kl": [], "total": []},
        "test": {"rec": [], "kl": [], "total": []}
    }
    best = {"index": -1, "model": None, "loss": inf}
    main_bar = tqdm(total=args.epoch, desc="Epoch: ")
    for e in range(args.epoch):
        # minor_bar = tqdm(total=len(train_dataloader)+len(test_dataloader))
        # train phase
        loss_objs = [Loss() for _ in range(3)]
        # minor_bar.set_description("Phase: train, Batch: ")
        model.train()
        for img, _ in train_dataloader:
            img = img.to(device)
            rec, mu, logvar = model(img)
            rec_loss = criterion_rec(rec, img)
            kl_loss = criterion_kl(mu, logvar)
            loss = rec_loss + kl_loss
            # 更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 记录训练的过程
            bs = img.size(0)
            for i, ls in enumerate([rec_loss, kl_loss, loss]):
                loss_objs[i].add(ls, bs)
            # minor_bar.update()
        # epoch loss，并更新tensorboard
        epoch_losses = [lo.value() for lo in loss_objs]
        # minor_bar.set_description("Phase: test, Batch: ")
        for i, s in enumerate(["rec", "kl", "total"]):
            history["train"][s].append(epoch_losses[i])
            writer.add_scalar("train/%s" % s, epoch_losses[i], e)

        # test phase
        loss_objs = [Loss() for _ in range(3)]
        model.eval()
        with torch.no_grad():
            for img, _ in test_dataloader:
                img = img.to(device)
                rec, mu, logsigma = model(img)
                rec_loss = criterion_rec(rec, img)
                kl_loss = criterion_kl(mu, logsigma)
                loss = rec_loss + kl_loss
                # 记录训练的过程
                bs = img.size(0)
                for i, ls in enumerate([rec_loss, kl_loss, loss]):
                    loss_objs[i].add(ls, bs)
                # minor_bar.update()
        # epoch loss，并更新tensorboard
        epoch_losses = [lo.value() for lo in loss_objs]
        for i, s in enumerate(["rec", "kl", "total"]):
            history["test"][s].append(epoch_losses[i])
            writer.add_scalar("test/%s" % s, epoch_losses[i], e)

        # 每5个epoch画一次图
        if e % 5 == 0:
            latents = torch.randn(64, args.latent_num, device=device)
            gen_imgs = model.decode(latents)
            if args.usefc:
                gen_imgs = gen_imgs.reshape(64, 1, 28, 28)
            writer.add_images("sampling", gen_imgs, e // 5)

        # best
        if epoch_losses[-1] < best["loss"]:
            best["index"] = e
            best["loss"] = epoch_losses[-1]
            best["model"] = deepcopy(model.state_dict())

        main_bar.update()

    # 保存结果
    print("")
    print("Best index: %d, Best Loss: %.4f" % (best["index"], best["loss"]))
    if not osp.exists(args.exper):
        mkdir(args.exper)
    model.load_state_dict(best["model"])
    torch.save(model, osp.join(args.exper, "model.pth"))
    with open(osp.join(args.exper, "hist.json"), "w") as f:
        json.dump(history, f)
    best.pop("model")
    with open(osp.join(args.exper, "best.json"), "w") as f:
        json.dump(best, f)
    with open(osp.join(args.exper, "args.json"), "w") as f:
        json.dump(args.__dict__, f)


if __name__ == "__main__":
    main()
