from os import mkdir
import os.path as osp
from datetime import datetime
from math import inf
from copy import deepcopy
import json

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Lambda, ToPILImage
from torchvision.utils import make_grid
import argparse
from tqdm import tqdm

from models import VanillaVAEfc, VanillaVAEcnn


def auto_name():
    return datetime.now().strftime("%b%d_%H-%M-%S")


def kl_cost(e, w0, wT, e0, eT):
    if e <= e0:
        return w0
    if e >= eT:
        return wT
    wdiff = wT - w0
    ediff = eT - e0
    w_per_step = wdiff / ediff
    return w0 + (e - e0) * w_per_step


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
    parser.add_argument("--epoch", default=200, type=int)
    parser.add_argument("--lr", default=0.005, type=float)
    parser.add_argument("--bs", default=256, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--log_dir", default="./runs", type=str)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--usefc", action="store_true")
    parser.add_argument("--latent_num", default=100, type=int)
    parser.add_argument("--kl_weight", default=0.001, type=float)
    parser.add_argument("--loss_type", default="mse", type=str)
    args = parser.parse_args()

    device = torch.device(args.device)
    kl_w = args.kl_weight
    topilimage = ToPILImage()

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
        model = VanillaVAEfc(
            28*28, args.latent_num, [1000, 500, 200]
        ).to(device)
    else:
        model = VanillaVAEcnn(
            1, args.latent_num, [32, 64, 128, 256, 512]
        )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 训练
    writer = SummaryWriter(osp.join(args.log_dir, osp.basename(args.exper)))
    history = {
        "train": {"rec": [], "kl": [], "total": []},
        "test": {"rec": [], "kl": [], "total": []}
    }
    best = {"index": -1, "model": None, "loss": inf}
    pilimgs = []
    for e in tqdm(range(args.epoch), desc="Epoch: "):
        writer.add_scalar("KL_weight", kl_w, e)
        # train phase
        loss_objs = [Loss() for _ in range(3)]
        model.train()
        for img, _ in train_dataloader:
            img = img.to(device)
            rec, mu, logvar = model(img)
            loss, rec_loss, kl_loss = model.criterion(
                rec, img, mu, logvar, kl_w, args.loss_type
            )
            # 更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 记录训练的过程
            bs = img.size(0)
            for i, ls in enumerate([rec_loss, kl_loss, loss]):
                loss_objs[i].add(ls, bs)
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
                rec, mu, logvar = model(img)
                loss, rec_loss, kl_loss = model.criterion(
                    rec, img, mu, logvar, kl_w, args.loss_type
                )
                # 记录训练的过程
                bs = img.size(0)
                for i, ls in enumerate([rec_loss, kl_loss, loss]):
                    loss_objs[i].add(ls, bs)
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
            pilimgs.append(topilimage(make_grid(gen_imgs).cpu()))

        # best
        if epoch_losses[-1] < best["loss"]:
            best["index"] = e
            best["loss"] = epoch_losses[-1]
            best["model"] = deepcopy(model.state_dict())

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

    # 采样的图像创建gif
    pilimgs[0].save(
        "samples_%s.gif" % args.loss_type, format="GIF",
        append_images=pilimgs[1:], save_all=True, duration=500, loop=0
    )


if __name__ == "__main__":
    main()
