# -*- coding: utf-8 -*-
import os
import argparse
import wandb 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST


from ban import config
from ban.updater import BANUpdater
from common.logger import Logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--n_epoch", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_gen", type=int, default=3)
    parser.add_argument("--resume_gen", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--outdir", type=str, default="snapshots")
    parser.add_argument("--print_interval", type=int, default=1)
    parser.add_argument("--randinit", type=str, default="true")
    parser.add_argument("--distloss", type=str, default="default")
    args = parser.parse_args()

    wandb.init(project='bans_compare', config=args)

    logger = Logger(args)
    logger.print_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    #else:
    #    device = "cpu"

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465),
        #                      (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == "cifar10":
        trainset = CIFAR10(root="./data",
                           train=True,
                           download=True,
                           transform=transform)
        testset = CIFAR10(root="./data",
                          train=False,
                          download=True,
                          transform=transform)
    else:
        trainset = MNIST(root="./data",
                         train=True,
                         download=True,
                         transform=transform)
        testset = MNIST(root="./data",
                        train=False,
                        download=True,
                        transform=transform)

    train_loader = DataLoader(trainset,
                              batch_size=args.batch_size,
                              shuffle=True)
    test_loader = DataLoader(testset,
                             batch_size=args.batch_size,
                             shuffle=False)

    model = config.get_model().to(device)
    if args.weight:
        model.load_state_dict(torch.load(args.weight))

    wandb.watch(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    kwargs = {
        "model": model,
        "optimizer": optimizer,
        "n_gen": args.n_gen,
        "distloss": args.distloss,
    }

    updater = BANUpdater(**kwargs)
    criterion = nn.CrossEntropyLoss()

    i = 0
    best_loss = 1e+9
    best_loss_list = []

    print("train...")
    for gen in range(args.resume_gen, args.n_gen):
        for epoch in range(args.n_epoch):
            print(device)
            model.to(device)
            train_loss = 0
            for idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                t_loss = updater.update(inputs, targets, criterion).item()
                train_loss += t_loss
                i += 1
                if i % args.print_interval == 0:
                    wandb.log({"train_loss": t_loss}, step=i)

                    val_loss = 0
                    val_samples = 0
                    val_correct = 0
                    with torch.no_grad():
                        for idx, (inputs, targets) in enumerate(test_loader):
                            inputs, targets = inputs.to(device), targets.to(device)
                            outputs = updater.model(inputs)
                            loss = criterion(outputs, targets).item()
                            val_loss += loss

                            _, predicted = torch.max(outputs.data, 1)
                            val_samples += targets.size(0)
                            val_correct += (predicted == targets).sum().item()

                    val_loss /= len(test_loader)
                    val_accuracy = val_correct / val_samples

                    if val_loss < best_loss:
                        best_loss = val_loss
                        last_model_weight = os.path.join(args.outdir,
                                                         "model"+str(gen)+".pth.tar")
                        torch.save(updater.model.state_dict(),
                                   last_model_weight)

                    wandb.log({"val_loss": val_loss}, step=i)
                    wandb.log({"val_accuracy": val_accuracy}, step=i)

                    print(epoch, i, train_loss / args.print_interval, val_loss)
                    train_loss = 0

        print("best loss: ", best_loss)
        print("Born Again...")
        updater.register_last_model(last_model_weight)
        updater.gen += 1
        best_loss_list.append(best_loss)
        best_loss = 1e+9
        if args.randinit == "true":
            print("new init")
            model = config.get_model().to(device)
        print(model.parameters.type())
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        updater.model = model
        updater.optimizer = optimizer

    for gen in range(args.n_gen):
        print("Gen: ", gen,
              ", best loss: ", best_loss_list[gen])


if __name__ == "__main__":
    main()
