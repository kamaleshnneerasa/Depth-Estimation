import argparse
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from midas.model_loader import load_model
from MyDataset import *
from unet_model import *
from loss import depth_loss
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for (batch_idx, sample) in enumerate(loop):
        data, targets = sample['image'], sample['depth']
        print(data.shape, targets.shape)
        data = data.to(device=device)
        targets = targets.float().to(device=device)
        targets = targets.unsqueeze(1)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            predictions = predictions.unsqueeze(1)
            print(predictions.shape, targets.shape)
            loss = depth_loss(predictions, targets)

        # backward
        print(f"Loss for batch-{batch_idx} is {loss}")
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main(num_epochs, lr):
    print(device)
    model, my_transform, net_w, net_h = load_model(device,
    "/Users/kamalesh/Desktop/Depth-Estimation/weights/dpt_swin2_large_384.pt",
    "dpt_swin2_large_384", False, 384, False)

    print(net_w, net_h)
    root = 'nyu_depth_v2_labeled.mat'
    nyu_dataset = MyDataset(root, transform=my_transform)

    val_size = int(0.2 * len(nyu_dataset))

    train_dataset, val_dataset = random_split(nyu_dataset, [len(nyu_dataset) - val_size, val_size])

    batch_size_train = 8
    batch_size_val = 4

    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(num_epochs):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', "--epochs", default=3, help="Number of epochs")
    parser.add_argument("--lr", default=1e-4, help="Load model from check point")
    args = parser.parse_args()
    main(args.epochs, args.lr)
