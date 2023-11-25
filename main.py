import argparse
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from nyu_dataset import *
from unet_model import *

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for (batch_idx, sample) in enumerate(loop):
        data, targets = sample['image'], sample['depth']
        data = data.to(device=device)
        targets = targets.float().to(device=device)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            print(data.shape, targets.shape, predictions.shape)
            loss = depth_loss(predictions, targets)
        # backward
        print(f"Loss for batch-{batch_idx} is {loss}")
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main(num_epochs, load_model, lr):
    train_transform = transforms.Compose([transforms.ToTensor()])

    root = 'nyu_depth_v2_labeled.mat'
    nyu_dataset = NyuDataset(root, transform=train_transform)

    val_size = int(0.2 * len(nyu_dataset))

    train_dataset, val_dataset = random_split(nyu_dataset, [len(nyu_dataset) - val_size, val_size])

    batch_size_train = 8
    batch_size_val = 4

    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False)
    
    model = UNET(in_channels=3, out_channels=1).to(device)
    if load_model:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
    
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(num_epochs):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', "--epochs", default=3, help="Number of epochs")
    parser.add_argument('-l', "--load_model", default=False, help="Load model from check point")
    parser.add_argument("--lr", default=1e-4, help="Load model from check point")
    args = parser.parse_args()
    main(args.epochs, args.load_model, args.lr)
