import torch
from torch.utils.data import DataLoader
from torch import optim, nn
from dataset import AudioDataset
from model import ConvNet
import os
from tqdm import tqdm
import time
import wandb


def train(x, y, model, optimizer, loss_function):
    model.train()
    optimizer.zero_grad()
    y_pred = model(x).squeeze()
    loss = loss_function(y_pred, y)
    loss.backward()
    optimizer.step()
    accuracy = ((y_pred > 0.5).int() == y).float().mean()
    return loss.item(), accuracy.item()


def validate(x, y, model, loss_function):
    model.eval()
    with torch.no_grad():
        y_pred = model(x).squeeze()
    loss = loss_function(y_pred, y)
    accuracy = ((y_pred > 0.5).int() == y).float().mean()
    return loss.item(), accuracy.item()


if __name__ == "__main__":
    torch.manual_seed(0)
    bar_fmt = "{desc}: {percentage:0.1f}% |{bar}| {n_fmt}/{total_fmt} | {elapsed}<{remaining}{postfix}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "n_epochs": 20,
        "dropout": [0.5, 0.3],
        "masking": [20, 10],
        "sample_rate": 22050,
        "n_mels": 128,
        "n_fft": 1024,
        "win_length": 512,
        "hop_length": 512,
        "augment": True
    }

    train_loader = DataLoader(
        AudioDataset(
            path=os.path.join("audio", "train"),
            sample_rate=config["sample_rate"],
            n_mels=config["n_mels"],
            n_fft=config["n_fft"],
            win_length=config["win_length"],
            hop_length=config["hop_length"],
            augment=config["augment"],
        ),
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        AudioDataset(
            path=os.path.join("audio", "validation"),
            sample_rate=config["sample_rate"],
            n_mels=config["n_mels"],
            n_fft=config["n_fft"],
            win_length=config["win_length"],
            hop_length=config["hop_length"],
        ),
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=True
    )

    # Initialize model, loss function, optimizers, and lr scheduler
    model = ConvNet(base=4)
    model.to(device)
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Initialize wandb
    wandb.init(project="torch", config=config)
    wandb.watch(model, log="all")

    # Start training
    for epoch in range(1, config["n_epochs"] + 1):
        print(f"Epoch {epoch}/{config['n_epochs']}")
        start_time = time.time()
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0
        # Training loop
        train_loop = tqdm(train_loader, desc="Training", ascii=True, bar_format=bar_fmt)
        for i, batch in enumerate(train_loop, 1):
            x_train, y_train = batch
            x_train, y_train = x_train.to(device), y_train.to(device)
            batch_loss, batch_acc = train(x_train, y_train.float(), model, optimizer, loss_fn)
            train_loss += batch_loss
            train_acc += batch_acc
            train_loop.set_postfix_str(
                f"loss: {round(train_loss/i, 4)} - acc: {round(train_acc/i, 4)}"
            )
            wandb.log({"train_loss_batch": batch_loss, "train_acc_batch": batch_acc})

        # Validation loop
        val_loop = tqdm(val_loader, desc="Validation", ascii=True, bar_format=bar_fmt)
        for i, batch in enumerate(val_loop, 1):
            x_val, y_val = batch
            x_val, y_val = x_val.to(device), y_val.to(device)
            batch_loss, batch_acc = validate(x_val, y_val.float(), model, loss_fn)
            val_loss += batch_loss
            val_acc += batch_acc
            wandb.log({"val_loss_batch": batch_loss, "val_acc_batch": batch_acc})
            val_loop.set_postfix_str(
                f"val_loss: {round(val_loss/i, 4)} - val_acc: {round(val_acc/i, 4)}"
            )

        # Change learning rate
        lr_scheduler.step()

        # Calculate and log averages
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc /= len(train_loader)
        val_acc /= len(val_loader)
        wandb.log(
            {
                "train_loss_epoch": train_loss,
                "train_acc_epoch": train_acc,
                "val_loss_epoch": val_loss,
                "val_acc_epoch": val_acc,
            }
        )
        total_time = round(time.time() - start_time, 1)
        print(f"Time per epoch: {total_time}s")

        # Save model
        torch.save(
            model.state_dict(),
            os.path.join(wandb.run.dir, f"model_{epoch}_{round(val_loss, 4)}.pth"),
        )
