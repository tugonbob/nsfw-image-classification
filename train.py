import argparse
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn

import wandb
from dataset import dataset
from resnet import resnet18


def train(checkpoint, starting_epoch, epochs, wan_db):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} processor")

    model = resnet18.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    loss_function = nn.CrossEntropyLoss()

    train_dataloader = DataLoader(
        dataset['train'], batch_size=512, shuffle=True)
    val_dataloader = DataLoader(dataset['val'], batch_size=512, shuffle=True)

    # Initializing in a separate cell so we can easily add more epochs to the same run
    if not os.path.exists('models'):
        os.makedirs('models')

    if checkpoint:
        print("Loading model from checkpoint...")
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Training on {len(dataset['train'])} images")
    print(f"Validating on {len(dataset['test'])} images")
    print(f"For {epochs - starting_epoch + 1} epochs...")

    for epoch in range(starting_epoch, epochs+1):
        print('EPOCH {}:'.format(epoch))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train()
        for i, data in enumerate(train_dataloader):
            # Every data instance is an input + label pair
            # images = torch.tensor(data['image'])
            images = data['image'].permute(0, 3, 1, 2).float().to(device)
            labels = data['label'].to(device)

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            predicted_label = model(images)

            # Compute the loss and its gradients
            loss = loss_function(predicted_label, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            if i % 10 == 0:
                print('Batch {} - Loss: {}'.format(i+1, loss))
                if wan_db:
                    wandb.log({"loss": loss.item()})

        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(val_dataloader):
                vimages = vdata['image'].permute(0, 3, 1, 2).float().to(device)
                vlabels = vdata['label'].to(device)
                vpredicted_labels = model(vimages)
                vloss = loss_function(vpredicted_labels, vlabels)

        print('LOSS: train {} val {}'.format(loss, vloss))
        if wan_db:
            wandb.log({"vloss": vloss.item()})

        # Track best performance, and save the model's state
        model_path = f'models/Epoch{epoch}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': vloss}, model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='MNIST handwritten number classification')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help="Path to your .pt model")
    parser.add_argument('--epochs', type=int, default=50,
                        help="Number of epochs you want to train for")
    parser.add_argument('--starting_epoch', type=int,
                        default=1, help="Starting point for epochs")
    parser.add_argument('--wandb', action='store_true',
                        help="Pass false, if you don't want to use wandb")
    args = parser.parse_args()

    if args.wandb:
        print("Initializing wandb...")
        wandb.init(
            project="nsfw-image-classification",
            config={
                "learning_rate": 0.03,
                "architecture": "resnet18",
                "dataset": "deepghs/nsfw_detect",
                "epochs": 50,
            },
        )

    train(args.checkpoint, args.starting_epoch, args.epochs, args.wandb)
