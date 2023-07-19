from datasets import load_dataset, DatasetDict
import torch
import os
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

print("Initializing database...")

dataset = load_dataset("deepghs/nsfw_detect")
train_val_dataset = dataset['train'].train_test_split(test_size=0.1)
train_val_test_dataset = train_val_dataset['test'].train_test_split(
    test_size=0.5)
dataset = DatasetDict({
    'train': train_val_dataset['train'],
    'val': train_val_test_dataset['test'],
    'test': train_val_test_dataset['train']
})

print("Applying transforms to database...")


def preprocess_dataset(dataset):
    transformed_dataset = dataset.map(
        transforms, batched=True)
    torch.save(transformed_dataset, "preprocessed_nsfw_detect.pt")
    return transformed_dataset


def transforms(examples):
    examples['image'] = [image.convert("RGB").resize(
        (224, 224)) for image in examples['image']]
    return examples


if not os.path.exists("preprocessed_nsfw_detect.pt"):
    # Preprocess the dataset and save the preprocessed version
    dataset = preprocess_dataset(dataset).with_format('torch')
else:
    # Load the preprocessed dataset
    dataset = torch.load("preprocessed_nsfw_detect.pt").with_format('torch')
