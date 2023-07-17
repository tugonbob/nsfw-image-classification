import torch
import torchvision.models as models

model = models.resnet18(weights=None)

for param in model.parameters():
    param.requires_grad = False

num_classes = 5
num_features = model.fc.in_features

model.fc = torch.nn.Linear(num_features, num_classes)

resnet18 = model
