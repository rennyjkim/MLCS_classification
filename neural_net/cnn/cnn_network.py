import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torchvision import transforms

data_dir = os.path.join('..', '002 CNN', 'cifar10')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                            (0.2023, 0.1994, 0.2010))])

train_data = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

test_data = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

path_dir = os.path.join('..', '002 CNN', 'model')
if not os.path.exists(path_dir):
    os.makedirs(path_dir)


DenseNet.load_state_dict(torch.load(path_dir))

DenseNet = models.densenet121(pretrained=True).features.to('cpu')

class Densenet(nn.Module):
    def __init__(self):
        super(Densenet, self).__init__()
        self.cnn_model = DenseNet
        self.classifier = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.cnn_model(x)
        x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

model = Densenet()

for param in model.parameters():
    param.requires_grad = False

for name, param in model.named_parameters():
    if name in ['classifier.weight', 'classifier.bias']:
        param.requires_grad = True

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_function = F.cross_entropy



# test
n = 0
test_losses = 0
test_acc = 0
model.eval()
for test_inputs, test_labels in test_dataloader:
    logits = model(test_inputs)
    test_losses += F.cross_entropy(logits, test_labels, reduction='sum')
    test_acc += (logits.argmax(dim=1) == test_labels).float().sum()
    n += test_inputs.size(0)

test_losses /= n
test_acc /= n
print('test_acc: ', test_acc)
print('test_loss', test_losses)

# model acc : 65.35