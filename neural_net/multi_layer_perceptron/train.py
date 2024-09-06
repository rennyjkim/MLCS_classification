import os
import torchvision
import torch
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

batch_size = 128
learning_rate = 0.001
max_epoch = 20 #typically 100+

data_dir = os.path.join('..', '001 MLP', 'MNIST')
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [49500, 10500])
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

test_dataset = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# train, valid, test -> 49000:10500:10500

class Classifier_1(nn.Module): # 5 deep
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10): #initializing network
        super(Classifier_1, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        outputs = self.layers(x)
        return outputs

class Classifier_2(nn.Module):  # 8 deep
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10): #initializing network
        super(Classifier_2, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        outputs = self.layers(x)
        return outputs

c1 = Classifier_1()
c2 = Classifier_2()

optimizer_1 = optim.Adam(c1.parameters(), lr=learning_rate)
optimizer_2 = optim.Adam(c2.parameters(), lr=learning_rate)

train1_loss = []
train2_loss = []
for epoch in range(max_epoch):
    c1.train()
    for inputs, labels in train_dataloader:
        inputs = inputs
        labels = labels
        outputs = c1(inputs)
        loss = F.cross_entropy(outputs, labels)
        optimizer_1.zero_grad()
        loss.backward()
        optimizer_1.step()
    train1_loss.append(loss)
print(train1_loss, 'train1')

for epoch in range(max_epoch):
    c2.train()
    for inputs, labels in train_dataloader:
        inputs = inputs
        labels = labels
        outputs = c2(inputs)
        loss = F.cross_entropy(outputs, labels)
        optimizer_2.zero_grad()
        loss.backward()
        optimizer_2.step()
    train2_loss.append(loss)
print(train2_loss, 'train2')
'''
n = 0.
valid_loss = 0.
valid_acc = 0.
c1.eval()
for valid_inputs, valid_labels in valid_dataloader:
    outputs = c1(valid_inputs)
    valid_loss += F.cross_entropy(outputs, valid_labels, reduction='sum')
    valid_acc += (outputs.argmax(dim=1) == valid_labels).float().sum()
    n += valid_inputs.size(0)

valid_loss /= n
valid_acc /= n
print('valid_acc: ', valid_acc)
print('valid_loss', valid_loss)
'''

# model 1 acc : 0.9797
# model 2 acc : 0.9732