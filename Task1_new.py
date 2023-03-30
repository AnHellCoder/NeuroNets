import torch
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision as tv
from torch.utils.data.dataloader import DataLoader
from torchvision import models as models
import os
import io
from PIL import Image
from tqdm import tqdm

class SimpsonsDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

        self.characters = open('annotation.txt', 'r').readlines()
        self.characters = [self.characters[i].split(',') for i in range(len(self.characters))]

        self.features = [tv.io.read_image(self.characters[i][0]) for i in range(len(self.characters))]
        self.labels = [self.characters[i][5].strip('\n') for i in range(len(self.characters))]

        flag_label, flag_char = 0, self.labels[0]
        for i in range(len(self.labels)):
            if flag_char != self.labels[i]:
                flag_char = self.labels[i]
                flag_label += 1
                self.labels[i] = flag_label
            else:
                self.labels[i] = flag_label
        
        for i in range(len(self.features)):
            self.features[i] = self.features[i].resize_(300, 300)
            self.features[i] = torch.tensor(self.features[i], dtype=torch.float32)
    
    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    
    def __len__(self):
        return len(self.characters)

class SimpsonsClassifier(nn.Module):
    def __init__(self, img_size = (16, 300, 300), num_classes = 43) -> None:
        super().__init__()
        in_channels = img_size[0]

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(15, 15), padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(15, 15), padding=1)
        self.fc3 = nn.Linear(in_features=276 * 276, out_features=num_classes)
    
    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc3(x)
        
        return x
    
def train_step():
    model.train()
    
    running_loss = 0.
    for images, labels in train_loader:
        
        # Удаляем накопленные ранее градиенты.
        # parameter.grad = 0
        optimizer.zero_grad()
        
        # Делаем проход (forward pass).
        # Состояние модели `train` обеспечивает сохранение промежуточных результатов вычислений.
        # Эти сохраненные значения будут использованы ниже для вычисления градиента функции потерь.
        output = model(images)
        
        # Вычисляем функцию потерь на основе предсказания модели.
        loss = criterion(output, labels)

        # Вычисляем градиент: направление, в котором функция потерь возрастает максимально быстро.
        # parameter.grad += dloss / dparameter
        loss.backward()

        # parameter += -lr * parameter.grad
        # 
        # PyTorch SGD:
        # velocity = momentum * velocity + parameter.grad
        # parameter += - lr * velocity
        optimizer.step()
        
        # Накапливаем статистику.
        running_loss += loss
    
    with torch.no_grad():
        train_loss = running_loss / len(sd)
    return train_loss.item()

def valid_step():
    model.eval()

    correct_total = 0.
    running_loss = 0.
    with torch.no_grad():
        for images, labels in valid_loader:
            
            output = model(images)
            
            prediction = output.argmax(dim=1)
            correct_total += prediction.eq(labels.view_as(prediction)).sum()
            
            loss = criterion(output, labels)
            running_loss += loss
        
    valid_loss = running_loss / len(valid_loader)
    accuracy = correct_total / len(valid_loader.dataset)
    return valid_loss.item(), accuracy.item()

sd = SimpsonsDataset()
#model = SimpsonsClassifier()
# print(sd[3])
# print(len(sd))
model = models.AlexNet()

train_loader, valid_loader = random_split(dataset=sd, lengths=[4726, 2026])

train_loader, valid_loader = DataLoader(dataset=train_loader, batch_size=16, shuffle=True), DataLoader(dataset=valid_loader, batch_size=16, shuffle=True)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    params=model.parameters(),
    lr=0.01,
    momentum=0.9,
)

epochs = 10
train_losses = []
valid_losses = []
valid_accs = []

for k in (pbar := tqdm(range(epochs))):
    train_loss = train_step()
    valid_loss, valid_acc = valid_step()
    
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    valid_accs.append(valid_acc)
    
    pbar.set_description(f'Avg. train/valid loss: {train_loss:.4f}/{valid_loss:.4f}')