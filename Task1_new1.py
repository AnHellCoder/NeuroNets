import torch
import numpy as np
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision as tv
from torch.utils.data.dataloader import DataLoader
from torchvision import models as models
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

#RCNN OCR
#https://github.com/open-mmlab/mmocr

class SimpsonsDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

        self.dataset = tv.datasets.ImageFolder('simpsons_dataset',
                            transform=tv.transforms.Compose([tv.transforms.ToTensor(),
                                                             tv.transforms.Resize([224, 224])]))
        #self.characters, self.labels = np.array(self.dataset.imgs)[0:, 0], np.array(self.dataset.imgs)[0:, 1]

        #self.characters = tv.io.read_image(self.characters)
        #print(self.dataset.transform)
        # print(self.dataset[0])

        # for i in range(len(self.characters)):
        #     self.characters[i] = self.characters[i].type(dtype=torch.float32)

        # self.characters = open('annotation.txt', 'r').readlines()
        # self.characters = [self.characters[i].split(',') for i in range(len(self.characters))]

        # self.features = [tv.io.read_image(self.characters[i][0]) for i in range(len(self.characters))]
        # self.labels = [self.characters[i][5].strip('\n') for i in range(len(self.characters))]

        # flag_label, flag_char = 0, self.labels[0]
        # for i in range(len(self.labels)):
        #     if flag_char != self.labels[i]:
        #         flag_char = self.labels[i]
        #         flag_label += 1
        #         self.labels[i] = flag_label
        #     else:
        #         self.labels[i] = flag_label
        
        # for i in range(len(self.features)):
        #     self.features[i] = self.features[i].resize_(300, 300)
        #     self.features[i] = torch.tensor(self.features[i], dtype=torch.float32)
    
    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)
    
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
        #output = model(images.unsqueeze(0))
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
        train_loss = running_loss / len(dataset)
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

dataset = SimpsonsDataset()

train_set, valid_set = random_split(dataset=dataset, lengths=[35586, 6280])

model = tv.models.resnet18()
model.fc = nn.Linear(in_features=512, out_features=43, bias=True)

train_loader, valid_loader = DataLoader(dataset=train_set, batch_size=3, shuffle=True), DataLoader(dataset=valid_set, batch_size=3, shuffle=True)
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

pbar = tqdm(range(epochs))

for _ in pbar:
    train_loss = train_step()
    valid_loss, valid_acc = valid_step()
    
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    valid_accs.append(valid_acc)
    
    pbar.set_description(f'Avg. train/valid loss: {train_loss:.4f}/{valid_loss:.4f}')
    print('epoch run successfully!')

fig = plt.figure(figsize=(16, 12))

plt.plot(train_losses[1:], label='train')
plt.plot(valid_losses[1:], label='valid')
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Loss')