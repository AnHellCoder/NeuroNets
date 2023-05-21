import os
import torch
import torchvision as tv
import pytorch_lightning as pl

from PIL import Image
from pprint import pprint
from torch.utils.data import DataLoader, Dataset
from road_model import RoadModel

class RoadsTrainset(Dataset):
    def __init__(self, transforms = tv.transforms.ToTensor()) -> None:
        super().__init__()

        self.image_path = './images/train/'
        self.label_path = './labels/train/'
        
        self.images = os.listdir(self.image_path)
        self.labels = os.listdir(self.label_path)

        for i in range(len(self.images)):
            self.images[i] = transforms(Image.open(self.image_path + self.images[i]))
            self.labels[i] = transforms(Image.open(self.label_path + self.labels[i]))

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        #return super().__getitem__(index)
        return self.images[index], self.labels[index]

class RoadsValidset(Dataset):
    def __init__(self, transforms = tv.transforms.ToTensor()) -> None:
        super().__init__()

        self.image_path = './images/val/'
        self.label_path = './labels/val/'

        self.images = os.listdir(self.image_path)
        self.labels = os.listdir(self.label_path)

        for i in range(len(self.images)):
            self.images[i] = transforms(Image.open(self.image_path + self.images[i]))
            self.labels[i] = transforms(Image.open(self.label_path + self.labels[i]))

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        #return super().__getitem__(index)
        return self.images[index], self.labels[index]
    
class RoadTestset(Dataset):
    def __init__(self, transforms = tv.transforms.ToTensor()) -> None:
        super().__init__()

        self.image_path = './images/test/'
        self.label_path = './labels/test/'

        self.images = os.listdir(self.image_path)
        self.labels = os.listdir(self.label_path)

        for i in range(len(self.images)):
            self.images[i] = transforms(Image.open(self.image_path + self.images[i]))
            self.labels[i] = transforms(Image.open(self.label_path + self.labels[i]))

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        #return super().__getitem__(index)
        return self.images[index], self.labels[index]
    
def create_sets():
    if 'train_dataset' in os.listdir('./'):
        train_dataset = torch.load('train_dataset')
    else:
        train_dataset = RoadsTrainset(transforms=transforms)
        torch.save(train_dataset, 'train_dataset')

    if 'valid_dataset' in os.listdir('./'):
        valid_dataset = torch.load('valid_dataset')
    else:
        valid_dataset = RoadsValidset(transforms=transforms)
        torch.save(valid_dataset, 'valid_dataset')

    if 'test_set' in os.listdir('./'):
        test_dataset = torch.load('test_dataset')
    else:
        test_dataset = RoadTestset(transforms=transforms)
        torch.save(test_dataset, 'test_dataset')

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=10, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=10, shuffle=True)

    return train_dataloader, valid_dataloader, test_dataloader

# def train_step():
#     model.train()
 
#     running_loss = 0.
#     for images, labels in train_dataloader:
#         optimizer.zero_grad()

#         output = model(images)

#         loss = model.loss_fn(output, labels)

#         optimizer.step()

#         running_loss += loss

#     with optimizer.no_grad():
#         train_loss = running_loss / len(train_dataloader.dataset)
#     return train_loss.item()

transforms = tv.transforms.Compose([tv.transforms.ToTensor(),
                                    tv.transforms.Resize([160, 160])])

train_dataloader, valid_dataloader, test_dataloader = create_sets()
model = RoadModel("FPN", "resnet34", in_channels=3, out_classes=1)

trainer = pl.Trainer( 
    max_epochs=5,
)

trainer.fit(
    model, 
    train_dataloaders=train_dataloader, 
    val_dataloaders=valid_dataloader,
)
# optimizer = model.configure_optimizers()

# train_dataset, valid_dataset, test_dataset = create_sets()

# train_dataloader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True)
# valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=10, shuffle=True)