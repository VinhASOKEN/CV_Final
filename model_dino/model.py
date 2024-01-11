import torch
import os
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from PIL import Image
from torchvision.models import *
import torch.nn as nn


def is_image(filename):
    return any(filename.endswith(end) for end in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def one_hot_label(num_class, index_class):
    label = [0 for i in range(num_class)]
    label[index_class] = 1
    return torch.Tensor(label)



"""
Xây dựng hàm DataLoader
"""
class LoadDataset(Dataset):
    def __init__(self, config):
        super(LoadDataset, self).__init__()
        self.config = config
        self.num_class = config['class']['num']
        self.images = []

        for classname in self.config['class']['name']:
            for filename in os.listdir(os.path.join(self.config['path'], classname)):
                if is_image(filename):
                    self.images.append(os.path.join(self.config['path'], classname, filename))

        self.transform = Compose([
            Resize(self.config['image_size']),
            transforms.RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        original_image = Image.open(self.images[index])
        if original_image.mode != 'RGB':
            original_image = original_image.convert('RGB')
            
        image = self.transform(original_image)

        class_name = self.images[index].split('/')[-2]
        label = one_hot_label(self.num_class, self.config['class']['name'].index(class_name))

        return image, label

    def __len__(self):
        return len(self.images)


"""
Xây dựng hàm loss
"""
def intit_loss():
    loss = torch.nn.CrossEntropyLoss()
    return loss



"""
Xây dựng hàm Optimizer
"""
def init_optimizeer(model, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    return optimizer


"""
Xây dựng model
"""
backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
num_classes = 30

class ClsfModel(nn.Module):
    def __init__(self):
        super(ClsfModel, self).__init__()

        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(384, num_classes),
            nn.Softmax(dim=1)
          )

    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output


def init_model_dino(config):
    model = ClsfModel()
    if config['load_checkpoint'] is not None:
        print("Loading Dino model checkpoint from " + config['load_checkpoint'] + ' ...')
        model.load_state_dict(torch.load(config['load_checkpoint']))
    
    return model

