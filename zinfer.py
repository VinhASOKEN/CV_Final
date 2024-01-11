import os
from model_dino.model import init_model_dino, is_image, one_hot_label
from model_res50.model import init_model_resnet50
from model_vit.model import init_model_vit

import model_dino.config as config_dino
import model_res50.config as config_resnet50
import model_vit.config as config_vit

from torchvision.transforms import Compose
from tqdm import tqdm
import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from PIL import Image


if __name__ == '__main__':

    test_folder = "/data/disk2/vinhnguyen/Dino/test"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    transform = Compose([
                Resize(config_dino.Testing_Config['image_size']),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
            ])

    model_dino = init_model_dino(config_dino.Testing_Config).to(device)
    model_resnet50 = init_model_resnet50(config_resnet50.Testing_Config).to(device)
    model_vit = init_model_vit(config_vit.Testing_Config).to(device)
    model_dino.eval()
    model_resnet50.eval()
    model_vit.eval()

    count_dino = 0
    count_resnet50 = 0
    count_vit = 0

    test_img_lst = []

    for class_folder in tqdm(os.listdir(test_folder)):
        for filename in os.listdir(os.path.join(test_folder, class_folder)):
            if is_image(filename):
                img_path = os.path.join(test_folder, class_folder, filename)
                test_img_lst.append(img_path)
                
                original_image = Image.open(img_path)
                if original_image.mode != 'RGB':
                    original_image = original_image.convert('RGB')
                    
                image = transform(original_image).unsqueeze(0)
                label = one_hot_label(config_dino.Class_Info['num'], config_dino.Class_Info['name'].index(class_folder)).unsqueeze(0)
                
                image, label = image.to(device), label.to(device)
                output_dino = model_dino(image)
                output_resnet50 = model_resnet50(image)
                output_vit = model_vit(image)
                
                check = label.argmax(dim=1, keepdim=True).item()
                
                if output_dino.argmax(dim=1, keepdim=True).item() == check:
                    count_dino += 1
                    
                if output_resnet50.argmax(dim=1, keepdim=True).item() == check:
                    count_resnet50 += 1
                    
                if output_vit.argmax(dim=1, keepdim=True).item() == check:
                    count_vit += 1

    print('Dino Test set accuracy     : {}/{}'.format(count_dino, len(test_img_lst)))
    print('Resnet50 Test set accuracy : {}/{}'.format(count_resnet50, len(test_img_lst)))
    print('ViT Test set accuracy      : {}/{}'.format(count_vit, len(test_img_lst)))