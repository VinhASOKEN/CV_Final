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
    test_folder = "/data/disk2/vinhnguyen/Dino/examples/test"
    result_folder = "/data/disk2/vinhnguyen/Dino/examples/results"
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
    
    for img_name in tqdm(os.listdir(test_folder)):
        img_path = os.path.join(test_folder, img_name)
        
        original_image = Image.open(img_path)
        if original_image.mode != 'RGB':
            original_image = original_image.convert('RGB')

        image = transform(original_image).unsqueeze(0)
        image = image.to(device)
        
        output_dino = model_dino(image)
        output_resnet50 = model_resnet50(image)
        output_vit = model_vit(image)
        
        class_name_dino_pred = config_dino.Class_Info['name'][output_dino.argmax(dim=1, keepdim=True).item()]
        class_name_res50_pred = config_resnet50.Class_Info['name'][output_resnet50.argmax(dim=1, keepdim=True).item()]
        class_name_vit_pred = config_vit.Class_Info['name'][output_vit.argmax(dim=1, keepdim=True).item()]

        with open(os.path.join(result_folder, f"{img_name}.txt"), 'w') as txt_file:
            txt_file.write(f"Dino     : {class_name_dino_pred}\n")
            txt_file.write(f"ResNet50 : {class_name_res50_pred}\n")
            txt_file.write(f"ViT      : {class_name_vit_pred}\n")
    