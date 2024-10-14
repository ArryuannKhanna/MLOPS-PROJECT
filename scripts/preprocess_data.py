import os
import torchvision.transforms as transforms
from PIL import Image

def preprocess_data(input_dir,output_dir):
    os.makedirs(output_dir,exist_ok=True)
    transform = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()])

    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir,img_name)
        img = Image.open(img_path)
        img = transform(img)
        img.save(os.path.join(output_dir,img_name))

if __name__ == '__main__':
    preprocess_data('raw_images','preprocess_images')
