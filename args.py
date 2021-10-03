import argparse
import os

def arguments():
    parser = argparse.ArgumentParser(description='Argument parser')

    parser.add_argument('--root_dir', type=str, default="data", help="Path of data folder in project directory")
    parser.add_argument('--img_folder', type=str, default="data/images")
    parser.add_argument('--model', type=str, default='inception_v3',
                        choices=['inception_v3', 'resnet18', 'resnet34', 'vgg16', 'vgg19'])
    parser.add_argument('--input_image', type=str)
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'])
    parser.add_argument('--metric', type=str, default='euclidian', choices=['euclidian', 'cosine'])
    parser.add_argument('--num_images', type=int, default=10)
    parser.add_argument('--batch_size',type=int, default=64)
    # parser.add_argument('--pretrained', type=bool, default=True)
    
    args = parser.parse_args()
    return args