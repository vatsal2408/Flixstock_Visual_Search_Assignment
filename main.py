import os
from os.path import join, splitext, basename
import torch
from args import arguments
import cv2
from similar_bottoms import *
from utilities.ugly_code import image_out

if __name__ == "__main__":
    args = arguments()
    
    txtfile, embed_flag = img_to_txt(args)
    # load/create image feature embeddings and model
    embedding_path = os.path.join(args.root_dir, 'embeddings\{0}_embeddings.pt'.format(args.model))
    model, embeddings = load_embed(args, embedding_path, txtfile, embed_flag)

    # Compare query image to the images in dataset
    orig, indexes = similar_images(args, model, embeddings)
    
    # Create grid of similar images for visualization
    input_img = cv2.resize(orig, None, fx=2, fy=2)
    out = image_out(args, input_img, indexes)
    save_path = join(args.root_dir, "results\{0}.jpg".format(splitext(basename(args.input_image))[0]))
    cv2.imwrite(save_path, out)
    print(colored("Results saved at: " + save_path, "green"))