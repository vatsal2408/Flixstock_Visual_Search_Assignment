import cv2
import os


def remove_dummy(image):
    '''
    takes 4 channel image as input and returns a 3 channel 
    image (RGB) with mask multiplied to every channel
    '''
    mask = image[:, :, -1]
    img = image[:, :, :3]
    image = cv2.bitwise_and(img, img, mask=mask)
    return image


def Diff(li1, li2):
    return list(set(li1) - set(li2)) + list(set(li2) - set(li1))


def img_to_txt(args):
    '''
    takes folder path as input and returns a txt with images names in it
    '''
    embed_flag = True
    new_list = os.listdir(args.img_folder)
    txt_path = os.path.join(args.root_dir, "img_names.txt")
    if os.path.exists(txt_path):
        old_list = open(txt_path, 'r').read().splitlines()
        if len(Diff(old_list, new_list)) == 0:
            img_list = old_list
            embed_flag = False
        else:
            img_list = new_list
    else:
        img_list = new_list
    
    if embed_flag:
        with open(txt_path, 'w') as f:
            for item in img_list:
                f.write("{0}\n".format(item))

    return os.path.join(args.root_dir, "img_names.txt"), embed_flag