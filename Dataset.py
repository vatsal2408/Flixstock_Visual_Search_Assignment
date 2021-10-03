from torch.utils.data import Dataset
import cv2
import os
import numpy as np
from utilities.utils import *


class MyDataset(Dataset):
    def __init__(self, args, txtfile, transform=None):
        self.args = args
        
        self.names = open(txtfile, 'r').read().splitlines()
        self.transform = transform
    
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.args.img_folder, self.names[index])
        img = cv2.imread(img_path, -1)
        img = remove_dummy(img)
        img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)
        if self.transform:
            image = self.transform(img)
        return image