import random

from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

# LasHeR dataset class
class LasHeR_dataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.video_dir = os.listdir(self.root_dir)
        img_ir_path = []
        for video in self.video_dir:
            current_path_ir = os.path.join(self.root_dir, video, 'infrared')
            img_ir_list = os.listdir(current_path_ir)
            img_ir_list_path = [current_path_ir + '/' + x for x in img_ir_list]
            img_ir_path = img_ir_path + img_ir_list_path
        self.img_ir_path = img_ir_path

    def __getitem__(self, idx):
        img_ir_name = self.img_ir_path[idx]
        img_vis_name = img_ir_name.replace('infrared', 'visible')
        img_ir = Image.open(img_ir_name)
        img_vis = Image.open(img_vis_name)
        if self.transform:
            # make same transform to ir and vis img
            seed = np.random.randint(21474836747)
            random.seed(seed)
            img_ir = self.transform(img_ir)
            random.seed(seed)
            img_vis = self.transform(img_vis)
        return img_ir, img_vis

    def __len__(self):
        return len(self.img_ir_path)

#root_dir = '/media/qiao/dataset/LasHeR'
#LasHeR = LasHeR_dataset(root_dir)
#img_ir, img_vis = LasHeR[0]