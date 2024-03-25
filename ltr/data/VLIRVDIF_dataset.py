from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms
import torchvision.transforms.functional as TF

# KAIST dataset class
class VLIRVDIF_dataset(Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.video_dir = os.listdir(self.root_dir)
        img_ir_path = []
        for video in self.video_dir:
            current_path_ir = os.path.join(self.root_dir, video, 'IR')
            img_ir_list = os.listdir(current_path_ir)
            img_ir_list_path = [current_path_ir + '/' + x for x in img_ir_list]
            img_ir_path = img_ir_path + img_ir_list_path
        self.img_ir_path = img_ir_path

    def transform(self, img_ir, img_vis):
        # random crop
        # i, j, h, w = torchvision.transforms.RandomCrop.get_params(img_ir, output_size=(64, 64))
        # img_ir = TF.crop(img_ir, i, j, h, w)
        # img_vis = TF.crop(img_vis, i, j, h, w)
        # reize
        img_ir = TF.resize(img_ir, [256, 256])
        img_vis = TF.resize(img_vis, [256, 256])
        # grayscale
        img_ir = TF.to_grayscale(img_ir)
        img_vis = TF.to_grayscale(img_vis)
        # to tensor
        img_ir = TF.to_tensor(img_ir)
        img_vis = TF.to_tensor(img_vis)
        # Normalize [-1, 1]
        img_ir = TF.normalize(img_ir, 0.5, 0.5)
        img_vis = TF.normalize(img_vis, 0.5, 0.5)

        return img_ir, img_vis

    def __getitem__(self, idx):
        img_ir_name = self.img_ir_path[idx]
        img_vis_name = img_ir_name.replace('IR', 'VIS')
        img_ir = Image.open(img_ir_name)
        img_vis = Image.open(img_vis_name)
        # self-defined transform
        img_ir, img_vis = self.transform(img_ir, img_vis)
        return img_ir, img_vis

    def __len__(self):
        return len(self.img_ir_path)




#root_dir = '/media/qiao/dataset/KAIST'
#KAIST = KAIST_dataset(root_dir)
#img_ir, img_vis = KAIST[0]