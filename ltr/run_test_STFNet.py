import os
import torch
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.utils import save_image
import admin.settings as ws_settings
from ltr.models.backbone import FusionUNet

# config some vars
setting = ws_settings.Settings()
# input size is changeable, e.g., 256, 512, 1024
input_size = 512
# method name
method = 'SSTUFourier'
# test dataset
dataset = 'TNO'
# if show fused image
show_fused_image = True
# set gpu id
torch.cuda.set_device(0)
# source image path
root_path = './images/' + dataset + '/'
# load all images
img_list = os.listdir(os.path.join(root_path, 'ir'))
# test for multiple epoch's checkpoint or single checkpoint
for i in range(3, 4):
    global start_time
    start_time = time.time()
    # path for save fused image
    fused_path = './FusedImg/' + method + '_' + str(i) + '_' + dataset + '/'
    # load model
    model = FusionUNet.Net()
    model_path = './checkpoints/STFNet_' + str(i) + '.pth'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['net'])
    # model para. size
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))
    # set for test
    model.cuda()
    model.eval()
    model.setting.batch_size = 1
    # load all images
    for img in img_list:
        img_ir_path = os.path.join(root_path, 'ir', img)
        img_vis_path = img_ir_path.replace('ir/', 'vi/')
        # read ir and vis images
        img_ir = Image.open(img_ir_path)
        img_vis = Image.open(img_vis_path)
        ori_size = img_ir.size

        # transform
        transform = T.Compose([T.Resize((input_size, input_size)), T.Grayscale(), T.ToTensor(), T.Normalize(0.5, 0.5)])
        img_ir = transform(img_ir)
        img_vis = transform(img_vis)

        img_ir = img_ir.view(1, 1, input_size, input_size).cuda()
        img_vis = img_vis.view(1, 1, input_size, input_size).cuda()
        # test
        with torch.no_grad():
            out = model(img_ir, img_vis)
        out = out.view(1, input_size, input_size)
        out = out / 2 + 0.5

        transforms = torch.nn.Sequential(
            T.Resize((ori_size[1], ori_size[0]))
        )
        fusion_img = transforms(out)

        # show fused image
        if show_fused_image:
            arrayImg_ir = torch.squeeze(img_ir).cpu().numpy()
            arrayImg_vis = torch.squeeze(img_vis).cpu().numpy()
            arrayImg_fused = torch.squeeze(out).cpu().detach().numpy()
            fig = plt.figure("Image")
            plt.subplot(131)
            plt.imshow(arrayImg_ir, cmap='gray')
            plt.subplot(132)
            plt.imshow(arrayImg_vis, cmap='gray')
            plt.subplot(133)
            plt.imshow(arrayImg_fused, cmap='gray')
            plt.pause(2)
            fig.clf()
        if not os.path.exists(fused_path):
            os.mkdir(fused_path)
        save_image(fusion_img, fused_path + img)
    # set show_fused_image is false to calculate fps
    total_time = time.time() - start_time
    fps = len(img_list) / total_time
    print('FPS is : {}'.format(fps))








