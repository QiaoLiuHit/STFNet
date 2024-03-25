import os
import torch
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from ltr.data.KAIST_dataset import KAIST_dataset
from ltr.data.LLVIP_dataset import LLVIP_dataset
from ltr.data.M3FD_dataset import M3FD_dataset
from ltr.data.VLIRVDIF_dataset import VLIRVDIF_dataset
import admin.settings as ws_settings
from ltr.models.loss import pytorch_msssim, loss_freq
from ltr.models.backbone import FusionUNet
from ltr import MultiGPU
import matplotlib.pyplot as plt


# config some vars
settings = ws_settings.Settings()

# prepare training dataset, please config your dataset path
root_kaist_dir = '/media/qiao/dataset/ImageFusionDatasets/KAIST'
root_LLVIP_train_dir = '/media/qiao/dataset/ImageFusionDatasets/LLVIP/infrared/train'
root_LLVIP_test_dir = '/media/qiao/dataset/ImageFusionDatasets/LLVIP/infrared/test'
root_M3FD_dir = '/media/qiao/dataset/ImageFusionDatasets/M3FD/Detection/IR'
root_MSRS_train_dir = '/media/qiao/dataset/ImageFusionDatasets/MSRS/train/IR'
root_MSRS_test_dir = '/media/qiao/dataset/ImageFusionDatasets/MSRS/test/IR'
root_RoadScene_dir = '/media/qiao/dataset/ImageFusionDatasets/RoadScene/IR'
root_VLIRVDIF_dir = '/media/qiao/dataset/ImageFusionDatasets/VLIVDIF'
# transform = transforms.Compose([transforms.Resize((256, 256)), transforms.Grayscale(), transforms.ToTensor()])
KAIST = KAIST_dataset(root_kaist_dir)
LLVIP_train = LLVIP_dataset(root_LLVIP_train_dir)
M3FD = M3FD_dataset(root_M3FD_dir)
MSRS_train = M3FD_dataset(root_MSRS_train_dir)
VLIRVDIF = VLIRVDIF_dataset(root_VLIRVDIF_dir)

TraningDatset = KAIST + LLVIP_train + M3FD + MSRS_train + VLIRVDIF
train_dataloader = DataLoader(TraningDatset, batch_size=settings.batch_size, drop_last=True, shuffle=True)

# Create STFNet network model
model = FusionUNet.Net()
device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Wrap the network for multi GPU training, default use single gpu
if settings.multi_gpu:
    model = MultiGPU(model, dim=0)

# write log
write = SummaryWriter("training_log")

# define loss
loss_mse = torch.nn.MSELoss(reduction='mean')
loss_msssim = pytorch_msssim.msssim
loss_l1 = torch.nn.L1Loss(reduction='mean')
loss_bce = torch.nn.BCELoss(reduction='mean')
gauss_size = 3
radius = 21
gauss_kernel = loss_freq.get_gaussian_kernel(gauss_size).cuda()
mask_h, mask_l = loss_freq.decide_circle(r=radius, N=int(settings.batch_size / 2), L=256)
mask_h, mask_l = mask_h.cuda(), mask_l.cuda()

# define optimizer
learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

# training epoch
epoch = 10
# if show fused images
show_fused_image = True
# resume training to load checkpoint
Resume = False
if Resume:
    path_checkpoint = "./checkpoints/STFNet_20.pth"
    checkpoints = torch.load(path_checkpoint)
    model.load_state_dict(checkpoints['net'])
    optimizer.load_state_dict(checkpoints['optimizer'])
    scheduler.load_state_dict(checkpoints['lr_scheduler'])
    start_epoch = checkpoints['epoch']

for i in range(1, epoch+1):
    total_train_step = 0
    print("training on {} epoch starting".format(i+1))
    model.train()
    for data in train_dataloader:
        img_ir, img_vis = data
        img_ir = Variable(img_ir.cuda())
        img_vis = Variable(img_vis.cuda())
        fusion_out = model(img_ir, img_vis).to(device)
        # training fusion network
        optimizer.zero_grad()
        # loss function
        fusion_out_gradient = abs(loss_freq.gradient(fusion_out))
        img_vis_gradient = abs(loss_freq.gradient(img_vis))
        img_ir_gradient = abs(loss_freq.gradient(img_ir))
        loss_grad = loss_l1(fusion_out_gradient, torch.maximum(img_vis_gradient, img_ir_gradient))
        loss_intensity = loss_l1(fusion_out-fusion_out_gradient, torch.maximum((img_ir-img_ir_gradient), (img_vis-img_vis_gradient)))
        loss_frequency = loss_grad + loss_intensity

        # loss for structure similarity
        loss_structure_vis = 1 - loss_msssim(fusion_out, img_vis, normalize=True)
        loss_structure_ir = 1 - loss_msssim(fusion_out, img_ir, normalize=True)
        loss_structure = (loss_structure_vis + loss_structure_ir) * 0.5

        # loss for fft high-frequency
        loss_fft = 0.5 * loss_freq.fft_L1_loss_mask(img_ir, fusion_out, mask_h) + 0.5 * loss_freq.fft_L1_loss_mask(img_vis, fusion_out, mask_h)

        # total loss
        loss = loss_structure + 10 * loss_frequency + loss_fft
        # loss = 10 * loss_structure + 20 * loss_intensity
        loss.backward()
        optimizer.step()

        # print training information
        total_train_step = total_train_step + 1
        if total_train_step % 10 == 0:
            print("training with {} epoch and {} times and total loss is {}, loss_structure is {}, "
                  "loss_frequecny is {}, loss_fft is {}".format(i, total_train_step, round(loss.item(), 3), round(loss_structure.item(), 3),
                                                                round(loss_frequency.item(), 3), round(loss_fft.item(), 3)))
            write.add_scalar("training_loss", loss.item(), total_train_step)
        # show fused images
        if show_fused_image:
            arrayImg_ir = img_ir[1].cpu().numpy()
            arrayImg_vis = img_vis[1].cpu().numpy()
            arrayImg_fused = fusion_out[1].cpu().detach().numpy()
            fig = plt.figure("ImageV2")
            plt.subplot(131)
            plt.imshow(arrayImg_ir.transpose(1, 2, 0), cmap='gray')
            plt.subplot(132)
            plt.imshow(arrayImg_vis.transpose(1, 2, 0), cmap='gray')
            plt.subplot(133)
            plt.imshow(arrayImg_fused.transpose(1, 2, 0), cmap='gray')
            plt.pause(0.0000001)
            fig.clf()
    scheduler.step()
    # save checkpoint
    checkpoints = {
        "net": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": scheduler.state_dict(),
        "epoch": i
    }
    torch.save(checkpoints, "./checkpoints/STFNet_{}.pth".format(i))
write.close()



