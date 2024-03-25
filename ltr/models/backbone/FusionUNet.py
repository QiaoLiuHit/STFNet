import os
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import ltr.admin.settings as ws_settings
from ltr.util.misc import (NestedTensor, nested_tensor_from_tensor,
                       nested_tensor_from_tensor_2,
                       accuracy)
import ltr.models.neck.position_encoding as pos_encoding
import ltr.models.neck.featurefusion_network as feature_network

# fusion network
class Net(nn.Module):
    def __init__(self, useBN=False):
        super(Net, self).__init__()
        self.setting = ws_settings.Settings()
        self.input_channel = 512
        self.hidden_dim = 256
        self.conv1 = add_conv_stage(1, 32)
        self.conv2 = add_conv_stage(32, 64)
        self.conv3 = add_conv_stage(64, 128)
        self.conv4 = add_conv_stage(128, 256)
        self.conv5 = add_conv_stage(256, 512)

        self.conv4m = add_conv_stage(512+256, 256)
        self.conv3m = add_conv_stage(256+128, 128)
        self.conv2m = add_conv_stage(128+64, 64)
        self.conv1m = add_conv_stage(64+32, 32)

        self.conv0 = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Tanh()
        )

        self.max_pool = nn.MaxPool2d(2)

        self.upsample54 = upsample(256, 256)
        self.upsample43 = upsample(256, 128)
        self.upsample32 = upsample(128, 64)
        self.upsample21 = upsample(64, 32)

        self.align5 = align(512)
        self.align4 = align(256)
        self.align3 = align(128)
        self.align2 = align(64)
        self.align1 = align(32)

        # 5th transformer
        self.pos_encoding5 = pos_encoding.PositionEmbeddingSine(num_pos_feats=self.hidden_dim // 2)
        self.featurefusion_network5 = feature_network.build_featurefusion_network(self.setting)
        self.input_proj5 = nn.Conv2d(self.input_channel, self.hidden_dim, kernel_size=1)


    def forward(self, img_ir, img_vis):
        # encoding img_ir
        conv1_out_ir = self.conv1(img_ir)
        conv2_out_ir = self.conv2(self.max_pool(conv1_out_ir))
        conv3_out_ir = self.conv3(self.max_pool(conv2_out_ir))
        conv4_out_ir = self.conv4(self.max_pool(conv3_out_ir))
        conv5_out_ir = self.conv5(self.max_pool(conv4_out_ir))

        # encoding img_vis
        conv1_out_vis = self.conv1(img_vis)
        conv2_out_vis = self.conv2(self.max_pool(conv1_out_vis))
        conv3_out_vis = self.conv3(self.max_pool(conv2_out_vis))
        conv4_out_vis = self.conv4(self.max_pool(conv3_out_vis))
        conv5_out_vis = self.conv5(self.max_pool(conv4_out_vis))

        # alignment
        conv5_out_vis = self.align5(conv5_out_ir, conv5_out_vis)
        conv4_out_vis = self.align4(conv4_out_ir, conv4_out_vis)
        conv3_out_vis = self.align3(conv3_out_ir, conv3_out_vis)
        conv2_out_vis = self.align2(conv2_out_ir, conv2_out_vis)
        conv1_out_vis = self.align1(conv1_out_ir, conv1_out_vis)

        # 5th layer's transformer 16 * 16
        if not isinstance(conv5_out_ir, NestedTensor):
            conv5_out_ir_nest = nested_tensor_from_tensor(conv5_out_ir)
        if not isinstance(conv5_out_vis, NestedTensor):
            conv5_out_vis_nest = nested_tensor_from_tensor(conv5_out_vis)

        feat_ir5, mask_ir5 = conv5_out_ir_nest.decompose()
        feat_vis5, mask_vis5 = conv5_out_vis_nest.decompose()

        pos_ir5 = self.pos_encoding5(conv5_out_ir_nest)
        pos_vis5 = self.pos_encoding5(conv5_out_vis_nest)

        hs5 = self.featurefusion_network5(self.input_proj5(feat_ir5), mask_ir5, self.input_proj5(feat_vis5), mask_vis5, pos_ir5, pos_vis5)
        # multi-gpu
        # fusion_out5 = hs5.contiguous().view(int(self.setting.batch_size/torch.cuda.device_count()), feat_ir5.shape[2], feat_ir5.shape[3], -1).permute(0, 3, 2, 1)
        # single gpu
        fusion_out5 = hs5.contiguous().view(int(self.setting.batch_size), feat_ir5.shape[2],
                                            feat_ir5.shape[3], -1).permute(0, 3, 2, 1)

        # decoder
        conv5m_out = torch.cat((self.upsample54(fusion_out5), conv4_out_ir, conv4_out_vis), 1)
        conv4m_out = self.conv4m(conv5m_out)
        conv4m_out_ = torch.cat((self.upsample43(conv4m_out), conv3_out_ir, conv3_out_vis), 1)
        conv3m_out = self.conv3m(conv4m_out_)
        conv3m_out_ = torch.cat((self.upsample32(conv3m_out), conv2_out_ir, conv2_out_vis), 1)
        conv2m_out = self.conv2m(conv3m_out_)
        conv2m_out_ = torch.cat((self.upsample21(conv2m_out), conv1_out_ir, conv1_out_vis), 1)
        conv1m_out = self.conv1m(conv2m_out_)
        conv0_out = self.conv0(conv1m_out)

        return conv0_out

# Down sampling module
def add_conv_stage(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.ReLU(),
        nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.ReLU()
    )

# Up sampling module
def upsample(ch_coarse, ch_fine):
    return nn.Sequential(
        nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
        nn.ReLU()
    )

# feature alignment network
class align(nn.Module):
    def __init__(self, input_channel):
        super(align, self).__init__()
        self.input_channel = input_channel
        self.conv1 = nn.Conv2d(2*input_channel, input_channel // 2, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(input_channel // 2, input_channel // 4, 3, padding=1, bias=True)
        self.offset_conv = nn.Conv2d(input_channel // 4, 2*3*3, 3, padding=1, bias=True)
        self.deformable_conv = torchvision.ops.DeformConv2d(input_channel, input_channel, 3, padding=1)

    def forward(self, feat_ir, feat_vis):
        fused_feat = torch.cat((feat_ir, feat_vis), dim=1)
        conv1_out = self.conv1(fused_feat)
        conv2_out = self.conv2(conv1_out)
        offsets = self.offset_conv(conv2_out)
        feat_vis_new = self.deformable_conv(feat_vis,  offsets)
        return feat_vis_new