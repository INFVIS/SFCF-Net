import sys

print(f"Executing with: {sys.version}")

from net import Restormer_Encoder_level_1, Restormer_Encoder_level_2, Restormer_Decoder, BaseFeatureExtraction, \
    DetailFeatureExtraction
from fusion_net import BaseFusion, Detail_Fusion
from utils.dataset import H5Dataset
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.loss import Fusionloss, cc
import kornia
from kornia.losses import ssim_loss as Loss_ssim
from utils.MM_Loss import Mutual_info_reg
from utils.min_mm import CustomLoss

'''
------------------------------------------------------------------------------
Configure our network
------------------------------------------------------------------------------
'''

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
criteria_fusion = Fusionloss()
model_str = 'SFCF_Net'

# . Set the hyper-parameters for training
num_epochs = 40  # total epoch
epoch_gap = 10  # epoches of Phase I

lr = 1e-4
weight_decay = 0
batch_size = 4
GPU_number = os.environ['CUDA_VISIBLE_DEVICES']
# Coefficients of the loss function
coeff_mse_loss_VF = 1.  # alpha1
coeff_mse_loss_IF = 1.
coeff_decomp = 2.  # alpha2 and alpha4
coeff_tv = 5.

clip_grad_norm_value = 0.01
optim_step = 20
optim_gamma = 0.5

# Model
device = torch.device('cuda')
DIDF_Encoder_level_1 = nn.DataParallel(Restormer_Encoder_level_1()).to(device)
DIDF_Encoder_level_2 = nn.DataParallel(Restormer_Encoder_level_2()).to(device)
DIDF_Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
BaseFuseLayer = nn.DataParallel(BaseFusion(dim=64)).to(device)
DetailFuseLayer = nn.DataParallel(Detail_Fusion(dim=64)).to(device)
Regulation = nn.DataParallel(Mutual_info_reg(64, 64, 8)).to(device)

# optimizer, scheduler and loss function
optimizer1_1 = torch.optim.Adam(
    DIDF_Encoder_level_1.parameters(), lr=lr, weight_decay=weight_decay)
optimizer1_2 = torch.optim.Adam(
    DIDF_Encoder_level_2.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = torch.optim.Adam(
    DIDF_Decoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer3 = torch.optim.Adam(
    BaseFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)
optimizer4 = torch.optim.Adam(
    DetailFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)
optimizer5 = torch.optim.Adam(
    Regulation.parameters(), lr=lr, weight_decay=weight_decay)

scheduler1_1 = torch.optim.lr_scheduler.StepLR(optimizer1_1, step_size=optim_step, gamma=optim_gamma)
scheduler1_2 = torch.optim.lr_scheduler.StepLR(optimizer1_2, step_size=optim_step, gamma=optim_gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)
scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=optim_step, gamma=optim_gamma)
scheduler5 = torch.optim.lr_scheduler.StepLR(optimizer5, step_size=optim_step, gamma=optim_gamma)

MSELoss = nn.MSELoss()
L1Loss = nn.L1Loss()

# data loader
trainloader = DataLoader(H5Dataset(r"data/MSRS_train_imgsize_128_stride_200.h5"),
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=0, drop_last=True)

loader = {'train': trainloader, }
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

'''
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
'''

step = 0
torch.backends.cudnn.benchmark = True
prev_time = time.time()

for epoch in range(num_epochs):
    ''' train '''
    for i, (data_VIS, data_IR) in enumerate(loader['train']):
        data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()
        DIDF_Encoder_level_1.train()
        DIDF_Encoder_level_2.train()
        DIDF_Decoder.train()
        BaseFuseLayer.train()
        DetailFuseLayer.train()
        Regulation.train()

        DIDF_Encoder_level_1.zero_grad()
        DIDF_Encoder_level_2.zero_grad()
        DIDF_Decoder.zero_grad()
        BaseFuseLayer.zero_grad()
        DetailFuseLayer.zero_grad()
        Regulation.zero_grad()

        optimizer1_1.zero_grad()
        optimizer1_2.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()
        optimizer5.zero_grad()

        if epoch < epoch_gap:  # Phase I
            out_enc_level1_V = DIDF_Encoder_level_1(data_VIS)
            out_enc_level1_I = DIDF_Encoder_level_1(data_IR)
            feature_V_B, feature_V_D = DIDF_Encoder_level_2(out_enc_level1_V)
            feature_I_B, feature_I_D = DIDF_Encoder_level_2(out_enc_level1_I)
            latent_loss = Regulation(feature_V_D, feature_I_D)
            data_VIS_hat, _ = DIDF_Decoder(data_VIS, feature_V_B, feature_V_D)
            data_IR_hat, _ = DIDF_Decoder(data_IR, feature_I_B, feature_I_D)

            cc_loss_B = cc(feature_V_B, feature_I_B)
            cc_loss_D = cc(feature_V_D, feature_I_D)
            mse_loss_V = 5 * Loss_ssim(data_VIS, data_VIS_hat, window_size=11, reduction='mean') + MSELoss(data_VIS,
                                                                                                           data_VIS_hat)
            mse_loss_I = 5 * Loss_ssim(data_IR, data_IR_hat, window_size=11, reduction='mean') + MSELoss(data_IR,
                                                                                                         data_IR_hat)

            Gradient_loss = L1Loss(kornia.filters.SpatialGradient()(data_VIS),
                                   kornia.filters.SpatialGradient()(data_VIS_hat))

            loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)

            loss = coeff_mse_loss_VF * mse_loss_V + coeff_mse_loss_IF * \
                   mse_loss_I + coeff_decomp * loss_decomp + coeff_tv * Gradient_loss + latent_loss
            loss.backward()
            nn.utils.clip_grad_norm_(
                DIDF_Encoder_level_1.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DIDF_Encoder_level_2.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                Regulation.parameters(), max_norm=clip_grad_norm_value, norm_type=2)

            optimizer1_1.step()
            optimizer1_2.step()
            optimizer2.step()
            optimizer5.step()


        else:  # Phase II
            out_enc_level1_V = DIDF_Encoder_level_1(data_VIS)
            out_enc_level1_I = DIDF_Encoder_level_1(data_IR)

            feature_V_B, feature_V_D = DIDF_Encoder_level_2(out_enc_level1_V)
            feature_I_B, feature_I_D = DIDF_Encoder_level_2(out_enc_level1_I)

            feature_F_B = BaseFuseLayer(feature_V_B, feature_I_B)
            feature_F_D = DetailFuseLayer(feature_V_D, feature_I_D)
            data_Fuse, feature_F = DIDF_Decoder(data_VIS, feature_F_B, feature_F_D)

            mse_loss_V = 5 * Loss_ssim(data_VIS, data_Fuse, window_size=11, reduction='mean') + MSELoss(data_VIS,
                                                                                                        data_Fuse)
            mse_loss_I = 5 * Loss_ssim(data_IR, data_Fuse, window_size=11, reduction='mean') + MSELoss(data_IR,
                                                                                                       data_Fuse)

            cc_loss_B = cc(feature_V_B, feature_I_B)
            cc_loss_D = cc(feature_V_D, feature_I_D)
            loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)
            fusionloss, _, _ = criteria_fusion(data_VIS, data_IR, data_Fuse)

            loss = fusionloss + coeff_decomp * loss_decomp
            loss.backward()
            nn.utils.clip_grad_norm_(
                DIDF_Encoder_level_1.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DIDF_Encoder_level_2.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                BaseFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DetailFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)

            optimizer1_1.step()
            optimizer1_2.step()
            optimizer2.step()
            optimizer3.step()
            optimizer5.step()

        # Determine approximate time left
        batches_done = epoch * len(loader['train']) + i
        batches_left = num_epochs * len(loader['train']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %.10s"
            % (
                epoch,
                num_epochs,
                i,
                len(loader['train']),
                loss.item(),
                time_left,
            )
        )

    # adjust the learning rate

    scheduler1_1.step()
    scheduler1_2.step()
    scheduler2.step()
    scheduler5.step()
    if not epoch < epoch_gap:
        scheduler3.step()
        optimizer4.step()

    if optimizer1_1.param_groups[0]['lr'] <= 1e-6:
        optimizer1_1.param_groups[0]['lr'] = 1e-6
    if optimizer1_2.param_groups[0]['lr'] <= 1e-6:
        optimizer1_2.param_groups[0]['lr'] = 1e-6
    if optimizer2.param_groups[0]['lr'] <= 1e-6:
        optimizer2.param_groups[0]['lr'] = 1e-6
    if optimizer3.param_groups[0]['lr'] <= 1e-6:
        optimizer3.param_groups[0]['lr'] = 1e-6
    if optimizer4.param_groups[0]['lr'] <= 1e-6:
        optimizer4.param_groups[0]['lr'] = 1e-6
    if optimizer5.param_groups[0]['lr'] <= 1e-6:
        optimizer5.param_groups[0]['lr'] = 1e-6

if True:
    checkpoint = {
        'DIDF_Encoder_level_1': DIDF_Encoder_level_1.state_dict(),
        'DIDF_Encoder_level_2': DIDF_Encoder_level_2.state_dict(),
        'DIDF_Decoder': DIDF_Decoder.state_dict(),
        'BaseFuseLayer': BaseFuseLayer.state_dict(),
        'DetailFuseLayer': DetailFuseLayer.state_dict(),
    }
    torch.save(checkpoint, os.path.join("models/hybrid_fusion_" + timestamp + '.pth'))
