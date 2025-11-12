from net import Restormer_Encoder_level_1, Restormer_Encoder_level_2, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction
from new_fusion_net import BaseFusion, Detail_Fusion
import os
import numpy as np
from utils.Evaluator import Evaluator
import torch
import torch.nn as nn
from utils.img_read_save import img_save,image_read_cv2
import warnings
import logging
import cv2
import re
import time
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ckpt_path= r"models/SFCF_Net.pth"

def count_parameters_in_MB(model):
    """
    计算模型的参数数量（以MB为单位）
    """
    return sum(p.numel() for p in model.state_dict().values()) / 1e6

def natural_sort_key(name):
    # 按照文件名中的数字顺序排序
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', name)]

for dataset_name in ["Test_TNO"]:
    print("\n"*2+"="*80)
    model_name="CDDFuse    "
    print("The test result of "+dataset_name+' :')
    # test_folder = os.path.join('.', 'new_test_data', dataset_name)
    test_folder = os.path.join('D:\\new_test_data', dataset_name)
    test_out_folder = os.path.join('.', 'new_test_result', dataset_name)
    # test_out_folder = os.path.join('C:\\Users\panda\Desktop\MRI_compare\\new_cdd')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Encoder_level_1 = nn.DataParallel(Restormer_Encoder_level_1()).to(device)
    Encoder_level_2 = nn.DataParallel(Restormer_Encoder_level_2()).to(device)
    Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
    BaseFuseLayer = nn.DataParallel(BaseFusion(dim=64,num_heads=4)).to(device)
    DetailFuseLayer = nn.DataParallel(Detail_Fusion(dim=64)).to(device)

    Encoder_level_1.load_state_dict(torch.load(ckpt_path)['DIDF_Encoder_level_1'])
    Encoder_level_2.load_state_dict(torch.load(ckpt_path)['DIDF_Encoder_level_2'])
    Decoder.load_state_dict(torch.load(ckpt_path)['DIDF_Decoder'])
    BaseFuseLayer.load_state_dict(torch.load(ckpt_path)['BaseFuseLayer'])
    DetailFuseLayer.load_state_dict(torch.load(ckpt_path)['DetailFuseLayer'])
    Encoder_level_1.eval()
    Encoder_level_2.eval()
    Decoder.eval()
    BaseFuseLayer.eval()
    DetailFuseLayer.eval()

    with torch.no_grad():
        ir_list = sorted(os.listdir(os.path.join(test_folder, "IR")), key=natural_sort_key)
        vis_list = sorted(os.listdir(os.path.join(test_folder, "VIS")), key=natural_sort_key)
        for idx, (img_name_ir, img_name_vis) in enumerate(zip(ir_list, vis_list), start=1):
            data_IR = image_read_cv2(os.path.join(test_folder, "IR", img_name_ir), mode='GRAY')[
                          np.newaxis, np.newaxis, ...] / 255.0
            data_VIS = image_read_cv2(os.path.join(test_folder, "VIS", img_name_vis), mode='GRAY')[
                           np.newaxis, np.newaxis, ...] / 255.0
            vis_bgr = cv2.imread(os.path.join(test_folder, "VIS", img_name_vis))

            vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
            vis_ycrcb = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2YCrCb)
            Cr = vis_ycrcb[:, :, 1]
            Cb = vis_ycrcb[:, :, 2]

            data_IR, data_VIS = torch.FloatTensor(data_IR), torch.FloatTensor(data_VIS)
            data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()

            feature_V_level1 = Encoder_level_1(data_VIS)
            feature_I_level1 = Encoder_level_1(data_IR)

            feature_V_B, feature_V_D = Encoder_level_2(feature_V_level1)
            feature_I_B, feature_I_D = Encoder_level_2(feature_I_level1)
            feature_F_B = BaseFuseLayer(feature_V_B, feature_I_B)
            feature_F_D = DetailFuseLayer(feature_V_D, feature_I_D)
            data_Fuse, _ = Decoder(base_feature=feature_F_B, detail_feature=feature_F_D, inp_img=None)

            data_Fuse = (data_Fuse - torch.min(data_Fuse)) / (torch.max(data_Fuse) - torch.min(data_Fuse))
            fi = np.squeeze((data_Fuse * 255).cpu().numpy()).astype(np.uint8)
            ycrcb_fused = cv2.merge([fi, Cr, Cb])
            fused_rgb = cv2.cvtColor(ycrcb_fused, cv2.COLOR_YCrCb2RGB)
            img_save(fused_rgb, idx, test_out_folder)

    print("finish")
