from torch.optim import Adam
from lib.loss import DiceCE, FocalLoss, DiceFocal, Dice_IoULoss_binary, Dice_IoULoss
import torch
from torch import nn, sigmoid
import logging
from lib.logger import get_logger
from torch.utils.data import DataLoader
import os
from lib.loader import CustomDataset_3d_gray, CustomDataset2_gray,CustomDataset, CustomDataset_3d_gray_out_1
import datetime
from lib.unet_model import UNet
import numpy as np
from utils import Report, transfer_weights
from lib.c_metrics import runningScore, averageMeter, get_score_from_all_slices_cherhoo
from datetime import datetime
from lib.DUnet import DUnet

logging.basicConfig(format='%(asctime)s-<%(funcName)s>-[line:%(lineno)d]-%(levelname)s: %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # define logging print level
logger.setLevel(logging.WARNING)  # define logging print level



def get_labels(test_path_root):
    print_interval_label = 100
    test_img_path = test_path_root + '/imgs/'
    test_mask_path = test_path_root + '/masks/'

    test_data_list = [test_img_path + i for i in os.listdir(test_img_path)]
    test_mask_list = [test_mask_path + i for i in os.listdir(test_mask_path)]

    test_data = DataLoader(CustomDataset2_gray(test_data_list, test_mask_list), batch_size=8, shuffle=True, num_workers=4)

    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_list = []
    for i, pack in enumerate(test_data):
        if (i + 1) % print_interval_label == 0:
            fmt_str = "Iter [{:d}/{:d}]"
            print_str = fmt_str.format(i + 1, len(test_data))
            print(print_str)
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        images, labels = pack
        labels = labels.to(device_).long()
        labels = labels.cpu().detach().numpy()
        labels_ = []
        for ii in range(len(labels)):
            label = labels[ii].transpose(1, 2, 0)
            labels_.append(label)
        labels_ = torch.tensor(labels_)
        if i == 0:
            label_list = labels_
        else:
            label_list = torch.cat([label_list, labels_], 0)

    return label_list



def convert_label_224(outputs):
    pred = np.zeros((224, 224, 1))
    for i in range(224):
        pred[i] = np.argmax(outputs[i], axis=-1).reshape((-1, 1))
    return pred


def get_3d_img_list(path, depth):
    names_3d_list = []
    img_depth = depth
    for name in os.listdir(path):
        name_tmps = name.split('_')
        name_id = name.split('_')[3].split(".")[0]
        name_id = int(name_id)

        num_i = name_id
        if num_i <= 189 - img_depth + 1:
            names_3d = []
            for d in range(img_depth):
                name_3d = name_tmps[0] + "_" + name_tmps[1] + "_" + name_tmps[2] + "_{:03d}".format(name_id + d) + ".png"
                names_3d.append(os.path.join(path, name_3d))
            names_3d_list.append(names_3d)
        #break
    return names_3d_list

def test(model, device):
    img_depth = 4
    img_depth_out_num = 2
    logdir = 'D:/soyeon_vision2022/01.Object_segmentation_start/03.code/old/ATLAS_models/DUnet_in2d_out2d_test'
    logger = get_logger(logdir)

    logger.info('test_strat')
    logger.info(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    test_img_path = test_path_root + '/imgs/'
    test_mask_path = test_path_root + '/masks/'

    test_data_list = get_3d_img_list(test_img_path, img_depth)
    test_mask_list = get_3d_img_list(test_mask_path, img_depth)

    #test_data_list = [test_img_path + i for i in os.listdir(test_img_path)]
    #test_mask_list = [test_mask_path + i for i in os.listdir(test_mask_path)]

    test_data = DataLoader(CustomDataset_3d_gray_out_1(test_data_list, test_mask_list, img_h=192, img_w=192, out_num=img_depth_out_num),batch_size=8, shuffle=True, num_workers=4)

    n_classes = 2
    running_metrics_val = runningScore(n_classes)

    mean_ctg = {}
    #num = 1
    print_interval = 10
    output_list = []
    for i, pack in enumerate(test_data):
        images, labels = pack
        images = images.to(device)
        outputs = model(images)
        """
        outputs = outputs.cpu().detach().numpy()
        outputs_ = []
        for ii in range(len(outputs)):
            output = outputs[ii].transpose(1, 2, 0)
            output = convert_label_224(output)
            outputs_.append(output)
        outputs_ = torch.tensor(outputs_)

        if i == 0:
            output_list = outputs_
        else:
            output_list = torch.cat([output_list, outputs_], 0)
        """
        pred = outputs.data.max(1)[1].cpu().numpy()
        gt = labels.data.cpu().numpy()

        running_metrics_val.update(gt, pred)

    if (i + 1) % print_interval == 0:
        fmt_str = "Iter [{:d}/{:d}]"
        print_str = fmt_str.format(i + 1, len(test_data))
        print(print_str)
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    #print('label_list_global: ',len(label_list_global))
    #print('output_list: ',len(output_list))
    #scores = get_score_from_all_slices_cherhoo(labels=label_list_global, predicts=output_list)
    score, class_iou, class_dice = running_metrics_val.get_scores()

    for k, v in score.items():
        logger.info("score {}: {}".format(k, v))
        #writer.add_scalar("val_metrics/{}".format(k), v, start_iter + 1)

    for k, v in class_iou.items():
        logger.info("class_iou: {}: {}".format(k, v))
        #writer.add_scalar("val_metrics/cls_{}".format(k), v, start_iter + 1)

    for k, v in class_dice.items():
        logger.info("class_dice: {}: {}".format(k, v))
        #writer.add_scalar("val_metrics/cls_{}".format(k), v, start_iter + 1)

    """
    mean_score = {}
    for key in scores.keys():
        mean_score[key] = np.mean(scores[key])
        if key not in mean_ctg.keys():
            mean_ctg[key] = mean_score[key]
        else:
            mean_ctg[key] += mean_score[key]

    ##################################
    json_all = {
        "scores": scores,
        "mean_score": mean_score
    }
    #logging.info(mean_score)
    
    return str(json_all['mean_score'])
    """


if __name__ == '__main__':
    test_path_root = 'D:/soyeon_vision2022/01.Object_segmentation_start/01.data/ATLAS_test/v2_png_1'
    model_path = "D-UNet_in2d_out2dD-UNet_ATLAS_test_v1_Dice_IoULoss_4/D-UNet_ATLAS_test_best_model.pth"
    img_depth = 4

    net = DUnet(in_channels=img_depth).cuda()
    transfer_weights(net, model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.cuda()
    net.train(False)
    torch.no_grad()

    #label_list_global = get_labels(test_path_root)

    #test(test_path_root)

    test(net, device)

