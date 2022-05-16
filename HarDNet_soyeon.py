
import logging

from torch.optim import Adam
from lib.loss import DiceCE, FocalLoss, DiceFocal, Dice_IoULoss_binary, Dice_IoULoss
import torch
from torch.utils.data import DataLoader
import os
from lib.loader import CustomDataset_3d_gray, CustomDataset2_gray,CustomDataset
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lib.logger import get_logger
from tqdm import tqdm
from lib.pytorchtools import EarlyStopping
from lib.c_metrics import runningScore, averageMeter, get_score_from_all_slices_cherhoo
import datetime
from lib.unet_model import UNet
from lib.c_hardnet_swin_ende_decoder import hardnet
import numpy as np


#from lib.trans_hardnet import TransHarDNet

import logging
logging.basicConfig(format='%(asctime)s-<%(funcName)s>-[line:%(lineno)d]-%(levelname)s: %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # define logging print level
logger.setLevel(logging.WARNING)  # define logging print level
# import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



#########################################################################soyeon



def get_labels():

    test_path_root = 'D:/soyeon_vision2022/01.Object_segmentation_start/01.data/ATLAS_test/v2_png'
    test_datas = {}
    ctg = 'val'

    test_img_path = test_path_root + '/imgs/'
    test_mask_path = test_path_root + '/masks/'

    test_data_list = [test_img_path + i for i in os.listdir(test_img_path)]
    test_mask_list = [test_mask_path + i for i in os.listdir(test_mask_path)]

    test_data = DataLoader(CustomDataset2_gray(test_data_list, test_mask_list), batch_size=8, shuffle=True, num_workers=4)


    #logging.info("val_data_list({}): {}  ::: {}".format(len(test_data_list),  test_data_list[0]))

    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_list = []
    for i, pack in enumerate(test_data):
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

    return test_datas, label_list


def convert_label_224(outputs):
    pred = np.zeros((224, 224, 1))
    for i in range(224):
        pred[i] = np.argmax(outputs[i], axis=-1).reshape((-1, 1))
    return pred

def test(model, device, label_list ):
    test_path_root = 'D:/soyeon_vision2022/01.Object_segmentation_start/01.data/ATLAS_test/v2_png'
    ctg = 'val'
    test_img_path = test_path_root + '/imgs/'
    test_mask_path = test_path_root + '/masks/'

    test_data_list = [test_img_path + i for i in os.listdir(test_img_path)]
    test_mask_list = [test_mask_path + i for i in os.listdir(test_mask_path)]

    test_datas = DataLoader(CustomDataset2_gray(test_data_list, test_mask_list), batch_size=8, shuffle=True,
                           num_workers=4)

    mean_ctg = {}
    num = 1
    criterion = DiceCE()
    loss_all = 0
    loss_n = 0

    output_list = []
    for i, pack in enumerate(test_datas):
        print(i)
        images, labels = pack
        #images = images.to(device)
        #outputs = model(images)
        #outputs = outputs.cpu().detach().numpy()
        #outputs_ = []
        #for ii in range(len(outputs)):
        #    output = outputs[ii].transpose(1, 2, 0)
        #    output = convert_label_224(output)
        #    outputs_.append(output)
        #outputs_ = torch.tensor(outputs_)
        #
        #if i == 0:
        #    output_list = outputs_
        #else:
        #    output_list = torch.cat([output_list, outputs_], 0)

        images_val = images.to(device)
        labels_val = labels.to(device).long().squeeze(1)

        outputs = model(images_val)
        val_loss = criterion(outputs, labels_val)

        pred = outputs.data.max(1)[1].cpu().numpy()
        gt = labels_val.data.cpu().numpy()

        writer.add_scalar("loss/val_loss", val_loss)
        logger.info("Val Loss: %.8f" % (val_loss))

    scores = get_score_from_all_slices_cherhoo(labels=pred, predicts=gt)

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
    logging.info(mean_score)
    #return str(json_all['mean_score'])
    return json_all









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

def train():
    batch_size = 8
    n_classes = 2
    learning_rate = 0.001
    img_depth = 1
    model_name = "HarDNet_ATLAS_test"  ##
    patience_scheduler = 5
    patience_early_stop = 10

    start_iter = 0
    num_epochs = 49
    ## save model log and check points
    #'D:/soyeon_vision2022/01.Object_segmentation_start/03.code/old/ATLAS_models/UNet_in2d_out2d'
    logdir = 'D:/soyeon_vision2022/01.Object_segmentation_start/03.code/old/ATLAS_models/HarDNet_in2d_out2d'+model_name+'_v1_Dice_IoULoss_'+str(img_depth)
    writer = SummaryWriter(log_dir=logdir)
    logger = get_logger(logdir)

    logger.info("Train Start")
    logger.info("========================================")
    logger.info("# parameters: ")
    logger.info("batch_size={}".format(batch_size))
    logger.info("n_classes={}".format(n_classes))
    logger.info("learning_rate={}".format(learning_rate))
    logger.info("patience_scheduler={}".format(patience_scheduler))
    logger.info("patience_early_stop={}".format(patience_early_stop))
    logger.info("logdir={}".format(logdir))
    logger.info("========================================")



    print_interval = 100
    #val_interval = 500

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("==========================================")
    print('Device:', device)
    print('Current cuda device:', torch.cuda.current_device())
    print("torch.cuda.device_count()=", torch.cuda.device_count())
    print("==========================================")
    device_ids = [0]
    torch.backends.cudnn.benchmark = True

    # train_img_path = 'D:/hl/data/0223_split/train/imgs/'
    # train_mask_path = 'D:/hl/data/0223_split/train/masks/'
    # val_img_path = 'D:/hl/data/0223_split/val/imgs/'
    # val_mask_path = 'D:/hl/data/0223_split/val/masks/'
    # train_img_path = 'C:/Users/user/Documents/python-code/11_c_paper/data/0223_split/train/imgs/'
    # train_mask_path = 'C:/Users/user/Documents/python-code/11_c_paper/data/0223_split/train/masks/'
    # val_img_path = 'C:/Users/user/Documents/python-code/11_c_paper/data/0223_split/val/imgs/'
    # val_mask_path = 'C:/Users/user/Documents/python-code/11_c_paper/data/0223_split/val/masks/'

    # train_img_path = 'D:/dataset/00_medical_ATLAS/train/imgs/'
    # train_mask_path = 'D:/dataset/00_medical_ATLAS/train/masks/'
    # val_img_path = 'D:/dataset/00_medical_ATLAS/val/imgs/'
    # val_mask_path = 'D:/dataset/00_medical_ATLAS/val/masks/'
    train_img_path = 'D:/soyeon_vision2022/01.Object_segmentation_start/01.data/ATLAS_test/v1_train_png/imgs/'
    train_mask_path = 'D:/soyeon_vision2022/01.Object_segmentation_start/01.data/ATLAS_test/v1_train_png/masks/'
    val_img_path = 'D:/soyeon_vision2022/01.Object_segmentation_start/01.data/ATLAS_test/v1_val_png/imgs/'
    val_mask_path = 'D:/soyeon_vision2022/01.Object_segmentation_start/01.data/ATLAS_test/v1_val_png/masks/'

    train_data_list = [train_img_path + i for i in os.listdir(train_img_path)]
    train_mask_list = [train_mask_path + i for i in os.listdir(train_mask_path)]
    val_data_list = [val_img_path + i for i in os.listdir(val_img_path)]
    val_mask_list = [val_mask_path + i for i in os.listdir(val_mask_path)]


    train_data = DataLoader(CustomDataset2_gray(train_data_list, train_mask_list), batch_size=batch_size, shuffle=True, num_workers=4)
    val_data   = DataLoader(CustomDataset2_gray(val_data_list, val_mask_list), batch_size=batch_size, shuffle=True, num_workers=4)

    logger.info("==========================================")
    logger.info("train_img_path: {}".format(train_img_path))
    logger.info("train_mask_path: {}".format(train_mask_path))
    logger.info("val_img_path: {}".format(val_img_path))
    logger.info("val_mask_path: {}".format(val_mask_path))
    logger.info("==========================================")

    if n_classes == 1:
        running_metrics_val = runningScore(2)
    else:
        running_metrics_val = runningScore(n_classes)

    criterion = DiceCE()
    # criterion = DiceFocal()
    #criterion = Dice_IoULoss_binary()
    #criterion = Dice_IoULoss()
    criterion.to(device)

    ## create model

    model = hardnet(n_classes=2, in_channels=1).cuda()
    #model = TransHarDNet(in_channels=img_depth, n_classes=n_classes).cuda()
    ##  load pre-trained model
    # model_path = "./runs/exp2_swin_hardnet_v2_dicece_0.01/swin_hardnet_v2_epoch_30_checkpoint.pkl"

    #model_path = os.path.join(logdir, model_name+"_best_model.pkl")
    #model.load_state_dict(torch.load(model_path)["model_state"])
    #model.eval()
    #model.to(device)

    # model = ResUnetPlusPlus(channel=1).cuda()
    # model = nn.DataParallel(model)
    total_params = sum(p.numel() for p in model.parameters())
    print('Parameters:', total_params)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=patience_scheduler, verbose=True)

    best_iou = -100.0
    loss_all = 0
    loss_n = 0
    flag = True
    val_loss_meter = averageMeter()
    train_loss_meter = averageMeter()
    time_meter = averageMeter()


    early_stopping = EarlyStopping(patience=patience_early_stop, verbose=True)
    val_loss_for_print = 0
    while start_iter <= num_epochs and flag:
        for i, pack in enumerate(train_data):
            i += 1
            model.train()
            images, labels = pack
            images = images.to(device)

            labels = labels.to(device).long()

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels.squeeze(1))
            # scheduler.step()
            loss.backward()
            optimizer.step()

            loss_all += loss.item()
            train_loss_meter.update(loss.item())  ## train loss update, 모니터링 용
            loss_n += 1

            if (i + 1) % print_interval == 0:
                fmt_str = "in epoch[{}]  Iter [{:d}/{:d}]  Loss: {:.8f} ;  val_loss:{:.8f}"
                print_str = fmt_str.format(start_iter, i + 1, len(train_data), loss_all / loss_n, val_loss_for_print)
                logger.info(print_str)
                # writer.add_scalar("loss/train_loss", loss.item(), i + 1)


        #torch.cuda.empty_cache()
        model.eval()
        loss_all = 0
        loss_n = 0
        with torch.no_grad():
            for i_val, (images_val, labels_val) in tqdm(enumerate(val_data)):
                images_val = images_val.to(device)
                labels_val = labels_val.to(device).long().squeeze(1)

                outputs = model(images_val)
                val_loss = criterion(outputs, labels_val)

                pred = outputs.data.max(1)[1].cpu().numpy()
                gt = labels_val.data.cpu().numpy()

                running_metrics_val.update(gt, pred)
                val_loss_meter.update(val_loss.item())

        writer.add_scalar("loss/val_loss", val_loss_meter.avg, start_iter + 1)
        logger.info("Iter %d Val Loss: %.8f" % (start_iter + 1, val_loss_meter.avg))



        ##  early stopping 및 learning 조정
        my_total_loss = train_loss_meter.avg + val_loss_meter.avg
        val_loss_for_print = val_loss_meter.avg
        logger.info("ecpch[{}] loss: {:.8f} ;  val_loss:{:.8f} my_total_loss:{:.8f}".format(start_iter, train_loss_meter.avg, val_loss_meter.avg, my_total_loss))
        scheduler.step(val_loss_meter.avg)
        early_stopping(val_loss_meter.avg, model)

        if early_stopping.early_stop:
            print("Early stopping : {}".format(datetime.datetime.now()))
            break

        score, class_iou, class_dice = running_metrics_val.get_scores()
        for k, v in score.items():
            logger.info("score {}: {}".format(k, v))
            writer.add_scalar("val_metrics/{}".format(k), v, start_iter + 1)

        for k, v in class_iou.items():
            logger.info("class_iou: {}: {}".format(k, v))
            writer.add_scalar("val_metrics/cls_{}".format(k), v, start_iter + 1)

        for k, v in class_dice.items():
            logger.info("class_dice: {}: {}".format(k, v))
            # writer.add_scalar("val_metrics/cls_{}".format(k), v, start_iter + 1)

        val_loss_meter.reset()
        train_loss_meter.reset()
        running_metrics_val.reset()

        save_path = os.path.join(writer.file_writer.get_logdir(),"{}_{}_checkpoint.pkl".format(model_name, 'epoch_' + str(start_iter + 1)))
        state = {
            "epoch": start_iter + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        }
        torch.save(state, save_path)

        if score["Mean IoU : \t"] >= best_iou:
            best_iou = score["Mean IoU : \t"]
            state = {
                "epoch": start_iter + 1,
                "model_state": model.state_dict(),
                "best_iou": best_iou,
            }
            save_path = os.path.join(
                writer.file_writer.get_logdir(),
                "{}_best_model.pkl".format(model_name))

            save_path2 = os.path.join(
                writer.file_writer.get_logdir(),
                "{}_best_model.pth".format(model_name))

            torch.save(state, save_path)
            torch.save(model, save_path2)

        torch.cuda.empty_cache()

        start_iter += 1

    return model, device, logdir, model_name, logger



if __name__ == '__main__':
    torch.cuda.empty_cache()
    test_score_list = []
    model_name = "HarDNet_ATLAS_test"
    logdir = 'D:/soyeon_vision2022/01.Object_segmentation_start/03.code/old/ATLAS_models/HarDNet_in2d_out2d' + model_name + '_v1_Dice_IoULoss_' + str(1)
    #model_name = "STHardnet_224_l3_s3_ATLAS-R2"  + "_v2"
    #ogdir = './runs/k_fold_'+model_name+'_D1006_dicece_0.001/main'
    writer = SummaryWriter(log_dir=logdir)
    logger_main = get_logger(logdir)
    print('start')

    model, device, logdir, model_name, logger_= train()

    ## print score in best model
    model_path = os.path.join(logdir, model_name + "_best_model.pkl")
    model.load_state_dict(torch.load(model_path)["model_state"])
    model.eval()
    model.to(device)
    logger_.info("============================")
    logger_.info("============= {} in best model ===============".format(model_name))
    logger_.info("============================\n\n")

    """
    model_path = os.path.join(logdir, model_name + "_best_model.pkl")
    model = torch.load(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_datas, label_list = get_labels()
    test_score = test(model, device, label_list)
    logger.info("============================")
    logger.info(test_score['mean_score'])
    logger.info("============================\n\n")
    test_score_list.append(test_score)
    
    ###  최종 성능  출력
    dice_list = []
    iou_list = []
    precision_list = []
    recall_list = []
    for i in range(2):
        logger_main.info("=============  in [{}] best model ===============".format(i + 1))
        logger_main.info(test_score_list[i]['mean_score'])
        logger_main.info("============================\n")
        dice_list.append(test_score_list[i]['mean_score']['dice'])
        iou_list.append(test_score_list[i]['mean_score']['iou'])
        precision_list.append(test_score_list[i]['mean_score']['precision'])
        recall_list.append(test_score_list[i]['mean_score']['recall'])

    logger_main.info("Average : ")
    logger_main.info("dice={}".format(np.mean(dice_list)))
    logger_main.info("iou={}".format(np.mean(iou_list)))
    logger_main.info("precision={}".format(np.mean(precision_list)))
    logger_main.info("recall={}".format(np.mean(recall_list)))
    """
####




