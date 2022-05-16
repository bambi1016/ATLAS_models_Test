import gc
from torch.optim import Adam
from lib.loss import DiceCE
import torch
from torch.utils.data import DataLoader
import os
from lib.loader import CustomDataset
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lib.logger import get_logger
from tqdm import tqdm
from lib.pytorchtools import EarlyStopping
from lib.metrics import runningScore, averageMeter

# from hl_seg.runs.unet.model import Unet
from lib.unet_model import UNet

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train():
    batch_size = 8
    n_classes = 6
    learning_rate = 0.01

    start_iter = 0
    num_epochs = 50

    ## save model log and check points
    logdir = './runs/unet'
    writer = SummaryWriter(log_dir=logdir)
    logger = get_logger(logdir)
    logger.info("Let the games begin")

    print_interval = 50
    val_interval = 500

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # GPU 할당 변경하기
    GPU_NUM = 0  # 원하는 GPU 번호 입력
    #device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cuda:[2,3]' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    #torch.cuda.set_device(device)  # change allocation of current GPU
    print('Current cuda device ', torch.cuda.current_device())  # check

    device_ids = [0]
    torch.backends.cudnn.benchmark = True

    train_img_path  = 'D:/soyeon_vision2022/01.Object_segmentation_start/01.data/ATLAS_test/v1_train_png/imgs/'
    train_mask_path = 'D:/soyeon_vision2022/01.Object_segmentation_start/01.data/ATLAS_test/v1_train_png/masks/'
    val_img_path =  'D:/soyeon_vision2022/01.Object_segmentation_start/01.data/ATLAS_test/v1_val_png/mgs/'
    val_mask_path = 'D:/soyeon_vision2022/01.Object_segmentation_start/01.data/ATLAS_test/v1_val_png/masks/'

    train_data_list = [train_img_path + i for i in os.listdir(train_img_path)]
    train_mask_list = [train_mask_path + i for i in os.listdir(train_mask_path)]
    val_data_list = [val_img_path + i for i in os.listdir(val_img_path)]
    val_mask_list = [val_mask_path + i for i in os.listdir(val_mask_path)]

    train_data = DataLoader(CustomDataset(train_data_list, train_mask_list), batch_size=batch_size, shuffle=True,
                            num_workers=8)
    val_data = DataLoader(CustomDataset(val_data_list, val_mask_list), batch_size=batch_size, shuffle=True,
                          num_workers=8)

    running_metrics_val = runningScore(n_classes)
    criterion = DiceCE()
    # criterion = DiceCELoss()
    criterion.to(device)

    ## create model
    model = UNet(n_classes=2, n_channels=1).cuda()

    ##  load pre-trained model
    # model_path = "./runs/exp22_hardnet_swin_transformer_dicece_0.01/hardnet_best_model.pkl"
    # model.load_state_dict(torch.load(model_path)["model_state"])
    model.eval()
    model.to(device)
    # model = ResUnetPlusPlus(channel=1).cuda()
    # model = nn.DataParallel(model)
    total_params = sum(p.numel() for p in model.parameters())
    print('Parameters:', total_params)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

    best_iou = -100.0
    loss_all = 0
    loss_n = 0
    flag = True
    val_loss_meter = averageMeter()
    time_meter = averageMeter()

    patience = 10
    early_stopping = EarlyStopping(patience=patience, verbose=True)

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
            loss_n += 1

            if (i + 1) % print_interval == 0:
                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.8f}"
                print_str = fmt_str.format(
                    i + 1,
                    len(train_data),
                    loss_all / loss_n)

                print(print_str)
                logger.info(print_str)
                # writer.add_scalar("loss/train_loss", loss.item(), i + 1)

        torch.cuda.empty_cache()
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

        scheduler.step(val_loss_meter.avg)
        early_stopping(val_loss_meter.avg, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        score, class_iou = running_metrics_val.get_scores()
        for k, v in score.items():
            print(k, v)
            logger.info("{}: {}".format(k, v))
            writer.add_scalar("val_metrics/{}".format(k), v, start_iter + 1)

        for k, v in class_iou.items():
            logger.info("{}: {}".format(k, v))
            writer.add_scalar("val_metrics/cls_{}".format(k), v, start_iter + 1)

        val_loss_meter.reset()
        running_metrics_val.reset()

        save_path = os.path.join(
            writer.file_writer.get_logdir(),
            "{}_{}_checkpoint.pkl".format('swin_transformer', 'epoch_' + str(start_iter + 1)))
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
                "{}_best_model.pkl".format('swin_transformer'))
            torch.save(state, save_path)
        torch.cuda.empty_cache()

        start_iter += 1


if __name__ == '__main__':
    torch.cuda.empty_cache()
    train()

