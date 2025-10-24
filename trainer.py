import os
import torch
import glob
import torch.utils.data as data
torch.cuda.set_device(2)
from tqdm import tqdm

from kits import metrics
from kits import SegMetrics
from kits import configure_loss
from kits import LR_Scheduler
from kits import Saver
from kits import TensorboardSummary
from data import get_segmentation_dataset
import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
import torchvision

class Trainer(object):
    def __init__(self, args, model):
        self.args = args
        self.device = torch.device('cuda')
        self.model = model
        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()


        train_img_paths = sorted(glob.glob("/media/tc/7810057410053B20/sy/BraTS2021/trainImage/*"))
        train_gt_paths = sorted(glob.glob("/media/tc/7810057410053B20/sy/BraTS2021/trainGt/*"))
        val_img_paths = sorted(glob.glob("/media/tc/7810057410053B20/sy/BraTS2021/testImage/*"))
        val_gt_paths = sorted(glob.glob("/media/tc/7810057410053B20/sy/BraTS2021/testGt/*"))

        #
        # train_img_paths = sorted(glob.glob("/home/tc/BraTS2020_npy/npy/trainImage/*"))
        # train_gt_paths = sorted(glob.glob("/home/tc/BraTS2020_npy/npy/trainGt/*"))
        #
        # val_img_paths = sorted(glob.glob("/home/tc/BraTS2020_npy/npy/testImage/*"))
        # val_gt_paths = sorted(glob.glob("/home/tc/BraTS2020_npy/npy/testGt/*"))

        train_dataset = get_segmentation_dataset('new', img_paths=train_img_paths, mask_paths=train_gt_paths, is_train=True)
        val_dataset = get_segmentation_dataset('new', img_paths=val_img_paths, mask_paths=val_gt_paths, is_train=False)

        self.train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_size=args.batch_size,
                                            num_workers=4,
                                            shuffle=True,
                                            pin_memory=True,
                                            drop_last=True)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_size=1,
                                          num_workers=4,
                                          pin_memory=True)

        self.optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                         weight_decay=args.weight_decay)

        # Define criterion
        self.criterion1 = configure_loss('dice')
        self.criterion2 = configure_loss('bce')

        # # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                      args.epochs, len(self.train_loader))

        # Define Evaluator
        self.evaluator1 = SegMetrics(2)
        self.evaluator2 = SegMetrics(2)
        self.evaluator3 = SegMetrics(2)

        self.evaluator1_1 = SegMetrics(2)
        self.evaluator1_2 = SegMetrics(2)
        self.evaluator1_3 = SegMetrics(2)

        self.evaluator2_1 = SegMetrics(2)
        self.evaluator2_2 = SegMetrics(2)
        self.evaluator2_3 = SegMetrics(2)

        # Resuming checkpoint
        self.train_loss = float('inf')
        self.best_pred = 0.0

        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']

            self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

    def training(self, epoch):
        # torch.manual_seed(13)
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for iteration, sample in enumerate(tbar):
            image = sample[0].to(self.device)

            target = sample[1].to(self.device)
            # print("tt", target.shape)
            self.scheduler(self.optimizer, iteration, epoch, self.best_pred)
            self.optimizer.zero_grad()
            if self.args.model == 'ACANet':
                GM1, GM2, output = self.model(image)
            elif self.args.modalities == 'all':
                output = self.model(image)
            else:
                raise RuntimeError("modalities error, chossen{}".format(self.args.modalities))

            if self.args.model == 'ACANet':
                GM1 = torch.sigmoid(GM1)
                GM2 = torch.sigmoid(GM2)
                output = torch.sigmoid(output)
                # print("Input shape:", GM1.shape)
                # print("Target shape:", target.shape)
                # print("Input shape:", GM2.shape)
                # print("put shape:", output.shape)
                # 在 trainer.py 第 121-122 行附近添加
                dice_loss_GM1 = self.criterion1(GM1, target)
                loss_GM1 = self.criterion2(GM1, target)
                dice_loss_GM2 = self.criterion1(GM2, target)
                loss_GM2 = self.criterion2(GM2, target)
                dice_loss_output = self.criterion1(output, target)
                loss_output = self.criterion2(output, target)
                loss_GM1 = dice_loss_GM1 + loss_GM1
                loss_GM2 = dice_loss_GM2 + loss_GM2
                loss_output = dice_loss_output + loss_output
                loss = loss_GM1 + loss_GM2 + loss_output
            else:
                output = torch.sigmoid(output)
                dice_loss_1 = self.criterion1(output, target)
                bce_loss_1 = self.criterion2(output, target)
                loss = dice_loss_1 + bce_loss_1

            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (iteration + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), iteration + num_img_tr * epoch)

            if iteration % (num_img_tr // 10) == 0:
                global_step = iteration + num_img_tr * epoch
                self.summary.visualize_image(self.writer, self.args.dataset, image, target,output, global_step)
        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, iteration * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % (train_loss / (iteration + 1)))

    def validation(self, epoch):
        self.model.eval()
        self.evaluator1.reset()
        self.evaluator2.reset()
        self.evaluator3.reset()
        tbar = tqdm(self.val_loader, desc='\r', ncols=80)
        test_loss = 0.0

        hausdorff_WT_sum = 0.0
        hausdorff_TC_sum = 0.0
        hausdorff_ET_sum = 0.0

        iou_WT_sum = 0.0
        iou_TC_sum = 0.0
        iou_ET_sum = 0.0

        for iteration, sample in enumerate(tbar):
            image = sample[0].to(self.device)
            target = sample[1].to(self.device)
            fn = sample[2]

            with torch.no_grad():
                if self.args.model == 'F2Net':
                    GM1, GM2, output = self.model(image)
                elif self.args.modalities == 'all':
                    output = self.model(image)
                else:
                    raise RuntimeError("modalities error, chossen{}".format(self.args.modalities))

            output = torch.sigmoid(output[2])
            dice_loss = self.criterion1(output, target)
            bce_loss = self.criterion2(output, target)
            loss = dice_loss + bce_loss
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (iteration + 1)))
            target1 = target.cpu()
            target1 = target1.numpy()

            pred = (output > 0.5).float()
            pred = pred.long().cpu()
            pred = pred.numpy()
            hausdorff_WT = metrics.hausdorff_95(pred[:, 0, :, :], target1[:, 0, :, :])
            hausdorff_TC = metrics.hausdorff_95(pred[:, 1, :, :], target1[:, 1, :, :])
            hausdorff_ET = metrics.hausdorff_95(pred[:, 2, :, :], target1[:, 2, :, :])
            hausdorff_WT_sum += hausdorff_WT
            hausdorff_TC_sum += hausdorff_TC
            hausdorff_ET_sum += hausdorff_ET
            self.evaluator1.update(pred[:, 0, None, :, :], target1[:, 0, None, :, :])
            self.evaluator2.update(pred[:, 1, None, :, :], target1[:, 1, None, :, :])
            self.evaluator3.update(pred[:, 2, None, :, :], target1[:, 2, None, :, :])



        # Fast test during the training
        dice_WT = self.evaluator1.dice()
        dice_TC = self.evaluator2.dice()
        dice_ET = self.evaluator3.dice()
        dice_avg = (dice_WT + dice_TC + dice_ET) / 3

        hausdorff_WT = hausdorff_WT_sum / (iteration + 1)
        hausdorff_TC = hausdorff_TC_sum / (iteration + 1)
        hausdorff_ET = hausdorff_ET_sum / (iteration + 1)
        hausdorff_avg = (hausdorff_WT + hausdorff_TC + hausdorff_ET) / 3

        iou_wt = self.evaluator1.IoU()
        iou_tc = self.evaluator2.IoU()
        iou_et = self.evaluator3.IoU()
        iou_avg = (iou_wt + iou_tc + iou_et) / 3

        self.writer.add_scalar('val/total_loss_epoch', test_loss/(iteration + 1), epoch)            #total test loss
        self.writer.add_scalar('val/dice_WT', dice_WT, epoch)
        self.writer.add_scalar('val/dice_TC', dice_TC, epoch)
        self.writer.add_scalar('val/dice_ET', dice_ET, epoch)
        self.writer.add_scalar('val/dice_avg', dice_avg, epoch)
        self.writer.add_scalar('val/hausdorff_WT', hausdorff_WT, epoch)
        self.writer.add_scalar('val/hausdorff_TC', hausdorff_TC, epoch)
        self.writer.add_scalar('val/hausdorff_ET', hausdorff_ET, epoch)
        self.writer.add_scalar('val/hausdorff_avg', hausdorff_avg, epoch)

        sensitivity_WT = self.evaluator1.sensitivity()
        sensitivity_TC = self.evaluator2.sensitivity()
        sensitivity_ET = self.evaluator3.sensitivity()
        sensitivity_avg = (sensitivity_WT + sensitivity_TC + sensitivity_ET) / 3

        specificity_WT = self.evaluator1.specificity()
        specificity_TC = self.evaluator2.specificity()
        specificity_ET = self.evaluator3.specificity()
        specificity_avg = (specificity_WT + specificity_TC + specificity_ET) / 3

        print('Validation:')
        print(f'[Epoch: {epoch}, numImages: {iteration * self.args.batch_size + image.data.shape[0]}]')
        print(f'dice: WT: {dice_WT:.4f}, TC: {dice_TC:.4f}, ET: {dice_ET:.4f}, avg: {dice_avg:.4f}')
        print(
            f'hausdorff: WT: {hausdorff_WT:.4f}, TC: {hausdorff_TC:.4f}, ET: {hausdorff_ET:.4f}, avg: {hausdorff_avg:.4f}')
        print(
            f'sensitivity: WT: {sensitivity_WT:.4f}, TC: {sensitivity_TC:.4f}, ET: {sensitivity_ET:.4f}, avg: {sensitivity_avg:.4f}')
        print(
            f'specificity: WT: {specificity_WT:.4f}, TC: {specificity_TC:.4f}, ET: {specificity_ET:.4f}, avg: {specificity_avg:.4f}')
        print(f'IoU: WT: {iou_wt:.4f}, TC: {iou_tc:.4f}, ET: {iou_et:.4f}, avg: {iou_avg:.4f}')
        print(f'Loss: {test_loss / (iteration + 1):.4f}')

        # 更新最佳模型
        new_pred = dice_avg
        if new_pred > self.best_pred:
            self.best_pred = new_pred
            is_best = True
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
                'train_loss': self.train_loss,
                'validation_metrics': {  # 保存更多验证指标
                    'dice_WT': dice_WT,
                    'dice_TC': dice_TC,
                    'dice_ET': dice_ET,
                    'dice_avg': dice_avg,
                    'hausdorff_WT': hausdorff_WT,
                    'hausdorff_TC': hausdorff_TC,
                    'hausdorff_ET': hausdorff_ET,
                    'hausdorff_avg': hausdorff_avg,
                    'sensitivity_WT': sensitivity_WT,
                    'sensitivity_TC': sensitivity_TC,
                    'sensitivity_ET': sensitivity_ET,
                    'sensitivity_avg': sensitivity_avg,
                    'specificity_WT': specificity_WT,
                    'specificity_TC': specificity_TC,
                    'specificity_ET': specificity_ET,
                    'specificity_avg': specificity_avg,
                    'iou_wt': iou_wt,
                    'iou_tc': iou_tc,
                    'iou_et': iou_et,
                    'iou_avg': iou_avg,
                    'test_loss': test_loss / (iteration + 1)
                }
            }, is_best)