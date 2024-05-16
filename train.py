import os

import torch
from torch.utils.data import Dataset, DataLoader

from models.net import U2NET
from utiles.dataset import MeterDataset
from utiles.loss_function import DiceLoss, FocalLoss
# from skimage.metrics import accuracy_score,f1_score

import matplotlib.pyplot as plt
import wandb
import numpy as np
from collections import deque

# from utiles.dice_score import multiclass_dice_coeff, dice_coeff
# import visualkeras



class Trainer(object):

    def __init__(self):
        print('GPU is available:',torch.cuda.is_available())
        
        # self.run = wandb.init(
        # project="U2net_project",
        # config={
        #     "epochs": 500,
        #     "lr": 0.00001,
        #     })

        #  input channel = 3, output channel = 1
        self.net = U2NET(3, 2)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        self.loss_function = FocalLoss(alpha=0.75)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.data_set_train = MeterDataset(mode='train')
        if not os.path.exists('weight/net.pt'):
            self.net.load_state_dict(torch.load('weight/net.pt', map_location='cpu'))
        self.data_set_val = MeterDataset(mode='val')
        self.net.to(self.device)
        

    def __call__(self):

        epoch_num = 500
        batch_size_train = 5
        batch_size_val = 1
        ite_num = 0
        data_loader_train = DataLoader(self.data_set_train, batch_size_train, True, num_workers=2)
        data_loader_val = DataLoader(self.data_set_val, batch_size_val, False, num_workers=2)
        loss_sum = 0
        running_tar_loss = 0
        save_frq = 2075

        model_dir = 'weight/net0515'
        # visualkeras.layered_view(self.net,to_file='outputvisual.png', legend=True ,  draw_volume=False, spacing =10)
    
        dice_score = 0
        iou_score = 0
        prec_score = 0
        acc_score = 0
        dequemaxlen = 415
        losslist= deque(maxlen=dequemaxlen)# batch:83
        batchloop=0
        total_dices = []
        epoch_count = 0
        bestloss=1

        for epoch in range(epoch_num):
            '--------train---------'
            self.net.train()
            epoch_count += 1
            loss_sum = 0
            running_tar_loss = 0
            n_classes = 2
            
            
            for i, (images, masks) in enumerate(data_loader_train):
                ite_num += 1
                images = images.to(self.device)
                masks = masks.to(self.device)
                d0, d1, d2, d3, d4, d5, d6 = self.net(images)
                
                loss, loss0 = self.calculate_loss(d0, d1, d2, d3, d4, d5, d6, masks)

                self.optimizer.zero_grad()
                # print(loss)
                loss.backward()
                self.optimizer.step()
                loss_sum += loss.item()
                running_tar_loss += loss0.item()

                # losslist.append([loss.item(), loss0.item()])
                losslist.append(float(loss.item()))
                mask_pred = d0

                dice_value=self.dice_coef(mask_pred, masks).item()
                # print(dice_value)
                total_dices.append(dice_value)

                del d0, d1, d2, d3, d4, d5, d6, loss
                print(f'epoch:{epoch}; batch:{i + 1}; train loss:{loss_sum / (i + 1)}; tar:{running_tar_loss / (i + 1)}')
                
                if ite_num % save_frq == 0:
                    model_dir_epoch = f'{model_dir}_{epoch}_{ite_num}.pt'
                    torch.save(self.net.state_dict(), model_dir_epoch)
                if loss_sum / (i + 1)<bestloss and epoch>10:
                    bestloss=loss_sum / (i + 1)
                    print("bestloss=",bestloss)
                    model_dir_epoch = f'{model_dir}_best.pt'
                    torch.save(self.net.state_dict(), model_dir_epoch)
                batchloop=i+1

            # '---------val----------'
            self.net.eval()
            for i,(images,masks) in enumerate(data_loader_val):
                images = images.to(self.device)
                masks = masks.to(self.device)
                d0, d1, d2, d3, d4, d5, d6 = self.net(images)
            print('==inference avg dice:{:.4f}=='.format(np.mean(total_dices)))
            print('==inference avg loss:{:.4f}=='.format(np.mean(list(losslist))))
            # 
            

            # if n_classes == 1:
            #     assert masks.min() >= 0 and masks.max() <= 1, 'True mask indices should be in [0, 1]'
            #     mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
            #     # compute the Dice score
            #     dice_score += dice_coeff(mask_pred, masks, reduce_batch_first=False)
            #     # compute the Iou score
            #     iou_score += iou_coeff(mask_pred, masks, reduce_batch_first=False)
            #     # compute the Precision score
            #     prec_score += precision(mask_pred, masks)
            #     # compute the Accuracy score
            #     acc_score += accuracy(mask_pred, masks)
            # else:
            #     assert masks.min() >= 0 and masks.max() < n_classes, 'True mask indices should be in [0, n_classes['
            #     # convert to one-hot format
            #     masks = F.one_hot(masks, n_classes).permute(0, 3, 1, 2).float()
            #     mask_pred = F.one_hot(mask_pred.argmax(dim=1), n_classes).permute(0, 3, 1, 2).float()
            #     # compute the Dice score, ignoring background
            #     dice_score += multiclass_dice_coeff(mask_pred[:, 1:], masks[:, 1:], reduce_batch_first=False)
            #     # compute the Iou score, ignoring background
            #     iou_score += multiclass_iou_coeff(mask_pred[:, 1:], masks[:, 1:], reduce_batch_first=False)
            #     # compute the Precision score, ignoring background
            #     prec_score += precision(mask_pred[:, 1:], masks[:, 1:])
            #     # compute the Accuracy score, ignoring background
            #     acc_score += accuracy(mask_pred[:, 1:], masks[:, 1:])

            # print(f'epoch:{epoch}; val dice:{dice_score / (i + 1)}; val iou:{iou_score / (i + 1)}; val prec:{prec_score / (i + 1)}; val acc:{acc_score / (i + 1)}')

            #plt.plot(range(1,60*(epoch+1)),([losslist,total_dices]))
            try:
                if 2+batchloop*(epoch+1)<=dequemaxlen:
                    plt.plot(range(1, len(list(losslist))+1), list(losslist), label='Loss')
                    #plt.plot(range(1, batchloop+1+(batchloop*epoch)), total_dices, label='Total Dices')
                else:
                    # 1+82*4=329
                    plt.plot(range(1+batchloop*(epoch-4), 1+batchloop*(epoch-4)+dequemaxlen), list(losslist), label='Loss')
                    #plt.plot(range(2+batchloop*epoch, 2+batchloop*(epoch+1)), total_dices, label='Total Dices')
                # predicted_labels=self.net.predict(images)
                plt.savefig('weight/loss_' + str(epoch) + '.jpg')
                plt.close()
            except Exception as e:
                print(e)
                print(f"\n{batchloop}*{epoch-5}")
                print(f"\n--PLT__error__{1+batchloop*(epoch-4)}--->{1+batchloop*(epoch-4)+dequemaxlen}_!=_{len(list(losslist))}\n",)


        # try:
        #     plt.plot(range(ite_num),losslist[0])
        #     plt.show()
        # except:
        #     print(losslist)


    def calculate_loss(self, d0, d1, d2, d3, d4, d5, d6, labels):
        loss0 = self.loss_function(d0, labels)
        loss1 = self.loss_function(d1, labels)
        loss2 = self.loss_function(d2, labels)
        loss3 = self.loss_function(d3, labels)
        loss4 = self.loss_function(d4, labels)
        loss5 = self.loss_function(d5, labels)
        loss6 = self.loss_function(d6, labels)
        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        return loss, loss0
    
    # 计算dice
    def dice_coef(self, output, target):  # output为预测结果 target为真实结果
        smooth = 1e-5  # 防止0除
        intersection = (output * target).sum()
        return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)

if __name__ == '__main__':
    trainer = Trainer()
    trainer()
