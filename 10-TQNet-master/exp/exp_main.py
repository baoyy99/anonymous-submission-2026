from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import TQNet
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
from utils.loss_factory import get_loss_function,PALL_EMA_TQA_MAELoss,trend_mean_vis 
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

import logging  
from logging.handlers import RotatingFileHandler  


def init_logger(log_file_path='./10-TQNet-master/TSout/training.log'):
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
  
    log_format = '%(asctime)s - %(levelname)s - %(message)s'

    logging.basicConfig(
        level=logging.INFO,  
        format=log_format,
        handlers=[
            RotatingFileHandler(  
                log_file_path,
                maxBytes=10*1024*1024,  
                backupCount=5,  
                encoding='utf-8'
            ),

        ]
    )
    return logging.getLogger(__name__)

logger = init_logger(log_file_path='./10-TQNet-master/TSout/training.log')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'TQNet': TQNet
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim


    def _select_criterion(self):
        criterion = get_loss_function(self.args)
        logger.info(f"🔹 loss function: {criterion.__class__.__name__}")
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'CycleNet', 'TQ'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'MLP', 'SegRNN', 'TST'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'CycleNet', 'TQ'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'SegRNN', 'TST'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()

                # loss = criterion(pred, true)
                if self.args.loss == 'psloss' or self.args.loss == 'psloss_taq': 
                    # print(self.model.output_proj.parameters)
                    mse_loss = nn.MSELoss()
                    loss = mse_loss(pred, true)
                else   :
                    loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # path = os.path.join(self.args.checkpoints, setting)
        path = os.path.join('./10-TQNet-master/TSout/checkpoints', setting)

        if not os.path.exists(path):
            os.makedirs(path)

        # logger.info(self.args.model_id)
        logger.info( setting)
        logger.info("loss: {} dataset: {} model: {}".format(self.args.loss, self.args.data_path, self.args.model))
        logger.info("begin training")

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            if epoch == 10:
                logger.info("===== Epoch 10: Running test with current checkpoint =====")
                temp_ckpt_path = os.path.join(path, 'temp_epoch10_checkpoint.pth')
                torch.save(self.model.state_dict(), temp_ckpt_path)
                self.model.load_state_dict(torch.load(temp_ckpt_path))
                self.test(setting, test=1)
                os.remove(temp_ckpt_path)
                logger.info("===== Epoch 10: Test completed, resume training =====")
  
                self.model.train()                 

            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            # max_memory = 0
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'CycleNet', 'TQ'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'MLP', 'SegRNN', 'TST'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if any(substr in self.args.model for substr in {'CycleNet', 'TQ'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'SegRNN', 'TST'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    if self.args.loss == 'psloss' or self.args.loss == 'psloss_taq':
                        loss = criterion(outputs, batch_y,self.model)
                    else   :
                        loss = criterion(outputs, batch_y)
                    
                    # loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 600 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

    
                    log_msg = (
                        "Ititers: {iters:5d}\t | "     
                        "Loss: {loss:.7f}\t | "
                        "Speed: {speed:.4f}s/iter\t | "
                        "Left Time: {left:.4f}s"
                    ).format(
                        iters=i + 1,
                        loss=loss.item(),
                        speed=speed,
                        left=left_time
                    )
                    logger.info(log_msg)

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()


                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
    
                logger.info("Early stopping")
       
                break

            logger.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            logger.info("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(epoch + 1, train_steps, train_loss, vali_loss, test_loss))
  

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))



        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')

            logger.info("loading model")

            self.model.load_state_dict(torch.load(os.path.join('./10-TQNet-master/TSout/checkpoints/' + setting, 'checkpoint.pth')))


        preds = []
        trues = []
        inputx = []
        folder_path = './10-TQNet-master/TSout/test_results/' + setting + '/'

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        total_mae = 0
        total_mse = 0

        total_tqa_00 = 0
        total_tqa_01 = 0
        total_tqa_10 = 0
        total_tqa_11 = 0
        total_tqa_00_6 = 0
        total_tqa_01_6 = 0
        total_tqa_10_6 = 0
        total_tqa_11_6 = 0

        total_samples = 0


        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'CycleNet', 'TQ'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'MLP', 'SegRNN', 'TST'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'CycleNet', 'TQ'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'SegRNN', 'TST'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                

                with torch.no_grad():
                    tqaloss_00 = PALL_EMA_TQA_MAELoss(alpha=0,beta=0,gamma=0.1,q=0.8)
                    tqaloss_01 = PALL_EMA_TQA_MAELoss(alpha=0,beta=1,gamma=0.1,q=0.8)
                    tqaloss_10 = PALL_EMA_TQA_MAELoss(alpha=1,beta=0,gamma=0.1,q=0.8)
                    tqaloss_11 = PALL_EMA_TQA_MAELoss(alpha=1,beta=1,gamma=0.1,q=0.8)
                    tqa_loss_00 = tqaloss_00(outputs, batch_y)  
                    tqa_loss_01 = tqaloss_01(outputs, batch_y)
                    tqa_loss_10 = tqaloss_10(outputs, batch_y)
                    tqa_loss_11 = tqaloss_11(outputs, batch_y)
                    tqaloss_00_6 = PALL_EMA_TQA_MAELoss(alpha=0,beta=0,gamma=0.1,q=0.66)
                    tqaloss_01_6 = PALL_EMA_TQA_MAELoss(alpha=0,beta=1,gamma=0.1,q=0.66)
                    tqaloss_10_6 = PALL_EMA_TQA_MAELoss(alpha=1,beta=0,gamma=0.1,q=0.66)
                    tqaloss_11_6 = PALL_EMA_TQA_MAELoss(alpha=1,beta=1,gamma=0.1,q=0.66)
                    tqa_loss_00_6 = tqaloss_00_6(outputs, batch_y)  
                    tqa_loss_01_6 = tqaloss_01_6(outputs, batch_y)
                    tqa_loss_10_6 = tqaloss_10_6(outputs, batch_y)
                    tqa_loss_11_6 = tqaloss_11_6(outputs, batch_y)

                
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()
                mae, mse, rmse, mape, mspe, rse, corr = metric(pred, true)
    
                bs = pred.shape[0]
                total_mae += mae * bs
                total_mse += mse * bs
                total_tqa_00 += tqa_loss_00 * bs
                total_tqa_01 += tqa_loss_01 * bs
                total_tqa_10 += tqa_loss_10 * bs
                total_tqa_11 += tqa_loss_11 * bs
                total_tqa_00_6 += tqa_loss_00_6 * bs
                total_tqa_01_6 += tqa_loss_01_6 * bs
                total_tqa_10_6 += tqa_loss_10_6 * bs
                total_tqa_11_6 += tqa_loss_11_6 * bs
                total_samples += bs
 

                if i % 10 == 0:
                    input = batch_x.detach().cpu().numpy()

                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)

                    trend_mean_vis(
                        pred=pd, 
                        true=gt, 
                        loss_type=self.args.loss.lower(), 
                        gamma=self.args.TQALoss_gamma,
                        batch_idx=i,
                        save_dir_base=folder_path
                    )

        if self.args.test_flop:
            test_params_flop(self.model, (batch_x.shape[1], batch_x.shape[2]))
            exit()
        mae = total_mae / total_samples
        mse = total_mse / total_samples
        tqa_00 = total_tqa_00 / total_samples
        tqa_01 = total_tqa_01 / total_samples
        tqa_10 = total_tqa_10 / total_samples
        tqa_11 = total_tqa_11 / total_samples
        tqa_00_6 = total_tqa_00_6 / total_samples
        tqa_01_6 = total_tqa_01_6 / total_samples
        tqa_10_6 = total_tqa_10_6 / total_samples
        tqa_11_6 = total_tqa_11_6 / total_samples
        # tqa = total_tqa / total_samples
        print('batch:tqa_00:{}, tqa_01:{}, tqa_10:{}, tqa_11:{}, mse:{}, mae:{}'.format(tqa_00, tqa_01, tqa_10, tqa_11, mse, mae))



        folder_path = './10-TQNet-master/TSout/results/' + setting + '/'

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


        timestamp = time.strftime("%Y-%m-%d %H:%M", time.localtime())   
        # logger.info(timestamp)
        # logger.info('0.66')
        logger.info('tqa_00_6:{}, tqa_01_6:{}, tqa_10_6:{}, tqa_11_6:{}'.format(tqa_00_6, tqa_01_6, tqa_10_6, tqa_11_6))
        logger.info('tqa_00_8:{}, tqa_01_8:{}, tqa_10_8:{}, tqa_11_8:{}'.format(tqa_00, tqa_01, tqa_10, tqa_11))
        logger.info('mse:{}, mae:{}'.format(mse, mae))

        # print('original:mse:{}, mae:{}'.format(mse, mae))
        f = open("./10-TQNet-master/TSout/result_sum.txt", 'a')

        f.write(timestamp + "  \n")

        f.write(setting + "  \n")
        f.write('tqa_00_6:{}, tqa_01_6:{}, tqa_10_6:{}, tqa_11_6:{}'.format(tqa_00_6, tqa_01_6, tqa_10_6, tqa_11_6))
        f.write("0.8  \n")
        f.write('tqa_00:{}, tqa_01:{}, tqa_10:{}, tqa_11:{}'.format(tqa_00, tqa_01, tqa_10, tqa_11))
        f.write(" \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()


        return
    
    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join('./10-TQNet-master/TSout/checkpoints/', setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))


        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(pred_loader):
              
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(
                    batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'CycleNet', 'TQ'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'MLP', 'SegRNN', 'TST'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'CycleNet', 'TQ'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'SegRNN', 'TST'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)
        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save

        folder_path = './10-TQNet-master/TSout/results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
