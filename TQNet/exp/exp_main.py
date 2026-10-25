from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, SegRNN, CycleNet, \
    iTransformer, TimeXer, TQNet, TQDLinear, TQPatchTST, TQiTransformer,CrossLinear
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from Loss_Metric.metric_factory import Metrics
from Loss_Metric.loss_factory import get_loss_function,Residual_vis 
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

# ====================
import logging  
from logging.handlers import RotatingFileHandler  
def init_logger(log_file_path):
    """Create one explicit logger; never touch legacy output at import time."""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    log_path = os.path.abspath(log_file_path)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    selected_logger = logging.getLogger(f'{__name__}:{log_path}')
    selected_logger.setLevel(logging.INFO)
    selected_logger.propagate = False
    if not selected_logger.handlers:
        handler = RotatingFileHandler(
            log_path, maxBytes=10 * 1024 * 1024,
            backupCount=5, encoding='utf-8'
        )
        handler.setFormatter(logging.Formatter(log_format))
        selected_logger.addHandler(handler)
    return selected_logger


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        global logger
        if getattr(args, 'evaluation_enabled', False):
            logger = init_logger(os.path.join(args.evaluation_run_dir, 'evaluation.log'))
        else:
            logger = init_logger('./TQNet/TSout/training.log')
        super(Exp_Main, self).__init__(args)
        self.training_recorder = None
        if getattr(args, 'evaluation_enabled', False):
            from evaluation.training import TrainingRecorder
            self.training_recorder = TrainingRecorder(args._evaluation_run)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'CrossLinear': CrossLinear,
            'PatchTST': PatchTST,
            'SegRNN': SegRNN,
            'CycleNet': CycleNet,
            'iTransformer': iTransformer,
            'TimeXer': TimeXer,
            'TQNet': TQNet,
            'TQDLinear': TQDLinear,
            'TQPatchTST': TQPatchTST,
            'TQiTransformer': TQiTransformer
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

    # def _select_criterion(self):
    #     criterion = nn.MSELoss()
    #     return criterion
    def _select_criterion(self):
        criterion = get_loss_function(self.args)
        logger.info(f"🔹 loss function: {criterion.__class__.__name__}")
        return criterion

    def _evaluation_forward_batch(self, batch, enable_grad=False):
        """Shared inference primitive for unified test and diagnostics."""
        batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle = batch
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)
        batch_cycle = batch_cycle.int().to(self.device)
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat(
            [batch_y[:, :self.args.label_len, :], dec_inp], dim=1
        ).float().to(self.device)
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if any(substr in self.args.model for substr in {'CycleNet', 'TQ'}):
                    outputs = self.model(batch_x, batch_cycle)
                elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'SegRNN', 'TST'}):
                    outputs = self.model(batch_x)
                elif self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if any(substr in self.args.model for substr in {'CycleNet', 'TQ'}):
                outputs = self.model(batch_x, batch_cycle)
            elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'SegRNN', 'TST'}):
                outputs = self.model(batch_x)
            elif self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        true = batch_y[:, -self.args.pred_len:, f_dim:]
        return outputs, true

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
                if self.args.loss == 'psloss' or self.args.loss == 'aei_mae': 
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

        if getattr(self.args, 'evaluation_enabled', False):
            path = os.path.join(self.args.evaluation_run_dir, 'checkpoints')
        else:
            path = os.path.join('./TQNet/TSout/checkpoints', setting)
        os.makedirs(path, exist_ok=True)

        # logger.info(self.args.model_id)
        logger.info( setting)
        logger.info("loss: {} dataset: {} model: {}".format(self.args.loss, self.args.data_path, self.args.model))
        logger.info("begin training")

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        clip_accumulator = None
        if self.training_recorder is not None:
            from evaluation.training import GradientClipAccumulator
            record_config = self.args._evaluation_run.config['training_record']
            clip_accumulator = GradientClipAccumulator(
                enabled=record_config.get('record_gradient_clipping', False),
                max_norm=record_config.get('gradient_clip_max_norm'),
            )

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        global_step = 0
        optimizer_step = 0
        last_epoch = -1
        for epoch in range(self.args.train_epochs):
            last_epoch = epoch
            # Legacy-only behavior. Unified runs use explicit diagnostic checkpoints.
            if epoch == 10 and not getattr(self.args, 'evaluation_enabled', False):
                logger.info("===== Epoch 10: Running test with current checkpoint =====")
                temp_ckpt_path = os.path.join(path, 'temp_epoch10_checkpoint.pth')
                torch.save(self.model.state_dict(), temp_ckpt_path)
                self.model.load_state_dict(torch.load(temp_ckpt_path, map_location=self.device))
                self.test(setting, test=1)
                os.remove(temp_ckpt_path)
                logger.info("===== Epoch 10: Test completed, resume training =====")
                self.model.train()

            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            diagnostic_checkpoint = None
            if clip_accumulator is not None:
                clip_accumulator.reset()
            if torch.cuda.is_available() and self.args.use_gpu:
                torch.cuda.reset_peak_memory_stats(self.device)

            for i, batch in enumerate(train_loader):
                iter_count += 1
                global_step += 1
                model_optim.zero_grad(set_to_none=True)
                outputs, batch_y = self._evaluation_forward_batch(batch, enable_grad=True)
                if self.args.loss == 'psloss' or self.args.loss == 'psloss_taq':
                    loss = criterion(outputs, batch_y, self.model)
                elif self.args.loss == 'aei_mae':
                    loss = criterion(outputs, batch_y, epoch, self.args.train_epochs)
                else:
                    loss = criterion(outputs, batch_y)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f'non-finite training objective at epoch={epoch + 1}, global_step={global_step}'
                    )
                train_loss.append(loss.item())

                if (i + 1) % 600 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()


                    log_msg = (
                        "Ititers: {iters:5d}\t | "     #加｜分割
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
                    if clip_accumulator is not None and clip_accumulator.enabled:
                        scaler.unscale_(model_optim)
                        clip_accumulator.observe_and_optionally_clip(self.model.parameters())
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    if clip_accumulator is not None:
                        clip_accumulator.observe_and_optionally_clip(self.model.parameters())
                    model_optim.step()
                optimizer_step += 1

                if self.training_recorder is not None:
                    saved = self.training_recorder.maybe_save_diagnostic(
                        self.model, model_optim, epoch + 1, optimizer_step
                    )
                    if saved is not None:
                        diagnostic_checkpoint = saved

                # current_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
                # max_memory = max(max_memory, current_memory)

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            epoch_elapsed = time.time() - epoch_time
            print("Epoch: {} cost time: {}".format(epoch + 1, epoch_elapsed))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            logger.info("Epoch: {} cost time: {}".format(epoch + 1, epoch_elapsed))
            logger.info("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            if self.training_recorder is not None:
                clip_summary = clip_accumulator.summary()
                peak_allocated = int(torch.cuda.max_memory_allocated(self.device)) if torch.cuda.is_available() and self.args.use_gpu else 0
                peak_reserved = int(torch.cuda.max_memory_reserved(self.device)) if torch.cuda.is_available() and self.args.use_gpu else 0
                self.training_recorder.record_epoch(
                    epoch=epoch + 1, global_step=global_step, optimizer_step=optimizer_step,
                    train_objective=float(train_loss),
                    validation_mse_or_objective=float(vali_loss),
                    test_mse_or_objective=float(test_loss),
                    learning_rate=float(model_optim.param_groups[0]['lr']),
                    epoch_time_seconds=epoch_elapsed,
                    samples_per_second=(train_steps * self.args.batch_size) / max(epoch_elapsed, 1e-12),
                    peak_allocated_bytes=peak_allocated,
                    peak_reserved_bytes=peak_reserved,
                    gradient_norm_preclip=clip_summary['pre'],
                    gradient_norm_postclip=clip_summary['post'],
                    clip_event_count=clip_summary['count'], clipped=clip_summary['clipped'],
                    nan_or_inf=False, diagnostic_checkpoint=diagnostic_checkpoint,
                )

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                logger.info("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        if self.training_recorder is not None:
            self.training_recorder.save_final(
                self.model, model_optim, last_epoch + 1, optimizer_step
            )
        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        if self.training_recorder is not None:
            self.training_recorder.register_best(best_model_path)

        return self.model

    def test(self, setting, test=0):
        if getattr(self.args, 'evaluation_enabled', False):
            from evaluation.evaluator import UnifiedEvaluator
            return UnifiedEvaluator(self, setting).run_evaluation()

        metrics = Metrics(
            ema_alpha=0.3,
            q=self.args.loss_gamma,
            alpha=self.args.loss_alpha,
            beta=self.args.loss_beta,
        )
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')

            logger.info("loading model")


            self.model.load_state_dict(torch.load(os.path.join('./TQNet/TSout/checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        
        folder_path = './TQNet/TSout/test_results/' + setting + '/'

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

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
                
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                metrics(pred, true)


                # inputx.append(batch_x.detach().cpu().numpy())
                if i % 10 == 0:
                    input = batch_x.detach().cpu().numpy()

                    gt = true[0, :, -1]
                    pd = pred[0, :, -1]

                    # visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                    # # np.savetxt(os.path.join(folder_path, str(i) + '.txt'), pd)
                    # # np.savetxt(os.path.join(folder_path, str(i) + 'true.txt'), gt)
                    # # ====================
                    Residual_vis(
                        pred=pd, 
                        true=gt, 
                        loss_type=self.args.loss.lower(), 
                        batch_idx=i,
                        save_dir_base=folder_path
                    )
                # ==========================================================

        (
            s_a_u_mae, s_a_u_mse,
            s_a_o_mae, s_a_o_mse,
            s_b_u_mae, s_b_u_mse,
            s_b_o_mae, s_b_o_mse,
            mae, mse,
        ) = metrics.finalize()

        if self.args.test_flop:
            test_params_flop(self.model, (batch_x.shape[1], batch_x.shape[2]))
            exit()


        # result save

        # folder_path = './results/' + setting + '/'
        folder_path = './TQNet/TSout/results/' + setting + '/'

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


        timestamp = time.strftime("%Y-%m-%d %H:%M", time.localtime())   
        logger.info('mse:{:.6f}, mae:{:.6f}'.format(mse, mae))
        logger.info(
            'S_A_U(MAE/MSE):{:.6f}/{:.6f}, '
            'S_A_O(MAE/MSE):{:.6f}/{:.6f}, '
            'S_B_U(MAE/MSE):{:.6f}/{:.6f}, '
            'S_B_O(MAE/MSE):{:.6f}/{:.6f}'.format(
                s_a_u_mae, s_a_u_mse,
                s_a_o_mae, s_a_o_mse,
                s_b_u_mae, s_b_u_mse,
                s_b_o_mae, s_b_o_mse,
            )
        )


        # print('original:mse:{}, mae:{}'.format(mse, mae))
        f = open("./TQNet/TSout/result_sum.txt", 'a')

        f.write(timestamp + "  \n")

        f.write(setting + "  \n")
        f.write('mse:{:.6f}, mae:{:.6f}'.format(mse, mae))
        f.write('\n')
        f.write(
            'S_A_U_MAE:{:.6f}, S_A_U_MSE:{:.6f}, '
            'S_A_O_MAE:{:.6f}, S_A_O_MSE:{:.6f}, '
            'S_B_U_MAE:{:.6f}, S_B_U_MSE:{:.6f}, '
            'S_B_O_MAE:{:.6f}, S_B_O_MSE:{:.6f}'.format(
                s_a_u_mae, s_a_u_mse,
                s_a_o_mae, s_a_o_mse,
                s_b_u_mae, s_b_u_mse,
                s_b_o_mae, s_b_o_mse,
            )
        )
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return
