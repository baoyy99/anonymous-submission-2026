import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

# fix_seed = 2
# random.seed(random_seed)
# torch.manual_seed(random_seed)
# np.random.seed(random_seed)
parser = argparse.ArgumentParser(description='Model family for Time Series Forecasting')

# random seed
parser.add_argument('--random_seed', type=int, default=2024, help='random seed')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
# parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='TQNet',
                    help='model name, options: [TQNet, Informer, Autoformer, ...]')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--task_name', type=str, default='long_term_forecast',
                    help='forecasting task used by CrossLinear')
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')  #fixed
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

# TQNet & CycleNet
parser.add_argument('--cycle', type=int, default=24, help='cycle length')
parser.add_argument('--model_type', type=str, default='mlp', help='model type, options: [linear, mlp]')
parser.add_argument('--use_revin', type=int, default=1, help='1: use revin or 0: no revin')

# CrossLinear (model parameters, independent of loss_alpha/loss_beta)
parser.add_argument('--alpha', type=float, default=1.0,
                    help='initial value of the learnable CrossLinear alpha coefficient')
parser.add_argument('--beta', type=float, default=1.0,
                    help='initial value of the learnable CrossLinear beta coefficient')

# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=0, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

# SegRNN
parser.add_argument('--rnn_type', default='gru', help='rnn_type')
parser.add_argument('--dec_way', default='pmf', help='decode way')
parser.add_argument('--seg_len', type=int, default=48, help='segment length')
parser.add_argument('--channel_id', type=int, default=1, help='Whether to enable channel position encoding')

# Formers 
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

#loss
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--loss_alpha', type=float, default=0, help='1')
parser.add_argument('--loss_beta', type=float, default=0, help='2')
parser.add_argument('--loss_gamma', type=float, default=0, help='3')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

# Unified, non-overwriting evaluation / lightweight training records.
parser.add_argument('--evaluation_enabled', action='store_true', default=False,
                    help='write only to a unique TQNet/evaluation run directory')
parser.add_argument('--evaluation_config', type=str, default=None,
                    help='optional dependency-free JSON evaluation config')
parser.add_argument('--evaluation_root', type=str, default='./TQNet/evaluation',
                    help='independent evaluation output root (must not be a TSout directory)')
parser.add_argument('--evaluation_experiment_id', type=str, default='UNIFIED_EVALUATION')
parser.add_argument('--evaluation_resume', action='store_true', default=False,
                    help='explicitly continue an existing evaluation run directory')
parser.add_argument('--evaluation_source_dir', type=str, default=None,
                    help='legacy result root, e.g. TSout_main or TSout_abl; checkpoint and result_sum are resolved from setting')

args = parser.parse_args()

random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main


def build_setting(current_args):
    return '{}_a{}_b{}_g{}_{}_{}_{}_ft{}_sl{}_pl{}_cycle{}_epo{}_{}_{}_lr{}_seed{}'.format(
        current_args.loss,
        current_args.loss_alpha,
        current_args.loss_beta,
        current_args.loss_gamma,
        current_args.data_path,
        current_args.model,
        current_args.data,
        current_args.features,
        current_args.seq_len,
        current_args.pred_len,
        current_args.cycle,
        current_args.train_epochs,
        current_args.patience,
        current_args.batch_size,
        current_args.learning_rate,
        current_args.random_seed,
    )


def execute_one(current_args, setting, training, evaluation_run_name=None):
    evaluation_run = None
    if current_args.evaluation_enabled:
        from evaluation.artifacts import EvaluationRun
        evaluation_run = EvaluationRun(
            current_args,
            setting,
            run_id_source=evaluation_run_name,
        )
        current_args._evaluation_run = evaluation_run
        current_args.evaluation_run_dir = str(evaluation_run.run_dir)
    try:
        exp = Exp(current_args)
        if evaluation_run:
            evaluation_run.update_manifest(
                total_parameters=sum(value.numel() for value in exp.model.parameters()),
                trainable_parameters=sum(value.numel() for value in exp.model.parameters() if value.requires_grad),
            )
        if training:
            if evaluation_run:
                evaluation_run.set_stage('training')
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
        if evaluation_run:
            evaluation_run.set_stage('evaluation')
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=0 if training else 1)
        if evaluation_run:
            evaluation_run.mark_success()
        torch.cuda.empty_cache()
    except BaseException as exc:
        if evaluation_run:
            evaluation_run.mark_failure(exc)
        raise


if args.is_training:
    for ii in range(args.itr):
        setting = build_setting(args)
        evaluation_run_name = setting if args.itr == 1 else '{}_itr{}'.format(setting, ii)
        execute_one(args, setting, training=True, evaluation_run_name=evaluation_run_name)
else:
    setting = build_setting(args)
    execute_one(args, setting, training=False)
