import os

import torch
import torch.optim as optim
import numpy as np

from torch.utils.data import RandomSampler
from torch.utils import data

from utils import mkdir_p, parse_args
from utils import get_lr, save_checkpoint, create_save_path

from solvers.runners import test_gbc, test_gbc_old, test_mc, test_tta
from solvers.loss import loss_dict

from models import model_dict
from datasets import dataloader_dict, dataset_nclasses_dict, dataset_classname_dict, dataset_dict

from time import localtime, strftime

import logging

import matplotlib
matplotlib.use('Agg')

import calibration_library.visualization as visualization
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math

from tqdm import tqdm
from utils import AverageMeter, accuracy
from calibration_library.metrics import ECELoss, SCELoss, ClassjECELoss, AdaptiveECELoss, GECELoss, GMECELoss

num_evals = 20
eval_mode = "tta"
num_mixture = 10
gece_sigma = 0.1
gece_distr = "gaussian"

plot_name = "metric_plot_ablation-distr"
plot_path =  plot_name + '_' + eval_mode + '-ci_' + gece_distr + '_sigma-' + str(gece_sigma) + '.png'

if __name__ == "__main__":
    
    args = parse_args()

    save_dir = "checkpoint/" + args.dataset + "/" + "metric_comparisons/"
    if not os.path.isdir(save_dir):
        mkdir_p(save_dir)

    data_metrics = pd.read_pickle(args.resume)

    for column_headers in data_metrics.columns: 
        print(column_headers)

    
    data_metrics = data_metrics.rename(columns={'RECE Gaussian': 'RECE-G Gaussian', 'RECE Cauchy': 'RECE-G Cauchy','RECE T': 'RECE-G T','RECE Exponential': 'RECE-G Exponential',})
    
    #For printing only ece plot
    #df_ece = data_metrics[['Data Size', 'ECE']].copy()

    sns.set_theme()
    sns.set(font_scale=1.5)

    #sns_plot = sns.lineplot(x='Data Size', y='value', hue='variable', data=pd.melt(df_ece, ['Data Size']))
    sns_plot = sns.lineplot(x='Data Size', y='value', hue='variable', data=pd.melt(data_metrics, ['Data Size']))
    #sns_plot.axhline(ece_score, linestyle = "--")
    #sns_plot.axhline(gece_score, linestyle = "--", color = "orange")
    #sns_plot.axhline(gece_sd_score, linestyle = "--", color = "green")
    fig = sns_plot.get_figure()
    sns_plot.legend(fontsize=15, title="Metric")
    plt.xlabel("Data Size")
    plt.ylabel("Abs. diff. with actual metric")
    fig.savefig(save_dir + '/' + plot_path ,bbox_inches='tight')
