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

batch_freq = 200
batch_start = 200
batch_end = 4000
perc_evals = [0.01, 0.05, 0.1, 0.25, 0.5]

num_evals = 20
eval_mode = "tta"
num_mixture = 10
gece_sigma = 0.1

plot_name = "metric_plot_distribution-ablation"
plot_path =  plot_name + '_' + eval_mode + '-ci_sigma-' + str(gece_sigma) + '.png'

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

@torch.no_grad()
def test_batchwise_mc(args, testset, model, criterion, cls_outputs, cls_targets, dataset_size, gbc = True, num_classes = 3, mc_evals = 10):
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    top5 = AverageMeter()
    
    ece_list = []
    sce_list = []
    aece_list = []
    jce_list = []
    gece_list = []
    gece_sd_list = []
    
    print("Running", str(num_evals), "Evaluations with MC:")
    for i in range(num_evals):

        all_targets = None
        all_outputs = None
        all_preds = None
        
        mc_targets = None
      
        # switch to train mode for MC dropout
        #model.train()
        
        testloader, random_indices = get_random_sample_loader(args, testset, dataset_size)
        
        n_classes = num_classes
        n_samples = len(random_indices)
      
        dropout_predictions = np.empty((0, n_samples, n_classes))
        softmax = torch.nn.Softmax(dim=1)
        
        for n in range(mc_evals):
            predictions = np.empty((0, n_classes))
            model.eval()
            enable_dropout(model)
            
            bar = tqdm(enumerate(testloader), total=len(testloader))
            for batch_idx, (inputs, targets) in bar:
        
                inputs, targets = inputs.squeeze(0).double().cuda(), targets.squeeze(0).cuda()
        
                # compute output
                outputs = model(inputs)
                outputs_softmax = softmax(outputs)
                _, preds = torch.max(outputs, dim=1)
                loss = criterion(outputs, targets)
                
                if gbc:
                    pred_label = torch.max(preds)
                    pred_idx = pred_label.item()
                    pred_label = pred_label.unsqueeze(0)
                    idx = torch.argmax(preds)
                
                prec1, prec3, prec5  = accuracy(outputs.data, targets.data, topk=(1, 3, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top3.update(prec3.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))
        
                targets = targets.detach().cpu().numpy()
                outputs = outputs.detach().cpu().numpy()
                preds = preds.detach().cpu().numpy()
                outputs_softmax = outputs_softmax.detach().cpu().numpy()
                
                if gbc:
                    if all_targets is None:
                        all_outputs = outputs[idx.item()][None, :]
                        all_targets = [targets[0]]
                        all_preds = [pred_label.squeeze(0).cpu().numpy()]
                    else:
                        all_targets = np.concatenate([all_targets, [targets[0]]], axis=0)
                        all_outputs = np.concatenate([all_outputs, outputs[idx.item()][None, :]], axis=0)
                        all_preds = np.concatenate([all_preds, [pred_label.squeeze(0).cpu().numpy()]], axis=0)
                    predictions = np.vstack((predictions, outputs_softmax[idx.item()][None, :]))
      
                else:
                    if all_targets is None:
                        all_outputs = outputs
                        all_targets = targets
                        all_preds = preds
                    else:
                        all_targets = np.concatenate([all_targets, targets], axis=0)
                        all_outputs = np.concatenate([all_outputs, outputs], axis=0)
                        all_preds = np.concatenate([all_preds, preds], axis=0)
                    predictions = np.vstack((predictions, outputs_softmax))
      
        
                # plot progress
                bar.set_postfix_str('({batch}/{size}) Loss: {loss:.8f} | top1: {top1: .4f} | top3: {top3: .4f} | top5: {top5: .4f}'.format(
                            batch=batch_idx + 1,
                            size=len(testloader),
                            loss=losses.avg,
                            top1=top1.avg,
                            top3=top3.avg,
                            top5=top5.avg,
                            ))
                            
            dropout_predictions = np.vstack((dropout_predictions, predictions[np.newaxis, :, :]))
            if mc_targets is None:
                mc_targets = all_targets
                
        # Calculating mean across multiple MCD forward passes 
        mean = np.mean(dropout_predictions, axis=0) # shape (n_samples, n_classes)
        
        #Prediction from average logits
        mc_preds = np.argmax(mean, axis=1)  # shape (n_samples)
        
        # Calculating standard deviation across MCD forward passes predictions
        std_mc = np.std(dropout_predictions, axis=0) # shape (n_samples, n_classes)
        std = std_mc[np.arange(len(std_mc)), mc_preds] # shape (n_samples)
              
        cls_outputs_subset, cls_targets_subset = cls_outputs[random_indices], cls_targets[random_indices]
        
        eces = ECELoss().loss(cls_outputs_subset, cls_targets_subset, n_bins=15)
        sces = SCELoss().loss(cls_outputs_subset, cls_targets_subset, n_bins=15)
        #jces = ClassjECELoss().loss(tta_mean, tta_targets, n_bins=15)
        aeces = AdaptiveECELoss().loss(cls_outputs_subset, cls_targets_subset, n_bins=15)
        geces = GECELoss().loss(cls_outputs_subset, cls_targets_subset, n_bins=15, distr_name=gece_distr, sigma=gece_sigma)
        geces_sd = GECELoss().loss(cls_outputs_subset, cls_targets_subset, n_bins=15, distr_name=gece_distr, sigma=std, sd=True)
        
        ece_list.append(eces)
        sce_list.append(sces)
        #jce_list.append(jces)
        aece_list.append(aeces)
        gece_list.append(geces)
        gece_sd_list.append(geces_sd)
        

    return (losses.avg, top1.avg, top3.avg, top5.avg, sce_list, ece_list, jce_list, aece_list, gece_list, gece_sd_list)
    
@torch.no_grad()
def test_batchwise_tta(args, testset, model, criterion, cls_outputs, cls_targets, dataset_size, gbc = True, num_classes = 3, tta_evals = 10):
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    top5 = AverageMeter()
    
    gaussian_list = []
    t_list = []
    cauchy_list = []
    exponential_list = []
    
    print("Running", str(num_evals), "Evaluations with TTA:")
    for i in range(num_evals):

        all_targets = None
        all_outputs = None
        all_preds = None
        
        tta_targets = None
      
        testloader, random_indices = get_random_sample_loader(args, testset, dataset_size)
      
        model.eval()
    
        n_samples = len(random_indices)
        n_total = len(cls_targets)
        tta_predictions = np.empty((0, n_samples, num_classes))
        softmax = torch.nn.Softmax(dim=1)
    
        for n in range(tta_evals):
            predictions = np.empty((0, num_classes))
            bar = tqdm(enumerate(testloader), total=len(testloader))
            for batch_idx, (inputs, targets) in bar:
        
                inputs, targets = inputs.squeeze(0).double().cuda(), targets.squeeze(0).cuda()
        
                # compute output
                outputs = model(inputs)
                outputs_softmax = softmax(outputs)
                _, preds = torch.max(outputs, dim=1)
                loss = criterion(outputs, targets)
                
                if gbc:
                    pred_label = torch.max(preds)
                    pred_idx = pred_label.item()
                    pred_label = pred_label.unsqueeze(0)
                    idx = torch.argmax(preds)
                
                prec1, prec3, prec5  = accuracy(outputs.data, targets.data, topk=(1, 3, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top3.update(prec3.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))
        
                targets = targets.detach().cpu().numpy()
                outputs = outputs.detach().cpu().numpy()
                preds = preds.detach().cpu().numpy()
                outputs_softmax = outputs_softmax.detach().cpu().numpy()
                
                if gbc:
                    if all_targets is None:
                        all_outputs = outputs[idx.item()][None, :]
                        all_targets = [targets[0]]
                        all_preds = [pred_label.squeeze(0).cpu().numpy()]
                    else:
                        all_targets = np.concatenate([all_targets, [targets[0]]], axis=0)
                        all_outputs = np.concatenate([all_outputs, outputs[idx.item()][None, :]], axis=0)
                        all_preds = np.concatenate([all_preds, [pred_label.squeeze(0).cpu().numpy()]], axis=0)
                    predictions = np.vstack((predictions, outputs[idx.item()][None, :]))
    
                else:
                    if all_targets is None:
                        all_outputs = outputs
                        all_targets = targets
                        all_preds = preds
                    else:
                        all_targets = np.concatenate([all_targets, targets], axis=0)
                        all_outputs = np.concatenate([all_outputs, outputs], axis=0)
                        all_preds = np.concatenate([all_preds, preds], axis=0)
                    predictions = np.vstack((predictions, outputs))
    
        
                # plot progress
                bar.set_postfix_str('({batch}/{size}) Loss: {loss:.8f} | top1: {top1: .4f} | top3: {top3: .4f} | top5: {top5: .4f}'.format(
                            batch=batch_idx + 1,
                            size=len(testloader),
                            loss=losses.avg,
                            top1=top1.avg,
                            top3=top3.avg,
                            top5=top5.avg,
                            ))
                            
            tta_predictions = np.vstack((tta_predictions, predictions[np.newaxis, :, :]))
            if tta_targets is None:
                tta_targets = all_targets
    
        tta_mean = np.mean(tta_predictions, axis=0) # shape (n_samples, n_classes)
        tta_preds = np.argmax(tta_mean, axis=1)
        std_val = np.std(tta_predictions, axis=0) # shape (n_samples, n_classes)
        tta_std = std_val[np.arange(len(std_val)), tta_preds] # shape (n_samples)
        
        cls_outputs_subset, cls_targets_subset = cls_outputs[random_indices], cls_targets[random_indices]        

        geces_gaussian = GECELoss().loss(cls_outputs_subset, cls_targets_subset, n_bins=15, distr_name="gaussian", sigma=gece_sigma)
        geces_cauchy = GECELoss().loss(cls_outputs_subset, cls_targets_subset, n_bins=15, distr_name="cauchy", sigma=gece_sigma)
        geces_t = GECELoss().loss(cls_outputs_subset, cls_targets_subset, n_bins=15, distr_name="t", sigma=gece_sigma)
        geces_exponential = GECELoss().loss(cls_outputs_subset, cls_targets_subset, n_bins=15, distr_name="exp-central", sigma=gece_sigma)

        gaussian_list.append(geces_gaussian)
        t_list.append(geces_t)
        cauchy_list.append(geces_cauchy)
        exponential_list.append(geces_exponential)

    return (losses.avg, top1.avg, top3.avg, top5.avg, gaussian_list, t_list, cauchy_list, exponential_list)
    
def get_random_sample_loader(args, testset, dataset_size):
    random_indices = np.random.choice(len(testset), dataset_size, replace=False)
    testset_subset = torch.utils.data.Subset(testset, random_indices)
    test_loader = data.DataLoader(testset_subset, batch_size=args.test_batch_size, num_workers=args.workers)
    return test_loader, random_indices

if __name__ == "__main__":
    
    args = parse_args()

    current_time = strftime("%d-%b", localtime())
    # prepare save path

    save_dir = "checkpoint/" + args.dataset + "/" + "metric_comparisons/"
    if not os.path.isdir(save_dir):
        mkdir_p(save_dir)

    file_logs = open(save_dir + plot_path[:-3] + '.txt', 'w')
    file_logs.write("Writing to logs: \n")

    num_classes = dataset_nclasses_dict[args.dataset]
    classes_name_list = dataset_classname_dict[args.dataset]
    
    # prepare model
    #logging.info(f"Using model : {args.model}")
    model = model_dict[args.model](num_classes=num_classes)
    
    if args.model == "gbcnet":
        model.net.load_state_dict(torch.load(args.checkpoint, map_location = 'cpu')['state_dict'])
        model = model.net.double().cuda()
    else:
        model.load_state_dict(torch.load(args.checkpoint, map_location = 'cpu')['state_dict'])
        model = model.double().cuda()
        
    # set up dataset
    #logging.info(f"Using dataset : {args.dataset}")
    trainloader, valloader, testloader = dataloader_dict[args.dataset](args)
    num_samples = len(testloader.dataset)

    #logging.info(f"Setting up optimizer : {args.optimizer}")

    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), 
                              lr=args.lr, 
                              momentum=args.momentum, 
                              weight_decay=args.weight_decay)

    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr,
                               weight_decay=args.weight_decay)
    
    criterion = loss_dict[args.loss](gamma=args.gamma, alpha=args.alpha, beta=args.beta, loss=args.loss, delta=args.delta)
    test_criterion = loss_dict["cross_entropy"]()
    
    print("Classification Evaluation:")
    print("Gaussian")
    if args.dataset == "gbc_usg" or args.dataset == "busi":    
        test_loss, top1, top3, top5, sce_score, ece_score, jce_score, aece_score, gece_gaussian_score, acc3, spec, sens, acc2, cls_acc, cf, cls_outputs, cls_targets = test_gbc(testloader, model, test_criterion, gece_sigma, "gaussian")
    else:
        test_loss, top1, top3, top5, sce_score, ece_score, jce_score, aece_score, gece_gaussian_score, acc3, spec, sens, acc2, cls_acc, cf, cls_outputs, cls_targets = test_gbc_old(testloader, model, test_criterion, gece_sigma, "gaussian")
        
    print("Single run only results:")
    print("Top 1: %.4f  Top 3: %.4f Top 5: %.4f"%(top1, top3, top5))
    print("2cls Acc %.4f Spec %.4f Sens %.4f 3cls Acc %.4f Cls wise Acc %.4f %.4f %.4f"%(acc2, spec, sens, acc3, cls_acc[0], cls_acc[1], cls_acc[2]))
    print("Calibration results:")
    print("SCE: %.4f AECE: %.4f ECE: %.4f GECE: %.4f"%(sce_score, aece_score, ece_score, gece_gaussian_score))

    file_logs.write("Classification Evaluation Gaussian: \n")
    file_logs.write("Single run only results: \n")
    file_logs.write("Top 1: %.4f  Top 3: %.4f Top 5: %.4f \n"%(top1, top3, top5))
    file_logs.write("2cls Acc %.4f Spec %.4f Sens %.4f 3cls Acc %.4f Cls wise Acc %.4f %.4f %.4f \n"%(acc2, spec, sens, acc3, cls_acc[0], cls_acc[1], cls_acc[2]))
    file_logs.write("Calibration results: \n")
    file_logs.write("SCE: %.4f AECE: %.4f ECE: %.4f GECE: %.4f \n"%(sce_score, aece_score, ece_score, gece_gaussian_score))

    file_logs.flush()

    print("Cauchy")
    if args.dataset == "gbc_usg" or args.dataset == "busi":    
        test_loss, top1, top3, top5, sce_score, ece_score, jce_score, aece_score, gece_cauchy_score, acc3, spec, sens, acc2, cls_acc, cf, cls_outputs, cls_targets = test_gbc(testloader, model, test_criterion, gece_sigma, "cauchy")
    else:
        test_loss, top1, top3, top5, sce_score, ece_score, jce_score, aece_score, gece_cauchy_score, acc3, spec, sens, acc2, cls_acc, cf, cls_outputs, cls_targets = test_gbc_old(testloader, model, test_criterion, gece_sigma, "cauchy")
        
    print("Single run only results:")
    print("Top 1: %.4f  Top 3: %.4f Top 5: %.4f"%(top1, top3, top5))
    print("2cls Acc %.4f Spec %.4f Sens %.4f 3cls Acc %.4f Cls wise Acc %.4f %.4f %.4f"%(acc2, spec, sens, acc3, cls_acc[0], cls_acc[1], cls_acc[2]))
    print("Calibration results:")
    print("SCE: %.4f AECE: %.4f ECE: %.4f GECE: %.4f"%(sce_score, aece_score, ece_score, gece_cauchy_score))

    file_logs.write("Classification Evaluation Cauchy: \n")
    file_logs.write("Single run only results: \n")
    file_logs.write("Top 1: %.4f  Top 3: %.4f Top 5: %.4f \n"%(top1, top3, top5))
    file_logs.write("2cls Acc %.4f Spec %.4f Sens %.4f 3cls Acc %.4f Cls wise Acc %.4f %.4f %.4f \n"%(acc2, spec, sens, acc3, cls_acc[0], cls_acc[1], cls_acc[2]))
    file_logs.write("Calibration results: \n")
    file_logs.write("SCE: %.4f AECE: %.4f ECE: %.4f GECE: %.4f \n"%(sce_score, aece_score, ece_score, gece_cauchy_score))

    file_logs.flush()

    print("T")
    if args.dataset == "gbc_usg" or args.dataset == "busi":    
        test_loss, top1, top3, top5, sce_score, ece_score, jce_score, aece_score, gece_t_score, acc3, spec, sens, acc2, cls_acc, cf, cls_outputs, cls_targets = test_gbc(testloader, model, test_criterion, gece_sigma, "t")
    else:
        test_loss, top1, top3, top5, sce_score, ece_score, jce_score, aece_score, gece_t_score, acc3, spec, sens, acc2, cls_acc, cf, cls_outputs, cls_targets = test_gbc_old(testloader, model, test_criterion, gece_sigma, "t")
        
    print("Single run only results:")
    print("Top 1: %.4f  Top 3: %.4f Top 5: %.4f"%(top1, top3, top5))
    print("2cls Acc %.4f Spec %.4f Sens %.4f 3cls Acc %.4f Cls wise Acc %.4f %.4f %.4f"%(acc2, spec, sens, acc3, cls_acc[0], cls_acc[1], cls_acc[2]))
    print("Calibration results:")
    print("SCE: %.4f AECE: %.4f ECE: %.4f GECE: %.4f"%(sce_score, aece_score, ece_score, gece_t_score))

    file_logs.write("Classification Evaluation T: \n")
    file_logs.write("Single run only results: \n")
    file_logs.write("Top 1: %.4f  Top 3: %.4f Top 5: %.4f \n"%(top1, top3, top5))
    file_logs.write("2cls Acc %.4f Spec %.4f Sens %.4f 3cls Acc %.4f Cls wise Acc %.4f %.4f %.4f \n"%(acc2, spec, sens, acc3, cls_acc[0], cls_acc[1], cls_acc[2]))
    file_logs.write("Calibration results: \n")
    file_logs.write("SCE: %.4f AECE: %.4f ECE: %.4f GECE: %.4f \n"%(sce_score, aece_score, ece_score, gece_t_score))

    file_logs.flush()

    print("Exponential")
    if args.dataset == "gbc_usg" or args.dataset == "busi":    
        test_loss, top1, top3, top5, sce_score, ece_score, jce_score, aece_score, gece_exponential_score, acc3, spec, sens, acc2, cls_acc, cf, cls_outputs, cls_targets = test_gbc(testloader, model, test_criterion, gece_sigma, "exp-central")
    else:
        test_loss, top1, top3, top5, sce_score, ece_score, jce_score, aece_score, gece_exponential_score, acc3, spec, sens, acc2, cls_acc, cf, cls_outputs, cls_targets = test_gbc_old(testloader, model, test_criterion, gece_sigma, "exp-central")
        
    print("Single run only results:")
    print("Top 1: %.4f  Top 3: %.4f Top 5: %.4f"%(top1, top3, top5))
    print("2cls Acc %.4f Spec %.4f Sens %.4f 3cls Acc %.4f Cls wise Acc %.4f %.4f %.4f"%(acc2, spec, sens, acc3, cls_acc[0], cls_acc[1], cls_acc[2]))
    print("Calibration results:")
    print("SCE: %.4f AECE: %.4f ECE: %.4f GECE: %.4f"%(sce_score, aece_score, ece_score, gece_exponential_score))

    file_logs.write("Classification Evaluation Exponential: \n")
    file_logs.write("Single run only results: \n")
    file_logs.write("Top 1: %.4f  Top 3: %.4f Top 5: %.4f \n"%(top1, top3, top5))
    file_logs.write("2cls Acc %.4f Spec %.4f Sens %.4f 3cls Acc %.4f Cls wise Acc %.4f %.4f %.4f \n"%(acc2, spec, sens, acc3, cls_acc[0], cls_acc[1], cls_acc[2]))
    file_logs.write("Calibration results: \n")
    file_logs.write("SCE: %.4f AECE: %.4f ECE: %.4f GECE: %.4f \n"%(sce_score, aece_score, ece_score, gece_exponential_score))

    file_logs.flush()
    
    if args.dataset == "gbc_usg" or args.dataset == "busi":
        gbc=True
    else:
        gbc=False
    
    gece_gaussian_mean = []
    gece_gaussian_std = []
    gece_cauchy_mean = []
    gece_cauchy_std = []
    gece_t_mean = []
    gece_t_std = []
    gece_exponential_mean = []
    gece_exponential_std = []
    
    
    batch_list_output = np.empty(0)
    gece_gaussian_mean_output = np.empty(0)
    gece_cauchy_mean_output = np.empty(0)
    gece_t_mean_output = np.empty(0)
    gece_exponential_mean_output = np.empty(0)
    
    batch_list = list(range(batch_start, batch_end, batch_freq))
    perc_vals = [math.ceil(num_samples * val) for val in perc_evals]
    if perc_vals[0] == 1:
        perc_vals[0] = 2
    batch_list.extend(perc_vals)
    print("Batchwise Eval:")
    for i in batch_list:
        print("Batch size", i)
        if eval_mode == "mc":
            train_dataset, test_dataset = dataset_dict[args.dataset](args)
            test_loss, top1, top3, top5, gaussian_list, t_list, cauchy_list, exponential_list = test_batchwise_mc(args, test_dataset, model, test_criterion, cls_outputs, cls_targets, i, gbc, num_classes, num_mixture)
        else:
            train_dataset, test_dataset = dataset_dict[args.dataset](args, tta=True)
            test_loss, top1, top3, top5, gaussian_list, t_list, cauchy_list, exponential_list = test_batchwise_tta(args, test_dataset, model, test_criterion, cls_outputs, cls_targets, i, gbc, num_classes, num_mixture)
            
        gaussian_list = np.array(gaussian_list)
        cauchy_list = np.array(cauchy_list)
        t_list = np.array(t_list)
        exponential_list = np.array(exponential_list)
        
        print("GECE Gaussian stats:", np.mean(gaussian_list), "+-", np.std(gaussian_list))
        print("GECE Cauchy stats:", np.mean(cauchy_list), "+-", np.std(cauchy_list))
        print("GECE T stats:", np.mean(t_list), "+-", np.std(t_list))
        print("GECE Exponential stats:", np.mean(exponential_list), "+-", np.std(exponential_list))
        

        file_logs.write("Batch size" + str(i) + "\n")
        file_logs.write("GECE Gaussian stats:" + str(np.mean(gaussian_list)) + "+-" + str(np.std(gaussian_list)) + "\n")
        file_logs.write("GECE Cauchy stats:" + str(np.mean(cauchy_list)) + "+-" + str(np.std(cauchy_list)) + "\n")
        file_logs.write("GECE T stats:" + str(np.mean(t_list)) + "+-" + str(np.std(t_list)) + "\n")
        file_logs.write("GECE Exponential stats:" + str(np.mean(exponential_list)) + "+-" + str(np.std(exponential_list)) + "\n")
        file_logs.write("\n")
        file_logs.flush()

        gece_gaussian_mean.append(np.mean(gaussian_list))
        gece_gaussian_std.append(np.std(gaussian_list))
        gece_cauchy_mean.append(np.mean(cauchy_list))
        gece_cauchy_std.append(np.std(cauchy_list))
        gece_t_mean.append(np.mean(t_list))
        gece_t_std.append(np.std(t_list))
        gece_exponential_mean.append(np.mean(exponential_list))
        gece_exponential_std.append(np.std(exponential_list))
        
        
        batch_list_output = np.concatenate((batch_list_output, np.full(num_evals, i)))
        gece_gaussian_mean_output = np.concatenate((gece_gaussian_mean_output, np.array(gaussian_list)))
        gece_cauchy_mean_output = np.concatenate((gece_cauchy_mean_output, np.array(cauchy_list)))
        gece_t_mean_output = np.concatenate((gece_t_mean_output, np.array(t_list)))
        gece_exponential_mean_output = np.concatenate((gece_exponential_mean_output, np.array(exponential_list)))

    
    #Calculate absolute value between batchwise metric and actual value
    gece_gaussian_mean_output = np.abs(gece_gaussian_mean_output - gece_gaussian_score)
    gece_cauchy_mean_output = np.abs(gece_cauchy_mean_output - gece_cauchy_score)
    gece_t_mean_output = np.abs(gece_t_mean_output - gece_t_score)
    gece_exponential_mean_output = np.abs(gece_exponential_mean_output - gece_exponential_score)

    data_metrics = pd.DataFrame({
        'Data Size': batch_list_output, 
        'RECE Gaussian': gece_gaussian_mean_output,
        'RECE Cauchy': gece_cauchy_mean_output,
        'RECE T': gece_t_mean_output,
        'RECE Exponential': gece_exponential_mean_output,
    })

    data_metrics.to_pickle(save_dir + plot_path[:-3] + '.pkl')

    file_logs.write("Perc Outputs: \n")
    file_logs.write("Gaussian Mean:" + str(gece_gaussian_mean[-len(perc_evals):]) + "\n")
    file_logs.write("Gaussian Std:" + str(gece_gaussian_std[-len(perc_evals):]) + "\n")
    file_logs.write("Cauchy Mean:" + str(gece_cauchy_mean[-len(perc_evals):]) + "\n")
    file_logs.write("Cauchy Std:" + str(gece_cauchy_std[-len(perc_evals):]) + "\n")
    file_logs.write("T Mean:" + str(gece_t_mean[-len(perc_evals):]) + "\n")
    file_logs.write("T Std:" + str(gece_t_std[-len(perc_evals):]) + "\n")
    file_logs.write("Exponential Mean:" + str(gece_exponential_mean[-len(perc_evals):]) + "\n")
    file_logs.write("Exponential Std:" + str(gece_exponential_std[-len(perc_evals):]) + "\n")

    file_logs.close()

    sns.set_theme()
    
    sns_plot = sns.lineplot(x='Data Size', y='value', hue='variable', data=pd.melt(data_metrics, ['Data Size']))
    #sns_plot.axhline(ece_score, linestyle = "--")
    #sns_plot.axhline(gece_score, linestyle = "--", color = "orange")
    #sns_plot.axhline(gece_sd_score, linestyle = "--", color = "green")
    fig = sns_plot.get_figure()
    plt.xlabel("Data Size")
    plt.ylabel("Absolute difference with actual metric")
    fig.savefig(save_dir + '/' + plot_path ,bbox_inches='tight')

    