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

batch_freq = 10
batch_start = 10
batch_end = 120
perc_evals = [0.01, 0.05, 0.1, 0.25, 0.5]

num_evals = 20
eval_mode = "tta"
num_mixture = 10
gece_sigma = 0.1
gece_distr = "gaussian"

plot_name = "metric_plot"
plot_path =  plot_name + '_' + eval_mode + '-ci_' + gece_distr + '_sigma-' + str(gece_sigma) + '.png'

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
    
    ece_list = []
    sce_list = []
    aece_list = []
    jce_list = []
    gece_list = []
    gece_sd_list = []
    gmece_list = []
    gmece_sd_list = []
    gmece_smd_list = []
    
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
        
        eces = ECELoss().loss(cls_outputs_subset, cls_targets_subset, n_bins=15)
        cces = SCELoss().loss(cls_outputs_subset, cls_targets_subset, n_bins=15)
        #jces = ClassjECELoss().loss(tta_mean, tta_targets, n_bins=15)
        aeces = AdaptiveECELoss().loss(tta_mean, tta_targets, n_bins=15)
        geces = GECELoss().loss(cls_outputs_subset, cls_targets_subset, n_bins=15, distr_name=gece_distr, sigma=gece_sigma)
        geces_sd = GECELoss().loss(cls_outputs_subset, cls_targets_subset, n_bins=15, distr_name=gece_distr, sigma=tta_std, sd=True)
        gmeces = GMECELoss().loss(tta_predictions, np.array([tta_targets]*tta_evals), cls_outputs_subset, cls_targets_subset, n_total, n_bins=15, distr_name=gece_distr, sigma=gece_sigma, mode="default")
        gmeces_sd = GMECELoss().loss(tta_predictions, np.array([tta_targets]*tta_evals), cls_outputs_subset, cls_targets_subset, n_total, n_bins=15, distr_name=gece_distr, sigma=tta_std, mode="sd")
        gmeces_smd = GMECELoss().loss(tta_predictions, np.array([tta_targets]*tta_evals), cls_outputs_subset, cls_targets_subset, n_total, n_bins=15, distr_name=gece_distr, sigma=tta_std, mode="gmm_estimate")

        ece_list.append(eces)
        sce_list.append(cces)
        #jce_list.append(jces)
        aece_list.append(aeces)
        gece_list.append(geces)
        gece_sd_list.append(geces_sd)
        gmece_list.append(gmeces)
        gmece_sd_list.append(gmeces_sd)
        gmece_smd_list.append(gmeces_smd)

    return (losses.avg, top1.avg, top3.avg, top5.avg, sce_list, ece_list, jce_list, aece_list, gece_list, gece_sd_list, gmece_list, gmece_sd_list, gmece_smd_list)
    
def get_random_sample_loader(args, testset, dataset_size):
    random_indices = np.random.choice(len(testset), dataset_size, replace=False)
    testset_subset = torch.utils.data.Subset(testset, random_indices)
    test_loader = data.DataLoader(testset_subset, batch_size=args.test_batch_size, num_workers=args.workers)
    return test_loader, random_indices

if __name__ == "__main__":
    
    args = parse_args()

    current_time = strftime("%d-%b", localtime())
    save_dir = "checkpoint/" + args.dataset + "/" + "metric_comparisons/"
    if not os.path.isdir(save_dir):
        mkdir_p(save_dir)

    file_logs = open(save_dir + plot_path[:-3] + '.txt', 'w')
    file_logs.write("Writing to logs: \n")

    num_classes = dataset_nclasses_dict[args.dataset]
    classes_name_list = dataset_classname_dict[args.dataset]
    
    # prepare model
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
    if args.dataset == "gbc_usg" or args.dataset == "busi":    
        test_loss, top1, top3, top5, sce_score, ece_score, jce_score, aece_score, gece_score, acc3, spec, sens, acc2, cls_acc, cf, cls_outputs, cls_targets = test_gbc(testloader, model, test_criterion, gece_sigma, gece_distr)
    else:
        test_loss, top1, top3, top5, sce_score, ece_score, jce_score, aece_score, gece_score, acc3, spec, sens, acc2, cls_acc, cf, cls_outputs, cls_targets = test_gbc_old(testloader, model, test_criterion, gece_sigma, gece_distr)
        
    print("Single run only results:")
    print("Top 1: %.4f  Top 3: %.4f Top 5: %.4f"%(top1, top3, top5))
    print("2cls Acc %.4f Spec %.4f Sens %.4f 3cls Acc %.4f Cls wise Acc %.4f %.4f %.4f"%(acc2, spec, sens, acc3, cls_acc[0], cls_acc[1], cls_acc[2]))
    print("Calibration results:")
    print("SCE: %.4f AECE: %.4f ECE: %.4f GECE: %.4f"%(sce_score, aece_score, ece_score, gece_score))

    file_logs.write("Classification Evaluation: \n")
    file_logs.write("Single run only results: \n")
    file_logs.write("Top 1: %.4f  Top 3: %.4f Top 5: %.4f \n"%(top1, top3, top5))
    file_logs.write("2cls Acc %.4f Spec %.4f Sens %.4f 3cls Acc %.4f Cls wise Acc %.4f %.4f %.4f \n"%(acc2, spec, sens, acc3, cls_acc[0], cls_acc[1], cls_acc[2]))
    file_logs.write("Calibration results: \n")
    file_logs.write("SCE: %.4f AECE: %.4f ECE: %.4f GECE: %.4f \n"%(sce_score, aece_score, ece_score, gece_score))

    file_logs.flush()
    
    print("Calibration Evaluation:")
    if eval_mode == "mc":
        if args.dataset == "gbc_usg" or args.dataset == "busi":    
            _, _, _, _, sce_score, ece_score, jce_score, gece_score, gece_sd_score, _, _, _, _, _, _, _, _ = test_mc(testloader, model, test_criterion, gece_sigma=gece_sigma, gece_distr=gece_distr)
        else:
            _, _, _, _, sce_score, ece_score, jce_score, gece_score, gece_sd_score, _, _, _, _, _, _, _, _ = test_mc(testloader, model, test_criterion, gbc=False, gece_sigma=gece_sigma, gece_distr=gece_distr)
    else:
        trainloader, valloader, testloader = dataloader_dict[args.dataset](args, tta=True)
        if args.dataset == "gbc_usg" or args.dataset == "busi":
            _, _, _, _, sce_score, ece_score, jce_score, gece_score, gece_sd_score, gmece_score, gmece_smd_score, _, _, _, _, _, _, _, _ = test_tta(testloader, model, test_criterion, cls_outputs, cls_targets, gbc = True, num_classes = num_classes, gece_sigma=gece_sigma, gece_distr=gece_distr)
        else:
            _, _, _, _, sce_score, ece_score, jce_score, gece_score, gece_sd_score, gmece_score, gmece_smd_score, _, _, _, _, _, _, _, _ = test_tta(testloader, model, test_criterion, cls_outputs, cls_targets, gbc = False, num_classes = num_classes, gece_sigma=gece_sigma, gece_distr=gece_distr)
            
    print("Results Combined:")
    print("Top 1: %.4f  Top 3: %.4f Top 5: %.4f"%(top1, top3, top5))
    print("2cls Acc %.4f Spec %.4f Sens %.4f 3cls Acc %.4f Cls wise Acc %.4f %.4f %.4f"%(acc2, spec, sens, acc3, cls_acc[0], cls_acc[1], cls_acc[2]))
    print("Calibration results:")
    print("SCE: %.4f AECE: %.4f ECE: %.4f GECE: %.4f GECE-SD: %.4f GMECE: %.4f GMECE-SMD: %.4f"%(sce_score, aece_score, ece_score, gece_score, gece_sd_score, gmece_score, gmece_smd_score))

    file_logs.write("Calibration Evaluation: \n")
    file_logs.write("Results Combined: \n")
    file_logs.write("Top 1: %.4f  Top 3: %.4f Top 5: %.4f \n"%(top1, top3, top5))
    file_logs.write("2cls Acc %.4f Spec %.4f Sens %.4f 3cls Acc %.4f Cls wise Acc %.4f %.4f %.4f \n"%(acc2, spec, sens, acc3, cls_acc[0], cls_acc[1], cls_acc[2]))
    file_logs.write("Calibration results: \n")
    file_logs.write("SCE: %.4f AECE: %.4f ECE: %.4f GECE: %.4f GECE-SD: %.4f GMECE: %.4f GMECE-SMD: %.4f \n"%(sce_score, aece_score, ece_score, gece_score, gece_sd_score, gmece_score, gmece_smd_score))

    file_logs.flush()
    
    
    if args.dataset == "gbc_usg" or args.dataset == "busi":
        gbc=True
    else:
        gbc=False
    
    ece_mean = []
    ece_std = []
    sce_mean = []
    sce_std = []
    aece_mean = []
    aece_std = []
    gece_mean = []
    gece_std = []
    gece_sd_mean = []
    gece_sd_std = []
    gmece_mean = []
    gmece_std = []
    gmece_sd_mean = []
    gmece_sd_std = []
    gmece_smd_mean = []
    gmece_smd_std = []
    
    batch_list_output = np.empty(0)
    ece_mean_output = np.empty(0)
    sce_mean_output = np.empty(0)
    aece_mean_output = np.empty(0)
    gece_mean_output = np.empty(0)
    gece_sd_mean_output = np.empty(0)
    gmece_mean_output = np.empty(0)
    gmece_sd_mean_output = np.empty(0)
    gmece_smd_mean_output = np.empty(0)
    
    batch_list = []
    perc_vals = [math.ceil(num_samples * val) for val in perc_evals]
    if perc_vals[0] == 1:
        perc_vals[0] = 2
    batch_list.extend(perc_vals)
    print("Batchwise Eval:")
    for i in batch_list:
        print("Batch size", i)
        if eval_mode == "mc":
            train_dataset, test_dataset = dataset_dict[args.dataset](args)
            test_loss, top1, top3, top5, sce_list, ece_list, jce_list, aece_list, gece_list, gece_sd_list, gmece_list, gmece_sd_list, gmece_smd_list = test_batchwise_mc(args, test_dataset, model, test_criterion, cls_outputs, cls_targets, i, gbc, num_classes, num_mixture)
        else:
            train_dataset, test_dataset = dataset_dict[args.dataset](args, tta=True)
            test_loss, top1, top3, top5, sce_list, ece_list, jce_list, aece_list, gece_list, gece_sd_list, gmece_list, gmece_sd_list, gmece_smd_list = test_batchwise_tta(args, test_dataset, model, test_criterion, cls_outputs, cls_targets, i, gbc, num_classes, num_mixture)
            
        ece_list = np.array(ece_list)
        sce_list = np.array(sce_list)
        aece_list = np.array(aece_list)
        gece_list = np.array(gece_list)
        gece_sd_list = np.array(gece_sd_list)
        gmece_list = np.array(gmece_list)
        gmece_sd_list = np.array(gmece_sd_list)
        gmece_smd_list = np.array(gmece_smd_list)
        print("ECE stats:", np.mean(ece_list), "+-", np.std(ece_list))
        print("SCE stats:", np.mean(sce_list), "+-", np.std(sce_list))
        print("AECE stats:", np.mean(aece_list), "+-", np.std(aece_list))
        print("GECE stats:", np.mean(gece_list), "+-", np.std(gece_list))
        print("GECE-SD stats:", np.mean(gece_sd_list), "+-", np.std(gece_sd_list))
        print("GMECE stats:", np.mean(gmece_list), "+-", np.std(gmece_list))
        print("GMECE-SD stats:", np.mean(gmece_sd_list), "+-", np.std(gmece_sd_list))
        print("GMECE-SMD stats:", np.mean(gmece_smd_list), "+-", np.std(gmece_smd_list))

        file_logs.write("Batch size" + str(i) + "\n")
        file_logs.write("ECE stats:" + str(np.mean(ece_list)) + "+-" + str(np.std(ece_list)) + "\n")
        file_logs.write("SCE stats:" + str(np.mean(sce_list)) + "+-" + str(np.std(sce_list)) + "\n")
        file_logs.write("AECE stats:" + str(np.mean(aece_list)) + "+-" + str(np.std(aece_list)) + "\n")
        file_logs.write("GECE stats:" + str(np.mean(gece_list)) + "+-" + str(np.std(gece_list)) + "\n")
        file_logs.write("GECE-SD stats:" + str(np.mean(gece_sd_list)) + "+-" + str(np.std(gece_sd_list)) + "\n")
        file_logs.write("GMECE stats:" + str(np.mean(gmece_list)) + "+-" + str(np.std(gmece_list)) + "\n")
        file_logs.write("GMECE-SD stats:" + str(np.mean(gmece_sd_list)) + "+-" + str(np.std(gmece_sd_list)) + "\n")
        file_logs.write("GMECE-SMD stats:" + str(np.mean(gmece_smd_list)) + "+-" + str(np.std(gmece_smd_list)) + "\n")
        file_logs.write("\n")
        file_logs.flush()

        ece_mean.append(np.mean(ece_list))
        ece_std.append(np.std(ece_list))
        sce_mean.append(np.mean(sce_list))
        sce_std.append(np.std(sce_list))
        aece_mean.append(np.mean(aece_list))
        aece_std.append(np.std(aece_list))
        gece_mean.append(np.mean(gece_list))
        gece_std.append(np.std(gece_list))
        gece_sd_mean.append(np.mean(gece_sd_list))
        gece_sd_std.append(np.std(gece_sd_list))
        gmece_mean.append(np.mean(gmece_list))
        gmece_std.append(np.std(gmece_list))
        gmece_sd_mean.append(np.mean(gmece_sd_list))
        gmece_sd_std.append(np.std(gmece_sd_list))
        gmece_smd_mean.append(np.mean(gmece_smd_list))
        gmece_smd_std.append(np.std(gmece_smd_list))
        
        batch_list_output = np.concatenate((batch_list_output, np.full(num_evals, i)))
        ece_mean_output = np.concatenate((ece_mean_output, np.array(ece_list)))
        sce_mean_output = np.concatenate((sce_mean_output, np.array(sce_list)))
        aece_mean_output = np.concatenate((aece_mean_output, np.array(aece_list)))
        gece_mean_output = np.concatenate((gece_mean_output, np.array(gece_list)))
        gece_sd_mean_output = np.concatenate((gece_sd_mean_output, np.array(gece_sd_list)))
        gmece_mean_output = np.concatenate((gmece_mean_output, np.array(gmece_list)))
        gmece_sd_mean_output = np.concatenate((gmece_sd_mean_output, np.array(gmece_sd_list)))
        gmece_smd_mean_output = np.concatenate((gmece_smd_mean_output, np.array(gmece_smd_list)))
    
    
    #Calculate absolute value between batchwise metric and actual value
    ece_mean_output = np.abs(ece_mean_output - ece_score)
    gece_mean_output = np.abs(gece_mean_output - gece_score)
    gece_sd_mean_output = np.abs(gece_sd_mean_output - gece_sd_score)
    sce_mean_output = np.abs(sce_mean_output - sce_score)
    aece_mean_output = np.abs(aece_mean_output - aece_score)
    gmece_mean_output = np.abs(gmece_mean_output - gmece_score)
    gece_mean_output = np.abs(gece_mean_output - gece_score)
    gmece_smd_mean_output = np.abs(gmece_smd_mean_output - gmece_smd_score)

    data_metrics = pd.DataFrame({
        'Data Size': batch_list_output, 
        'ECE': ece_mean_output,
        #'RECE Mean': gece_sd_mean_output,
        'SCE': sce_mean_output,
        'AECE': aece_mean_output,
        'RECE': gece_mean_output,
        'RECE-M': gmece_smd_mean_output,
        #'RECE Estimate SD': gece_sd_mean_output,
        #'RECE-M Fixed SD': gmece_mean_output,
        #'RECE-M Estimate SD': gmece_sd_mean_output,
    })

    data_metrics.to_pickle(save_dir + plot_path[:-3] + '.pkl')

    file_logs.write("Perc Outputs: \n")
    file_logs.write("ECE Mean:" + str(ece_mean[-len(perc_evals):]) + "\n")
    file_logs.write("ECE Std:" + str(ece_std[-len(perc_evals):]) + "\n")
    file_logs.write("SCE Mean:" + str(sce_mean[-len(perc_evals):]) + "\n")
    file_logs.write("SCE Std:" + str(sce_std[-len(perc_evals):]) + "\n")
    file_logs.write("AECE Mean:" + str(aece_mean[-len(perc_evals):]) + "\n")
    file_logs.write("AECE Std:" + str(aece_std[-len(perc_evals):]) + "\n")
    file_logs.write("GECE Mean:" + str(gece_mean[-len(perc_evals):]) + "\n")
    file_logs.write("GECE Std:" + str(gece_std[-len(perc_evals):]) + "\n")
    file_logs.write("GECE-SD Mean:" + str(gece_sd_mean[-len(perc_evals):]) + "\n")
    file_logs.write("GECE-SD Std:" + str(gece_sd_std[-len(perc_evals):]) + "\n")
    file_logs.write("GMECE Mean:" + str(gmece_mean[-len(perc_evals):]) + "\n")
    file_logs.write("GMECE Std:" + str(gmece_std[-len(perc_evals):]) + "\n")
    file_logs.write("GMECE-SD Mean:" + str(gmece_sd_mean[-len(perc_evals):]) + "\n")
    file_logs.write("GMECE-SD Std:" + str(gmece_sd_std[-len(perc_evals):]) + "\n")
    file_logs.write("GMECE-SMD Mean:" + str(gmece_smd_mean[-len(perc_evals):]) + "\n")
    file_logs.write("GMECE-SMD Std:" + str(gmece_smd_std[-len(perc_evals):]) + "\n")

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

    