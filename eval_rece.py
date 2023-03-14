import os

import torch
import torch.optim as optim

from utils import mkdir_p, parse_args
from utils import get_lr, save_checkpoint, create_save_path

from solvers.runners import test_gbc, test_gbc_old, test_tta
from solvers.loss import loss_dict

from models import model_dict
from datasets import dataloader_dict, dataset_nclasses_dict, dataset_classname_dict

from time import localtime, strftime

import logging

import calibration_library.visualization as visualization
import matplotlib.pyplot as plt

gece_sigma = 0.1
gece_distr = "gaussian"

if __name__ == "__main__":
    
    args = parse_args()

    current_time = strftime("%d-%b", localtime())
    # prepare save path
    
    save_dir = "checkpoint/" + args.dataset + "/" + args.checkpoint.split('/')[-1][:-4]
    if not os.path.isdir(save_dir):
        mkdir_p(save_dir)

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
    trainloader, valloader, testloader = dataloader_dict[args.dataset](args, tta=True)

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
    
    trainloader, valloader, testloader = dataloader_dict[args.dataset](args, tta=False)
    
    print("Classification Evaluation:")
    if args.dataset == "gbc_usg" or args.dataset == "busi":    
        test_loss, top1, top3, top5, sce_score, ece_score, jce_score, aece_score, gece_score, acc3, spec, sens, acc2, cls_acc, cf, cls_outputs, cls_targets = test_gbc(testloader, model, test_criterion, gece_sigma, gece_distr)
    else:
        test_loss, top1, top3, top5, sce_score, ece_score, jce_score, aece_score, gece_score, acc3, spec, sens, acc2, cls_acc, cf, cls_outputs, cls_targets = test_gbc_old(testloader, model, test_criterion, gece_sigma, gece_distr)
        
    print("Single run only results:")
    print("Top 1: %.4f  Top 3: %.4f Top 5: %.4f"%(top1, top3, top5))
    print("2cls Acc %.4f Spec %.4f Sens %.4f 3cls Acc %.4f Cls wise Acc %.4f %.4f %.4f"%(acc2, spec, sens, acc3, cls_acc[0], cls_acc[1], cls_acc[2]))
    print("Calibration results:")
    print("SCE: %.4f AECE: %.4f ECE: %.4f RECE-G: %.4f"%(sce_score, aece_score, ece_score, gece_score))
    print("Class-j ECE:" + str(jce_score))
    print("Confusion matrix")
    print(cf)
    
    trainloader, valloader, testloader = dataloader_dict[args.dataset](args, tta=True)
    
    print("TTA Evaluation:")    
    if args.dataset == "gbc_usg" or args.dataset == "busi":
        _, _, _, _, sce_score, ece_score, jce_score, gece_score, gece_sd_score, gmece_score, gmece_smd_score, _, _, _, _, _, _, _, _ = test_tta(testloader, model, test_criterion, cls_outputs, cls_targets, gbc = True, num_classes = num_classes, gece_sigma=gece_sigma, gece_distr=gece_distr)
    else:
        _, _, _, _, sce_score, ece_score, jce_score, gece_score, gece_sd_score, gmece_score, gmece_smd_score, _, _, _, _, _, _, _, _ = test_tta(testloader, model, test_criterion, cls_outputs, cls_targets, gbc = False, num_classes = num_classes, gece_sigma=gece_sigma, gece_distr=gece_distr)
        
    print("Results Combined:")
    print("Top 1: %.4f  Top 3: %.4f Top 5: %.4f"%(top1, top3, top5))
    print("2cls Acc %.4f Spec %.4f Sens %.4f 3cls Acc %.4f Cls wise Acc %.4f %.4f %.4f"%(acc2, spec, sens, acc3, cls_acc[0], cls_acc[1], cls_acc[2]))
    print("Calibration results:")
    print("SCE: %.4f AECE: %.4f ECE: %.4f RECE-G: %.4f RECE-G-SD: %.4f RECE-M-Fixed: %.4f RECE-M: %.4f"%(sce_score, aece_score, ece_score, gece_score, gece_sd_score, gmece_score, gmece_smd_score))
    print("Class-j ECE:" + str(jce_score))
    print("Confusion matrix")
    print(cf)

    print("Results:")
    print("%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f"%(acc3, spec, sens, acc2, ece_score, sce_score, gece_score, aece_score, gmece_smd_score))
    print("Class-j ECE:")
    print(str(jce_score))
    
    if num_classes <=3:
        
        conf_hist = visualization.ConfidenceHistogram()
        plt_test = conf_hist.plot(cls_outputs, cls_targets, logits=False, title="TTA Confidence Histogram")
        plt_test.savefig(save_dir + '/conf_histogram_tta.png',bbox_inches='tight')
        
        rel_diagram = visualization.ReliabilityDiagram()
        plt_test_2 = rel_diagram.plot(cls_outputs, cls_targets, logits=False, title="TTA Reliability Diagram")
        plt_test_2.savefig(save_dir + '/rel_diagram_tta.png',bbox_inches='tight')
    
        pred_diagram = visualization.AugmentedReliabilityDiagram()
        plt_test_3 = pred_diagram.plot(cls_outputs, cls_targets, logits=False, title="TTA Augmented Reliabilty Diagram")
        plt_test_3.savefig(save_dir + '/pred_diagram_tta.png',bbox_inches='tight')
        
        perc_diagram = visualization.AugmentedConfidenceDiagram()
        plt_test_4 = perc_diagram.plot(cls_outputs, cls_targets, logits=False, title="TTA Augmented Confidence Diagram")
        plt_test_4.savefig(save_dir + '/perc_diagram_tta.png',bbox_inches='tight')
        
        per_diagram = visualization.ClasswiseAugmentedConfidenceDiagram()
        plt_test_5 = per_diagram.plot(cls_outputs, cls_targets, logits=False, title="TTA Classwise Augmented Confidence Diagram")
        plt_test_5.savefig(save_dir + '/per_diagram_tta.png',bbox_inches='tight')  
        
        cls_diagram = visualization.ClasswiseAugmentedReliabilityDiagram()
        plt_test_6 = cls_diagram.plot(cls_outputs, cls_targets, logits=False, title="TTA Classwise Augmented Reliabilty Diagram")
        plt_test_6.savefig(save_dir + '/cls_diagram_tta.png',bbox_inches='tight')

    