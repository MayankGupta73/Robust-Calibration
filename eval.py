import os

import torch
import torch.optim as optim

from utils import mkdir_p, parse_args
from utils import get_lr, save_checkpoint, create_save_path

from solvers.runners import test_gbc, test_gbc_old
from solvers.loss import loss_dict

from models import model_dict
from datasets import dataloader_dict, dataset_nclasses_dict, dataset_classname_dict

from time import localtime, strftime

import logging

import calibration_library.visualization as visualization
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')


if __name__ == "__main__":
    
    args = parse_args()

    current_time = strftime("%d-%b", localtime())
    # prepare save path

    save_dir = "checkpoint/" + args.dataset + "/" + args.checkpoint.split('/')[-1][:-4]
    if not os.path.isdir(save_dir):
        mkdir_p(save_dir)
        
    logging.basicConfig(level=logging.INFO, 
                        format="%(levelname)s:  %(message)s",
                        handlers=[
                            logging.FileHandler(filename=os.path.join(save_dir, "eval.log")),
                            logging.StreamHandler()
                        ])

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
        
    print(model)

    # set up dataset
    #logging.info(f"Using dataset : {args.dataset}")
    trainloader, valloader, testloader = dataloader_dict[args.dataset](args)

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
    
    if args.dataset == "gbc_usg" or args.dataset == "busi":    
        test_loss, top1, top3, top5, sce_score, ece_score, jce_score, aece_score, gece_score, acc3, spec, sens, acc2, cls_acc, cf, all_outputs, all_targets = test_gbc(testloader, model, test_criterion)
    else:
        test_loss, top1, top3, top5, sce_score, ece_score, jce_score, aece_score, gece_score, acc3, spec, sens, acc2, cls_acc, cf, all_outputs, all_targets = test_gbc_old(testloader, model, test_criterion)
    
    print("Eval results:")
    print("2cls Acc %.4f Spec %.4f Sens %.4f 3cls Acc %.4f Cls wise Acc %.4f %.4f %.4f"%(acc2, spec, sens, acc3, cls_acc[0], cls_acc[1], cls_acc[2]))
    print("Calibration results:")
    print("SCE: %.4f  ECE: %.4f GECE: %.4f"%(sce_score, ece_score, gece_score))
    print("Adaptive ECE: %.4f"%(aece_score))
    print("Confusion matrix")
    print(cf)
    
    logging.info("Results:")
    logging.info("%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f"%(acc3, spec, sens, acc2, ece_score, sce_score, gece_score, aece_score))
    logging.info("Class-j ECE:")
    logging.info(str(jce_score))
    
    
    conf_hist = visualization.ConfidenceHistogram()
    plt_test = conf_hist.plot(all_outputs, all_targets, title="Confidence Histogram")
    plt_test.savefig(save_dir + '/conf_histogram_test.png',bbox_inches='tight')
    
    rel_diagram = visualization.ReliabilityDiagram()
    plt_test_2 = rel_diagram.plot(all_outputs, all_targets, title="Reliability Diagram")
    plt_test_2.savefig(save_dir + '/rel_diagram_test.png',bbox_inches='tight')
    
    pred_diagram = visualization.AugmentedReliabilityDiagram()
    plt_test_3 = pred_diagram.plot(all_outputs, all_targets, title="Augmented Reliabilty Diagram")
    plt_test_3.savefig(save_dir + '/aug-pred_diagram_test.png',bbox_inches='tight')
    
    perc_diagram = visualization.AugmentedConfidenceDiagram()
    plt_test_4 = perc_diagram.plot(all_outputs, all_targets, title="Augmented Confidence Diagram")
    plt_test_4.savefig(save_dir + '/aug-conf_diagram_test.png',bbox_inches='tight')
    
    arel_diagram = visualization.AdaptiveReliabilityDiagram()
    plt_test_5 = arel_diagram.plot(all_outputs, all_targets, title="Adaptive Reliability Diagram")
    plt_test_5.savefig(save_dir + '/arel_diagram_test.png',bbox_inches='tight')
    
    gconf_diagram = visualization.GaussianConfidenceHistogram()
    plt_test_6 = gconf_diagram.plot(all_outputs, all_targets, title="Gaussian Confidence Histogram")
    plt_test_6.savefig(save_dir + '/gconf_histogram_test.png',bbox_inches='tight')
    
    grel_diagram = visualization.GaussianReliabilityDiagram()
    plt_test_7 = grel_diagram.plot(all_outputs, all_targets, title="Gaussian Reliability Diagram")
    plt_test_7.savefig(save_dir + '/grel_diagram_test.png',bbox_inches='tight')

    