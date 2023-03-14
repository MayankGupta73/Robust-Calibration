import os

import torch
from utils import Logger, parse_args

from solvers.runners import test_gbc, test_gbc_old, test_tta

from models import model_dict
from datasets import dataloader_dict, dataset_nclasses_dict, dataset_classname_dict

from calibration_library.calibrators import TemperatureScaling, DirichletScaling
from calibration_library.recalibration import ModelWithTemperature
import calibration_library.visualization as visualization

import matplotlib
matplotlib.use('Agg')

import logging

gece_sigma = 0.1
gece_distr = "gaussian"

if __name__ == "__main__":
    
    args = parse_args()

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    checkpoint_folder = args.checkpoint[:-4]
    logging.basicConfig(level=logging.INFO, 
                        format="%(levelname)s:  %(message)s",
                        handlers=[
                            logging.FileHandler(filename=os.path.join(checkpoint_folder, "train_post-hoc.log")),
                            logging.StreamHandler()
                        ])

    num_classes = dataset_nclasses_dict[args.dataset]
    classes_name_list = dataset_classname_dict[args.dataset]
    
    # prepare model
    logging.info(f"Using model : {args.model}")
    assert args.checkpoint, "Please provide a trained model file"
    assert os.path.isfile(args.checkpoint)
    logging.info(f'Resuming from saved checkpoint: {args.checkpoint}')
   
    
    saved_model_dict = torch.load(args.checkpoint)

    model = model_dict[args.model](num_classes=num_classes)

    if args.model == "gbcnet":
        model.net.load_state_dict(torch.load(args.checkpoint, map_location = 'cpu')['state_dict'])
        model = model.net.double().cuda()
    else:
        model.load_state_dict(torch.load(args.checkpoint, map_location = 'cpu')['state_dict'])
        model = model.double().cuda()

    # set up dataset
    logging.info(f"Using dataset : {args.dataset}")
    trainloader, valloader, testloader = dataloader_dict[args.dataset](args)
    
    criterion = torch.nn.CrossEntropyLoss()

    # set up loggers
    metric_log_path = os.path.join(checkpoint_folder, 'temperature.txt')
    logger = Logger(metric_log_path, resume=False)

    logger.set_names(['temperature', 'ECE', 'SCE', 'RECE', 'RECE-M'])

    print("Cls Performance")
    if args.dataset == "gbc_usg" or args.dataset == "busi":    
        test_loss, top1, top3, top5, sce_score, ece_score, jce_score, aece_score, gece_score, acc3, spec, sens, acc2, cls_acc, cf, cls_outputs, cls_targets = test_gbc(testloader, model, criterion, gece_sigma, gece_distr)
    else:
        test_loss, top1, top3, top5, sce_score, ece_score, jce_score, aece_score, gece_score, acc3, spec, sens, acc2, cls_acc, cf, cls_outputs, cls_targets = test_gbc_old(testloader, model, criterion, gece_sigma, gece_distr)
    
    print("Calibration Performance")
    trainloader, valloader, testloader = dataloader_dict[args.dataset](args, tta=True)
    if args.dataset == "gbc_usg" or args.dataset == "busi":
        _, _, _, _, sce_score, ece_score, jce_score, gece_score, gece_sd_score, gmece_score, gmece_smd_score, _, _, _, _, _, _, _, _ = test_tta(testloader, model, criterion, cls_outputs, cls_targets, gbc = True, num_classes = num_classes, gece_sigma=gece_sigma, gece_distr=gece_distr)
    else:
        _, _, _, _, sce_score, ece_score, jce_score, gece_score, gece_sd_score, gmece_score, gmece_smd_score, _, _, _, _, _, _, _, _ = test_tta(testloader, model, criterion, cls_outputs, cls_targets, gbc = False, num_classes = num_classes, gece_sigma=gece_sigma, gece_distr=gece_distr)
        
    logger.append(["1.0", ece_score, sce_score, gece_score, gmece_smd_score])
    logging.info("Uncalibrated: 2cls Acc %.4f Spec %.4f Sens %.4f 3cls Acc %.4f Cls wise Acc %.4f %.4f %.4f"%(acc2, spec, sens, acc3, cls_acc[0], cls_acc[1], cls_acc[2]))
    logging.info("CM: " + str(cf))
    logging.info("Stats: loss : {:.4f} | top1 : {:.4f} | ECE : {:.5f} | SCE : {:.5f} | RECE : {:.5f} | RECE-M : {:.5f}".format(
            test_loss,
            top1,
            ece_score,
            sce_score,
            gece_score,
            gmece_smd_score
        ))
    logging.info("Class-j ECE:" + str(jce_score))
    
    conf_hist = visualization.ConfidenceHistogram()
    plt_test = conf_hist.plot(cls_outputs, cls_targets, title="Uncalibrated Confidence Histogram")
    plt_test.savefig(checkpoint_folder + '/conf_histogram_uncal.png',bbox_inches='tight')
    
    rel_diagram = visualization.ReliabilityDiagram()
    plt_test_2 = rel_diagram.plot(cls_outputs, cls_targets, title="Uncalibrated Reliability Diagram")
    plt_test_2.savefig(checkpoint_folder + '/rel_diagram_uncal.png',bbox_inches='tight')

    pred_diagram = visualization.AugmentedReliabilityDiagram()
    plt_test_3 = pred_diagram.plot(cls_outputs, cls_targets, title="Uncalibrated Augmented Reliabilty Diagram")
    plt_test_3.savefig(checkpoint_folder + '/pred_diagram_uncal.png',bbox_inches='tight')
    
    perc_diagram = visualization.AugmentedConfidenceDiagram()
    plt_test_4 = perc_diagram.plot(cls_outputs, cls_targets, title="Uncalibrated Augmented Confidence Diagram")
    plt_test_4.savefig(checkpoint_folder + '/perc_diagram_uncal.png',bbox_inches='tight')
    
    per_diagram = visualization.ClasswiseAugmentedConfidenceDiagram()
    plt_test_5 = per_diagram.plot(cls_outputs, cls_targets, title="Uncalibrated Classwise Augmented Confidence Diagram")
    plt_test_5.savefig(checkpoint_folder + '/per_diagram_uncal.png',bbox_inches='tight')  
    
    cls_diagram = visualization.ClasswiseAugmentedReliabilityDiagram()
    plt_test_6 = cls_diagram.plot(cls_outputs, cls_targets, title="Uncalibrated Classwise Augmented Reliabilty Diagram")
    plt_test_6.savefig(checkpoint_folder + '/cls_diagram_uncal.png',bbox_inches='tight')  
      
    # Set up temperature scaling
    temperature_model = TemperatureScaling(base_model=model)
    temperature_model.cuda()
    
    logging.info("Running temp scaling:")
    temperature_model.calibrate(valloader)

    trainloader, valloader, testloader = dataloader_dict[args.dataset](args, tta=False)
    
    print("Cls Performance")
    if args.dataset == "gbc_usg" or args.dataset == "busi":    
        test_loss, top1, top3, top5, sce_score, ece_score, jce_score, aece_score, gece_score, acc3, spec, sens, acc2, cls_acc, cf, cls_outputs, cls_targets = test_gbc(testloader, temperature_model, criterion, gece_sigma, gece_distr)
    else:
        test_loss, top1, top3, top5, sce_score, ece_score, jce_score, aece_score, gece_score, acc3, spec, sens, acc2, cls_acc, cf, cls_outputs, cls_targets = test_gbc_old(testloader, temperature_model, criterion, gece_sigma, gece_distr)

    print("Calibration Performance")    
    trainloader, valloader, testloader = dataloader_dict[args.dataset](args, tta=True)
    if args.dataset == "gbc_usg" or args.dataset == "busi":
        _, _, _, _, sce_score, ece_score, jce_score, gece_score, gece_sd_score, gmece_score, gmece_smd_score, _, _, _, _, _, _, _, _ = test_tta(testloader, temperature_model, criterion, cls_outputs, cls_targets, gbc = True, num_classes = num_classes, gece_sigma=gece_sigma, gece_distr=gece_distr)
    else:
        _, _, _, _, sce_score, ece_score, jce_score, gece_score, gece_sd_score, gmece_score, gmece_smd_score, _, _, _, _, _, _, _, _ = test_tta(testloader, temperature_model, criterion, cls_outputs, cls_targets, gbc = False, num_classes = num_classes, gece_sigma=gece_sigma, gece_distr=gece_distr)
        
    logger.append(["{:.2f}".format(temperature_model.T), ece_score, sce_score, gece_score, gmece_smd_score])
    print("Min temp:" , temperature_model.T)
    logging.info("TS Model: 2cls Acc %.4f Spec %.4f Sens %.4f 3cls Acc %.4f Cls wise Acc %.4f %.4f %.4f"%(acc2, spec, sens, acc3, cls_acc[0], cls_acc[1], cls_acc[2]))
    logging.info("CM: " + str(cf))
    logging.info("Stats: loss : {:.4f} | top1 : {:.4f} | ECE : {:.5f} | SCE : {:.5f} | RECE : {:.5f} | RECE-M : {:.5f}".format(
            test_loss,
            top1,
            ece_score,
            sce_score,
            gece_score,
            gmece_smd_score
        ))
    logging.info("Class-j ECE:" + str(jce_score))

    logger.close()
    
    conf_hist = visualization.ConfidenceHistogram()
    plt_test = conf_hist.plot(cls_outputs, cls_targets, title="Temp Scaled Confidence Histogram")
    plt_test.savefig(checkpoint_folder + '/conf_histogram_ts.png',bbox_inches='tight')
    
    rel_diagram = visualization.ReliabilityDiagram()
    plt_test_2 = rel_diagram.plot(cls_outputs, cls_targets, title="Temp Scaled Reliability Diagram")
    plt_test_2.savefig(checkpoint_folder + '/rel_diagram_ts.png',bbox_inches='tight')

    pred_diagram = visualization.AugmentedReliabilityDiagram()
    plt_test_3 = pred_diagram.plot(cls_outputs, cls_targets, title="Temp Scaled Augmented Reliabilty Diagram")
    plt_test_3.savefig(checkpoint_folder + '/pred_diagram_ts.png',bbox_inches='tight')
    
    perc_diagram = visualization.AugmentedConfidenceDiagram()
    plt_test_4 = perc_diagram.plot(cls_outputs, cls_targets, title="Temp Scaled Augmented Confidence Diagram")
    plt_test_4.savefig(checkpoint_folder + '/perc_diagram_ts.png',bbox_inches='tight')
    
    per_diagram = visualization.ClasswiseAugmentedConfidenceDiagram()
    plt_test_5 = per_diagram.plot(cls_outputs, cls_targets, title="Temp Scaled Classwise Augmented Confidence Diagram")
    plt_test_5.savefig(checkpoint_folder + '/per_diagram_ts.png',bbox_inches='tight')
    
    cls_diagram = visualization.ClasswiseAugmentedReliabilityDiagram()
    plt_test_6 = cls_diagram.plot(cls_outputs, cls_targets, title="Temp Scaled Classwise Augmented Reliabilty Diagram")
    plt_test_6.savefig(checkpoint_folder + '/cls_diagram_ts.png',bbox_inches='tight')

    # Set up dirichlet scaling
    logging.info("Running dirichlet scaling:")
    #lambdas = [0, 0.01, 0.1, 1, 10, 0.005, 0.05, 0.5, 5, 0.0025, 0.025, 0.25, 2.5]
    lambdas = [0, 0.01, 0.1, 1, 10]
    mus = [0, 0.01, 0.1, 1, 10]

    # set up loggers
    metric_log_path = os.path.join(checkpoint_folder, 'dirichlet.txt')
    logger = Logger(metric_log_path, resume=False)
    logger.set_names(['method', 'test_nll', 'top1', 'top3', 'top5', 'ECE', 'SCE', 'RECE', 'RECE-M'])

    min_stats = {}
    min_error = float('inf')
    outputs, targets = None, None
    stat_vals = ""
    cal_vals = ""

    for l in lambdas:
        for m in mus:
            # Set up dirichlet model
            dir_model = DirichletScaling(base_model=model, num_classes=num_classes, optim=args.optimizer, Lambda=l, Mu=m)
            dir_model.double().cuda()

            trainloader, valloader, testloader = dataloader_dict[args.dataset](args, tta=False)

            # calibrate
            dir_model.calibrate(valloader, lr=args.lr, epochs=args.epochs, patience=args.patience)

            if args.dataset == "gbc_usg" or args.dataset == "busi":    
                val_nll, top1, top3, top5, sce_score, ece_score, jce_score, aece_score, gece_score, acc3, spec, sens, acc2, cls_acc, cf, cls_outputs, cls_targets = test_gbc(valloader, dir_model, criterion, gece_sigma, gece_distr)
            else:
                val_nll, top1, top3, top5, sce_score, ece_score, jce_score, aece_score, gece_score, acc3, spec, sens, acc2, cls_acc, cf, cls_outputs, cls_targets = test_gbc_old(valloader, dir_model, criterion, gece_sigma, gece_distr)

            trainloader, valloader, testloader = dataloader_dict[args.dataset](args, tta=False)
            if args.dataset == "gbc_usg" or args.dataset == "busi":    
                test_loss, top1, top3, top5, sce_score, ece_score, jce_score, aece_score, gece_score, acc3, spec, sens, acc2, cls_acc, cf, cls_outputs, cls_targets = test_gbc(testloader, dir_model, criterion, gece_sigma, gece_distr)
            else:
                test_loss, top1, top3, top5, sce_score, ece_score, jce_score, aece_score, gece_score, acc3, spec, sens, acc2, cls_acc, cf, cls_outputs, cls_targets = test_gbc_old(testloader, dir_model, criterion, gece_sigma, gece_distr)

            trainloader, valloader, testloader = dataloader_dict[args.dataset](args, tta=True)
            if args.dataset == "gbc_usg" or args.dataset == "busi":
                _, _, _, _, sce_score, ece_score, jce_score, gece_score, gece_sd_score, gmece_score, gmece_smd_score, _, _, _, _, _, _, _, _ = test_tta(testloader, dir_model, criterion, cls_outputs, cls_targets, gbc = True, num_classes = num_classes, gece_sigma=gece_sigma, gece_distr=gece_distr)
            else:
                _, _, _, _, sce_score, ece_score, jce_score, gece_score, gece_sd_score, gmece_score, gmece_smd_score, _, _, _, _, _, _, _, _ = test_tta(testloader, dir_model, criterion, cls_outputs, cls_targets, gbc = False, num_classes = num_classes, gece_sigma=gece_sigma, gece_distr=gece_distr)

            logging.info("DC Model l,m {%.2f},{%.2f}: 2cls Acc %.4f Spec %.4f Sens %.4f 3cls Acc %.4f Cls wise Acc %.4f %.4f %.4f"%(l, m, acc2, spec, sens, acc3, cls_acc[0], cls_acc[1], cls_acc[2]))
            logging.info("Stats: loss : {:.4f} | top1 : {:.4f} | ECE : {:.5f} | SCE : {:.5f} | RECE : {:.5f} | RECE-M : {:.5f}".format(test_loss, top1, ece_score, sce_score, gece_score, gmece_smd_score))

            if val_nll < min_error:
                min_error = val_nll
                min_stats = {
                    "test_loss" : test_loss,
                    "top1" : top1,
                    "top3" : top3,
                    "top5" : top5,
                    "ece_score" : ece_score,
                    "sce_score" : sce_score,
                    "rece_score" : gece_score,
                    "rece_m_score" : gmece_smd_score,
                    "pair" : (l, m)
                }
                outputs, targets = cls_outputs, cls_targets
                stat_vals = "DC Model l,m {%.2f},{%.2f}: 2cls Acc %.4f Spec %.4f Sens %.4f 3cls Acc %.4f Cls wise Acc %.4f %.4f %.4f"%(l, m, acc2, spec, sens, acc3, cls_acc[0], cls_acc[1], cls_acc[2])
                cal_vals = "Stats: loss : {:.4f} | top1 : {:.4f} | ECE : {:.5f} | SCE : {:.5f} | RECE : {:.5f} | RECE-M : {:.5f}".format(test_loss, top1, ece_score, sce_score, gece_score, gmece_smd_score)
            
            logger.append(["Dir=({:.2f},{:.2f})".format(l, m), test_loss, top1, top3, top5, ece_score, sce_score, gece_score, gmece_smd_score])
    
    logger.append(["Best_Dir={}".format(min_stats["pair"]), 
                                            min_stats["test_loss"], 
                                            min_stats["top1"], 
                                            min_stats["top3"], 
                                            min_stats["top5"], 
                                            min_stats["ece_score"], 
                                            min_stats["sce_score"],
                                            min_stats["rece_score"],
                                            min_stats["rece_m_score"]])
                                            
    print("Best DS:", stat_vals)
    print(cal_vals)
    conf_hist = visualization.ConfidenceHistogram()
    plt_test = conf_hist.plot(outputs, targets, title="Dirichlet Scaled Confidence Histogram")
    plt_test.savefig(checkpoint_folder + '/conf_histogram_ds.png',bbox_inches='tight')
    
    rel_diagram = visualization.ReliabilityDiagram()
    plt_test_2 = rel_diagram.plot(outputs, targets, title="Dirichlet Scaled Reliability Diagram")
    plt_test_2.savefig(checkpoint_folder + '/rel_diagram_ds.png',bbox_inches='tight')

