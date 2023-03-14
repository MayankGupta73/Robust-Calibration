import os

import torch
import torch.optim as optim

from utils import mkdir_p, parse_args
from utils import get_lr, save_checkpoint, create_save_path

from solvers.runners import train, test, test_gbc, test_gbc_old
from solvers.loss import loss_dict

from models import model_dict
from datasets import dataloader_dict, dataset_nclasses_dict, dataset_classname_dict

from time import localtime, strftime

import calibration_library.visualization as visualization

import logging

if __name__ == "__main__":
    
    args = parse_args()

    current_time = strftime("%d-%b", localtime())
    # prepare save path
    model_save_pth = f"{args.checkpoint}/{args.dataset}/{current_time}{create_save_path(args)}"
    checkpoint_dir_name = model_save_pth

    if not os.path.isdir(model_save_pth):
        mkdir_p(model_save_pth)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(level=logging.INFO, 
                        format="%(levelname)s:  %(message)s",
                        handlers=[
                            logging.FileHandler(filename=os.path.join(model_save_pth, "train.log")),
                            logging.StreamHandler()
                        ])
    logging.info(f"Setting up logging folder : {model_save_pth}")

    num_classes = dataset_nclasses_dict[args.dataset]
    classes_name_list = dataset_classname_dict[args.dataset]
    
    # prepare model
    logging.info(f"Using model : {args.model}")
    model = model_dict[args.model](num_classes=num_classes)
    
    if args.model == "gbcnet":
        #loading from ckpt
        if num_classes == 3:
            model.load_state_dict(torch.load("gbcnet_init_weights.pth"))
        else:
            sd = torch.load("gbcnet_init_weights.pth")
            #print(sd)
            new_sd = {k: v for k, v in sd.items() if not k.startswith("net.fc.") }
            model.load_state_dict(new_sd, strict=False)
        model = model.net.double().cuda()
    else:
        model = model.double().cuda()

    if args.resume != '':
        print("Resuming from ", args.resume)
        model.load_state_dict(torch.load(args.resume)['state_dict'])


    # set up dataset
    logging.info(f"Using dataset : {args.dataset}")
    trainloader, valloader, testloader = dataloader_dict[args.dataset](args)

    logging.info(f"Setting up optimizer : {args.optimizer}")

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
    
    logging.info(f"Step sizes : {args.schedule_steps} | lr-decay-factor : {args.lr_decay_factor}")
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule_steps, gamma=args.lr_decay_factor)

    start_epoch = args.start_epoch
    
    best_acc = 0.
    best_gece = 100.0
    best_acc_stats = {"top1" : 0.0}
    
    logging.info(f"Save condition : {args.save_condition}")

    for epoch in range(start_epoch, args.epochs):

        logging.info('Epoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, get_lr(optimizer)))
        
        if "GMDCA4" in args.loss:
            train_loss, top1_train = train(trainloader, model, optimizer, criterion, epoch + 1)
        else:
            train_loss, top1_train = train(trainloader, model, optimizer, criterion)
            
        
        if args.dataset == "gbc_usg" or args.dataset == "busi":
            val_loss, top1_val, _, _, sce_score_val, ece_score_val, jce_score_val, aece_score_val, gece_score_val, _, _, _, _, _, _, _, _  = test_gbc(valloader, model, test_criterion)
            test_loss, top1, top3, top5, sce_score, ece_score, jce_score, aece_score, gece_score, acc3, spec, sens, acc2, cls_acc, cf, all_outputs, all_targets = test_gbc(testloader, model, test_criterion)
        else:
            val_loss, top1_val, _, _, sce_score_val, ece_score_val, jce_score_val, aece_score_val, gece_score_val, _, _, _, _, _, _, _, _  = test_gbc_old(valloader, model, test_criterion)
            test_loss, top1, top3, top5, sce_score, ece_score, jce_score, aece_score, gece_score, acc3, spec, sens, acc2, cls_acc, cf, all_outputs, all_targets = test_gbc_old(testloader, model, test_criterion)

        scheduler.step()
        
        if args.save_plots and (epoch + 1) % 10 == 0:
            #Plotting conf
            conf_hist = visualization.ConfidenceHistogram()
            plt_test = conf_hist.plot(all_outputs, all_targets, title="Epoch {} Confidence Histogram".format(epoch))
            plt_test.savefig(model_save_pth + '/conf_histogram_' + str(epoch) + '.png',bbox_inches='tight')
            
            rel_diagram = visualization.ReliabilityDiagram()
            plt_test_2 = rel_diagram.plot(all_outputs, all_targets, title="Epoch {} Reliability Diagram".format(epoch))
            plt_test_2.savefig(model_save_pth + '/rel_diagram_' + str(epoch) + '.png',bbox_inches='tight')
          
            pred_diagram = visualization.PredictionConfidenceDiagram()
            plt_test_3 = pred_diagram.plot(all_outputs, all_targets, title="Epoch {} Augmented Reliabilty Diagram".format(epoch))
            plt_test_3.savefig(model_save_pth + '/pred_diagram_' + str(epoch) + '.png',bbox_inches='tight')
            
            perc_diagram = visualization.PercentageDiagram()
            plt_test_4 = perc_diagram.plot(all_outputs, all_targets, title="Epoch {} Augmented Confidence Diagram".format(epoch))
            plt_test_4.savefig(model_save_pth + '/perc_diagram_' + str(epoch) + '.png',bbox_inches='tight')

        logging.info("End of epoch {} stats: train_loss : {:.4f} | val_loss : {:.4f} | top1_train : {:.4f} | top1 : {:.4f} | SCE : {:.5f} | ECE : {:.5f} | AECE : {:.5f} | GECE : {:.5f}".format(
            epoch+1,
            train_loss,
            test_loss,
            top1_train,
            top1,
            sce_score,
            ece_score,
            aece_score,
            gece_score
        ))
        
        logging.info("2cls Acc %.4f Spec %.4f Sens %.4f 3cls Acc %.4f Cls wise Acc %.4f %.4f %.4f"%(acc2, spec, sens, acc3, cls_acc[0], cls_acc[1], cls_acc[2]))

        # save best accuracy model
        if args.save_condition == "acc":
            is_best = acc3 > best_acc
            best_acc = max(best_acc, acc3)
        elif args.save_condition == "top1":
            is_best = top1 > best_acc
            best_acc = max(best_acc, top1)
        elif args.save_condition == "acc+gece":
            if acc3 > best_acc:
                is_best = True
                best_acc = max(best_acc, acc3)
                best_gece = min(best_gece, gece_score)
            elif acc3 == best_acc and gece_score < best_gece:
                is_best = True
                best_acc = max(best_acc, acc3)
                best_gece = min(best_gece, gece_score)
        else:
            if epoch >= 10:
                is_best = gece_score < best_gece
                best_gece = min(best_gece, gece_score)
            else:
                is_best = False

        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict(),
                'dataset' : args.dataset,
                'model' : args.model
            }, is_best, checkpoint=model_save_pth)
        
        # Update best stats
        if is_best:
            best_acc_stats = {
                "top1" : top1,
                "top3" : top3,
                "top5" : top5,
                "SCE" : sce_score,
                "ECE" : ece_score,
                "GECE": gece_score
            }

    logging.info("training completed...")
    logging.info("The stats for best trained model on test set are as below:")
    logging.info(best_acc_stats)