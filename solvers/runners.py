import torch
from tqdm import tqdm
from utils import AverageMeter, accuracy
import numpy as np
from calibration_library.metrics import ECELoss, SCELoss, ClassjECELoss, AdaptiveECELoss, GECELoss, GMECELoss
from sklearn.metrics import confusion_matrix 
import sys

def train(trainloader, model, optimizer, criterion, epoch = 0):
    # switch to train mode
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()

    bar = tqdm(enumerate(trainloader), total=len(trainloader))
    for batch_idx, (inputs, targets) in bar:
        
        inputs, targets = inputs.double().cuda(), targets.cuda()

        # compute output
        outputs = model(inputs)
        
        if epoch == 0:
            loss = criterion(outputs, targets)
        else:
            loss = criterion(outputs, targets, curr_epoch = epoch)

        # measure accuracy and record loss
        prec1, = accuracy(outputs.data, targets.data, topk=(1, ))
        losses.update(loss.item(), inputs.size(0))

        top1.update(prec1.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # plot progress
        bar.set_postfix_str('({batch}/{size}) Loss: {loss:.8f} | top1: {top1: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    loss=losses.avg,
                    top1=top1.avg
                    ))

    return (losses.avg, top1.avg)
    
def train_le(trainloader, model, optimizer, criterion):
    # switch to train mode
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()

    bar = tqdm(enumerate(trainloader), total=len(trainloader))
    for batch_idx, (inputs, targets) in bar:
        
        inputs, targets = inputs.double().cuda(), targets.cuda()

        # compute output
        outputs = model(inputs)
        loss = 0
        output_losses = [criterion(output, targets) for _, output in outputs.items()]
        output_list = [output for _, output in outputs.items()]
        for l in output_losses:
            loss = loss + l
            
        final_output = torch.mean(torch.stack(output_list), dim=0)

        # measure accuracy and record loss
        prec1, = accuracy(final_output.data, targets.data, topk=(1, ))
        losses.update(loss.item(), inputs.size(0))

        top1.update(prec1.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # plot progress
        bar.set_postfix_str('({batch}/{size}) Loss: {loss:.8f} | top1: {top1: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    loss=losses.avg,
                    top1=top1.avg
                    ))

    return (losses.avg, top1.avg)

@torch.no_grad()
def test(testloader, model, criterion):

    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    top5 = AverageMeter()

    all_targets = None
    all_outputs = None

    # switch to evaluate mode
    model.eval()

    bar = tqdm(enumerate(testloader), total=len(testloader))
    for batch_idx, (inputs, targets) in bar:

        inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        prec1, prec3, prec5  = accuracy(outputs.data, targets.data, topk=(1, 3, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top3.update(prec3.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        targets = targets.cpu().numpy()
        outputs = outputs.cpu().numpy()

        if all_targets is None:
            all_outputs = outputs
            all_targets = targets
        else:
            all_targets = np.concatenate([all_targets, targets], axis=0)
            all_outputs = np.concatenate([all_outputs, outputs], axis=0)

        # plot progress
        bar.set_postfix_str('({batch}/{size}) Loss: {loss:.8f} | top1: {top1: .4f} | top3: {top3: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    loss=losses.avg,
                    top1=top1.avg,
                    top3=top3.avg,
                    top5=top5.avg,
                    ))

    eces = ECELoss().loss(all_outputs, all_targets, n_bins=15)
    cces = SCELoss().loss(all_outputs, all_targets, n_bins=15)

    return (losses.avg, top1.avg, top3.avg, top5.avg, cces, eces)
    
@torch.no_grad()
def test_gbc(testloader, model, criterion, gece_sigma=0.05, gece_distr = "gaussian"):

    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    top5 = AverageMeter()

    all_targets = None
    all_outputs = None
    all_preds = None

    # switch to evaluate mode
    model.eval()

    bar = tqdm(enumerate(testloader), total=len(testloader))
    for batch_idx, (inputs, targets) in bar:

        inputs, targets = inputs.squeeze(0).double().cuda(), targets.squeeze(0).cuda()

        # compute output
        outputs = model(inputs)
        _, preds = torch.max(outputs, dim=1)
        loss = criterion(outputs, targets)
        
        pred_label = torch.max(preds)
        pred_idx = pred_label.item()
        pred_label = pred_label.unsqueeze(0)
        idx = torch.argmax(preds)
        
        prec1, prec3, prec5  = accuracy(outputs.data, targets.data, topk=(1, 3, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top3.update(prec3.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        targets = targets.cpu().numpy()
        outputs = outputs.cpu().numpy()
        preds = preds.cpu().numpy()
        
        if all_targets is None:
            all_outputs = outputs[idx.item()][None, :]
            all_targets = [targets[0]]
            all_preds = [pred_label.squeeze(0).cpu().numpy()]
        else:
            all_targets = np.concatenate([all_targets, [targets[0]]], axis=0)
            all_outputs = np.concatenate([all_outputs, outputs[idx.item()][None, :]], axis=0)
            all_preds = np.concatenate([all_preds, [pred_label.squeeze(0).cpu().numpy()]], axis=0)

        # plot progress
        bar.set_postfix_str('({batch}/{size}) Loss: {loss:.8f} | top1: {top1: .4f} | top3: {top3: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    loss=losses.avg,
                    top1=top1.avg,
                    top3=top3.avg,
                    top5=top5.avg,
                    ))
    
    eces = ECELoss().loss(all_outputs, all_targets, n_bins=15)
    cces = SCELoss().loss(all_outputs, all_targets, n_bins=15)
    jces = ClassjECELoss().loss(all_outputs, all_targets, n_bins=15)
    aeces = AdaptiveECELoss().loss(all_outputs, all_targets, n_bins=15)
    geces = GECELoss().loss(all_outputs, all_targets, n_bins=15, sigma=gece_sigma, distr_name=gece_distr)
    
    cf = confusion_matrix(all_targets, all_preds)
    
    if cf.shape[0] == 2:
        acc3 = acc2 = (cf[0][0]+cf[1][1])/np.sum(cf)
        spec = (cf[0][0])/(np.sum(cf[0]))
        sens = cf[1][1]/np.sum(cf[1])
        cls_acc = [cf[0][0]/np.sum(cf[0]), cf[1][1]/np.sum(cf[1]), 0]
    else:
        acc3 = (cf[0][0] + cf[1][1] + cf[2][2])/np.sum(cf)
        spec = (cf[0][0]+cf[0][1]+cf[1][0]+cf[1][1])/(np.sum(cf[0])+np.sum(cf[1]))
        sens = cf[2][2]/np.sum(cf[2])
        acc2 = (cf[0][0]+cf[0][1]+cf[1][0]+cf[1][1]+cf[2][2])/np.sum(cf)
        cls_acc = [cf[0][0]/np.sum(cf[0]), cf[1][1]/np.sum(cf[1]), cf[2][2]/np.sum(cf[2])]

    return (losses.avg, top1.avg, top3.avg, top5.avg, cces, eces, jces, aeces, geces, acc3, spec, sens, acc2, cls_acc, cf, all_outputs, all_targets)
    
@torch.no_grad()
def test_gbc_old(testloader, model, criterion, gece_sigma=0.05, gece_distr = "gaussian"):

    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    top5 = AverageMeter()

    all_targets = None
    all_outputs = None
    all_preds = None

    # switch to evaluate mode
    model.eval()

    bar = tqdm(enumerate(testloader), total=len(testloader))
    for batch_idx, (inputs, targets) in bar:

        inputs, targets = inputs.double().cuda(), targets.cuda()

        # compute output
        outputs = model(inputs)
        _, preds = torch.max(outputs, dim=1)
        loss = criterion(outputs, targets)
        
        prec1, prec3, prec5  = accuracy(outputs.data, targets.data, topk=(1, 3, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top3.update(prec3.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        targets = targets.cpu().numpy()
        outputs = outputs.cpu().numpy()
        preds = preds.cpu().numpy()

        if all_targets is None:
            all_outputs = outputs
            all_targets = targets
            all_preds = preds
        else:
            all_targets = np.concatenate([all_targets, targets], axis=0)
            all_outputs = np.concatenate([all_outputs, outputs], axis=0)
            all_preds = np.concatenate([all_preds, preds], axis=0)

        # plot progress
        bar.set_postfix_str('({batch}/{size}) Loss: {loss:.8f} | top1: {top1: .4f} | top3: {top3: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    loss=losses.avg,
                    top1=top1.avg,
                    top3=top3.avg,
                    top5=top5.avg,
                    ))

    eces = ECELoss().loss(all_outputs, all_targets, n_bins=15)
    cces = SCELoss().loss(all_outputs, all_targets, n_bins=15)
    jces = ClassjECELoss().loss(all_outputs, all_targets, n_bins=15)
    aeces = AdaptiveECELoss().loss(all_outputs, all_targets, n_bins=15)
    geces = GECELoss().loss(all_outputs, all_targets, n_bins=15, sigma=gece_sigma, distr_name=gece_distr)
    
    cf = confusion_matrix(all_targets, all_preds)
    
    if cf.shape[0] == 2:
        acc3 = acc2 = (cf[0][0]+cf[1][1])/np.sum(cf)
        spec = (cf[0][0])/(np.sum(cf[0]))
        sens = cf[1][1]/np.sum(cf[1])
        cls_acc = [cf[0][0]/np.sum(cf[0]), cf[1][1]/np.sum(cf[1]), 0]
    else:
        acc3 = (cf[0][0] + cf[1][1] + cf[2][2])/np.sum(cf)
        spec = (cf[0][0]+cf[0][1]+cf[1][0]+cf[1][1])/(np.sum(cf[0])+np.sum(cf[1]))
        sens = cf[2][2]/np.sum(cf[2])
        acc2 = (cf[0][0]+cf[0][1]+cf[1][0]+cf[1][1]+cf[2][2])/np.sum(cf)
        cls_acc = [cf[0][0]/np.sum(cf[0]), cf[1][1]/np.sum(cf[1]), cf[2][2]/np.sum(cf[2])]

    return (losses.avg, top1.avg, top3.avg, top5.avg, cces, eces, jces, aeces, geces, acc3, spec, sens, acc2, cls_acc, cf, all_outputs, all_targets)
    
    
@torch.no_grad()
def test_tta(testloader, model, criterion, cls_outputs, cls_targets, gbc = True, num_classes = 3, num_evals = 10, gece_sigma=0.05, gece_distr = "gaussian"):

    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    top5 = AverageMeter()

    all_targets = None
    all_outputs = None
    all_preds = None
    
    tta_targets = None

    # switch to evaluate mode
    model.eval()
    
    n_samples = len(testloader.dataset)
    tta_predictions = np.empty((0, n_samples, num_classes))
    softmax = torch.nn.Softmax(dim=1)

    for n in range(num_evals):
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
    
    eces = ECELoss().loss(cls_outputs, cls_targets, n_bins=15)
    cces = SCELoss().loss(cls_outputs, cls_targets, n_bins=15)
    jces = ClassjECELoss().loss(cls_outputs, cls_targets, n_bins=15)
    geces = GECELoss().loss(cls_outputs, cls_targets, n_bins=15, sigma=gece_sigma, distr_name=gece_distr)
    geces_sd = GECELoss().loss(cls_outputs, cls_targets, n_bins=15, distr_name=gece_distr, sigma=tta_std, sd=True)
    gmece = GMECELoss().loss(tta_predictions, np.array([tta_targets]*num_evals), cls_outputs, cls_targets, n_samples, n_bins=15, distr_name=gece_distr, sigma=gece_sigma, mode="default")
    gmece_smd = GMECELoss().loss(tta_predictions, np.array([tta_targets]*num_evals), cls_outputs, cls_targets, n_samples, n_bins=15, distr_name=gece_distr, sigma=tta_std, mode="gmm_estimate")
    
    cls_preds = np.argmax(cls_outputs, axis=1)
    cf = confusion_matrix(cls_targets, cls_preds)
    
    if cf.shape[0] == 2:
        acc3 = acc2 = (cf[0][0]+cf[1][1])/np.sum(cf)
        spec = (cf[0][0])/(np.sum(cf[0]))
        sens = cf[1][1]/np.sum(cf[1])
        cls_acc = [cf[0][0]/np.sum(cf[0]), cf[1][1]/np.sum(cf[1]), 0]
    else:
        acc3 = (cf[0][0] + cf[1][1] + cf[2][2])/np.sum(cf)
        spec = (cf[0][0]+cf[0][1]+cf[1][0]+cf[1][1])/(np.sum(cf[0])+np.sum(cf[1]))
        sens = cf[2][2]/np.sum(cf[2])
        acc2 = (cf[0][0]+cf[0][1]+cf[1][0]+cf[1][1]+cf[2][2])/np.sum(cf)
        cls_acc = [cf[0][0]/np.sum(cf[0]), cf[1][1]/np.sum(cf[1]), cf[2][2]/np.sum(cf[2])]

    return (losses.avg, top1.avg, top3.avg, top5.avg, cces, eces, jces, geces, geces_sd, gmece, gmece_smd, acc3, spec, sens, acc2, cls_acc, cf, tta_mean, tta_targets)

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
    
@torch.no_grad()
def test_mc(testloader, model, criterion, gbc = True, num_evals = 20, gece_sigma=0.05, gece_distr = "gaussian"):

    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    top5 = AverageMeter()

    all_targets = None
    all_outputs = None
    all_preds = None
    
    mc_targets = None
    
    if gbc:
        n_classes = 3
        n_samples = len(testloader)
    else:
        n_classes = 10
        n_samples = len(testloader.dataset)

    dropout_predictions = np.empty((0, n_samples, n_classes))
    softmax = torch.nn.Softmax(dim=1)
    
    for n in range(num_evals):
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

    # Calculating variance across multiple MCD forward passes 
    variance = np.var(dropout_predictions, axis=0) # shape (n_samples, n_classes)

    epsilon = sys.float_info.min
    # Calculating entropy across multiple MCD forward passes 
    entropy = -np.sum(mean*np.log(mean + epsilon), axis=-1) # shape (n_samples,)

    # Calculating mutual information across multiple MCD forward passes 
    mutual_info = entropy - np.mean(np.sum(-dropout_predictions*np.log(dropout_predictions + epsilon),
                                            axis=-1), axis=0) # shape (n_samples,)
                                            
    print("MC Dropout Entropy:", np.mean(entropy))
    print("MC Dropout Mutual Info:", mutual_info)
    
    #Prediction from average logits
    mc_preds = np.argmax(mean, axis=1)  # shape (n_samples)
    
    # Calculating standard deviation across MCD forward passes predictions
    std_mc = np.std(dropout_predictions, axis=0) # shape (n_samples, n_classes)
    std = std_mc[np.arange(len(std_mc)), mc_preds] # shape (n_samples)
    
    print("Dropout output", dropout_predictions)
    print("Mean", mean)
    
    print("Mc preds", mc_preds)

    eces = ECELoss().loss(mean, mc_targets, n_bins=15, logits=False)
    cces = SCELoss().loss(mean, mc_targets, n_bins=15, logits=False)
    jces = ClassjECELoss().loss(mean, mc_targets, n_bins=15, logits=False)
    geces = GECELoss().loss(mean, mc_targets, n_bins=15, logits=False, distr_name=gece_distr, sigma=gece_sigma)
    geces_sd = GECELoss().loss(mean, mc_targets, n_bins=15, logits=False, distr_name=gece_distr, sigma=std, sd=True)

    cf = confusion_matrix(mc_targets, mc_preds)
    
    if cf.shape[0] == 2:
        acc3 = acc2 = (cf[0][0]+cf[1][1])/np.sum(cf)
        spec = (cf[0][0])/(np.sum(cf[0]))
        sens = cf[1][1]/np.sum(cf[1])
        cls_acc = [cf[0][0]/np.sum(cf[0]), cf[1][1]/np.sum(cf[1]), 0]
    else:
        acc3 = (cf[0][0] + cf[1][1] + cf[2][2])/np.sum(cf)
        spec = (cf[0][0]+cf[0][1]+cf[1][0]+cf[1][1])/(np.sum(cf[0])+np.sum(cf[1]))
        sens = cf[2][2]/np.sum(cf[2])
        acc2 = (cf[0][0]+cf[0][1]+cf[1][0]+cf[1][1]+cf[2][2])/np.sum(cf)
        cls_acc = [cf[0][0]/np.sum(cf[0]), cf[1][1]/np.sum(cf[1]), cf[2][2]/np.sum(cf[2])]

    return (losses.avg, top1.avg, top3.avg, top5.avg, cces, eces, jces, geces, geces_sd, acc3, spec, sens, acc2, cls_acc, cf, mean, mc_targets)
    
    
@torch.no_grad()
def test_le(testloader, model, criterion, gbc = True):

    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    top5 = AverageMeter()

    all_targets = None
    all_outputs = None
    all_preds = None
    all_stds = None

    # switch to evaluate mode
    model.eval()
    
    bar = tqdm(enumerate(testloader), total=len(testloader))
    for batch_idx, (inputs, targets) in bar:

        inputs, targets = inputs.squeeze(0).double().cuda(), targets.squeeze(0).cuda()

        # compute output
        outputs = model(inputs)
        
        loss = 0
        output_losses = [criterion(output, targets) for _, output in outputs.items()]
        output_list = [output for _, output in outputs.items()]
        for l in output_losses:
            loss = loss + l
            
        final_output = torch.mean(torch.stack(output_list), dim=0)
        final_std = torch.std(torch.stack(output_list), dim=0)
        _, preds = torch.max(final_output, dim=1)
        
        if gbc:
            pred_label = torch.max(preds)
            pred_idx = pred_label.item()
            pred_label = pred_label.unsqueeze(0)
            idx = torch.argmax(preds)
        
        prec1, prec3, prec5  = accuracy(final_output.data, targets.data, topk=(1, 3, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top3.update(prec3.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        targets = targets.cpu().numpy()
        outputs = final_output.cpu().numpy()
        preds = preds.cpu().numpy()
        stds = final_std.cpu().numpy()

        if gbc:
            if all_targets is None:
                all_outputs = outputs[idx.item()][None, :]
                all_targets = [targets[0]]
                all_preds = [pred_label.squeeze(0).cpu().numpy()]
                all_stds = stds[idx.item()][None, :]
            else:
                all_targets = np.concatenate([all_targets, [targets[0]]], axis=0)
                all_outputs = np.concatenate([all_outputs, outputs[idx.item()][None, :]], axis=0)
                all_preds = np.concatenate([all_preds, [pred_label.squeeze(0).cpu().numpy()]], axis=0)
                all_stds = np.concatenate([all_stds, stds[idx.item()][None, :]], axis=0)
        else:
            if all_targets is None:
                all_outputs = outputs
                all_targets = targets
                all_preds = preds
                all_stds = stds
            else:
                all_targets = np.concatenate([all_targets, targets], axis=0)
                all_outputs = np.concatenate([all_outputs, outputs], axis=0)
                all_preds = np.concatenate([all_preds, preds], axis=0)
                all_stds = np.concatenate([all_stds, stds], axis=0)

        # plot progress
        bar.set_postfix_str('({batch}/{size}) Loss: {loss:.8f} | top1: {top1: .4f} | top3: {top3: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    loss=losses.avg,
                    top1=top1.avg,
                    top3=top3.avg,
                    top5=top5.avg,
                    ))
        
    std = all_stds[np.arange(len(all_stds)), all_preds]

    eces = ECELoss().loss(all_outputs, all_targets, n_bins=15)
    cces = SCELoss().loss(all_outputs, all_targets, n_bins=15)
    jces = ClassjECELoss().loss(all_outputs, all_targets, n_bins=15)
    aeces = AdaptiveECELoss().loss(all_outputs, all_targets, n_bins=15)
    geces = GECELoss().loss(all_outputs, all_targets, n_bins=15)
    geces_sd = GECELoss().loss(all_outputs, all_targets, n_bins=15, sigma=std, sd=True)
    
    cf = confusion_matrix(all_targets, all_preds)
    
    if cf.shape[0] == 2:
        acc3 = acc2 = (cf[0][0]+cf[1][1])/np.sum(cf)
        spec = (cf[0][0])/(np.sum(cf[0]))
        sens = cf[1][1]/np.sum(cf[1])
        cls_acc = [cf[0][0]/np.sum(cf[0]), cf[1][1]/np.sum(cf[1]), 0]
    else:
        acc3 = (cf[0][0] + cf[1][1] + cf[2][2])/np.sum(cf)
        spec = (cf[0][0]+cf[0][1]+cf[1][0]+cf[1][1])/(np.sum(cf[0])+np.sum(cf[1]))
        sens = cf[2][2]/np.sum(cf[2])
        acc2 = (cf[0][0]+cf[0][1]+cf[1][0]+cf[1][1]+cf[2][2])/np.sum(cf)
        cls_acc = [cf[0][0]/np.sum(cf[0]), cf[1][1]/np.sum(cf[1]), cf[2][2]/np.sum(cf[2])]

    return (losses.avg, top1.avg, top3.avg, top5.avg, cces, eces, jces, geces, geces_sd, acc3, spec, sens, acc2, cls_acc, cf, all_outputs, all_targets)

@torch.no_grad()
def test_le_eval(testloader, model, criterion, gbc = True):

    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    top5 = AverageMeter()

    all_targets = None
    all_outputs = None
    all_preds = None

    # switch to evaluate mode
    model.eval()

    bar = tqdm(enumerate(testloader), total=len(testloader))
    for batch_idx, (inputs, targets) in bar:

        inputs, targets = inputs.squeeze(0).double().cuda(), targets.squeeze(0).cuda()

        # compute output
        outputs = model(inputs)
        
        loss = 0
        output_losses = [criterion(output, targets) for _, output in outputs.items()]
        output_list = [output for _, output in outputs.items()]
        for l in output_losses:
            loss = loss + l
                    
        for op in output_list:
            _, preds = torch.max(op, dim=1)
        
            if gbc:
                pred_label = torch.max(preds)
                pred_idx = pred_label.item()
                pred_label = pred_label.unsqueeze(0)
                idx = torch.argmax(preds)
            
            prec1, prec3, prec5  = accuracy(op.data, targets.data, topk=(1, 3, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top3.update(prec3.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
    
            target = targets.cpu().numpy()
            outputs = op.cpu().numpy()
            preds = preds.cpu().numpy()
    
            if gbc:
                if all_targets is None:
                    all_outputs = outputs[idx.item()][None, :]
                    all_targets = [target[0]]
                    all_preds = [pred_label.squeeze(0).cpu().numpy()]
                else:
                    all_targets = np.concatenate([all_targets, [target[0]]], axis=0)
                    all_outputs = np.concatenate([all_outputs, outputs[idx.item()][None, :]], axis=0)
                    all_preds = np.concatenate([all_preds, [pred_label.squeeze(0).cpu().numpy()]], axis=0)
            else:
                if all_targets is None:
                    all_outputs = outputs
                    all_targets = target
                    all_preds = preds
                else:
                    all_targets = np.concatenate([all_targets, target], axis=0)
                    all_outputs = np.concatenate([all_outputs, outputs], axis=0)
                    all_preds = np.concatenate([all_preds, preds], axis=0)

        # plot progress
        bar.set_postfix_str('({batch}/{size}) Loss: {loss:.8f} | top1: {top1: .4f} | top3: {top3: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    loss=losses.avg,
                    top1=top1.avg,
                    top3=top3.avg,
                    top5=top5.avg,
                    ))
    

    eces = ECELoss().loss(all_outputs, all_targets, n_bins=15)
    cces = SCELoss().loss(all_outputs, all_targets, n_bins=15)
    jces = ClassjECELoss().loss(all_outputs, all_targets, n_bins=15)
    
    cf = confusion_matrix(all_targets, all_preds)
    
    if cf.shape[0] == 2:
        acc3 = acc2 = (cf[0][0]+cf[1][1])/np.sum(cf)
        spec = (cf[0][0])/(np.sum(cf[0]))
        sens = cf[1][1]/np.sum(cf[1])
        cls_acc = [cf[0][0]/np.sum(cf[0]), cf[1][1]/np.sum(cf[1]), 0]
    else:
        acc3 = (cf[0][0] + cf[1][1] + cf[2][2])/np.sum(cf)
        spec = (cf[0][0]+cf[0][1]+cf[1][0]+cf[1][1])/(np.sum(cf[0])+np.sum(cf[1]))
        sens = cf[2][2]/np.sum(cf[2])
        acc2 = (cf[0][0]+cf[0][1]+cf[1][0]+cf[1][1]+cf[2][2])/np.sum(cf)
        cls_acc = [cf[0][0]/np.sum(cf[0]), cf[1][1]/np.sum(cf[1]), cf[2][2]/np.sum(cf[2])]

    return (losses.avg, top1.avg, top3.avg, top5.avg, cces, eces, jces, acc3, spec, sens, acc2, cls_acc, cf, all_outputs, all_targets)
    
@torch.no_grad()
def get_logits_from_model_dataloader(testloader, model):
    """Returns torch tensor of logits and targets on cpu"""
    # switch to evaluate mode
    model.eval()

    all_targets = None
    all_outputs = None

    bar = tqdm(testloader, total=len(testloader), desc="Evaluating logits")
    for inputs, targets in bar:
        inputs = inputs.cuda()
        # compute output
        outputs = model(inputs)
        # to numpy
        targets = targets.cpu().numpy()
        outputs = outputs.cpu().numpy()

        if all_targets is None:
            all_outputs = outputs
            all_targets = targets
        else:
            all_targets = np.concatenate([all_targets, targets], axis=0)
            all_outputs = np.concatenate([all_outputs, outputs], axis=0)

    return torch.from_numpy(all_outputs), torch.from_numpy(all_targets)

    
