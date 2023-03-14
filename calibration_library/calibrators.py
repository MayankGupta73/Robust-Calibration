import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.std import tqdm
import numpy as np
from calibration_library import metrics

from utils import EarlyStopping, AverageMeter

def _freeze_model(model : nn.Module):
    for params in model.parameters():
        params.requires_grad = False

class TemperatureScaling(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.T = 1.0
        _freeze_model(self.base_model)
        self.base_model.eval()

        # set up temperature list
        self.temp_list = []
        t = 0.1
        while t <= 10:
            self.temp_list.append(t)
            t += 0.1
        
        #t = 0.01    
        #while t < 1:
        #    self.temp_list.append(t)
        #    t += 0.01

    def forward(self, x):
        x = self.base_model(x)
        x /= self.T
        return x
    
    def calibrate(self, train_loader, **kwargs):

        min_error = float('inf')
        min_T = 1.0

        criterion = nn.CrossEntropyLoss()

        for T in tqdm(self.temp_list, desc="Running temp scaling"):
            error = 0.
            logits_list = []
            labels_list = []
            ece, sce = 1.0, 1.0
            for images, targets in train_loader:
                labels_list.append(targets.squeeze(0)[0].unsqueeze(0))
                images, targets = images.squeeze(0).double().cuda(), targets.squeeze(0).cuda()
                #images, targets = images.cuda(), targets.cuda()
                outputs = self.base_model(images)
                outputs /= T
                
                _, preds = torch.max(outputs, dim=1)
                pred_label = torch.max(preds)
                pred_idx = pred_label.item()
                pred_label = pred_label.unsqueeze(0)
                idx = torch.argmax(preds)
                
                logits_list.append(outputs[idx].unsqueeze(0).cpu().detach())

                cur_error = criterion(outputs[idx], targets[0])
                #cur_error = criterion(outputs, targets)
                error += cur_error.item()
            
            logits = torch.cat(logits_list)
            labels = torch.cat(labels_list)
            logits_np = logits.numpy()
            labels_np = labels.numpy()
            
            ece_criterion = metrics.ECELoss()
            ece_loss = ece_criterion.loss(logits_np,labels_np, 15)
            
            sce_criterion = metrics.SCELoss()
            sce_loss = sce_criterion.loss(logits_np,labels_np, 15)
            
            if error < min_error:
                min_T = T
                min_error = error
            
            #if ece_loss < ece and error < min_error:
            #    ece = ece_loss
            #    sce = sce_loss
            #    min_T = T
            #    min_error = error
        
        self.T = min_T

class DirichletScaling(nn.Module):
    def __init__(self, base_model, num_classes, optim='adam', Lambda=0., Mu=0.):
        super().__init__()

        self.base_model = base_model
        self.num_classes = num_classes

        self.optim = optim
        self.Lambda = Lambda
        self.Mu = Mu

        _freeze_model(self.base_model)
        self.setup_model()

    def setup_model(self):
        self.fc = nn.Linear(self.num_classes, self.num_classes)
    
    def forward(self, x):
        x = self.base_model(x)
        x = torch.log_softmax(x, dim=1)
        x = self.fc(x)
        return x

    def regularizer(self):
        k = self.num_classes
        W, b = self.fc.parameters()

        # keep loss value 
        w_loss = ((W**2).sum() - (torch.diagonal(W, 0)**2).sum())/(k*(k-1))
        b_loss = ((b**2).sum())/k

        return self.Lambda*w_loss + self.Mu*b_loss

    def loss_func(self, outputs, targets):
        crit = nn.CrossEntropyLoss()
        return crit(outputs, targets) + self.regularizer()

    def give_params(self):
        return self.fc.parameters()

    def fit(self, train_loader, lr=0.001, epochs=25, patience=10):

        self.train()

        # if self.optim == "sgd":
        #     optimizer = optim.SGD(self.give_params(), 
        #                         lr=lr,
        #                         weight_decay=0.0)

        # elif self.optim == "adam":
        optimizer = optim.Adam(self.give_params(),
                            lr=lr,
                            weight_decay=0.0)
        
        scheduler = EarlyStopping(patience=patience)

        # send model to gpu
        self.cuda()

        last_loss = 0.0

        bar = tqdm(range(epochs), desc="running dir for ({:.2f},{:.2f})".format(self.Lambda, self.Mu))
        for i in bar:
        # for i in range(epochs):
            avg_loss = AverageMeter()
            for imgs, labels in train_loader:
                optimizer.zero_grad()
                imgs, labels = imgs.squeeze(0).double().cuda(), labels.squeeze(0).cuda()

                outs = self.forward(imgs)
                loss = self.loss_func(outs, labels)

                loss.backward()
                optimizer.step()

                avg_loss.update(loss.item())
            
            last_loss = avg_loss.avg
            bar.set_postfix_str("loss : {:.5f} | lr : {:.5f}".format(avg_loss.avg, lr))
            if scheduler.step(avg_loss.avg):
                break
        
        return last_loss
    
    def calibrate(self, train_loader, lr=0.001, epochs=25, double_fit=True, patience=10):

        loss = self.fit(train_loader, lr, epochs, patience)

        if double_fit:
            print("Trying to double fit...")
            lr /= 10
            loss = self.fit(train_loader, lr, epochs, patience)
        
        return loss


