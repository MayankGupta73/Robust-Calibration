import numpy as np
#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt

import calibration_library.metrics as metrics

class ConfidenceHistogram(metrics.MaxProbCELoss):

    def plot(self, output, labels, n_bins = 15, logits = True, title = None):
        super().loss(output, labels, n_bins, logits)
        #scale each datapoint
        n = len(labels)
        w = np.ones(n)/n

        plt.rcParams["font.family"] = "serif"
        #size and axis limits 
        plt.figure(figsize=(5,5))
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        #plot grid
        plt.grid(color='tab:grey', linestyle=(0, (1, 5)), linewidth=1,zorder=0)    
        #plot histogram
        plt.hist(self.confidences,n_bins,weights = w,color='b',range=(0.0,1.0),edgecolor = 'k')

        #plot vertical dashed lines
        acc = np.mean(self.accuracies)
        conf = np.mean(self.confidences)                
        plt.axvline(x=acc, color='tab:grey', linestyle='--', linewidth = 3)
        plt.axvline(x=conf, color='tab:grey', linestyle='--', linewidth = 3)
        if acc > conf:
            plt.text(acc+0.03,0.9,'Accuracy',rotation=90,fontsize=11)
            plt.text(conf-0.07,0.9,'Avg. Confidence',rotation=90, fontsize=11)
        else:
            plt.text(acc-0.07,0.9,'Accuracy',rotation=90,fontsize=11)
            plt.text(conf+0.03,0.9,'Avg. Confidence',rotation=90, fontsize=11)

        plt.ylabel('% of Samples',fontsize=10)
        plt.xlabel('Confidence',fontsize=10)
        plt.tight_layout()
        if title is not None:
            plt.title(title,fontsize=12)
        return plt

class ReliabilityDiagram(metrics.MaxProbCELoss):

    def plot(self, output, labels, n_bins = 15, logits = True, title = None):
        super().loss(output, labels, n_bins, logits)

        #computations
        delta = 1.0/n_bins
        x = np.arange(0,1,delta)
        mid = np.linspace(delta/2,1-delta/2,n_bins)
        error = np.abs(np.subtract(mid,self.bin_acc))

        plt.rcParams["font.family"] = "serif"
        #size and axis limits
        plt.figure(figsize=(5,5))
        plt.xlim(0,1)
        plt.ylim(0,1)
        #plot grid
        plt.grid(color='tab:grey', linestyle=(0, (1, 5)), linewidth=1,zorder=0)
        #plot bars and identity line
        plt.bar(x, self.bin_acc, color = 'b', width=delta,align='edge',edgecolor = 'k',label='Outputs',zorder=5)
        plt.bar(x, error, bottom=np.minimum(self.bin_acc,mid), color = 'mistyrose', alpha=0.5, width=delta,align='edge',edgecolor = 'r',hatch='/',label='Gap',zorder=10)
        ident = [0.0, 1.0]
        plt.plot(ident,ident,linestyle='--',color='tab:grey',zorder=15)
        #labels and legend
        plt.ylabel('Accuracy',fontsize=10)
        plt.xlabel('Confidence',fontsize=10)
        plt.legend(loc='upper left',framealpha=1.0,fontsize='medium')
        if title is not None:
            plt.title(title,fontsize=12)
        plt.tight_layout()
        
        #print("Rel diagram values:", self.bin_acc)

        return plt

class AdaptiveReliabilityDiagram(metrics.AdaptiveMaxProbCELoss):

    def plot(self, output, labels, n_bins = 15, logits = True, title = None):
        super().loss(output, labels, n_bins, logits)

        #computations
        delta = 1.0/n_bins
        x = np.arange(0,1,delta)
        mid = np.linspace(delta/2,1-delta/2,n_bins)
        error = np.abs(np.subtract(mid,self.bin_acc))

        plt.rcParams["font.family"] = "serif"
        #size and axis limits
        plt.figure(figsize=(5,5))
        plt.xlim(0,1)
        plt.ylim(0,1)
        #plot grid
        plt.grid(color='tab:grey', linestyle=(0, (1, 5)), linewidth=1,zorder=0)
        #plot bars and identity line
        plt.bar(x, self.bin_acc, color = 'b', width=delta,align='edge',edgecolor = 'k',label='Outputs',zorder=5)
        plt.bar(x, error, bottom=np.minimum(self.bin_acc,mid), color = 'mistyrose', alpha=0.5, width=delta,align='edge',edgecolor = 'r',hatch='/',label='Gap',zorder=10)
        ident = [0.0, 1.0]
        plt.plot(ident,ident,linestyle='--',color='tab:grey',zorder=15)
        #labels and legend
        plt.ylabel('Accuracy',fontsize=10)
        plt.xlabel('Confidence',fontsize=10)
        plt.legend(loc='upper left',framealpha=1.0,fontsize='medium')
        if title is not None:
            plt.title(title,fontsize=12)
        plt.tight_layout()
        
        return plt

class AugmentedReliabilityDiagram(metrics.MaxProbCELoss):

    def plot(self, output, labels, n_bins = 15, logits = True, title = None):
        super().loss(output, labels, n_bins, logits)

        #computations
        delta = 1.0/n_bins
        x = np.arange(0,1,delta)
        mid = np.linspace(delta/2,1-delta/2,n_bins)
        error = np.abs(np.subtract(mid,self.bin_acc_flatten))

        plt.rcParams["font.family"] = "serif"
        #size and axis limits
        plt.figure(figsize=(5,5))
        plt.xlim(0,1)
        plt.ylim(0,1)
        #plot grid
        plt.grid(color='tab:grey', linestyle=(0, (1, 5)), linewidth=1,zorder=0)
        #plot bars and identity line
        plt.bar(x, self.bin_acc_flatten, color = 'b', width=delta,align='edge',edgecolor = 'k',label='Outputs',zorder=5)
        plt.bar(x, error, bottom=np.minimum(self.bin_acc_flatten,mid), color = 'mistyrose', alpha=0.5, width=delta,align='edge',edgecolor = 'r',hatch='/',label='Gap',zorder=10)
        ident = [0.0, 1.0]
        plt.plot(ident,ident,linestyle='--',color='tab:grey',zorder=15)
        #labels and legend
        plt.ylabel('% Sample',fontsize=10)
        plt.xlabel('Confidence',fontsize=10)
        plt.legend(loc='upper left',framealpha=1.0,fontsize='medium')
        if title is not None:
            plt.title(title,fontsize=12)
        plt.tight_layout()
        
        return plt
        
class AugmentedConfidenceDiagram(metrics.MaxProbCELoss):

    def plot(self, output, labels, n_bins = 15, logits = True, title = None):
        super().loss(output, labels, n_bins, logits)

        #computations
        delta = 1.0/n_bins
        x = np.arange(0,1,delta)
        mid = np.linspace(delta/2,1-delta/2,n_bins)
        error = np.abs(np.subtract(mid,self.bin_perc))

        plt.rcParams["font.family"] = "serif"
        #size and axis limits
        plt.figure(figsize=(5,5))
        plt.xlim(0,1)
        plt.ylim(0,1)
        #plot grid
        plt.grid(color='tab:grey', linestyle=(0, (1, 5)), linewidth=1,zorder=0)
        #plot bars and identity line
        plt.bar(x, self.bin_perc, color = 'b', width=delta,align='edge',edgecolor = 'k',label='Outputs',zorder=5)
        #plt.bar(x, error, bottom=np.minimum(self.bin_perc,mid), color = 'mistyrose', alpha=0.5, width=delta,align='edge',edgecolor = 'r',hatch='/',label='Gap',zorder=10)
        ident = [0.0, 1.0]
        plt.plot(ident,ident,linestyle='--',color='tab:grey',zorder=15)
        #labels and legend
        plt.ylabel('% of total samples',fontsize=10)
        plt.xlabel('Confidence',fontsize=10)
        plt.legend(loc='upper left',framealpha=1.0,fontsize='medium')
        if title is not None:
            plt.title(title,fontsize=12)
        plt.tight_layout()

        return plt
        
class ClasswiseAugmentedReliabilityDiagram(metrics.MaxProbCELoss):

    def plot(self, output, labels, n_bins = 15, logits = True, title = None):
        super().loss(output, labels, n_bins, logits)

        fig, ax = plt.subplots(1, 3, figsize=(15,5))
        plt.rcParams["font.family"] = "serif"
        #size and axis limits
        #fig.figure(figsize=(15,5))
        plt.xlim(0,1)
        plt.ylim(0,1)
        
        n_cls = output.shape[1]

        for i in range(n_cls):
            plot = ax[i]

            #computations
            delta = 1.0/n_bins
            x = np.arange(0,1,delta)
            mid = np.linspace(delta/2,1-delta/2,n_bins)
            error = np.abs(np.subtract(mid,self.bin_acc_cls[i]))
    
            #plot grid
            plot.grid(color='tab:grey', linestyle=(0, (1, 5)), linewidth=1,zorder=0)
            #plot bars and identity line
            plot.bar(x, self.bin_acc_cls[i], color = 'b', width=delta,align='edge',edgecolor = 'k',label='Outputs',zorder=5)
            plot.bar(x, error, bottom=np.minimum(self.bin_acc_cls[i],mid), color = 'mistyrose', alpha=0.5, width=delta,align='edge',edgecolor = 'r',hatch='/',label='Gap',zorder=10)
            ident = [0.0, 1.0]
            plot.plot(ident,ident,linestyle='--',color='tab:grey',zorder=15)
        
            #labels and legend
            plot.set_ylabel('Accuracy',fontsize=10)
            plot.set_xlabel('Confidence',fontsize=10)
            plot.legend(loc='upper left',framealpha=1.0,fontsize='medium')
            if title is not None:
                plot.set_title(title,fontsize=12)
        fig.tight_layout()
        
        return fig
        
class ClasswiseAugmentedConfidenceDiagram(metrics.MaxProbCELoss):

    def plot(self, output, labels, n_bins = 15, logits = True, title = None):
        super().loss(output, labels, n_bins, logits)

        fig, ax = plt.subplots(1, 3, figsize=(15,5))
        plt.rcParams["font.family"] = "serif"
        #size and axis limits
        #fig.figure(figsize=(15,5))
        plt.xlim(0,1)
        plt.ylim(0,1)
        
        n_cls = output.shape[1]

        for i in range(n_cls):
            plot = ax[i]

            #computations
            delta = 1.0/n_bins
            x = np.arange(0,1,delta)
            mid = np.linspace(delta/2,1-delta/2,n_bins)
            error = np.abs(np.subtract(mid,self.bin_acc_cls[i]))
    
            #plot grid
            plot.grid(color='tab:grey', linestyle=(0, (1, 5)), linewidth=1,zorder=0)
            #plot bars and identity line
            plot.bar(x, self.bin_prop_cls[i], color = 'b', width=delta,align='edge',edgecolor = 'k',label='Outputs',zorder=5)
            #plot.bar(x, error, bottom=np.minimum(self.bin_prop_cls[i],mid), color = 'mistyrose', alpha=0.5, width=delta,align='edge',edgecolor = 'r',hatch='/',label='Gap',zorder=10)
            ident = [0.0, 1.0]
            plot.plot(ident,ident,linestyle='--',color='tab:grey',zorder=15)
        
            #labels and legend
            plot.set_ylabel('% of total samples',fontsize=10)
            plot.set_xlabel('Confidence',fontsize=10)
            plot.legend(loc='upper left',framealpha=1.0,fontsize='medium')
            if title is not None:
                plot.set_title(title,fontsize=12)
        fig.tight_layout()

        return fig
        
class GaussianConfidenceHistogram(metrics.MaxProbGCELoss):

    def plot(self, output, labels, n_bins = 15, logits = True, title = None):
        super().loss(output, labels, n_bins, logits)
        #scale each datapoint
        n = len(labels)
        w = np.ones(n)/n
        
        y = self.bin_prop_gaussian / np.sum(self.bin_prop_gaussian)

        plt.rcParams["font.family"] = "serif"
        #size and axis limits 
        plt.figure(figsize=(5,5))
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        #plot grid
        plt.grid(color='tab:grey', linestyle=(0, (1, 5)), linewidth=1,zorder=0)    
        #plot histogram
        plt.hist(np.linspace(0.0, 1.0, num=15), n_bins, weights = y ,color='b',range=(0.0,1.0),edgecolor = 'k')
        #plt.bar(np.linspace(0.0, 1.0, num=15), y)

        plt.ylabel('% of Samples',fontsize=10)
        plt.xlabel('Confidence',fontsize=10)
        plt.tight_layout()
        if title is not None:
            plt.title(title,fontsize=12)
            
        return plt
        
class GaussianReliabilityDiagram(metrics.MaxProbGCELoss):

    def plot(self, output, labels, n_bins = 15, logits = True, title = None):
        super().loss(output, labels, n_bins, logits)

        #computations
        delta = 1.0/n_bins
        x = np.arange(0,1,delta)
        mid = np.linspace(delta/2,1-delta/2,n_bins)
        error = np.abs(np.subtract(mid,self.bin_acc_gaussian))

        plt.rcParams["font.family"] = "serif"
        #size and axis limits
        plt.figure(figsize=(5,5))
        plt.xlim(0,1)
        plt.ylim(0,1)
        #plot grid
        plt.grid(color='tab:grey', linestyle=(0, (1, 5)), linewidth=1,zorder=0)
        #plot bars and identity line
        plt.bar(x, self.bin_acc_gaussian, color = 'b', width=delta,align='edge',edgecolor = 'k',label='Outputs',zorder=5)
        plt.bar(x, error, bottom=np.minimum(self.bin_acc_gaussian,mid), color = 'mistyrose', alpha=0.5, width=delta,align='edge',edgecolor = 'r',hatch='/',label='Gap',zorder=10)
        ident = [0.0, 1.0]
        plt.plot(ident,ident,linestyle='--',color='tab:grey',zorder=15)
        #labels and legend
        plt.ylabel('Accuracy',fontsize=10)
        plt.xlabel('Confidence',fontsize=10)
        plt.legend(loc='upper left',framealpha=1.0,fontsize='medium')
        if title is not None:
            plt.title(title,fontsize=12)
        plt.tight_layout()
        
        return plt