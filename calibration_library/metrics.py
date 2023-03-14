# taken from https://github.com/Jonathan-Pearce/calibration_library/blob/master/metrics.py

import numpy as np
from scipy.special import softmax, expit
from scipy.stats import norm, truncnorm, cauchy, t, expon, rv_continuous, rv_discrete
from sklearn.mixture import GaussianMixture
from distfit import distfit

class BrierScore():
    def __init__(self) -> None:
        pass

    def loss(self, outputs, targets):
        K = outputs.shape[1]
        one_hot = np.eye(K)[targets]
        probs = softmax(outputs, axis=1)
        return np.mean( np.sum( (probs - one_hot)**2 , axis=1) )


class CELoss(object):

    def compute_bin_boundaries(self, probabilities = np.array([])):

        #uniform bin spacing
        if probabilities.size == 0:
            bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
            self.bin_lowers = bin_boundaries[:-1]
            self.bin_uppers = bin_boundaries[1:]
        else:
            #size of bins 
            bin_n = int(self.n_data/self.n_bins)

            bin_boundaries = np.array([])

            probabilities_sort = np.sort(probabilities)  

            for i in range(0,self.n_bins):
                bin_boundaries = np.append(bin_boundaries,probabilities_sort[i*bin_n])
            bin_boundaries = np.append(bin_boundaries,1.0)

            self.bin_lowers = bin_boundaries[:-1]
            self.bin_uppers = bin_boundaries[1:]


    def get_probabilities(self, output, labels, logits):
        #If not probabilities apply softmax!
        if logits:
            self.probabilities = softmax(output, axis=1)
        else:
            self.probabilities = output

        self.labels = labels
        self.confidences = np.max(self.probabilities, axis=1)
        self.predictions = np.argmax(self.probabilities, axis=1)
        self.accuracies = np.equal(self.predictions,labels)
        
        self.num_classes = np.max(labels) + 1
        self.num_samples = labels.shape[0]
        
        self.conf_flatten = self.probabilities.reshape(-1)
        self.label_onehot = np.eye(self.num_classes)[labels]
        self.label_flatten = self.label_onehot.reshape(-1)
        
        self.labels_binary = self.label_onehot[:, -1]
        #self.accuracies_flatten = np.equal(self.predictions,label_flatten)
        
        #print("prob:", self.probabilities.shape)
        #print("Conf Flatten:", self.conf_flatten)
        #print("Labels",self.labels )
        #print("labels one hot", self.label_onehot)

    def binary_matrices(self):
        idx = np.arange(self.n_data)
        #make matrices of zeros
        pred_matrix = np.zeros([self.n_data,self.n_class])
        label_matrix = np.zeros([self.n_data,self.n_class])
        pred_matrix[idx,self.predictions] = 1
        label_matrix[idx,self.labels] = 1

        self.acc_matrix = np.equal(pred_matrix, label_matrix)


    def compute_bins(self, index = None):
        self.bin_prop = np.zeros(self.n_bins)
        self.bin_acc = np.zeros(self.n_bins)
        self.bin_conf = np.zeros(self.n_bins)
        self.bin_score = np.zeros(self.n_bins)
        
        self.bin_acc_flatten = np.zeros(self.n_bins)
        self.bin_perc = np.zeros(self.n_bins)
        self.bin_acc_cls = np.zeros((self.num_classes, self.n_bins))
        self.bin_acc_binary_cls = np.zeros((2, self.n_bins))
        self.bin_prop_cls = np.zeros((self.num_classes, self.n_bins))

        if index == None:
            confidences = self.confidences
            accuracies = self.accuracies
        else:
            confidences = self.probabilities[:,index]
            accuracies = (self.labels == index).astype("float")


        for i, (bin_lower, bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
            # Calculated |confidence - accuracy| in each bin
            in_bin = np.greater(confidences,bin_lower.item()) * np.less_equal(confidences,bin_upper.item())
            self.bin_prop[i] = np.mean(in_bin)

            if self.bin_prop[i].item() > 0:
                self.bin_acc[i] = np.mean(accuracies[in_bin])
                self.bin_conf[i] = np.mean(confidences[in_bin])
                self.bin_score[i] = np.abs(self.bin_conf[i] - self.bin_acc[i])
                
            in_bin_flatten = np.greater(self.conf_flatten ,bin_lower.item()) * np.less_equal(self.conf_flatten ,bin_upper.item())
            bin_prop_flatten = np.mean(in_bin_flatten)
        
            # if bin_prop_flatten.item() > 0:
            #     self.bin_acc_flatten[i] = np.mean(self.label_flatten[in_bin_flatten])
            #     self.bin_perc[i] = np.sum(self.label_flatten[in_bin_flatten]) / self.num_samples
            #     for c in range(self.num_classes):
            #         in_bin_cls = in_bin_flatten.reshape((self.num_samples, self.num_classes))[:, c]
            #         #print("label onehot", self.label_onehot.shape)
            #         #print("in bin cls:", in_bin_cls.shape, in_bin_cls)
            #         self.bin_prop_cls[c, i] = self.label_onehot[in_bin_cls, c].sum() / self.num_samples
            #         #if np.mean(in_bin_cls).item() > 0:
            #             #self.bin_acc_cls[c, i] = np.mean(self.label_onehot[in_bin_cls, c])
            #             #label_cls = np.equal(self.labels, c) 
            #             #self.bin_acc_cls[c, i] = np.mean(self.label_onehot[in_bin_cls, c])
                        
            # for c in range(self.num_classes):
            #     label_cls = np.equal(self.labels, c)
            #     #print(self.label_onehot)
            #     in_bin_cls = in_bin_flatten.reshape((self.num_samples, self.num_classes))[label_cls, :].reshape(-1)
            #     bin_cls_flatten = np.mean(in_bin_cls)
            #     if bin_cls_flatten.item() > 0:
            #         self.bin_acc_cls[c, i] = np.mean(self.label_onehot[label_cls, :].reshape(-1)[in_bin_cls])
            
            #for c in range(2):
            #    in_bin_cls = in_bin_flatten.reshape((self.num_samples, self.num_classes))[self.label_onehot[:, c], :].reshape(-1)
            #    bin_cls_flatten = np.mean(in_bin_cls)
            #    if bin_cls_flatten.item() > 0:
            #        self.bin_acc_cls[c, i] = np.mean(self.label_onehot[self.label_onehot[:, c], :].reshape(-1)[in_bin_cls])
                
class MaxProbCELoss(CELoss):
    def loss(self, output, labels, n_bins = 15, logits = True):
        self.n_bins = n_bins
        super().compute_bin_boundaries()
        super().get_probabilities(output, labels, logits)
        super().compute_bins()

class AdaptiveMaxProbCELoss(CELoss):
    def loss(self, output, labels, n_bins = 15, logits = True):
        self.n_bins = n_bins
        self.n_data = len(labels)
        super().get_probabilities(output, labels, logits)
        super().compute_bin_boundaries(self.confidences)
        super().compute_bins()

#http://people.cs.pitt.edu/~milos/research/AAAI_Calibration.pdf
class ECELoss(MaxProbCELoss):

    def loss(self, output, labels, n_bins = 15, logits = True):
        super().loss(output, labels, n_bins, logits)
        return np.dot(self.bin_prop,self.bin_score)
        
#http://people.cs.pitt.edu/~milos/research/AAAI_Calibration.pdf
class AdaptiveECELoss(AdaptiveMaxProbCELoss):

    def loss(self, output, labels, n_bins = 15, logits = True):
        super().loss(output, labels, n_bins, logits)
        return np.dot(self.bin_prop,self.bin_score)

class MCELoss(MaxProbCELoss):
    
    def loss(self, output, labels, n_bins = 15, logits = True):
        super().loss(output, labels, n_bins, logits)
        return np.max(self.bin_score)

#https://arxiv.org/abs/1905.11001
#Overconfidence Loss (Good in high risk applications where confident but wrong predictions can be especially harmful)
class OELoss(MaxProbCELoss):

    def loss(self, output, labels, n_bins = 15, logits = True):
        super().loss(output, labels, n_bins, logits)
        return np.dot(self.bin_prop,self.bin_conf * np.maximum(self.bin_conf-self.bin_acc,np.zeros(self.n_bins)))


#https://arxiv.org/abs/1904.01685
class SCELoss(CELoss):

    def loss(self, output, labels, n_bins = 15, logits = True):
        sce = 0.0
        self.n_bins = n_bins
        self.n_data = len(output)
        self.n_class = len(output[0])

        super().compute_bin_boundaries()
        super().get_probabilities(output, labels, logits)
        super().binary_matrices()

        for i in range(self.n_class):
            super().compute_bins(i)
            sce += np.dot(self.bin_prop, self.bin_score)

        return sce/self.n_class

class ClassjECELoss(CELoss):

    def loss(self, output, labels, n_bins = 15, logits = True):
        jce = []
        self.n_bins = n_bins
        self.n_data = len(output)
        self.n_class = len(output[0])

        super().compute_bin_boundaries()
        super().get_probabilities(output, labels, logits)
        super().binary_matrices()

        for i in range(self.n_class):
            super().compute_bins(i)
            jce.append(np.dot(self.bin_prop, self.bin_score))

        return jce

class TACELoss(CELoss):

    def loss(self, output, labels, threshold = 0.01, n_bins = 15, logits = True):
        tace = 0.0
        self.n_bins = n_bins
        self.n_data = len(output)
        self.n_class = len(output[0])

        super().get_probabilities(output, labels, logits)
        self.probabilities[self.probabilities < threshold] = 0
        super().binary_matrices()

        for i in range(self.n_class):
            super().compute_bin_boundaries(self.probabilities[:,i]) 
            super().compute_bins(i)
            tace += np.dot(self.bin_prop,self.bin_score)

        return tace/self.n_class

#create TACELoss with threshold fixed at 0
class ACELoss(TACELoss):

    def loss(self, output, labels, n_bins = 15, logits = True):
        return super().loss(output, labels, 0.0 , n_bins, logits)
        
class GaussianCELoss(object):

    def compute_bin_boundaries(self, probabilities = np.array([])):

        #uniform bin spacing
        if probabilities.size == 0:
            bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
            self.bin_lowers = bin_boundaries[:-1]
            self.bin_uppers = bin_boundaries[1:]
        else:
            #size of bins 
            bin_n = int(self.n_data/self.n_bins)

            bin_boundaries = np.array([])

            probabilities_sort = np.sort(probabilities)  

            for i in range(0,self.n_bins):
                bin_boundaries = np.append(bin_boundaries,probabilities_sort[i*bin_n])
            bin_boundaries = np.append(bin_boundaries,1.0)

            self.bin_lowers = bin_boundaries[:-1]
            self.bin_uppers = bin_boundaries[1:]


    def get_probabilities(self, output, labels, logits):
        #If not probabilities apply softmax!
        if logits:
            self.probabilities = softmax(output, axis=1)
        else:
            self.probabilities = output

        self.labels = labels
        self.confidences = np.max(self.probabilities, axis=1)
        self.predictions = np.argmax(self.probabilities, axis=1)
        self.accuracies = np.equal(self.predictions,labels)
        
        self.num_classes = np.max(labels) + 1
        self.num_samples = labels.shape[0]
        
        self.conf_flatten = self.probabilities.reshape(-1)
        self.label_onehot = np.eye(self.num_classes)[labels]
        self.label_flatten = self.label_onehot.reshape(-1)
        
        self.labels_binary = self.label_onehot[:, -1]
    

    def get_distr(self, mean, sigma):
        if self.distr_name == "gaussian":
            distr = norm(loc = mean, scale = sigma)
        elif self.distr_name == "cauchy":
            distr = cauchy(loc = mean, scale = sigma)
        elif self.distr_name == "t":
            distr = t(loc = mean, scale = sigma, df=3)
        elif self.distr_name == "exp-central":
            distr = exp_central(loc = mean, scale = sigma)

        return distr

    def generate_distr_weights(self):
        total_weights = np.zeros([self.n_bins, self.n_bins])
        self.bin_centers = np.linspace(0.0, 1.0, num=self.n_bins)
        if self.sd:
            sc = 0.1
        else:
            sc = self.gece_sigma
        for i in range(len(self.bin_centers)):
            distr = self.get_distr(mean = self.bin_centers[i], sigma = sc)
            total_weights[i] = distr.pdf(self.bin_centers)
            
        total_weights = total_weights / np.sum(total_weights, axis=1)
        return total_weights
        

    def binary_matrices(self):
        idx = np.arange(self.n_data)
        #make matrices of zeros
        pred_matrix = np.zeros([self.n_data,self.n_class])
        label_matrix = np.zeros([self.n_data,self.n_class])
        pred_matrix[idx,self.predictions] = 1
        label_matrix[idx,self.labels] = 1

        self.acc_matrix = np.equal(pred_matrix, label_matrix)


    def compute_bins(self, index = None):
        self.bin_prop = np.zeros(self.n_bins)
        self.bin_count = np.zeros(self.n_bins)
        self.sample_bin_map = np.zeros(self.num_samples)
        self.bin_acc = np.zeros(self.n_bins)
        self.bin_conf = np.zeros(self.n_bins)
        self.bin_score = np.zeros(self.n_bins)
        
        self.bin_acc_flatten = np.zeros(self.n_bins)
        self.bin_perc = np.zeros(self.n_bins)
        self.bin_acc_cls = np.zeros((self.num_classes, self.n_bins))
        self.bin_acc_binary_cls = np.zeros((2, self.n_bins))
        self.bin_prop_cls = np.zeros((self.num_classes, self.n_bins))
        
        self.bin_acc_gaussian = np.zeros(self.n_bins)
        self.bin_conf_gaussian = np.zeros(self.n_bins)
        self.bin_conf_gaussian_temp = np.zeros(self.n_bins)
        self.bin_prop_gaussian = np.zeros(self.n_bins)
        
        self.gaussian_weights = self.generate_distr_weights()

        if index == None:
            confidences = self.confidences
            accuracies = self.accuracies
        else:
            confidences = self.probabilities[:,index]
            accuracies = (self.labels == index).astype("float")
            
        for i, (bin_lower, bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
            in_bin = np.greater(confidences,bin_lower.item()) * np.less_equal(confidences,bin_upper.item())
            self.bin_count[i] = np.sum(in_bin)
            self.sample_bin_map[in_bin] = i
        
        g = np.zeros((self.num_samples, self.n_bins))
        for i in range(self.num_samples):
            if self.sd:
                distr = self.get_distr(mean = self.confidences[i], sigma = self.gece_sigma[i]/15)
            else:
                distr = self.get_distr(mean = self.confidences[i], sigma = self.gece_sigma)
            
            g[i] = distr.cdf(self.bin_uppers) - distr.cdf(self.bin_lowers)
            if np.sum(g[i]) == 0:
                g[i] = np.zeros(self.n_bins)
                g[i][int(self.sample_bin_map[i])] = 1.0
                
            self.bin_conf_gaussian += self.confidences[i] * (g[i] / np.sum(g[i]))
            self.bin_acc_gaussian += self.accuracies[i] * (g[i] / np.sum(g[i]))

        for i, (bin_lower, bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
            # Calculated |confidence - accuracy| in each bin
            in_bin = np.greater(confidences,bin_lower.item()) * np.less_equal(confidences,bin_upper.item())
            self.bin_prop[i] = np.mean(in_bin)

            if self.bin_prop[i].item() > 0:
                self.bin_acc[i] = np.mean(accuracies[in_bin])
                self.bin_conf[i] = np.mean(confidences[in_bin])
                self.bin_prop_gaussian += np.sum(in_bin) * self.gaussian_weights[i]
                
                self.bin_score[i] =  np.abs(self.bin_conf_gaussian[i] - self.bin_acc_gaussian[i])

class exp_central(rv_continuous):
    def __init__(self, loc, scale):
        self.mean=loc
        self.sigma = scale
        super().__init__()
    
    def _pdf(self, x):
        exp = expon(loc = self.mean, scale = self.sigma)
        op = np.where(x > self.mean, exp.pdf(x), exp.pdf(2*self.mean - x))
        return op
                    
class MaxProbGCELoss(GaussianCELoss):
    def loss(self, output, labels, n_bins = 15, logits = True, distr_name = "t", sigma=0.1, sd = False):
        self.n_bins = n_bins
        self.gece_sigma = sigma
        self.sd = sd
        self.distr_name = distr_name
        super().compute_bin_boundaries()
        super().get_probabilities(output, labels, logits)
        super().compute_bins()
        
class GECELoss(MaxProbGCELoss):

    def loss(self, output, labels, n_bins = 15, logits = True, distr_name = "t", sigma=0.1, sd = False):
        super().loss(output, labels, n_bins, logits, distr_name, sigma, sd)
        return np.sum(self.bin_score) / self.num_samples

class GaussianMixtureCELoss(object):

    def compute_bin_boundaries(self, probabilities = np.array([])):

        #uniform bin spacing
        if probabilities.size == 0:
            bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
            self.bin_lowers = bin_boundaries[:-1]
            self.bin_uppers = bin_boundaries[1:]
        else:
            #size of bins 
            bin_n = int(self.n_data/self.n_bins)

            bin_boundaries = np.array([])

            probabilities_sort = np.sort(probabilities)  

            for i in range(0,self.n_bins):
                bin_boundaries = np.append(bin_boundaries,probabilities_sort[i*bin_n])
            bin_boundaries = np.append(bin_boundaries,1.0)

            self.bin_lowers = bin_boundaries[:-1]
            self.bin_uppers = bin_boundaries[1:]


    def get_ce(self, probabilities, target):
        ll = - np.log(probabilities[:, np.arange(len(target)), target]) * 100
        ll = expit(ll) / 10
        return ll

    def get_probabilities(self, output, labels, logits):
        #If not probabilities apply softmax!
        if logits:
            self.probabilities = softmax(output, axis=2)
            self.cls_output = softmax(self.cls_output, axis=1)
        else:
            self.probabilities = output

        self.labels = labels
        self.confidences = np.max(self.probabilities, axis=2)
        self.predictions = np.argmax(self.probabilities, axis=2)
        self.accuracies = np.equal(self.predictions, labels)
        self.mixture_ce = self.get_ce(self.probabilities, self.cls_target)

        self.cls_confidences = np.max(self.cls_output, axis=1)
        self.cls_predictions = np.argmax(self.cls_output, axis=1)
        self.cls_accuracies = np.equal(self.cls_predictions, self.cls_target)
        
        self.num_samples = labels.shape[1]
        self.num_mixture = labels.shape[0]

    def get_distr(self, mean, sigma):
        if self.distr_name == "gaussian":
            distr = norm(loc = mean, scale = sigma)
        elif self.distr_name == "cauchy":
            distr = cauchy(loc = mean, scale = sigma)
        elif self.distr_name == "t":
            distr = t(loc = mean, scale = sigma, df=3)
        elif self.distr_name == "exp-central":
            distr = exp_central(loc = mean, scale = sigma)

        return distr
    
    def infer_distr(self, confidences):
        if self.distr_name == "gaussian-single":
            dfit = distfit(distr='norm')
            loc, scale = dfit.model['loc'], dfit.model['scale']
            distr = norm(loc = loc, scale = scale)
        elif self.distr_name == "cauchy":
            dfit = distfit(distr='cauchy')
            loc, scale = dfit.model['loc'], dfit.model['scale']
            distr = cauchy(loc = loc, scale = scale)
        elif self.distr_name == "t":
            dfit = distfit(distr='t')
            loc, scale = dfit.model['loc'], dfit.model['scale']
            distr = t(loc = loc, scale = scale)

        return distr


    def generate_distr_weights(self):
        total_weights = np.zeros([self.n_bins, self.n_bins])
        self.bin_centers = np.linspace(0.0, 1.0, num=self.n_bins)
        if self.mode != "default":
            sc = 0.1
        else:
            sc = self.gece_sigma 
        for i in range(len(self.bin_centers)):
            distr = self.get_distr(mean = self.bin_centers[i], sigma = sc)
            total_weights[i] = distr.pdf(self.bin_centers)
            
        total_weights = total_weights / np.sum(total_weights, axis=1)
        return total_weights        

    def binary_matrices(self):
        idx = np.arange(self.n_data)
        #make matrices of zeros
        pred_matrix = np.zeros([self.n_data,self.n_class])
        label_matrix = np.zeros([self.n_data,self.n_class])
        pred_matrix[idx,self.predictions] = 1
        label_matrix[idx,self.labels] = 1

        self.acc_matrix = np.equal(pred_matrix, label_matrix)


    def compute_bins(self, index = None):
        self.bin_prop = np.zeros(self.n_bins)
        self.bin_count = np.zeros(self.n_bins)
        self.sample_bin_map = np.zeros(self.num_samples)
        self.bin_acc = np.zeros(self.n_bins)
        self.bin_conf = np.zeros(self.n_bins)
        self.bin_score = np.zeros(self.n_bins)
        
        self.bin_acc_gaussian = np.zeros(self.n_bins)
        self.bin_conf_gaussian = np.zeros(self.n_bins)
        self.bin_prop_gaussian = np.zeros(self.n_bins)
        
        self.gaussian_weights = self.generate_distr_weights()

        if index == None:
            confidences = self.confidences
            accuracies = self.accuracies
        else:
            confidences = self.probabilities[:,index]
            accuracies = (self.labels == index).astype("float")
            
        for i, (bin_lower, bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
            in_bin = np.greater(self.cls_confidences, bin_lower.item()) * np.less_equal(self.cls_confidences, bin_upper.item())
            self.bin_count[i] = np.sum(in_bin)
            self.sample_bin_map[in_bin] = i
    
        g = np.zeros((self.num_samples, self.n_bins))
        for i in range(self.num_samples):
            if self.mode == "default":
                for j in range(self.num_mixture):
                    distr = self.get_distr(mean = self.confidences[j, i], sigma = self.gece_sigma)
                    wts = distr.cdf(self.bin_uppers) - distr.cdf(self.bin_lowers)
                    if np.sum(wts) == 0:
                        wts = np.zeros(self.n_bins)
                        wts[int(self.sample_bin_map[i])] = 1.0
                    g[i] += wts / self.num_mixture
                    
                self.bin_conf_gaussian += self.cls_confidences[i] * (g[i] / np.sum(g[i]))
                self.bin_acc_gaussian += self.cls_accuracies[i] * (g[i] / np.sum(g[i]))
            elif self.mode == "gmm_estimate":
                if self.distr_name == "gaussian":
                    cf = confidences[:, i].reshape(-1, 1)
                    gmm = GaussianMixture(n_components = 3)
                    gmm.fit(cf)
                    gmm_wts, gmm_means, gmm_stds = gmm.weights_.reshape(-1), gmm.means_.reshape(-1), gmm.covariances_.reshape(-1)
                    for j in range(3):
                        distr = self.get_distr(mean = gmm_means[j], sigma = gmm_stds[j])
                        wts = distr.cdf(self.bin_uppers) - distr.cdf(self.bin_lowers)
                        if np.sum(wts) == 0:
                            wts = np.zeros(self.n_bins)
                            wts[int(self.sample_bin_map[i])] = 1.0
                        g[i] += gmm_wts[j] * wts / self.num_mixture
                else:
                    distr = self.infer_distr(confidences[:, i])
                    wts = distr.cdf(self.bin_uppers) - distr.cdf(self.bin_lowers)
                    if np.sum(wts) == 0:
                        wts = np.zeros(self.n_bins)
                        wts[int(self.sample_bin_map[i])] = 1.0
                    g[i] += wts

                self.bin_conf_gaussian += self.cls_confidences[i] * (g[i] / np.sum(g[i]))
                self.bin_acc_gaussian += self.cls_accuracies[i] * (g[i] / np.sum(g[i]))
            elif self.mode == "sd":
                for j in range(self.num_mixture):
                    distr = self.get_distr(mean = self.confidences[j, i], sigma = self.gece_sigma[i]/15)
                    wts = distr.cdf(self.bin_uppers) - distr.cdf(self.bin_lowers)
                    if np.sum(wts) == 0:
                        wts = np.zeros(self.n_bins)
                        wts[int(self.sample_bin_map[i])] = 1.0
                    g[i] += wts / self.num_mixture
                
                self.bin_conf_gaussian += self.cls_confidences[i] * (g[i] / np.sum(g[i]))
                self.bin_acc_gaussian += self.cls_accuracies[i] * (g[i] / np.sum(g[i]))
            elif self.mode == "smd":
                for j in range(self.num_mixture):
                    distr = self.get_distr(mean = self.confidences[j, i], sigma = self.mixture_ce[j, i])
                    wts = distr.pdf(self.bin_centers)
                    if np.sum(wts) == 0:
                        wts = np.zeros(self.n_bins)
                        wts[int(self.sample_bin_map[i])] = 1.0
                    g[i] += wts / self.num_mixture
                
                self.bin_conf_gaussian += self.cls_confidences[i] * (g[i] / np.sum(g[i]))
                self.bin_acc_gaussian += self.cls_accuracies[i] * (g[i] / np.sum(g[i]))


        for i, (bin_lower, bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
            # Calculated |confidence - accuracy| in each bin
            in_bin = np.greater(self.cls_confidences, bin_lower.item()) * np.less_equal(self.cls_confidences, bin_upper.item())
            self.bin_prop[i] = np.mean(in_bin)

            if self.bin_prop[i].item() > 0:
                self.bin_prop_gaussian += np.sum(in_bin) * self.gaussian_weights[i]
                self.bin_score[i] =  np.abs(self.bin_conf_gaussian[i] - self.bin_acc_gaussian[i])
                    
class MaxProbGMCELoss(GaussianMixtureCELoss):
    def loss(self, output, labels, cls_output, cls_target, n_total, n_bins = 15, logits = True, distr_name = "gaussian", sigma=0.1, mode = "default"):
        self.cls_output = cls_output
        self.cls_target = cls_target
        self.n_total = n_total
        self.n_bins = n_bins
        self.gece_sigma = sigma
        self.distr_name = distr_name
        self.mode = mode
        super().compute_bin_boundaries()
        super().get_probabilities(output, labels, logits)
        super().compute_bins()
        
class GMECELoss(MaxProbGMCELoss):
    def loss(self, output, labels, cls_output, cls_target, n_total, n_bins = 15, logits = True, distr_name = "gaussian", sigma=0.1, mode = "default"):
        super().loss(output, labels, cls_output, cls_target, n_total, n_bins, logits, distr_name, sigma, mode)
        return np.sum(self.bin_score) / self.num_samples