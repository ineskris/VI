from torch.autograd import Variable
import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import *
from torch.distributions.utils import probs_to_logits, logits_to_probs
from torch.nn.functional import binary_cross_entropy_with_logits
from scipy.special import expit as sigmoid

import numpy as np
from torch.autograd import Variable
#Simulation
n = 2000
p = 1 
X = np.random.randn(n,p) #standard normal distribution
w_real = np.array([1]) #to infer 
y = np.random.binomial(1,sigmoid(np.dot(X,w_real)),n)


class bbvi(torch.nn.Module):
    def __init__(self, p=1, nb_sample=10):
        self.p = p #1 for now 
        self.nb_sample = nb_sample

        super(bbvi, self).__init__()
        # mu and sigma of q distrib for w 
        self.mu = torch.nn.Parameter(torch.randn(1,p), requires_grad=True)
        self.std = torch.nn.Parameter(torch.randn(1,p), requires_grad=True)
        self.softplus = torch.nn.Softplus() #for positive variance
        self.sigmoid = torch.nn.Sigmoid()
        
        #prior
        self.prior_m = Variable(torch.randn(1,p), requires_grad=False)
        self.prior_s = Variable(torch.randn(1,p), requires_grad=False)
        
    def gen_sample(self):
        normal = Normal(self.mu,self.std)
        sample = normal.sample(sample_shape=torch.Size([self.nb_sample]))
        return(sample)
    
    def reparam(self, eps):
        eps = Variable(torch.FloatTensor(eps))
        return  eps.mul(self.softplus(self.std)).add(self.mu)
    
    def compute_elbo(self, X, y ):
        eps = self.gen_sample()
        z = self.reparam(eps) 
        
        q_likelihood = Normal(self.mu,self.softplus(self.std)).log_prob(z).mean(0)
        prior = Normal(self.prior_m, self.softplus(self.prior_s)).log_prob(z).mean(0)
        kld = q_likelihood - prior

        loglike = binary_cross_entropy_with_logits(self.sigmoid(torch.matmul(X,self.mu).reshape([len(y)])), y) #self.mu ? 
        #Function that measures Binary Cross Entropy between target and output logits. 

        elbo = loglike - kld
        return elbo
    

bb = bbvi(p=1)
optimizer = torch.optim.Adagrad(bb.parameters(),lr=0.01) #lr = learning rate lr 
# input as Tensor
X = torch.Tensor(X)
y= torch.Tensor(y)

for i in range(4001):
    loss = -bb.compute_elbo(X, y)
    # clear out the gradients of all Variables in this optimizer (i.e. W, b)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 400 ==0:
        print(bb.mu.data.numpy(), (bb.softplus(bb.std).data**2).numpy())
        print(-bb.compute_elbo(X, y ))

print("w_real", w_real)
   