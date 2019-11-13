


# Simulation
from scipy.special import expit as sigmoid
from scipy.stats import norm
n = 1000
p = 1
X = np.random.randn(n,p) #standard normal distribution
z_real =  np.array([-2])
y = np.random.binomial(1,sigmoid(np.dot(X,z_real)))


#lets take p =1 t0 make it simple. 

class VAE(nn.Module):
    '''
    X : Tensor nxp given
    y : Tensor n given
    z : Tensor s x p (s=number of sample , z~q(z|y,x) distribution)
    '''
    def __init__(self):
        super(VAE, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()
       
        self.mu = torch.nn.Parameter(torch.randn(1,p), requires_grad=True)
        self.logsigma = torch.nn.Parameter(torch.randn(1,p), requires_grad=True)
        #prior
        self.prior_m = Variable(torch.zeros(1,p), requires_grad=False)
        self.prior_log_sigma = Variable(torch.ones(1,p), requires_grad=False)
        
    def encoder(self, y):
        p = y.mean() #estimator
        x = p.log()-(1-p).log()
        mu = xz
       
        return mu
    
    def reparam(self, mu):
        eps = Variable(torch.FloatTensor( torch.randn(1,1)))
        sigma = self.logsigma.exp()
        return  eps.mul(sigma).add(self.mu)
    
    
    def decoder(self,z,X):
        mult = torch.matmul(X,z.transpose(0,1)).mean(1)
        sigmoid = self.sigmoid(mult)
        return Binomial(1, sigmoid).sample()
    
    def loss_function(self, q_log_likelihood ,log_prior, recon_y, y ):
        KLD = q_log_likelihood - log_prior
        BCE = F.binary_cross_entropy(recon_y, y, reduction ='sum')
        return BCE + KLD
    
    def forward(self, y, X):
        mu = self.encoder(y)
        z = self.reparam(mu)
        
        sigma = self.logsigma.exp()
        prior_sigma =self.prior_log_sigma.exp()
        
        q_log_likelihood = Normal(mu,sigma).log_prob(z).sum(0).mean()
        log_prior =  Normal(self.prior_m, prior_sigma).log_prob(z).sum(0).mean()
        
        recon_y = self.decoder(z, X)
        loss = self.loss_function(q_log_likelihood ,log_prior, recon_y, y)
        return loss
   

model = VAE()
optimizer = optim.Adagrad(model.parameters(), lr=0.01)



# input as Tensor
X = Variable(torch.Tensor(X), requires_grad=False) 
y = Variable(torch.Tensor(y), requires_grad=False)

for i in range(1000):
    loss = model(y, X)
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    if i % 200 ==0:
        print(model.mu.data.numpy())
