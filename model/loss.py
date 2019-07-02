import torch
import sys

def loss_function(x, mean, logvar, p_x_given_h, indices):

    KLD = -0.5 * torch.sum((1 + logvar - (mean ** 2) - torch.exp(logvar)),1)
    nll_term = -torch.sum(torch.mul(indices,torch.log(torch.mul(indices,p_x_given_h)+1e-32)),1)
    #avitm
#     nll_term = -torch.sum( x * (p_x_given_h+1e-10).log()+1)
    loss = KLD+nll_term
    # add an L1 penalty to the decoder terms
#     penalty = l1_strength * (torch.sum(torch.abs(parameter)).data[0])
    penalty = 0
    return loss,nll_term, KLD, penalty