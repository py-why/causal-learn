import torch

def sample_logistic(shape, out=None):
    U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
    return torch.log(U) - torch.log(1-U)


def gumbel_sigmoid(logits, tau=1):
    dims = logits.dim()
    logistic_noise = sample_logistic(logits.size(), out=logits.data.new())
    y = logits + logistic_noise
    return torch.sigmoid(y / tau)

def matrix_poly(matrix, d, device):
    x = torch.eye(d, device=device, dtype=matrix.dtype)+ torch.div(matrix, d)
    return torch.matrix_power(x, d)