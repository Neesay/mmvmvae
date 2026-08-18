import torch
import torch.nn as nn
import torch.nn.functional as F

class OrthogMat(nn.Module):
    def __init__(self, cfg):
        super(OrthogMat, self).__init__()
      
        # Parametrize the eigenvectors and eigenvalues
        self.latent_dim = cfg.model.latent_dim
        self.eigenvectorsU = nn.Parameter(torch.rand(self.latent_dim,
                                                     self.latent_dim))  # Uniform initialization
        # self.eigenvectorsV = nn.Parameter(torch.rand(self.latent_dim,
        #                                              self.latent_dim))  # Uniform initialization
        self.eigenvalues = nn.Parameter(torch.rand(self.latent_dim))  # Uniform initialization
        self.alpha_scalar = cfg.model.alpha_scalar  # scalar for orthogonalization
        self.register_buffer("eye", torch.eye(self.latent_dim))

    def forward(self, x):
        # Orthogonalize the eigenvectors (using QR decomposition)
        # U: upper triangular part, subtract transpose to create a skew-symmetric matrix
        U = torch.triu(self.eigenvectorsU, diagonal=1)
        A = U - U.T

        # Cayley transform O_U = (I + A)(I - A)^-1. Since A is skew-symmetric,
        # (I - A) and (I + A) commute, so (I + A)(I - A)^-1 == (I - A)^-1(I + A),
        # which lets us use a linear solve instead of an explicit matrix inverse.
        O_U = torch.linalg.solve(self.eye - A, self.eye + A)

        # V: similarly orthogonalize eigenvectorsV
        # V = torch.triu(self.eigenvectorsV, diagonal=1)
        # B = V - V.T
        # O_V = torch.matmul(torch.eye(self.latent_dim).to(x.device) + B, torch.linalg.inv(torch.eye(self.latent_dim).to(x.device) - B))
        
        # Apply the orthogonal matrices to the input x
        # return torch.matmul(O_U, x), torch.sigmoid(self.eigenvalues), O_V
        return self.alpha_scalar * torch.matmul(O_U, x)