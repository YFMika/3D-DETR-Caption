import torch
from torch import nn

from utils.misc import seed_all
from models.detector_Vote2Cap_DETRv2.point_diffusion.autoencoder import AutoEncoder

# Model

class Diffusion(nn.Module):
    def __init__(self, ckpt, latent_dim):
        
        super().__init__()
        
        self.ckpt = torch.load(ckpt)
        seed_all(self.ckpt['args'].seed)

        self.ckpt['args'].latent_dim = latent_dim
        
        self.diffusion = AutoEncoder(self.ckpt['args'])

        for param in self.diffusion.parameters():
            param.requires_grad = False

  
    def forward(self, latent_features, npoints=2048):
        """
        Function to decode latent features into point clouds.

        Args:
            latent_features (torch.Tensor): Input features of shape (batch, feature_dim), 

        Returns:
            torch.Tensor: Decoded point clouds of shape (batch, npoints, 3).
        """
        self.diffusion.load_state_dict(self.ckpt['state_dict'])
        self.diffusion.eval()

        # Decode latent features into point clouds
        decoded_point_clouds = self.diffusion.decode(
            code=latent_features, 
            num_points=npoints, 
            flexibility=self.ckpt['args'].flexibility
        )

        return decoded_point_clouds
    