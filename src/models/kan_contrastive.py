import torch
import torch.nn as nn
import torch.nn.functional as F
from .efficient_kan import KAN

class KANEncoder2D(nn.Module):
    """
    Encoder that processes 2D Wavelet Scalograms using a combination of 
    CNN for feature extraction and KAN for high-level representation.
    """
    def __init__(
        self, 
        image_size: int = 64, 
        hidden_dim: int = 128, 
        projection_dim: int = 64
    ):
        super().__init__()
        
        # CNN Backbone to reduce 2D image to a feature vector
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), # (16, image_size/2, image_size/2)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # (32, image_size/4, image_size/4)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), # (32, 4, 4)
            nn.Flatten() # 32 * 4 * 4 = 512
        )
        
        # KAN for non-linear feature mapping
        # Input: 512, Output: hidden_dim
        self.kan = KAN(layers_hidden=[512, hidden_dim], grid_size=5, spline_order=3)
        
        # Projection head for contrastive learning
        self.projection = nn.Linear(hidden_dim, projection_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, 1, image_size, image_size)
        Returns:
            Projected embeddings of shape (batch, projection_dim)
        """
        # Ensure input has channel dimension
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        features = self.cnn(x)
        embeddings = self.kan(features)
        projected = self.projection(embeddings)
        
        return projected

class KANEncoder(nn.Module):
    """
    Encoder that processes 1D time series directly using KAN.
    """
    def __init__(
        self, 
        input_dim: int = 384, 
        hidden_dim: int = 128, 
        projection_dim: int = 64
    ):
        super().__init__()
        # Use KAN directly on the 1D input
        self.kan = KAN(layers_hidden=[input_dim, hidden_dim], grid_size=5, spline_order=3)
        self.projection = nn.Linear(hidden_dim, projection_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, input_dim)
        Returns:
            Projected embeddings of shape (batch, projection_dim)
        """
        # Ensure input is 2D (batch, input_dim)
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
            
        embeddings = self.kan(x)
        projected = self.projection(embeddings)
        
        return projected

class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss for contrastive learning.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, query: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: Embeddings from view 1 (batch, dim)
            positive: Embeddings from view 2 (batch, dim)
        """
        batch_size = query.shape[0]
        
        # Normalize embeddings
        query = F.normalize(query, dim=1)
        positive = F.normalize(positive, dim=1)
        
        # Compute similarity matrix
        logits = torch.matmul(query, positive.T) / self.temperature
        
        # Labels are the diagonal (positive pairs)
        labels = torch.arange(batch_size, device=query.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss
