import torch
import torch.nn as nn

class GroupVocalClassifier(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, num_members=4, dropout=0.3):
        """
        A highly generalizable, lightweight Multi-Label Vocal Classifier.
        
        Args:
            input_dim (int): The dimension of your raw MuQ embeddings (default 1024).
            hidden_dim (int): The capacity of the shared vocal extraction backbone.
            num_members (int): The number of output switches (e.g., 4 for aespa, 6 for IVE).
            dropout (float): Probability of neural dropout to prevent memorization.
        """
        super().init()
        pooled_dim = input_dim * 2
        
        # We concatenate Mean + Std pooled features, doubling the input size.
        pooled_dim = input_dim * 2 
        
        # Shared Feature Extraction Backbone
        self.backbone = nn.Sequential(
            nn.BatchNorm1d(pooled_dim),
            nn.Linear(pooled_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Final classification layer
        # Maps the 256-dim bottleneck representation directly to binary logits for each member.
        self.classifier = nn.Linear(hidden_dim // 2, num_members)

    def forward(self, x):
        """
        Forward pass.
        x expected shape: (Batch, TimeSteps, FeatureDim) e.g., (B, 25, 1024).
        """
        # 1. Temporal Pooling over the frame/time dimension (dim=1)
        mean_pooled = x.mean(dim=1)              # Shape: (Batch, FeatureDim) 
        std_pooled = x.std(dim=1, unbiased=False) # Shape: (Batch, FeatureDim)
        
        # Concatenate mean and std to build the timbral fingerprint 
        fingerprint = torch.cat([mean_pooled, std_pooled], dim=-1) # Shape: (Batch, FeatureDim * 2)
        
        # 2. Extract deep, compressed representations
        features = self.backbone(fingerprint) # Shape: (Batch, hidden_dim // 2)
        
        # 3. Predict raw logits for each member's independent Sigmoid switch
        logits = self.classifier(features) # Shape: (Batch, num_members)
        
        return logits