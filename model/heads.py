import torch.nn as nn

class PresenceHead(nn.Module):
    """
    Predicts members + silence from fused embeddings.
    No harmony/adlib outputs.
    """
    def __init__(self, emb_dim_fused: int, num_members: int, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.num_members = num_members

        self.trunk = nn.Sequential(
            nn.Linear(emb_dim_fused, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # members + silence
        self.out = nn.Linear(hidden, num_members + 1)

    def forward(self, emb_fused):
        h = self.trunk(emb_fused)
        return self.out(h)  # logits: (B, num_members+1)