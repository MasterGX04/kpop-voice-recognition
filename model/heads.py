import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiMemberBinaryHead(nn.Module):
    """
    One shared trunk + per-member binary classifier weights.
    Forward returns (B,) logits for ONE member if memberIdx is given,
    otherwise returns (B, M) logits for all members.
    """
    def __init__(self, embDim: int, numMembers: int, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.numMembers = numMembers

        self.trunk = nn.Sequential(
            nn.Linear(embDim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Each member has its own binary classifier: w_m, b_m
        self.memberW = nn.Parameter(torch.empty(numMembers, hidden))
        self.memberB = nn.Parameter(torch.zeros(numMembers))
        nn.init.xavier_uniform_(self.memberW)

    def forward(self, emb, memberIdx=None):
        """
        emb: (B, embDim)
        memberIdx: int or None
        """
        h = self.trunk(emb)  # (B, hidden)

        if memberIdx is None:
            # (B, M) logits
            return h @ self.memberW.t() + self.memberB

        # (B,) logits for one member
        w = self.memberW[memberIdx] # (hidden,)
        b = self.memberB[memberIdx] # ()
        return (h * w).sum(dim=-1) + b