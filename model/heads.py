import torch
import torch.nn as nn

class MultiMemberBinaryHead(nn.Module):
    """
    Independent MLP pathways for each member.
    Prevents "easy" members from warping the latent space of "hard" members.
    """
    def __init__(self, embDim: int, numMembers: int, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.numMembers = numMembers

        # Create a completely independent network for EACH member.
        # nn.ModuleList registers all the parameters properly.
        self.member_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embDim, hidden),
                nn.GELU(), # GELU is slightly smoother for complex boundaries than ReLU
                nn.Dropout(dropout),
                nn.Linear(hidden, 1) # Final binary projection
            ) for _ in range(numMembers)
        ])

    def forward(self, emb, memberIdx=None):
        """
        emb: (B, embDim)
        memberIdx: int or None
        """
        if memberIdx is not None:
            # Forward pass through ONLY the specified member's private network
            # Output is (B, 1), so we squeeze it to (B,)
            return self.member_heads[memberIdx](emb).squeeze(-1)

        # If no memberIdx is given, run all of them and stack (useful for eval/inference)
        logits = []
        for m in range(self.numMembers):
            logits.append(self.member_heads[m](emb)) # Each is (B, 1)
        
        return torch.cat(logits, dim=-1) # Returns (B, M)