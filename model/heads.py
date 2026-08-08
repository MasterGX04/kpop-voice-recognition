import torch
import torch.nn as nn

class MultiMemberBinaryHead(nn.Module):
    """
    Independent MLP pathways for each member.
    Prevents "easy" members from warping the latent space of "hard" members.
    """
    def __init__(self, embDim, numMembers, memberNames, member_configs, default_config):
        super().__init__()
        self.numMembers = numMembers
        self.member_heads = nn.ModuleList()

        for name in memberNames:
            # Grab specific config or fallback to defaults
            conf = member_configs.get(name, default_config)
            
            hidden = conf.get("hidden", default_config["hidden"])
            dropout = conf.get("dropout", default_config["dropout"])

            self.member_heads.append(
                nn.Sequential(
                    nn.BatchNorm1d(embDim),
                    nn.Linear(embDim, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, 1)
                )
            )

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