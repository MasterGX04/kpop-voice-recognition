import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import numpy as np

# ---------------------------
# Model: MuQ encoder (frozen or trainable) + linear head
# ---------------------------
class MuQEncoderWrapper(nn.Module):
    def __init__(self, muq_model, pooling="mean", debug=False, topk_frac=0.3):
        super().__init__()
        self.muq = muq_model
        self.pooling = pooling
        self.debug = debug
        self.topk_frac = topk_frac
        self._debug_printed = False

        # Optional PCA projection (set later)
        self.pcaK = None           # int

    def setPca(self, mean_np: np.ndarray, W_np: np.ndarray):
        mean = torch.from_numpy(mean_np).float()
        W = torch.from_numpy(W_np).float()

        # If buffers already exist, just overwrite them
        if hasattr(self, "pcaMean"):
            self.pcaMean.data.copy_(mean)
        else:
            self.register_buffer("pcaMean", mean)

        if hasattr(self, "pcaW"):
            self.pcaW.data.copy_(W)
        else:
            self.register_buffer("pcaW", W)

        self.pcaK = W.shape[1]

    def _pool_feats(self, feats):
        if self.pooling == "mean":
            return feats.mean(dim=1)
        if self.pooling == "cls":
            return feats[:, 0, :]
        if self.pooling == "topk":
            scores = feats.norm(p=2, dim=-1)
            T = feats.size(1)
            k = max(1, int(T * self.topk_frac))
            idx = scores.topk(k, dim=1).indices
            idx = idx.unsqueeze(-1).expand(-1, -1, feats.size(-1))
            top_feats = feats.gather(dim=1, index=idx)
            return top_feats.mean(dim=1)
        raise ValueError(self.pooling)

    def _applyPcaIfSet(self, emb: torch.Tensor) -> torch.Tensor:
        """
        emb: (B, D)
        returns: (B, K) if PCA set else (B, D)
        """
        if self.pcaW is None or self.pcaMean is None:
            return emb
        # center then project
        return (emb - self.pcaMean) @ self.pcaW

    @torch.no_grad()
    def encode_batch(self, wavs: torch.Tensor, ctx_frac: float = 0.2):
        if wavs.ndim != 2:
            raise ValueError(f"MuQ expects (B,T), got {wavs.shape}")

        out = self.muq(wavs, output_hidden_states=False)

        feats = getattr(out, "last_hidden_state", None)
        if feats is None and isinstance(out, dict):
            feats = out.get("last_hidden_state", None)
        if feats is None:
            if torch.is_tensor(out):
                feats = out
            else:
                raise RuntimeError("MuQ output has no last_hidden_state")

        # feats: (B, T', D)
        B, Tprime, D = feats.shape

        if not self._debug_printed:
            print(f"[MuQ] wavs: {tuple(wavs.shape)} -> feats: {tuple(feats.shape)} (B,T',D); D={D}")
            self._debug_printed = True

        emb_main = self._pool_feats(feats)  # (B,D)

        ctx_len = max(1, int(round(Tprime * ctx_frac)))
        mid = Tprime // 2
        half = ctx_len // 2
        start = max(0, mid - half)
        end = min(Tprime, start + ctx_len)
        start = max(0, end - ctx_len)

        emb_ctx = self._pool_feats(feats[:, start:end, :])  # (B,D)

        # ✅ Apply PCA projection AFTER pooling (important)
        emb_main = self._applyPcaIfSet(emb_main)
        emb_ctx = self._applyPcaIfSet(emb_ctx)

        return emb_main, emb_ctx

class FusedEncoder(nn.Module):
    """
    Produces a single fused embedding per example.
    Fusion = normalize -> concat -> linear projection -> normalize
    """
    """
    Shared trunk -> normalized hidden embedding -> cosine similarity to per-member prototypes.

    - forward(emb, memberIdx=i) returns (B,) logits for that member
    - forward(emb) returns (B, M) logits for all members
    """
    def __init__(self, embDim: int, numMembers: int, hidden: int = 256, dropout: float = 0.2, scale: float = 30.0):
        super().__init__()
        self.numMembers = numMembers
        self.scale = float(scale)

        self.trunk = nn.Sequential(
            nn.Linear(embDim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Per-member prototypes in hidden space (learnable "reference embeddings")
        self.memberProto = nn.Parameter(torch.empty(numMembers, hidden))
        nn.init.xavier_uniform_(self.memberProto)

        # Optional bias per member (often you can omit this; keep it if you like thresholding flexibility)
        self.memberB = nn.Parameter(torch.zeros(numMembers))

    def forward(self, emb: torch.Tensor, memberIdx: int):
        h = self.trunk(emb)                  # (B, hidden)
        h = F.normalize(h, dim=-1)           # (B, hidden)

        proto = F.normalize(self.memberProto, dim=-1)  # (M, hidden)

        if memberIdx is None:
            # cosine sim for all members: (B, M)
            cos = h @ proto.t()
            logits = self.scale * cos + self.memberB
            return logits

        # cosine sim for one member: (B,)
        cos = (h * proto[memberIdx]).sum(dim=-1)
        logits = self.scale * cos + self.memberB[memberIdx]
        return logits