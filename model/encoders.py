import torch
import torch.nn as nn
import torchaudio

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

        emb_main = self._pool_feats(feats)  # (B,D)

        ctx_len = max(1, int(round(Tprime * ctx_frac)))
        mid = Tprime // 2
        half = ctx_len // 2
        start = max(0, mid - half)
        end = min(Tprime, start + ctx_len)
        start = max(0, end - ctx_len)

        emb_ctx = self._pool_feats(feats[:, start:end, :])  # (B,D)
        return emb_main, emb_ctx

class FusedEncoder(nn.Module):
    """
    Produces a single fused embedding per example: (B, D_fused)
    """
    def __init__(self, muq_encoder: MuQEncoderWrapper):
        super().__init__()
        self.muq = muq_encoder

    @torch.no_grad()
    def encode_batch(self, wav24: torch.Tensor, ctx_frac: float = 0.2):
        emb_main, emb_ctx = self.muq.encode_batch(wav24, ctx_frac=ctx_frac)  # (B,D), (B,D)
        emb_fused = torch.cat([emb_main, emb_ctx], dim=1)                    # (B,2D)
        return emb_fused
    
class FusedEncoderWithECAPA(nn.Module):
    def __init__(self, muq_encoder, ecapa_encoder, sr_muq=24000, sr_ecapa=16000):
        super().__init__()
        self.muq = muq_encoder
        self.ecapa = ecapa_encoder
        self.resample = torchaudio.transforms.Resample(sr_muq, sr_ecapa)
        self.sr_muq = sr_muq
        self.sr_ecapa = sr_ecapa

    @torch.no_grad()
    def encode_batch(self, wav24: torch.Tensor, ctx_frac: float = 0.2):
        # MuQ embeddings (B,D)
        emb_main, emb_ctx = self.muq.encode_batch(wav24, ctx_frac=ctx_frac)
        muq_fused = torch.cat([emb_main, emb_ctx], dim=1)  # (B,2D)

        # ECAPA embedding (B, D_ecapa)
        wav16 = self.resample(wav24) # (B,T16)
        ecapa_emb = self.ecapa.encode_batch(wav16) # make it return (B,D)

        # Final fusion
        return torch.cat([muq_fused, ecapa_emb], dim=1) # (B, 2D + D_ecapa)