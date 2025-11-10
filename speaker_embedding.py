import torch, numpy as np, torchcrepe, torchaudio
from pathlib import Path
from typing import List, Tuple
from speechbrain.inference.speaker import EncoderClassifier

# Prefer CUDA if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Change this later to increase sr to improve quality
def load_wav_mono(wav_path: str, target_sr: int = 16000) -> torch.Tensor:
    wav, sr = torchaudio.load(wav_path)
    if wav.shape[0] > 1: # change to mono
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0)

def slice_ms(wav: torch.Tensor, sr: int, start_ms: int, end_ms: int) -> torch.Tensor:
    s = int(start_ms * sr / 1000)
    e = int(end_ms   * sr / 1000)
    s = max(0, min(s, wav.numel()))
    e = max(0, min(e, wav.numel()))
    if e <= s:
        return torch.empty(0)
    return wav[s:e]

def compute_ecapa_embeddings(wav: torch.Tensor, segments_ms: List[Tuple[int, int]], sr: int = 16000) -> np.ndarray:
    """
    Returns (N, D) numpy array of L2-normalized embeddings for each provided segment.
    """
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={'device': str(DEVICE)}
    )
    
    embeddings = []
    for (st, en) in segments_ms:
        segment = slice_ms(wav, sr, st, en)
        if segment.numel() < int(0.5 * sr): # skip too short segments
            continue
        
        with torch.no_grad():
            segment = segment.to(DEVICE)
            embed = classifier.encode_batch(segment.unsqueeze(0))
            embed = embed.squeeze(0).squeeze(0) # [D]
            embed = embed / (embed.norm(p=2) + 1e-9) # L2 Normalize
            embeddings.append(embed.detach().cpu().float().numpy())
            
        if not embeddings:
            raise ValueError("No valid segments produced embeddings; check your segment list or durations.")
        return np.stack(embeddings, axis=0)
    
def aggregate_speaker_embedding(embs: np.ndarray) -> np.ndarray:
    """
    A simple and effective aggregator: mean of L2-normalized vectors, then renormalize.
    """
    mean = embs.mean(axis=0)
    mean = mean / (np.linalg.norm(mean) + 1e-9)
    return mean

def build_singer_embedding(singer_name: str, wav_path: str, segments_ms: List[Tuple[int, int]], save_path: str = "spk_embed.npy") -> str:
    wav = load_wav_mono(wav_path, target_sr=16000)
    save_path = f"{singer_name}_spk_embed.npy"
    embeddings = compute_ecapa_embeddings(wav, segments_ms, sr=16000)
    speaker_embed = aggregate_speaker_embedding(embeddings)
    np.save(save_path, speaker_embed.astype(np.float32))
    
    return save_path
    
def extract_f0_torchcrepe(vocal_wav_path: str, hop_ms: float = 10.0, model='full') -> torch.Tensor:
    wav, sr = torchaudio.load(vocal_wav_path)  # [C, T]
    wav = wav.mean(dim=0, keepdim=True)        # [1, T]
    wav = torchaudio.functional.resample(wav, sr, 16000)
    sr = 16000

    hop_length = int(sr * (hop_ms / 1000.0))
    f0 = torchcrepe.predict(
        wav.to(DEVICE),
        sr,
        hop_length,
        fmin=50.0, fmax=1100.0,
        model=model,
        batch_size=2048,
        device=DEVICE,
        return_periodicity=False
    )  # [1, frames]
    return f0.squeeze(0).cpu()  # [frames], Hz


def extract_content_wav2vec2(vocal_wav_path: str):
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    model = bundle.get_model().to(DEVICE).eval()
    wav, sr = torchaudio.load(vocal_wav_path)
    wav = wav.mean(dim=0, keepdim=True)
    if sr != bundle.sample_rate:
        wav = torchaudio.functional.resample(wav, sr, bundle.sample_rate)
        sr = bundle.sample_rate
    with torch.inference_mode():
        feats, _ = model.extract_features(wav.to(DEVICE))  # list of layers
    return feats[-1].squeeze(0).cpu()  # [T, C] top layer content