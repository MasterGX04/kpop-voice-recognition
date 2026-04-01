from torch.utils.data import Dataset
import torch
import os
import numpy as np

class FastEmbeddingDataset(Dataset):
    def __init__(self, examples, cache_dir: str, sr_out: int, ctx_frac: float, chunk_sec: float):
        self.examples = examples
        self.cache_dir = cache_dir
        self.sr_out = sr_out
        self.ctx_frac = ctx_frac
        self.chunk_sec = chunk_sec
        self.pca_tag = "pca256"
        self.encoder_tag = "muq-large-msd-iter"
        
        # Each CPU worker gets its own memory cache to prevent file locking
        self._worker_cache = {}

    def __len__(self):
        return len(self.examples)
        
    def _get_song_path(self, song_id: str) -> str:
        cfg = f"sr{self.sr_out}_ctx{self.ctx_frac}_chunk{self.chunk_sec}_{self.pca_tag}_{self.encoder_tag}"
        safe_song = song_id.replace(os.sep, "_")
        return os.path.join(self.cache_dir, f"{safe_song}__{cfg}.npy")

    def __getitem__(self, idx):
        ex = self.examples[idx]
        song_id = ex["songId"]
        center = ex["centerChunk"]
        
        # Format labels and weights immediately
        y = torch.tensor([ex["label"]], dtype=torch.float32)
        w = torch.tensor([ex["weight"]], dtype=torch.float32)

        # Load the song matrix into the worker's RAM if not already present
        if song_id not in self._worker_cache:
            path = self._get_song_path(song_id)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Pre-encoded embedding missing: {path}. Run pre-encoding first.")
            
            # Loading without mmap_mode="r" places the entire array in RAM for instant slicing
            self._worker_cache[song_id] = np.load(path)
            
            # Optional: Cap worker memory to ~50 songs to prevent RAM overflow on school VMs
            if len(self._worker_cache) > 50:
                self._worker_cache.pop(next(iter(self._worker_cache)))

        # Slice the specific chunk row and convert to Tensor
        emb_row = self._worker_cache[song_id][center]
        emb = torch.from_numpy(emb_row).float()

        return emb, y, w