import os
import torch
import torchaudio
from collections import OrderedDict

class AudioHandler:
    """
    Handles all low-level audio operations: loading, resampling, 
    LRU caching, and window slicing.
    """
    def __init__(self, sr_out: int, max_cached_songs: int = 64):
        self.sr_out = sr_out
        self.max_cached_songs = max_cached_songs
        
        # LRU cache: path -> waveform (1, T_out)
        self._wave_cache = OrderedDict()
        # Resampler cache: (sr_src, sr_out) -> Resample object
        self._resamplers = {}

    def get_resampler(self, sr_src: int):
        key = (sr_src, self.sr_out)
        if key not in self._resamplers:
            self._resamplers[key] = torchaudio.transforms.Resample(sr_src, self.sr_out)
        return self._resamplers[key]

    def load_song_wave(self, path: str) -> torch.Tensor:
        """
        Load + mono + resample full song ONCE and cache it (LRU).
        Returns: Tensor shape (1, T_out) at self.sr_out
        """
        # ---- Already cached? Refresh LRU and return ----
        if path in self._wave_cache:
            wav = self._wave_cache.pop(path)
            self._wave_cache[path] = wav  # move to end (most recently used)
            return wav

        # ---- Load from disk (slow path) ----
        wav, sr_src = torchaudio.load(path)

        # Convert to mono
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Resample once
        if sr_src != self.sr_out:
            resampler = self.get_resampler(sr_src)
            wav = resampler(wav)

        wav = wav.contiguous()

        # ---- Store in LRU ----
        self._wave_cache[path] = wav

        # Evict oldest if over limit
        if len(self._wave_cache) > self.max_cached_songs:
            self._wave_cache.popitem(last=False)

        return wav

    def load_window(self, path: str, center_c: int, chunk_ms: int, context_sec: float, samples_per_window: int):
        """
        Slice a fixed-length window centered at chunk index center_c.
        Uses cached full-song waveform.
        """
        # Convert center chunk -> time
        center_ms = (center_c + 0.5) * chunk_ms
        half_ms = (context_sec * 1000.0) / 2.0

        start_ms = max(0.0, center_ms - half_ms)
        start_sample = int(round((start_ms / 1000.0) * self.sr_out))
        
        # ---- Load full waveform from cache ----
        wav = self.load_song_wave(path)  # (1, T)

        end_sample = start_sample + samples_per_window

        # ---- Slice ----
        if end_sample > wav.size(1):
            pad = end_sample - wav.size(1)
            chunk = torch.nn.functional.pad(
                wav[:, start_sample:], (0, pad)
            )
        else:
            chunk = wav[:, start_sample:end_sample]

        return chunk