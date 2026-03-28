

class MuQEmbeddingCache:
    def __init__(self, cacheDir: str):
        self.cacheDir = cacheDir
        os.makedirs(self.cacheDir, exist_ok=True)

    def _keyToPath(self, key: str) -> str:
        return os.path.join(self.cacheDir, f"{key}.pt")

    def has(self, key: str) -> bool:
        return os.path.exists(self._keyToPath(key))

    def load(self, key: str) -> torch.Tensor:
        return torch.load(self._keyToPath(key), map_location="cpu")

    def save(self, key: str, emb: torch.Tensor):
        torch.save(emb.detach().to("cpu"), self._keyToPath(key))

def makeEmbeddingKey(
    *,
    songId: str,
    centerChunk: int,
    memberName: str,
    srOut: int,
    ctxFrac: float,
    chunkSec: float,
    pcaTag: str = "pca256",
    encoderTag: str = "muq-large-msd-iter",
):
    raw = f"{songId}|{centerChunk}|{memberName}|sr={srOut}|ctx={ctxFrac}|chunk={chunkSec}|{pcaTag}|{encoderTag}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()