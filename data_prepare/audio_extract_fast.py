"""Fast audio extraction helpers: threaded prefetch, noise corpus cache, async saves."""
from __future__ import annotations

import os
import random
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from audio_utils import read_audio

NOISE_LIST = ["Home", "Music", "TV", "Store", "WindAirCon", "WindFan", "babble_noise"]
NOISE_DIR_MAP = {"Home": "GenHome", "Music": "GenMusic"}
NOISE_WEIGHTS = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.70]


def read_audio_f32(wav_path: str) -> np.ndarray:
    """Read mono PCM wav and return float32 in [-1, 1]."""
    return read_audio(wav_path).astype(np.float32) / 32768.0


class DirCache:
    """Avoid repeated os.makedirs calls for the same output directory."""

    __slots__ = ("_seen",)

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def ensure(self, path: str) -> None:
        parent = os.path.dirname(path)
        if not parent or parent in self._seen:
            return
        os.makedirs(parent, exist_ok=True)
        self._seen.add(parent)


class AsyncNpySaver:
    """Write npy files on background threads while the GPU encodes the next batch."""

    def __init__(self, max_workers: int = 2, max_pending: int = 64) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max(1, int(max_workers)))
        self._futures: list[Future] = []
        self._max_pending = max(4, int(max_pending))
        self.dirs = DirCache()

    def save(self, path: str, arr: np.ndarray) -> None:
        self.dirs.ensure(path)
        self._futures.append(self._pool.submit(np.save, path, arr))
        if len(self._futures) >= self._max_pending:
            self.flush()

    def flush(self) -> None:
        for fut in self._futures:
            fut.result()
        self._futures.clear()

    def shutdown(self) -> None:
        self.flush()
        self._pool.shutdown(wait=True)


class AudioPrefetcher:
    """Prefetch wav decode/normalize on CPU threads while the GPU encodes."""

    def __init__(self, max_workers: int = 4) -> None:
        self.max_workers = max(1, int(max_workers))
        self._pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self._futures: Dict[str, Future] = {}

    def schedule(self, key: str, wav_path: str) -> None:
        if key in self._futures:
            return
        self._futures[key] = self._pool.submit(read_audio_f32, wav_path)

    def get(self, key: str) -> np.ndarray:
        if key not in self._futures:
            raise KeyError(key)
        return self._futures.pop(key).result()

    def discard(self, key: str) -> None:
        fut = self._futures.pop(key, None)
        if fut is None:
            return
        if fut.done():
            try:
                fut.result()
            except Exception:
                pass

    def shutdown(self) -> None:
        for fut in self._futures.values():
            if fut.done():
                try:
                    fut.result()
                except Exception:
                    pass
        self._futures.clear()
        self._pool.shutdown(wait=True)


class NoiseCorpusPool:
    """Cache noise corpus listings and pick noise files without repeated os.listdir."""

    def __init__(self, noise_root: str) -> None:
        self.noise_root = noise_root
        self._by_corpus: Dict[str, List[str]] = {}

    def _files_for(self, corpus_name: str) -> List[str]:
        if corpus_name not in self._by_corpus:
            corpus = os.path.join(
                self.noise_root, NOISE_DIR_MAP.get(corpus_name, corpus_name)
            )
            self._by_corpus[corpus_name] = [
                os.path.join(corpus, w)
                for w in os.listdir(corpus)
                if w.endswith(".wav")
            ]
            if not self._by_corpus[corpus_name]:
                raise FileNotFoundError(f"No .wav in {corpus}")
        return self._by_corpus[corpus_name]

    def pick(self, rng: random.Random) -> str:
        corpus_name = rng.choices(NOISE_LIST, weights=NOISE_WEIGHTS, k=1)[0]
        files = self._files_for(corpus_name)
        return files[rng.randrange(len(files))]


def prefetch_ahead(
    prefetcher: Optional[AudioPrefetcher],
    keys: Sequence[str],
    paths: Sequence[str],
    start: int,
    ahead: int,
) -> None:
    if prefetcher is None:
        return
    for j in range(start, min(start + ahead, len(paths))):
        prefetcher.schedule(keys[j], paths[j])
