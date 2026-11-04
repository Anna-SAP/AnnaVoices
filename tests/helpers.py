"""Small builders shared by the test modules."""
from __future__ import annotations

import numpy as np

from voiceextract import pipeline


def seg(id=0, spk="S1", start=0.0, end=5.0, text="你好", sim=0.8, margin=0.4, **extra) -> dict:
    """One clean segment, shaped like those in result.json."""
    return {"id": id, "spk": spk, "start": start, "end": end, "text": text, "sim": sim, "margin": margin, **extra}


class FakeEmbedder:
    """Stands in for CAM++: ``x16`` holds a voice index per sample.

    A window's voiceprint is the share of each voice in it, so two voices are
    orthogonal and a window straddling a change is a blend, like the real model.
    """

    VOICES = 4

    def many(self, chunks, report=None):
        if not chunks:
            return np.zeros((0, self.VOICES), np.float32)
        shares = [np.bincount(np.asarray(c).astype(int), minlength=self.VOICES) for c in chunks]
        return pipeline.normalize(np.array(shares, np.float32))


def voices(*spans: tuple[int, float]) -> np.ndarray:
    """16 kHz signal from (voice index, seconds) spans, for ``FakeEmbedder``."""
    return np.concatenate([np.full(int(sec * pipeline.SR16), v, np.float32) for v, sec in spans])


def blobs(*groups: int, dim: int = 8, noise: float = 0.03) -> tuple[np.ndarray, np.ndarray]:
    """Unit-norm points around orthogonal centres, with their true group."""
    rng = np.random.default_rng(1)
    X = np.vstack([np.eye(dim)[i] + rng.normal(0, noise, (n, dim)) for i, n in enumerate(groups)])
    truth = np.repeat(np.arange(len(groups)), groups)
    return pipeline.normalize(X), truth
