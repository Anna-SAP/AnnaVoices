"""Audio pipeline: video -> per-speaker clean speech -> merged MP3.

Stages (see ``analyze``):
  1. extract   ffmpeg pulls the audio track (44.1 kHz float).
  2. separate  Demucs htdemucs keeps the vocal stem (drops BGM / effects).
  3. diarize   pyannote-3.0 segmentation + 3D-Speaker CAM++ (zh-cn) embeddings,
               agglomerative clustering (sherpa-onnx).
  4. refine    overlap removal, Silero VAD trimming, sliding-window voiceprint
               purity check against each speaker centroid.
  5. asr       SenseVoice transcript + audio-event tag per clean segment.
  6. assets    preview clips, video thumbnails, a browsable vocals MP3.

``export`` then concatenates the chosen speaker(s), optionally denoises with
DPDFNet (48 kHz), loudness-normalises and encodes MP3 (+ SRT, + slice zip).
"""
from __future__ import annotations

import difflib
import json
import math
import os
import re
import subprocess
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
import sherpa_onnx as so
import soundfile as sf
from scipy.ndimage import binary_dilation
from scipy.signal import resample_poly

from .models import ensure_models, model_path

SR = 44100  # working / export sample rate
SR16 = 16000  # analysis sample rate
FPS = 100  # timeline resolution for masks (10 ms frames)
HOP16 = SR16 // FPS
NUM_THREADS = max(2, (os.cpu_count() or 4) - 4)

Report = Callable[[str, float, str], None]  # (stage, fraction 0..1, message)

STAGES = [
    ("extract", "提取音轨", 0.03),
    ("separate", "人声分离 (Demucs)", 0.55),
    ("diarize", "说话人日志 (pyannote + CAM++)", 0.20),
    ("refine", "重叠剔除 / VAD / 声纹提纯", 0.08),
    ("asr", "语音识别 (SenseVoice)", 0.08),
    ("assets", "生成试听片段与缩略图", 0.06),
]

DEFAULT_OPTS = {
    "separate": True,  # run Demucs vocal separation
    "num_speakers": 0,  # 0 = auto
    "max_speakers": 10,  # upper bound for the automatic estimate
    "merge_sim": 0.7,  # clusters whose centroids are this similar are one speaker
    "new_speaker_sim": 0.3,  # segments below this similarity to every speaker start a new one
    "change_sim": 0.4,  # voiceprints before/after a point below this similarity -> speaker change
    "min_minor_share": 0.01,  # a minor voice is listed only with >= 1 % of all clean speech
    "min_segment": 1.0,  # seconds; shorter clean pieces are dropped
    "max_segment": 15.0,  # seconds; longer turns are split at pauses
    "min_sim": 0.45,  # frame must be at least this similar to its own centroid
    "min_margin": 0.2,  # ...and this much closer to it than to any other speaker
}


# --------------------------------------------------------------------------- utils


def run_ffmpeg(*args: str, input: bytes | None = None) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
        input=input,
        check=True,
        capture_output=True,
    )


def probe_media(src: Path) -> dict:
    """Duration / stream info from ``ffmpeg -i`` (ffprobe is not always installed)."""
    p = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(src)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    err = p.stderr
    m = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", err)
    duration = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]) if m else 0.0
    has_video = re.search(r"Stream #\S+.*Video: (?!mjpeg|png|bmp)", err) is not None
    has_audio = re.search(r"Stream #\S+.*Audio:", err) is not None
    return {"duration": duration, "has_video": has_video, "has_audio": has_audio}


def encode_mp3(samples: np.ndarray, dst: Path, sr: int = SR, bitrate: str = "320k", af: str | None = None) -> None:
    args = ["-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0"]
    if af:
        args += ["-af", af]
    args += ["-ar", str(sr), "-c:a", "libmp3lame", "-b:a", bitrate, str(dst)]
    run_ffmpeg(*args, input=np.ascontiguousarray(samples, dtype=np.float32).tobytes())


def loudnorm_filter(samples: np.ndarray, sr: int, target: float = -18.0, tp: float = -1.5, lra: float = 11.0) -> str:
    """Two-pass EBU R128: measure first, return a linear (static-gain) loudnorm filter."""
    p = subprocess.run(
        ["ffmpeg", "-hide_banner", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
         "-af", f"loudnorm=I={target}:TP={tp}:LRA={lra}:print_format=json", "-f", "null", "-"],
        input=np.ascontiguousarray(samples, dtype=np.float32).tobytes(),
        capture_output=True,
        check=True,
    )
    stats = json.loads(re.findall(r"\{[^{}]*\}", p.stderr.decode("utf-8", "replace"))[-1])
    return (
        f"loudnorm=I={target}:TP={tp}:LRA={lra}:linear=true"
        f":measured_I={stats['input_i']}:measured_TP={stats['input_tp']}"
        f":measured_LRA={stats['input_lra']}:measured_thresh={stats['input_thresh']}"
        f":offset={stats['target_offset']}"
    )


def to_mask(segs, n: int) -> np.ndarray:
    m = np.zeros(n, bool)
    for s, e in segs:
        m[max(0, int(s * FPS)) : max(0, min(n, int(math.ceil(e * FPS))))] = True
    return m


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def dilate(mask: np.ndarray, frames: int) -> np.ndarray:
    return binary_dilation(mask, iterations=frames) if frames > 0 else mask


def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def fade(y: np.ndarray, sr: int, ms: float = 10.0) -> np.ndarray:
    k = min(len(y) // 2, int(sr * ms / 1000))
    if k > 0:
        r = np.linspace(0.0, 1.0, k, dtype=np.float32)
        y = y.copy()
        y[:k] *= r
        y[-k:] *= r[::-1]
    return y


# ------------------------------------------------------------------ model wrappers


class Embedder:
    """CAM++ speaker embeddings; one extractor per worker thread."""

    def __init__(self, workers: int = 4):
        self.workers = workers
        self._local = threading.local()

    def _ext(self):
        if not hasattr(self._local, "ext"):
            cfg = so.SpeakerEmbeddingExtractorConfig(
                model=str(model_path("embedding")), num_threads=max(1, NUM_THREADS // self.workers)
            )
            self._local.ext = so.SpeakerEmbeddingExtractor(cfg)
        return self._local.ext

    def one(self, x16: np.ndarray) -> np.ndarray:
        ext = self._ext()
        s = ext.create_stream()
        s.accept_waveform(SR16, np.ascontiguousarray(x16, dtype=np.float32))
        s.input_finished()
        return normalize(np.asarray(ext.compute(s), np.float32))

    def many(self, chunks: list[np.ndarray], report: Callable[[float], None] | None = None) -> np.ndarray:
        out = [None] * len(chunks)
        done = 0
        lock = threading.Lock()

        def work(i):
            nonlocal done
            out[i] = self.one(chunks[i])
            with lock:
                done += 1
                if report and done % 50 == 0:
                    report(done / len(chunks))

        with ThreadPoolExecutor(self.workers) as pool:
            list(pool.map(work, range(len(chunks))))
        return np.stack(out) if out else np.zeros((0, 192), np.float32)


def vad_segments(x16: np.ndarray, pad: float = 0.08, report: Callable[[float], None] | None = None) -> list[tuple[float, float]]:
    """Silero VAD speech regions in seconds (padded)."""
    cfg = so.VadModelConfig()
    cfg.silero_vad.model = str(model_path("vad"))
    cfg.silero_vad.threshold = 0.45
    cfg.silero_vad.min_silence_duration = 0.20
    cfg.silero_vad.min_speech_duration = 0.20
    cfg.silero_vad.max_speech_duration = 30.0
    cfg.sample_rate = SR16
    cfg.num_threads = 1
    vad = so.VoiceActivityDetector(cfg, buffer_size_in_seconds=len(x16) / SR16 + 60)
    segs = []

    def drain():
        while not vad.empty():
            f = vad.front
            segs.append((f.start / SR16 - pad, (f.start + len(f.samples)) / SR16 + pad))
            vad.pop()

    # the detector must be fed exactly one model window per call
    step = cfg.silero_vad.window_size
    x16 = np.ascontiguousarray(x16, dtype=np.float32)
    every = step * 2000
    for i in range(0, len(x16), step):
        vad.accept_waveform(x16[i : i + step])
        drain()
        if report and i % every == 0:
            report(i / len(x16))
    vad.flush()
    drain()
    return segs


# ------------------------------------------------------------------------- stages


def separate_vocals(x: np.ndarray, report: Callable[[float], None]) -> np.ndarray:
    """Demucs htdemucs on 60 s blocks (5 s context each side, 0.5 s crossfades)."""
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    torch.set_num_threads(NUM_THREADS)
    model = get_model("htdemucs")
    model.eval()
    vi = model.sources.index("vocals")
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    mono = x.mean(1)
    mean, std = float(mono.mean()), float(mono.std()) or 1.0
    n = len(x)
    block, ctx, h = 60 * SR, 5 * SR, SR // 4
    starts = list(range(0, n, block))
    if len(starts) > 1 and n - starts[-1] < 2 * SR:
        starts.pop()
    out = np.zeros(n, np.float32)
    wsum = np.zeros(n, np.float32)
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else n
        a, b = max(0, start - ctx), min(n, end + ctx)
        chunk = torch.from_numpy(((x[a:b].T - mean) / std).astype(np.float32))[None]
        with torch.no_grad():
            y = apply_model(model, chunk, split=True, overlap=0.25, progress=False, num_workers=0)[0, vi]
        y = y.mean(0).numpy() * std + mean
        ks, ke = max(0, start - h), min(n, end + h)
        w = np.ones(ke - ks, np.float32)
        if start > 0:
            w[: 2 * h] = np.linspace(0, 1, 2 * h, dtype=np.float32)
        if end < n:
            w[-2 * h :] = np.minimum(w[-2 * h :], np.linspace(1, 0, 2 * h, dtype=np.float32))
        out[ks:ke] += y[ks - a : ke - a] * w
        wsum[ks:ke] += w
        report((i + 1) / len(starts))
    return out / np.maximum(wsum, 1e-6)


def diarize(x16: np.ndarray, report: Callable[[float], None]):
    """pyannote-3.0 segmentation + CAM++ via sherpa-onnx.

    The clustering threshold is deliberately tight (over-splits speakers): we
    only rely on this pass for speech activity, turn boundaries and overlap;
    speaker identities are re-derived in ``refine``.
    """
    cfg = so.OfflineSpeakerDiarizationConfig(
        segmentation=so.OfflineSpeakerSegmentationModelConfig(
            pyannote=so.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(model_path("segmentation")), window_shift_ratio=0.2
            ),
            num_threads=NUM_THREADS,
        ),
        embedding=so.SpeakerEmbeddingExtractorConfig(model=str(model_path("embedding")), num_threads=NUM_THREADS),
        clustering=so.FastClusteringConfig(num_clusters=-1, threshold=0.6),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    sd = so.OfflineSpeakerDiarization(cfg)

    def cb(done, total):
        report(done / max(total, 1))
        return 0

    res = sd.process(np.ascontiguousarray(x16, dtype=np.float32), callback=cb).sort_by_start_time()
    return [(float(s.start), float(s.end), int(s.speaker)) for s in res]


def _window_spans(a: int, b: int, win: int, hop: int) -> list[tuple[int, int]]:
    if b - a <= win + hop // 2:
        return [(a, b)]
    nw = int(math.ceil((b - a - win) / hop)) + 1
    return [(int(s), int(s) + win) for s in np.linspace(a, b - win, nw).astype(int)]


def split_on_change(x16: np.ndarray, segs: list[tuple[int, int]], emb: "Embedder", thresh: float, report: Callable[[float], None]) -> list[tuple[int, int]]:
    """Cut segments where the voice changes; pyannote misses quick hand-overs in cross-talk.

    1 s CAM++ windows every 0.25 s; at each candidate point the mean voiceprint of
    up to 1.75 s before it is compared with up to 1.75 s after it (windows never
    straddle the point). Local minima below ``thresh`` become cuts, each piece
    keeping at least 1 s. A needless cut is harmless (both halves get the same
    label later); a missed one would let a second voice leak into a speaker.
    """
    W, H = FPS, FPS // 4
    jobs = [(i, t) for i, (a, b) in enumerate(segs) if b - a >= 2 * W for t in range(a, b - W + 1, H)]
    if not jobs:
        return segs
    E = emb.many([x16[t * HOP16 : (t + W) * HOP16] for _, t in jobs], report)
    seg_of = np.array([i for i, _ in jobs])
    out = []
    for i, (a, b) in enumerate(segs):
        idx = np.flatnonzero(seg_of == i)
        if not len(idx):
            out.append((a, b))
            continue
        Es = E[idx]
        starts = np.array([jobs[j][1] for j in idx])
        sims = np.full(len(idx), 1.0)
        for k in range(4, len(idx)):  # cut at starts[k]: left windows end by then, right ones begin there
            left, right = Es[max(0, k - 7) : k - 3], Es[k : k + 4]
            sims[k] = float(normalize(left.mean(0)) @ normalize(right.mean(0)))
        cuts, last = [], a
        for k in np.argsort(sims):
            c = int(starts[k])
            if sims[k] >= thresh:
                break
            if c - a >= W and b - c >= W and all(abs(c - x) >= W for x in cuts):
                cuts.append(c)
        for c in sorted(cuts):
            out.append((last, c))
            last = c
        out.append((last, b))
    return out


def spectral_cluster(X: np.ndarray, max_k: int = 10, oracle: int | None = None, pval: float = 0.022, max_points: int = 3000) -> np.ndarray:
    """Self-tuning spectral clustering of unit-norm embeddings (as in 3D-Speaker / NeMo).

    Each row of the cosine affinity keeps only its strongest ~2 % links (at least
    6), the number of speakers is read off the largest gap in the Laplacian's
    eigenvalues, and k-means runs on the leading eigenvectors. No absolute
    similarity threshold is involved, so it adapts to recording length, language
    and embedding model.
    """
    from scipy.linalg import eigh
    from sklearn.cluster import KMeans

    n = len(X)
    if n < 3:
        return np.zeros(n, int)
    idx = np.sort(np.random.default_rng(0).choice(n, max_points, replace=False)) if n > max_points else np.arange(n)
    Xs = X[idx]
    m = len(Xs)
    A = Xs @ Xs.T
    keep = min(m - 1, max(6, int(round(pval * m))))
    nn = np.argpartition(-A, keep, axis=1)[:, :keep]
    rows = np.arange(m)[:, None]
    P = np.zeros_like(A)
    P[rows, nn] = A[rows, nn]
    P = np.maximum(0.5 * (P + P.T), 0.0)
    np.fill_diagonal(P, 0.0)
    lam, vec = eigh(np.diag(P.sum(1)) - P)
    k = oracle or int(np.argmax(np.diff(lam[: max_k + 1]))) + 1
    k = max(1, min(k, m))
    lab = KMeans(k, n_init=10, random_state=0).fit_predict(vec[:, :k]) if k > 1 else np.zeros(m, int)
    if m == n:
        return lab
    C = normalize(np.stack([Xs[lab == i].mean(0) for i in range(k)]))
    return (X @ C.T).argmax(1)


def refine(x16: np.ndarray, diar, opts: dict, report: Callable[[float], None]):
    """Raw diarization -> clean single-speaker segments with purity scores.

    1. exclusive speech: diarization activity (nudged 0.25 s into unclaimed
       frames, so word onsets survive) minus overlapped speech (+0.1 s guard),
       intersected with Silero VAD; short in-turn pauses are bridged and long
       turns are split in their longest pause; segments are also cut wherever
       the voice changes (``split_on_change``).
    2. CAM++ embeddings on 1.5 s windows (hop 0.75 s) -> self-tuning spectral
       clustering for the main speakers; near-duplicate clusters are merged.
    3. minor speakers: segments that resemble none of the main speakers are
       grouped into extra speakers instead of being absorbed by the closest one
       (a guest with a few seconds of speech must not leak into the target);
       leftovers join when one is closer to the other than to any main voice.
    4. every frame is labelled with the speaker its covering windows match best
       and kept only if that match is strong (min_sim) and unambiguous
       (min_margin over the runner-up); segments are cut at label changes.
    """
    from sklearn.cluster import AgglomerativeClustering

    n = int(math.ceil(len(x16) / HOP16))
    raw_ids = sorted({k for _, _, k in diar})
    masks = {k: to_mask([(s, e) for s, e, kk in diar if kk == k], n) for k in raw_ids}
    count = np.sum([m.astype(np.int8) for m in masks.values()], axis=0) if masks else np.zeros(n, np.int8)
    overlap = dilate(count >= 2, 10)
    vad = to_mask(vad_segments(x16, pad=0.15, report=lambda f: report(0.2 * f)), n)
    pad = np.zeros(n * HOP16, np.float32)
    pad[: len(x16)] = x16[: n * HOP16]
    energy = 10 * np.log10((pad.reshape(n, HOP16) ** 2).mean(1) + 1e-10)

    # 1) exclusive speech segments
    claimed = count >= 1
    ext = {k: masks[k] | (dilate(masks[k], 25) & ~claimed) for k in raw_ids}
    ext_count = np.sum([m.astype(np.int8) for m in ext.values()], axis=0) if ext else np.zeros(n, np.int8)
    max_f, min_raw = int(opts["max_segment"] * FPS), int(0.4 * FPS)
    turns = []
    for k in raw_ids:
        others = overlap | (ext_count - ext[k].astype(np.int8) > 0)
        cur = None
        for a, b in runs(ext[k] & (ext_count == 1) & ~overlap & vad):
            if cur and a - cur[1] <= 50 and b - cur[0] <= max_f and not others[cur[1] : a].any():
                cur[1] = b
            else:
                if cur:
                    turns.append(cur)
                cur = [a, b]
        if cur:
            turns.append(cur)
    segs = []
    for a, b in sorted(turns):
        while b - a > max_f:
            lo, hi = a + max_f // 2, a + max_f
            gaps = [(gb - ga, ga, gb) for ga, gb in runs(~vad[lo:hi])]
            if gaps:  # cut in the middle of the longest pause
                _, ga, gb = max(gaps)
                cut = lo + (ga + gb) // 2
            else:
                cut = lo + int(np.argmin(energy[lo:hi]))
            segs.append((a, cut))
            a = cut
        if b - a >= min_raw:
            segs.append((a, b))
    report(0.25)
    emb = Embedder()
    segs = split_on_change(x16, segs, emb, float(opts["change_sim"]), lambda f: report(0.25 + 0.25 * f))

    # 2) window embeddings + spectral clustering
    win, hop = int(1.5 * SR16), int(0.75 * SR16)
    windows = [(i, wa, wb) for i, (a, b) in enumerate(segs) for wa, wb in _window_spans(a * HOP16, b * HOP16, win, hop)]
    WE = emb.many([x16[a:b] for _, a, b in windows], lambda f: report(0.5 + 0.35 * f))
    if not len(WE):
        return [], {}, {}, float((count >= 2).sum()) / FPS, 0
    win_seg = np.array([w[0] for w in windows])
    # A requested speaker count only caps the estimate: forcing more clusters
    # would split the dominant voice rather than isolate someone who spoke for a
    # few seconds; missing speakers are filled from the minor voices in analyze().
    oracle = int(opts["num_speakers"]) or None
    lab = spectral_cluster(WE, max_k=int(opts["max_speakers"]))
    if oracle and len(np.unique(lab)) > oracle:
        lab = spectral_cluster(WE, max_k=int(opts["max_speakers"]), oracle=oracle)
    C = normalize(np.stack([WE[lab == i].mean(0) for i in np.unique(lab)]))
    if not oracle:  # merge clusters that are really the same voice
        while len(C) > 1:
            S = C @ C.T
            np.fill_diagonal(S, -1)
            i, j = np.unravel_index(np.argmax(S), S.shape)
            if S[i, j] < opts["merge_sim"]:
                break
            lab = (WE @ C.T).argmax(1)
            lab[lab == j] = i
            C = normalize(np.stack([WE[lab == c].mean(0) for c in np.unique(lab)]))
    for _ in range(2):  # k-means style refinement on the windows
        lab = (WE @ C.T).argmax(1)
        C = normalize(np.stack([WE[lab == c].mean(0) if (lab == c).any() else C[c] for c in range(len(C))]))
    report(0.88)

    # 3) minor speakers: segments unlike every main speaker form their own groups
    n_main = len(C)
    seg_E = normalize(np.stack([WE[win_seg == i].mean(0) for i in range(len(segs))]))
    dur = np.array([(b - a) / FPS for a, b in segs])
    best = (seg_E @ C.T).max(1)
    cand = np.flatnonzero((best < opts["new_speaker_sim"]) & (dur >= 0.8))
    if len(cand):
        if len(cand) > 1:
            # two leftovers are one voice when at least one of them resembles the
            # other more than any main speaker (complete linkage: every pair must).
            # Grouping leftovers never moves audio into a main speaker.
            Sc = seg_E[cand] @ seg_E[cand].T
            D = np.where(Sc > np.minimum.outer(best[cand], best[cand]), 1 - Sc, 3.0)
            np.fill_diagonal(D, 0.0)
            groups = AgglomerativeClustering(
                n_clusters=None, metric="precomputed", linkage="complete", distance_threshold=2.5
            ).fit_predict(D)
        else:
            groups = np.zeros(1, int)
        extra = [
            normalize(WE[np.isin(win_seg, cand[groups == g])].mean(0))
            for g in np.unique(groups)
            if dur[cand[groups == g]].sum() >= opts["min_segment"]
        ]
        if extra:
            C = np.vstack([C, np.stack(extra)])

    # 4) frame-level labels and purity
    S = WE @ C.T
    acc = np.zeros((n, len(C)), np.float32)
    cnt = np.zeros(n, np.float32)
    for w, (_, a, b) in enumerate(windows):
        fa, fb = a // HOP16, b // HOP16
        acc[fa:fb] += S[w]
        cnt[fa:fb] += 1
    has = cnt > 0
    acc[has] /= cnt[has, None]
    lab_f = acc.argmax(1)
    srt = np.sort(acc, axis=1)
    own_f = srt[:, -1]
    mar_f = own_f - (srt[:, -2] if len(C) > 1 else 0.0)
    good = has & (own_f >= opts["min_sim"]) & (mar_f >= opts["min_margin"])

    min_f = int(opts["min_segment"] * FPS)
    clean = []
    talk = np.zeros(len(C))
    for a, b in segs:
        talk += np.bincount(lab_f[a:b][has[a:b]], minlength=len(C)) / FPS
        key = np.where(good[a:b], lab_f[a:b], -1)
        edges = np.flatnonzero(np.diff(key)) + 1
        for ra, rb in zip(np.r_[0, edges], np.r_[edges, b - a]):
            if key[ra] >= 0 and rb - ra >= min_f:
                fa, fb = a + ra, a + rb
                clean.append(
                    {
                        "cluster": int(key[ra]),
                        "start": fa / FPS,
                        "end": fb / FPS,
                        "sim": round(float(own_f[fa:fb].mean()), 3),
                        "margin": round(float(mar_f[fa:fb].mean()), 3),
                    }
                )
    clean.sort(key=lambda s: s["start"])
    report(1.0)
    centroids = {i: C[i] for i in range(len(C))}
    raw_seconds = {i: float(talk[i]) for i in range(len(C))}
    return clean, centroids, raw_seconds, float((count >= 2).sum()) / FPS, n_main


_EVENT_RE = re.compile(r"<\|([^|]+)\|>")


def transcribe(x16: np.ndarray, segments: list[dict], report: Callable[[float], None]) -> None:
    d = model_path("asr").parent
    rec = so.OfflineRecognizer.from_sense_voice(
        model=str(model_path("asr")), tokens=str(d / "tokens.txt"), use_itn=True, num_threads=NUM_THREADS
    )
    batch = 16
    for i in range(0, len(segments), batch):
        group = segments[i : i + batch]
        streams = []
        for seg in group:
            s = rec.create_stream()
            s.accept_waveform(SR16, x16[int(seg["start"] * SR16) : int(seg["end"] * SR16)])
            streams.append(s)
        rec.decode_streams(streams)
        for seg, s in zip(group, streams):
            r = s.result
            seg["text"] = r.text.strip()
            ev = _EVENT_RE.findall(r.event or "")
            em = _EVENT_RE.findall(r.emotion or "")
            seg["event"] = ev[0] if ev else (r.event or "Speech")
            seg["emotion"] = em[0] if em else (r.emotion or "")
        report(min(1.0, (i + batch) / max(len(segments), 1)))


# ----------------------------------------------------------------------- analyze


def _stage_reporter(report: Report, active: list[str]):
    weights = {k: w for k, _, w in STAGES if k in active}
    total = sum(weights.values())
    offsets, acc = {}, 0.0
    for k, _, _ in STAGES:
        if k in active:
            offsets[k] = acc
            acc += weights[k] / total

    def make(stage: str, label: str):
        def r(frac: float, msg: str = ""):
            report(stage, offsets[stage] + weights[stage] / total * max(0.0, min(1.0, frac)), msg or label)

        return r

    return make


def analyze(src: Path, job_dir: Path, opts: dict | None, report: Report) -> dict:
    opts = {**DEFAULT_OPTS, **(opts or {})}
    ensure_models()
    job_dir.mkdir(parents=True, exist_ok=True)
    info = probe_media(src)
    if not info["has_audio"]:
        raise ValueError("文件中没有音轨")
    active = [k for k, _, _ in STAGES if k != "separate" or opts["separate"]]
    stage = _stage_reporter(report, active)
    labels = {k: lab for k, lab, _ in STAGES}

    # 1+2. extract / separate (cached: re-analysis with other settings skips this)
    vocals_path = job_dir / "vocals_44k.wav"
    flag = job_dir / ("vocals.demucs" if opts["separate"] else "vocals.mix")
    if vocals_path.exists() and flag.exists():
        vocals, _ = sf.read(vocals_path, dtype="float32")
        stage("extract", "复用已分离的人声")(1.0)
        if opts["separate"]:
            stage("separate", "复用已分离的人声")(1.0)
    else:
        r = stage("extract", labels["extract"])
        r(0.0)
        wav = job_dir / "audio_44k.wav"
        run_ffmpeg("-i", str(src), "-vn", "-map", "0:a:0", "-ac", "2", "-ar", str(SR), "-c:a", "pcm_f32le", str(wav))
        x, _ = sf.read(wav, dtype="float32", always_2d=True)
        r(1.0)
        if opts["separate"]:
            r = stage("separate", labels["separate"])
            vocals = separate_vocals(x, lambda f: r(f, f"{labels['separate']} {f:.0%}"))
        else:
            vocals = x.mean(1)
        del x
        wav.unlink(missing_ok=True)
        sf.write(vocals_path, vocals, SR, subtype="FLOAT")
        for old in job_dir.glob("vocals.*"):
            if old.suffix in (".demucs", ".mix"):
                old.unlink()
        flag.touch()
    x16 = resample_poly(vocals, 160, 441).astype(np.float32)

    # 3. diarize
    r = stage("diarize", labels["diarize"])
    diar_path = job_dir / "diarization.json"
    if diar_path.exists() and diar_path.stat().st_mtime >= vocals_path.stat().st_mtime:
        diar = json.loads(diar_path.read_text(encoding="utf-8"))
        r(1.0, "复用已有的说话人日志")
    else:
        diar = diarize(x16, lambda f: r(f, f"{labels['diarize']} {f:.0%}"))
        diar_path.write_text(json.dumps(diar), encoding="utf-8")

    # 4. refine
    r = stage("refine", labels["refine"])
    clean, centroids, raw_seconds, overlap_seconds, n_main = refine(x16, diar, opts, r)

    # Main speakers are always listed. Minor voices found among the leftovers are
    # listed only if they hold a real share of the speech; smaller groups are
    # laughter / cross-talk fragments. Either way they stayed in the purity check
    # as competitors, so none of that audio leaked into a listed speaker.
    totals = {}
    for s in clean:
        totals[s["cluster"]] = totals.get(s["cluster"], 0.0) + s["end"] - s["start"]
    minors = sorted((k for k in totals if k >= n_main), key=lambda k: -totals[k])
    if int(opts["num_speakers"]) > 0:  # user knows the count: top up with the largest minor voices
        listed = {k for k in totals if k < n_main} | set(minors[: max(0, int(opts["num_speakers"]) - n_main)])
    else:
        min_minor = max(opts["min_segment"], opts["min_minor_share"] * sum(totals.values()))
        listed = {k for k in totals if k < n_main} | {k for k in minors if totals[k] >= min_minor}
    fragments = [s for s in clean if s["cluster"] not in listed]
    clean = [s for s in clean if s["cluster"] in listed]
    if not clean:
        raise ValueError("没有检测到可用的语音，请确认文件里有清晰的人声")

    # 5. asr
    r = stage("asr", labels["asr"])
    transcribe(x16, clean, r)

    # name speakers by amount of clean speech (S1 = most)
    order = sorted(listed, key=lambda k: -totals[k])
    name = {k: f"S{i + 1}" for i, k in enumerate(order)}
    for i, s in enumerate(clean):
        s["id"] = i
        s["spk"] = name[s.pop("cluster")]
    np.save(job_dir / "centroids.npy", np.stack([centroids[k] for k in order]))
    C = np.stack([centroids[k] for k in order])
    sim = C @ C.T

    speakers = []
    for i, k in enumerate(order):
        mine = [s for s in clean if s["spk"] == name[k]]
        speakers.append(
            {
                "id": name[k],
                "label": f"说话人 {i + 1}",
                "raw_seconds": round(raw_seconds.get(k, 0.0), 1),
                "clean_seconds": round(totals.get(k, 0.0), 1),
                "num_segments": len(mine),
                "similar": [
                    {"id": name[order[j]], "sim": round(float(sim[i, j]), 3)}
                    for j in np.argsort(-sim[i])
                    if j != i and sim[i, j] > 0.6
                ],
            }
        )

    # 6. assets
    r = stage("assets", labels["assets"])
    make_assets(src, job_dir, vocals, info, speakers, clean, r)

    result = {
        "source": src.name,
        "duration": round(len(vocals) / SR, 2),
        "has_video": info["has_video"],
        "options": opts,
        "overlap_seconds": round(overlap_seconds, 1),
        "fragments": {
            "groups": len({s["cluster"] for s in fragments}),
            "seconds": round(sum(s["end"] - s["start"] for s in fragments), 1),
        },
        "speakers": speakers,
        "segments": clean,
    }
    (job_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


def pick_preview_segments(segs: list[dict], target: float = 20.0) -> list[dict]:
    """Highest-purity 3-10 s speech segments, spread across the recording."""
    cand = [s for s in segs if 3.0 <= s["end"] - s["start"] <= 10.0 and s.get("event", "Speech") == "Speech" and s.get("text")]
    if not cand:
        cand = sorted(segs, key=lambda s: -(s["end"] - s["start"]))[:3]
    cand = sorted(cand, key=lambda s: -(s["sim"] + s["margin"]))[: max(3, len(cand) // 4)]
    picked, total = [], 0.0
    for s in sorted(cand, key=lambda s: s["start"])[:: max(1, len(cand) // 6)]:
        picked.append(s)
        total += s["end"] - s["start"]
        if total >= target:
            break
    return picked


def make_assets(src: Path, job_dir: Path, vocals: np.ndarray, info: dict, speakers, clean, r) -> None:
    prev_dir, thumb_dir = job_dir / "previews", job_dir / "thumbs"
    prev_dir.mkdir(exist_ok=True)
    thumb_dir.mkdir(exist_ok=True)
    gap = np.zeros(int(0.35 * SR), np.float32)
    steps = len(speakers) + 1
    for i, spk in enumerate(speakers):
        mine = [s for s in clean if s["spk"] == spk["id"]]
        picks = pick_preview_segments(mine)
        spk["preview_segments"] = [s["id"] for s in picks]
        if picks:
            parts = []
            for s in picks:
                parts += [fade(vocals[int(s["start"] * SR) : int(s["end"] * SR)], SR), gap]
            encode_mp3(np.concatenate(parts), prev_dir / f"{spk['id']}.mp3", bitrate="128k")
        spk["thumbs"] = []
        if info["has_video"]:
            longest = sorted(mine, key=lambda s: -(s["end"] - s["start"]))[:12]
            longest = sorted(longest, key=lambda s: s["start"])[:: max(1, len(longest) // 4)][:4]
            for j, s in enumerate(longest):
                t = (s["start"] + s["end"]) / 2
                out = thumb_dir / f"{spk['id']}_{j}.jpg"
                try:
                    run_ffmpeg("-ss", f"{t:.2f}", "-i", str(src), "-frames:v", "1", "-q:v", "3", str(out))
                    spk["thumbs"].append({"file": out.name, "time": round(t, 2)})
                except subprocess.CalledProcessError:
                    pass
        r((i + 1) / steps)
    # whole vocal track, small, for click-to-play of any segment in the browser
    encode_mp3(vocals, job_dir / "vocals_browse.mp3", bitrate="64k")
    r(1.0)


# ------------------------------------------------------------------------ matching


def ref_audio_embedding(path: Path) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", str(SR16), "-f", "f32le", "pipe:1"],
        capture_output=True,
        check=True,
    ).stdout
    x16 = np.frombuffer(raw, np.float32)
    speech = [x16[int(max(0, a) * SR16) : int(b * SR16)] for a, b in vad_segments(x16)]
    x = np.concatenate(speech) if speech else x16
    if len(x) < SR16:
        raise ValueError("参考音频中有效语音不足 1 秒")
    win, hop = 3 * SR16, int(1.5 * SR16)
    chunks = [x[i : i + win] for i in range(0, max(1, len(x) - win + 1), hop)] or [x]
    return normalize(Embedder(workers=2).many(chunks).mean(0))


def match_audio(job_dir: Path, ref: Path) -> list[dict]:
    result = json.loads((job_dir / "result.json").read_text(encoding="utf-8"))
    C = np.load(job_dir / "centroids.npy")
    sims = C @ ref_audio_embedding(ref)
    return sorted(
        [{"id": s["id"], "score": round(float(v), 3)} for s, v in zip(result["speakers"], sims)],
        key=lambda d: -d["score"],
    )


_PUNCT = re.compile(r"[\s\W_]+", re.UNICODE)
_CN_NUM = {c: i for i, c in enumerate("零一二三四五六七八九")} | {"〇": 0, "两": 2}
_CN_UNIT = {"十": 10, "百": 100, "千": 1000, "万": 10000}
_CN_RUN = re.compile("[零〇一二两三四五六七八九十百千万]+")


def _cn_number(s: str) -> str:
    """十四 -> 14, 五一 -> 51, 一百零五 -> 105 (ASR writes numbers either way)."""
    if not any(c in _CN_UNIT for c in s):
        return "".join(str(_CN_NUM[c]) for c in s)
    total = section = num = 0
    for c in s:
        if c in _CN_NUM:
            num = _CN_NUM[c]
        elif c == "万":
            total += (section + num) * 10000
            section = num = 0
        else:
            section += (num or 1) * _CN_UNIT[c]
            num = 0
    return str(total + section + num)


def _norm_text(t: str) -> str:
    return _CN_RUN.sub(lambda m: _cn_number(m[0]), _PUNCT.sub("", t).lower())


def search_text(result: dict, query: str, limit: int = 20) -> dict:
    q = _norm_text(query)
    if not q:
        return {"hits": [], "votes": []}
    hits = []
    for s in result["segments"]:
        t = _norm_text(s.get("text", ""))
        if not t:
            continue
        if q in t:
            score = 1.0
        else:
            m = difflib.SequenceMatcher(None, q, t, autojunk=False)
            score = sum(b.size for b in m.get_matching_blocks()) / len(q)
        if score >= 0.6:
            hits.append({"segment": s["id"], "spk": s["spk"], "start": s["start"], "end": s["end"], "text": s["text"], "score": round(score, 3)})
    hits.sort(key=lambda h: (-h["score"], h["start"]))
    votes: dict[str, float] = {}
    for h in hits:
        votes[h["spk"]] = votes.get(h["spk"], 0.0) + h["score"]
    return {
        "hits": hits[:limit],
        "votes": sorted([{"id": k, "score": round(v, 3)} for k, v in votes.items()], key=lambda d: -d["score"]),
    }


_face_lock = threading.Lock()


def _face_models():
    import cv2

    ensure_models(["face_det", "face_rec"])
    det = cv2.FaceDetectorYN.create(str(model_path("face_det")), "", (320, 320), 0.6, 0.3, 5000)
    rec = cv2.FaceRecognizerSF.create(str(model_path("face_rec")), "")
    return det, rec


def _face_features(img, det, rec, upscale_to: int = 720) -> list[np.ndarray]:
    import cv2

    h, w = img.shape[:2]
    if max(h, w) < upscale_to:  # tiny video frames: upscale so YuNet finds the faces
        f = upscale_to / max(h, w)
        img = cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_CUBIC)
    det.setInputSize((img.shape[1], img.shape[0]))
    _, faces = det.detect(img)
    feats = []
    for face in faces if faces is not None else []:
        aligned = rec.alignCrop(img, face)
        feats.append((float(face[2] * face[3]), normalize(rec.feature(aligned).flatten())))
    return [f for _, f in sorted(feats, key=lambda t: -t[0])]


def _grab_frame(src: Path, t: float):
    import cv2

    p = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", str(src), "-frames:v", "1", "-f", "image2pipe", "-c:v", "png", "pipe:1"],
        capture_output=True,
    )
    if p.returncode or not p.stdout:
        return None
    return cv2.imdecode(np.frombuffer(p.stdout, np.uint8), cv2.IMREAD_COLOR)


def match_image(job_dir: Path, src: Path, ref: Path, frames_per_speaker: int = 16) -> list[dict]:
    """Which speaker's speaking turns show the face in the reference photo?

    For each speaker we sample frames in the middle of their longest clean
    segments (the camera usually shows whoever is talking) and compare every
    detected face with the reference face (SFace cosine similarity).
    """
    import cv2

    result = json.loads((job_dir / "result.json").read_text(encoding="utf-8"))
    if not result.get("has_video"):
        raise ValueError("该文件没有视频画面，无法按照片匹配")
    with _face_lock:
        det, rec = _face_models()
        img = cv2.imdecode(np.fromfile(str(ref), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("无法读取图片")
        ref_feats = _face_features(img, det, rec)
        if not ref_feats:
            raise ValueError("参考图片中没有检测到人脸")
        target = ref_feats[0]  # largest face in the photo
        out = []
        for spk in result["speakers"]:
            mine = sorted((s for s in result["segments"] if s["spk"] == spk["id"]), key=lambda s: -(s["end"] - s["start"]))
            mine = mine[: frames_per_speaker * 3][::3] or mine[:frames_per_speaker]
            scores = []
            for s in mine[:frames_per_speaker]:
                frame = _grab_frame(src, (s["start"] + s["end"]) / 2)
                if frame is None:
                    continue
                feats = _face_features(frame, det, rec)
                scores.append(max((float(f @ target) for f in feats), default=0.0))
            scores = np.array(scores) if scores else np.zeros(1)
            out.append(
                {
                    "id": spk["id"],
                    "score": round(float(scores.mean()), 3),
                    "match_rate": round(float((scores >= 0.363).mean()), 3),  # SFace cosine threshold
                    "frames": int(len(scores)),
                }
            )
    return sorted(out, key=lambda d: -d["score"])


# -------------------------------------------------------------------------- export


def _denoise_segments(pieces: list[np.ndarray], atten_db: float, report: Callable[[float], None]) -> list[np.ndarray]:
    """DPDFNet 48 kHz on each segment, several denoisers in parallel."""
    workers = 4
    local = threading.local()
    done = 0
    lock = threading.Lock()

    def get():
        if not hasattr(local, "dn"):
            cfg = so.OfflineSpeechDenoiserConfig(
                model=so.OfflineSpeechDenoiserModelConfig(
                    dpdfnet=so.OfflineSpeechDenoiserDpdfNetModelConfig(model=str(model_path("denoise")), attenuation_limit_db=atten_db),
                    num_threads=max(1, NUM_THREADS // workers),
                )
            )
            local.dn = so.OfflineSpeechDenoiser(cfg)
        return local.dn

    def work(y):
        nonlocal done
        y48 = resample_poly(y, 160, 147).astype(np.float32)
        out = np.asarray(get()(y48, 48000).samples, np.float32)
        out = resample_poly(out, 147, 160).astype(np.float32)[: len(y)]
        if len(out) < len(y):
            out = np.pad(out, (0, len(y) - len(out)))
        with lock:
            done += 1
            report(done / len(pieces))
        return out

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(work, pieces))


def _srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def select_segments(result: dict, speakers: list[str], opts: dict) -> list[dict]:
    exclude = set(opts.get("exclude", []))
    keep_events = {"Speech"} if opts.get("skip_events", True) else None
    out = []
    for s in result["segments"]:
        if s["spk"] not in speakers or s["id"] in exclude:
            continue
        if s["sim"] < opts.get("min_sim", 0.0) or s["margin"] < opts.get("min_margin", -1.0):
            continue
        if s["end"] - s["start"] < opts.get("min_segment", 0.0):
            continue
        if keep_events is not None and s.get("event", "Speech") not in keep_events:
            continue
        if opts.get("skip_empty_text", True) and not s.get("text"):
            continue
        out.append(s)
    return sorted(out, key=lambda s: s["start"])


def export(job_dir: Path, speakers: list[str], opts: dict, report: Callable[[float, str], None], name: str | None = None) -> dict:
    """Merge the chosen speakers' clean segments into one MP3 (+ SRT, optional slice zip)."""
    result = json.loads((job_dir / "result.json").read_text(encoding="utf-8"))
    segs = select_segments(result, speakers, opts)
    if not segs:
        raise ValueError("没有符合条件的片段")
    out_dir = job_dir / "export"
    out_dir.mkdir(exist_ok=True)
    stem = name or "_".join(speakers)
    report(0.02, "读取片段")
    pad = 0.05
    with sf.SoundFile(job_dir / "vocals_44k.wav") as f:
        pieces = []
        for s in segs:
            a = max(0, int((s["start"] - pad) * SR))
            f.seek(a)
            pieces.append(f.read(int((s["end"] + pad) * SR) - a, dtype="float32"))
    if opts.get("denoise", True):
        pieces = _denoise_segments(pieces, float(opts.get("denoise_db", 15.0)), lambda fr: report(0.05 + 0.75 * fr, f"DPDFNet 降噪 {fr:.0%}"))
    pieces = [fade(p, SR) for p in pieces]
    gap = np.zeros(int(float(opts.get("gap", 0.5)) * SR), np.float32)
    parts, srt, t = [], [], 0.0
    for i, (s, p) in enumerate(zip(segs, pieces)):
        if i:
            parts.append(gap)
            t += len(gap) / SR
        parts.append(p)
        srt.append(f"{i + 1}\n{_srt_time(t)} --> {_srt_time(t + len(p) / SR)}\n{s.get('text', '')}\n")
        t += len(p) / SR
    audio = np.concatenate(parts)
    report(0.82, "响度标准化 (EBU R128)")
    af = loudnorm_filter(audio, SR, target=float(opts.get("lufs", -18.0))) if opts.get("loudnorm", True) else None
    report(0.88, "编码 MP3")
    mp3 = out_dir / f"{stem}.mp3"
    encode_mp3(audio, mp3, bitrate=opts.get("bitrate", "320k"), af=af)
    (out_dir / f"{stem}.srt").write_text("\n".join(srt), encoding="utf-8")
    files = {"mp3": mp3.name, "srt": f"{stem}.srt"}
    if opts.get("slices", False):
        report(0.94, "打包切片")
        zpath = out_dir / f"{stem}_slices.zip"
        gain = 1.0
        if af:  # apply the same static gain to the slices
            m = re.search(r"measured_I=(-?[\d.]+)", af)
            gain = 10 ** ((float(opts.get("lufs", -18.0)) - float(m[1])) / 20) if m else 1.0
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            lines = []
            for i, (s, p) in enumerate(zip(segs, pieces)):
                fn = f"{stem}_{i:04d}.wav"
                buf = _wav_bytes(np.clip(p * gain, -1, 1))
                z.writestr(f"wavs/{fn}", buf)
                lines.append(f"wavs/{fn}|{stem}|ZH|{s.get('text', '')}")
            z.writestr("list.txt", "\n".join(lines) + "\n")
        files["zip"] = zpath.name
    meta = {
        "speakers": speakers,
        "segments": len(segs),
        "speech_seconds": round(sum(len(p) for p in pieces) / SR, 1),
        "file_seconds": round(len(audio) / SR, 1),
        "files": files,
        "options": opts,
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    report(1.0, "完成")
    return meta


def _wav_bytes(y: np.ndarray) -> bytes:
    import io

    buf = io.BytesIO()
    sf.write(buf, y, SR, subtype="PCM_16", format="WAV")
    return buf.getvalue()
