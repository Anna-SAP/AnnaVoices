"""Model registry and downloader.

All models are ONNX files published on the sherpa-onnx GitHub releases page,
so nothing here needs an account or access token. Demucs weights are fetched
by demucs itself on first use (into the torch hub cache).
"""
from __future__ import annotations

import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"

_GH = "https://github.com/k2-fsa/sherpa-onnx/releases/download"

# name -> (url, path inside MODEL_DIR that must exist afterwards)
MODELS = {
    "segmentation": (
        f"{_GH}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
        "sherpa-onnx-pyannote-segmentation-3-0/model.onnx",
    ),
    # CAM++ trained on Chinese + English: separates non-Mandarin speakers far better
    # than the zh-cn-only model, with identical results on Mandarin.
    "embedding": (
        f"{_GH}/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
        "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
    ),
    "vad": (
        f"{_GH}/asr-models/silero_vad_v5.onnx",
        "silero_vad_v5.onnx",
    ),
    "asr": (
        f"{_GH}/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09.tar.bz2",
        "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/model.int8.onnx",
    ),
    "denoise": (
        f"{_GH}/speech-enhancement-models/dpdfnet2_48khz_hr.onnx",
        "dpdfnet2_48khz_hr.onnx",
    ),
}

# Only needed for "match by reference photo"; fetched on first use.
_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
FACE_MODELS = {
    "face_det": (
        f"{_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "face_detection_yunet_2023mar.onnx",
    ),
    "face_rec": (
        f"{_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "face_recognition_sface_2021dec.onnx",
    ),
}
MODELS.update(FACE_MODELS)
CORE_MODELS = [k for k in MODELS if k not in FACE_MODELS]


def model_path(name: str) -> Path:
    return MODEL_DIR / MODELS[name][1]


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url}", flush=True)
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / total:6.1%} of {total / 1e6:.0f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def ensure_models(names: list[str] | None = None) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name in names or CORE_MODELS:
        url, rel = MODELS[name]
        target = MODEL_DIR / rel
        if target.exists():
            continue
        print(f"[models] {name}", flush=True)
        fname = url.rsplit("/", 1)[1]
        if fname.endswith(".tar.bz2"):
            archive = MODEL_DIR / fname
            _download(url, archive)
            with tarfile.open(archive, "r:bz2") as tf:
                tf.extractall(MODEL_DIR, filter="data")
            archive.unlink()
            # drop bundled test wavs etc.
            test_dir = target.parent / "test_wavs"
            if test_dir.exists():
                shutil.rmtree(test_dir)
        else:
            _download(url, target)
        if not target.exists():
            sys.exit(f"model {name} missing after download: {target}")


def models_ready(names: list[str] | None = None) -> bool:
    return all(model_path(n).exists() for n in names or CORE_MODELS)


if __name__ == "__main__":
    ensure_models(sys.argv[1:] or list(MODELS))
    print("all models ready in", MODEL_DIR)
