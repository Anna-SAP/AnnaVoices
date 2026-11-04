"""The export and analyze flows with every model and ffmpeg call stubbed.

These check the bookkeeping around the models (timing, file names, speaker
naming, caching). Whether speakers are separated well is not covered here.
"""
from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from helpers import FakeEmbedder, seg, voices
from voiceextract import pipeline as p

QUIET = {"denoise": False, "loudnorm": False}


def write_vocals(job_dir: Path, seconds: float, level: float = 0.1) -> None:
    sf.write(job_dir / "vocals_44k.wav", np.full(int(seconds * p.SR), level, np.float32), p.SR, subtype="FLOAT")


@pytest.fixture
def job_dir(tmp_path):
    d = tmp_path / "job"
    d.mkdir()
    write_vocals(d, 10.0)
    segments = [
        seg(id=0, start=1.0, end=3.0, text="第一句"),
        seg(id=1, start=5.0, end=7.0, text="第二句"),
        seg(id=2, spk="S2", start=7.0, end=9.0, text="别人说的"),
    ]
    (d / "result.json").write_text(json.dumps({"segments": segments}, ensure_ascii=False), encoding="utf-8")
    return d


@pytest.fixture
def encoded(monkeypatch):
    """Replaces the MP3 encoder and records what it was given."""
    calls = []

    def encode(samples, dst, sr=p.SR, bitrate="320k", af=None):
        calls.append({"samples": np.array(samples), "dst": Path(dst), "bitrate": bitrate, "af": af})
        Path(dst).write_bytes(b"mp3")

    monkeypatch.setattr(p, "encode_mp3", encode)
    return calls


def export(job_dir, speakers=("S1",), name=None, **opts):
    return p.export(job_dir, list(speakers), {**QUIET, **opts}, lambda f, m: None, name=name)


def test_export_without_matching_segments_writes_nothing(job_dir):
    with pytest.raises(ValueError, match="没有符合条件的片段"):
        export(job_dir, speakers=["S9"])
    assert not (job_dir / "export").exists()


def test_export_writes_the_expected_files(job_dir, encoded):
    meta = export(job_dir)
    names = sorted(f.name for f in (job_dir / "export").iterdir())
    assert names == ["S1.json", "S1.mp3", "S1.srt"]
    assert meta["files"] == {"mp3": "S1.mp3", "srt": "S1.srt"}
    assert json.loads((job_dir / "export" / "S1.json").read_text(encoding="utf-8")) == meta


@pytest.mark.parametrize(
    "speakers, name, stem",
    [(["S1"], None, "S1"), (["S1", "S2"], None, "S1_S2"), (["S1"], "姜文", "姜文")],
)
def test_export_file_stem(job_dir, encoded, speakers, name, stem):
    assert export(job_dir, speakers=speakers, name=name)["files"]["mp3"] == f"{stem}.mp3"


def test_export_timeline_counts_the_padding_and_the_gaps(job_dir, encoded):
    meta = export(job_dir, gap=0.5)
    # each 2 s segment gains 0.05 s of padding on both sides
    assert (meta["segments"], meta["speech_seconds"], meta["file_seconds"]) == (2, 4.2, 4.7)
    assert len(encoded[0]["samples"]) == pytest.approx(4.7 * p.SR, abs=2)
    assert (job_dir / "export" / "S1.srt").read_text(encoding="utf-8") == (
        "1\n00:00:00,000 --> 00:00:02,100\n第一句\n\n"
        "2\n00:00:02,600 --> 00:00:04,700\n第二句\n"
    )


def test_export_with_no_gap(job_dir, encoded):
    assert export(job_dir, gap=0.0)["file_seconds"] == 4.2


def test_export_honours_excluded_segments(job_dir, encoded):
    assert export(job_dir, exclude=[0])["segments"] == 1


def test_export_clips_padding_at_the_recording_start(tmp_path, encoded):
    write_vocals(tmp_path, 3.0)
    (tmp_path / "result.json").write_text(json.dumps({"segments": [seg(start=0.0, end=2.0)]}), encoding="utf-8")
    export(tmp_path)
    assert len(encoded[0]["samples"]) == pytest.approx(2.05 * p.SR, abs=2)  # no padding before 0


def test_export_tolerates_a_segment_past_the_end_of_the_audio(tmp_path, encoded):
    write_vocals(tmp_path, 3.0)
    (tmp_path / "result.json").write_text(json.dumps({"segments": [seg(start=2.0, end=9.0)]}), encoding="utf-8")
    export(tmp_path)
    assert len(encoded[0]["samples"]) == pytest.approx(1.05 * p.SR, abs=2)  # stops where the audio ends


def test_export_passes_bitrate_and_filter_to_the_encoder(job_dir, encoded, monkeypatch):
    monkeypatch.setattr(p, "loudnorm_filter", lambda audio, sr, target: f"loudnorm={target}")
    export(job_dir, loudnorm=True, lufs=-20.0, bitrate="128k")
    assert (encoded[0]["bitrate"], encoded[0]["af"]) == ("128k", "loudnorm=-20.0")


def test_export_runs_the_denoiser_only_when_asked(job_dir, encoded, monkeypatch):
    calls = []

    def denoise(pieces, atten_db, report):
        calls.append((len(pieces), atten_db))
        return pieces

    monkeypatch.setattr(p, "_denoise_segments", denoise)
    export(job_dir)
    assert calls == []
    export(job_dir, denoise=True, denoise_db=12.0)
    assert calls == [(2, 12.0)]


def slices(job_dir, stem="S1"):
    with zipfile.ZipFile(job_dir / "export" / f"{stem}_slices.zip") as z:
        wavs = {n: sf.read(io.BytesIO(z.read(n)), dtype="float32") for n in z.namelist() if n.endswith(".wav")}
        return wavs, z.read("list.txt").decode("utf-8")


def test_export_only_builds_slices_when_asked(job_dir, encoded):
    assert "zip" not in export(job_dir)["files"]


def test_export_slices_are_listed_for_tts_training(job_dir, encoded):
    meta = export(job_dir, name="姜文", slices=True)
    assert meta["files"]["zip"] == "姜文_slices.zip"
    wavs, listing = slices(job_dir, "姜文")
    assert sorted(wavs) == ["wavs/姜文_0000.wav", "wavs/姜文_0001.wav"]
    assert all(sr == p.SR for _, sr in wavs.values())
    assert listing == "wavs/姜文_0000.wav|姜文|ZH|第一句\nwavs/姜文_0001.wav|姜文|ZH|第二句\n"


def test_export_slices_keep_their_level_without_loudnorm(job_dir, encoded):
    export(job_dir, slices=True)
    y, _ = slices(job_dir)[0]["wavs/S1_0000.wav"]
    assert y[len(y) // 2] == pytest.approx(0.1, abs=1e-3)


def test_export_slices_get_the_same_gain_as_the_mp3(job_dir, encoded, monkeypatch):
    monkeypatch.setattr(p, "loudnorm_filter", lambda audio, sr, target: "loudnorm=I=-18:measured_I=-24.0:linear=true")
    export(job_dir, loudnorm=True, lufs=-18.0, slices=True)
    y, _ = slices(job_dir)[0]["wavs/S1_0000.wav"]
    assert y[len(y) // 2] == pytest.approx(0.1 * 10 ** (6 / 20), abs=1e-3)


def test_export_slices_clip_rather_than_wrap(job_dir, encoded, monkeypatch):
    monkeypatch.setattr(p, "loudnorm_filter", lambda audio, sr, target: "measured_I=-70.0")
    export(job_dir, loudnorm=True, lufs=-18.0, slices=True)
    y, _ = slices(job_dir)[0]["wavs/S1_0000.wav"]
    assert y.max() <= 1.0
    assert y[len(y) // 2] == pytest.approx(1.0, abs=1e-3)


# ------------------------------------------------------------------------ analyze


def clean(cluster, start, end):
    return {"cluster": cluster, "start": start, "end": end, "sim": 0.8, "margin": 0.4}


@pytest.fixture
def stages(tmp_path, monkeypatch):
    """A job folder with cached vocals; model stages are stubbed, refine is scripted."""
    d = tmp_path / "job"
    d.mkdir()
    write_vocals(d, 1.0)
    (d / "vocals.mix").touch()
    (d / "diarization.json").write_text("[]", encoding="utf-8")

    centroids = {k: np.eye(4)[k] for k in range(4)}
    script = {"clean": [], "centroids": centroids, "n_main": 2}

    def refine(x16, diar, opts, report):
        return script["clean"], script["centroids"], {k: 99.0 for k in script["centroids"]}, 1.5, script["n_main"]

    def transcribe(x16, segments, report):
        for s in segments:
            s["text"] = "你好"

    monkeypatch.setattr(p, "ensure_models", lambda *a: None)
    monkeypatch.setattr(p, "probe_media", lambda src: {"duration": 1.0, "has_video": False, "has_audio": True})
    monkeypatch.setattr(p, "refine", refine)
    monkeypatch.setattr(p, "transcribe", transcribe)
    monkeypatch.setattr(p, "make_assets", lambda *a: None)
    script["dir"] = d
    return script


def analyze(script, **opts):
    return p.analyze(Path("clip.mp4"), script["dir"], {"separate": False, **opts}, lambda *a: None)


def test_analyze_refuses_a_file_without_audio(stages, monkeypatch):
    monkeypatch.setattr(p, "probe_media", lambda src: {"duration": 0.0, "has_video": False, "has_audio": False})
    with pytest.raises(ValueError, match="文件中没有音轨"):
        analyze(stages)


def test_analyze_names_speakers_by_amount_of_clean_speech(stages):
    stages["clean"] = [clean(0, 0.0, 1.0), clean(1, 1.0, 3.0)]
    res = analyze(stages)
    assert [s["id"] for s in res["speakers"]] == ["S1", "S2"]
    assert [s["clean_seconds"] for s in res["speakers"]] == [2.0, 1.0]
    assert [(s["id"], s["spk"]) for s in res["segments"]] == [(0, "S2"), (1, "S1")]
    assert all("cluster" not in s for s in res["segments"])


def test_analyze_lists_a_minor_voice_with_a_real_share(stages):
    stages["clean"] = [clean(0, 0.0, 50.0), clean(1, 50.0, 90.0), clean(2, 90.0, 95.0)]
    assert len(analyze(stages)["speakers"]) == 3


def test_analyze_moves_tiny_minor_voices_to_fragments(stages):
    stages["clean"] = [clean(0, 0.0, 500.0), clean(1, 500.0, 900.0), clean(2, 900.0, 902.0)]
    res = analyze(stages)
    assert len(res["speakers"]) == 2
    assert res["fragments"] == {"groups": 1, "seconds": 2.0}
    assert all(s["end"] <= 900.0 for s in res["segments"])


def test_analyze_tops_up_to_the_requested_count_from_the_largest_minor_voices(stages):
    stages["clean"] = [clean(0, 0.0, 500.0), clean(1, 500.0, 900.0), clean(2, 900.0, 901.0), clean(3, 901.0, 903.0)]
    res = analyze(stages, num_speakers=3)
    assert len(res["speakers"]) == 3
    assert res["fragments"] == {"groups": 1, "seconds": 1.0}
    assert res["speakers"][2]["clean_seconds"] == 2.0


def test_analyze_reports_similar_voices(stages):
    a, b = np.array([1.0, 0, 0, 0]), p.normalize(np.array([0.8, 0.6, 0, 0]))
    stages["centroids"] = {0: a, 1: b}
    stages["clean"] = [clean(0, 0.0, 2.0), clean(1, 2.0, 3.0)]
    assert analyze(stages)["speakers"][0]["similar"] == [{"id": "S2", "sim": 0.8}]


def test_analyze_saves_its_result_and_centroids(stages):
    stages["clean"] = [clean(0, 0.0, 1.0), clean(1, 1.0, 3.0)]
    res = analyze(stages)
    assert json.loads((stages["dir"] / "result.json").read_text(encoding="utf-8")) == json.loads(json.dumps(res))
    saved = np.load(stages["dir"] / "centroids.npy")
    assert saved.shape == (2, 4)
    assert np.array_equal(saved[0], stages["centroids"][1])  # S1 is cluster 1


def test_analyze_explains_a_recording_without_speech(stages):
    stages.update(clean=[], centroids={}, n_main=0)
    try:
        analyze(stages)
    except ValueError as e:
        assert "need at least one array" not in str(e)


@pytest.mark.parametrize("found", [[], [clean(2, 0.0, 0.2)]], ids=["no-speech", "only-a-fragment"])
def test_analyze_stops_before_transcribing_when_no_speech_is_usable(stages, monkeypatch, found):
    def must_not_run(*args):
        raise AssertionError("nothing to transcribe")

    monkeypatch.setattr(p, "transcribe", must_not_run)
    stages.update(clean=found, centroids={}, n_main=0)
    with pytest.raises(ValueError, match="没有检测到可用的语音"):
        analyze(stages)
    assert not (stages["dir"] / "result.json").exists()


def test_analyze_reuses_cached_stages(stages, monkeypatch):
    def must_not_run(*args):
        raise AssertionError("cached stage ran again")

    monkeypatch.setattr(p, "run_ffmpeg", must_not_run)
    monkeypatch.setattr(p, "diarize", must_not_run)
    stages["clean"] = [clean(0, 0.0, 1.0)]
    assert analyze(stages)["duration"] == 1.0


def test_analyze_redoes_a_diarization_older_than_the_vocals(stages, monkeypatch):
    calls = []
    monkeypatch.setattr(p, "diarize", lambda x16, report: calls.append(1) or [])
    os.utime(stages["dir"] / "diarization.json", (1, 1))
    stages["clean"] = [clean(0, 0.0, 1.0)]
    analyze(stages)
    assert calls == [1]


def test_analyze_does_not_reuse_vocals_made_with_other_settings(stages, monkeypatch):
    calls = []

    def fake_ffmpeg(*args):
        calls.append(args)
        sf.write(args[-1], np.zeros((p.SR, 2), np.float32), p.SR, subtype="FLOAT")

    monkeypatch.setattr(p, "run_ffmpeg", fake_ffmpeg)
    monkeypatch.setattr(p, "separate_vocals", lambda x, report: x.mean(1))
    monkeypatch.setattr(p, "diarize", lambda x16, report: [])
    stages["clean"] = [clean(0, 0.0, 1.0)]
    analyze(stages, separate=True)
    assert len(calls) == 1
    d = stages["dir"]
    assert (d / "vocals.demucs").exists() and not (d / "vocals.mix").exists()
    assert not (d / "audio_44k.wav").exists()


# ------------------------------------------------------------------------- refine


@pytest.fixture
def fake_models(monkeypatch):
    monkeypatch.setattr(p, "Embedder", FakeEmbedder)


def refine(x, diar):
    return p.refine(x, diar, p.DEFAULT_OPTS, lambda f: None)


def test_refine_of_a_recording_without_speech(fake_models, monkeypatch):
    monkeypatch.setattr(p, "vad_segments", lambda *a, **k: [])
    assert refine(np.zeros(16000, np.float32), []) == ([], {}, {}, 0.0, 0)


def test_refine_separates_two_clean_voices(fake_models, monkeypatch):
    monkeypatch.setattr(p, "vad_segments", lambda *a, **k: [(0.0, 20.0)])
    x = voices((0, 10), (1, 10))
    out, centroids, raw_seconds, overlap, n_main = refine(x, [(0.0, 10.0, 0), (10.0, 20.0, 1)])
    assert n_main == 2
    assert overlap == 0.0
    by_voice = {}
    for s in out:
        by_voice.setdefault(s["cluster"], []).append((s["start"], s["end"]))
    assert len(by_voice) == 2
    assert all(e <= 10.0 for _, e in min(by_voice.values()))
    assert all(a >= 10.0 for a, _ in max(by_voice.values()))


def test_refine_drops_overlapped_speech(fake_models, monkeypatch):
    monkeypatch.setattr(p, "vad_segments", lambda *a, **k: [(0.0, 20.0)])
    x = voices((0, 10), (1, 10))
    out, _, _, overlap, _ = refine(x, [(0.0, 12.0, 0), (8.0, 20.0, 1)])
    assert overlap == 4.0
    assert all(s["end"] <= 7.9 or s["start"] >= 12.1 for s in out)
