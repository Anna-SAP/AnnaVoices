"""Model registry and installer (no real downloads) and the command line."""
from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path, PurePosixPath

import pytest

from helpers import seg
from voiceextract import cli, models, pipeline
from voiceextract.jobs import JobStore

# ------------------------------------------------------------------------- models


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    d = tmp_path / "sandbox" / "models"
    monkeypatch.setattr(models, "MODEL_DIR", d)
    return d


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"model")


def tar_with(dest: Path, files: dict[str, bytes]) -> None:
    with tarfile.open(dest, "w:bz2") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def test_model_path_of_an_unknown_model():
    with pytest.raises(KeyError):
        models.model_path("nope")


def test_face_models_stay_optional():
    assert not set(models.CORE_MODELS) & set(models.FACE_MODELS)
    assert set(models.CORE_MODELS) | set(models.FACE_MODELS) == set(models.MODELS)


@pytest.mark.parametrize("name", list(models.MODELS))
def test_registry_entries_are_https_and_stay_inside_the_model_folder(name):
    url, rel = models.MODELS[name]
    assert url.startswith("https://")
    assert not PurePosixPath(rel).is_absolute() and ".." not in PurePosixPath(rel).parts


def test_models_ready_needs_every_core_model(model_dir):
    assert models.models_ready() is False
    for name in models.CORE_MODELS[:-1]:
        touch(models.model_path(name))
    assert models.models_ready() is False
    touch(models.model_path(models.CORE_MODELS[-1]))
    assert models.models_ready() is True


def test_models_ready_can_check_a_subset(model_dir):
    touch(models.model_path("vad"))
    assert models.models_ready(["vad"]) is True
    assert models.models_ready(["vad", "face_det"]) is False


def test_ensure_models_leaves_existing_files_alone(model_dir):
    """The sealed test run fails any download attempt, so this proves none happened."""
    touch(models.model_path("vad"))
    models.ensure_models(["vad"])
    assert models.model_path("vad").read_bytes() == b"model"


def test_ensure_models_fetches_a_plain_file(model_dir, monkeypatch):
    monkeypatch.setattr(models, "_download", lambda url, dest: dest.write_bytes(b"new"))
    models.ensure_models(["vad"])
    assert models.model_path("vad").read_bytes() == b"new"


def test_ensure_models_unpacks_archives_and_drops_the_extras(model_dir, monkeypatch):
    rel = models.MODELS["asr"][1]
    folder = rel.split("/")[0]
    monkeypatch.setattr(models, "_download", lambda url, dest: tar_with(dest, {rel: b"m", f"{folder}/test_wavs/a.wav": b"w"}))
    models.ensure_models(["asr"])
    assert models.model_path("asr").read_bytes() == b"m"
    assert not (models.model_path("asr").parent / "test_wavs").exists()
    assert [p.name for p in model_dir.iterdir()] == [folder]


@pytest.mark.parametrize("member", ["../pwned.txt", "../../pwned.txt"])
def test_ensure_models_refuses_an_archive_that_escapes(model_dir, monkeypatch, member):
    monkeypatch.setattr(models, "_download", lambda url, dest: tar_with(dest, {member: b"pwned"}))
    with pytest.raises(tarfile.OutsideDestinationError):
        models.ensure_models(["asr"])
    assert list(model_dir.parent.rglob("pwned.txt")) == []


def test_ensure_models_stops_when_a_download_lacks_the_model(model_dir, monkeypatch):
    monkeypatch.setattr(models, "_download", lambda url, dest: None)
    with pytest.raises(SystemExit):
        models.ensure_models(["vad"])


class FakeResponse(io.BytesIO):
    headers = {"Content-Length": "4"}


def test_download_writes_the_file_in_one_step(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(models.urllib.request, "urlopen", lambda url: FakeResponse(b"data"))
    dest = tmp_path / "model.onnx"
    models._download("https://example.invalid/model.onnx", dest)
    assert dest.read_bytes() == b"data"
    assert [p.name for p in tmp_path.iterdir()] == ["model.onnx"]


def test_a_failed_download_leaves_no_finished_file(tmp_path, monkeypatch, capsys):
    class Dropped(FakeResponse):
        def read(self, *args):
            raise OSError("connection lost")

    monkeypatch.setattr(models.urllib.request, "urlopen", lambda url: Dropped(b"data"))
    dest = tmp_path / "model.onnx"
    with pytest.raises(OSError):
        models._download("https://example.invalid/model.onnx", dest)
    assert not dest.exists()


# ---------------------------------------------------------------------------- CLI


@pytest.fixture
def s(tmp_path, monkeypatch):
    store = JobStore(tmp_path / "jobs")
    monkeypatch.setattr(cli, "JobStore", lambda: store)
    return store


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["nope"],
        ["analyze", "clip.mp4"],
        ["analyze", "--job", "demo"],
        ["analyze", "clip.mp4", "--job", "demo", "--speakers", "abc"],
        ["identify"],
        ["export", "--job", "demo"],
        ["export", "--speaker", "S1"],
    ],
)
def test_bad_command_lines_exit_with_usage(argv, capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(argv)
    assert e.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["analyze", "clip.mp4", "--job", "../evil"],
        ["analyze", "clip.mp4", "--job", "..\\evil"],
        ["identify", "--job", "../evil", "--text", "x"],
        ["export", "--job", "../evil", "--speaker", "S1"],
    ],
)
def test_illegal_job_names_are_refused_without_writing(s, argv):
    with pytest.raises(KeyError):
        cli.main(argv)
    assert list(s.root.iterdir()) == []
    assert not (s.root.parent / "evil").exists()


def test_analyze_records_success(s, tmp_path, monkeypatch, capsys):
    speaker = {"id": "S1", "clean_seconds": 1.0, "raw_seconds": 2.0, "num_segments": 3, "similar": []}

    def analyze(src, job_dir, opts, report):
        assert opts == {"separate": False, "num_speakers": 2}
        assert job_dir == s.dir("demo")
        report("asr", 0.5, "转写中")
        assert s.get("demo")["stage"] == "asr"
        return {"speakers": [speaker]}

    monkeypatch.setattr(pipeline, "analyze", analyze)
    cli.main(["analyze", "clip.mp4", "--job", "demo", "--speakers", "2", "--no-separate"])
    job = s.get("demo")
    assert job["status"] == "done"
    assert job["name"] == "clip.mp4"
    assert Path(job["source_path"]).is_absolute()
    assert "S1: clean" in capsys.readouterr().out


def test_analyze_records_failure_and_still_raises(s, monkeypatch):
    def analyze(*args):
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline, "analyze", analyze)
    with pytest.raises(RuntimeError):
        cli.main(["analyze", "clip.mp4", "--job", "demo"])
    assert s.get("demo")["status"] == "error"
    assert s.get("demo")["error"] == "boom"


def test_identify_before_analysis_fails_clearly(s):
    s.create("a.mp4", {}, job_id="demo")
    with pytest.raises(FileNotFoundError):
        cli.main(["identify", "--job", "demo", "--text", "x"])


def test_identify_prints_text_matches(s, capsys):
    s.create("a.mp4", {}, job_id="demo")
    result = {"segments": [seg(spk="S2", text="今天天气不错")]}
    (s.dir("demo") / "result.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    cli.main(["identify", "--job", "demo", "--text", "天气"])
    out = capsys.readouterr().out
    assert "S2" in out and "今天天气不错" in out


@pytest.fixture
def exported(s, monkeypatch):
    """A job whose export writes real files, as pipeline.export would."""
    s.create("a.mp4", {}, job_id="demo")

    def export(job_dir, speakers, opts, report, name=None):
        out = job_dir / "export"
        out.mkdir()
        files = {"mp3": f"{name}.mp3", "srt": f"{name}.srt", "zip": f"{name}_slices.zip"}
        for fn in files.values():
            (out / fn).write_text(fn, encoding="utf-8")
        report(1.0, "完成")
        return {"files": files, "opts": opts}

    monkeypatch.setattr(pipeline, "export", export)
    return s


def test_export_registers_the_file_for_the_web_app(exported, capsys):
    cli.main(["export", "--job", "demo", "--speaker", "S1", "--speaker", "S2"])
    record = exported.get("demo")["exports"]["S1_S2"]
    assert record["status"] == "done"
    assert record["speakers"] == ["S1", "S2"]
    assert exported.get("demo")["names"] == {}


def test_export_name_also_names_a_single_speaker(exported, capsys):
    cli.main(["export", "--job", "demo", "--speaker", "S1", "--name", "姜文"])
    assert exported.get("demo")["names"] == {"S1": "姜文"}


def test_export_maps_its_flags_to_options(exported, capsys):
    cli.main(["export", "--job", "demo", "--speaker", "S1", "--no-denoise", "--slices"])
    assert exported.get("demo")["exports"]["S1"]["meta"]["opts"] == {"denoise": False, "slices": True}


def test_export_out_copies_every_file_next_to_the_target(exported, tmp_path, capsys):
    out = tmp_path / "output" / "deep" / "target.mp3"
    cli.main(["export", "--job", "demo", "--speaker", "S1", "--name", "x", "--out", str(out)])
    assert sorted(p.name for p in out.parent.iterdir()) == ["target.mp3", "target.srt", "target_slices.zip"]
    assert out.read_text(encoding="utf-8") == "x.mp3"
    assert (out.parent / "target_slices.zip").read_text(encoding="utf-8") == "x_slices.zip"
