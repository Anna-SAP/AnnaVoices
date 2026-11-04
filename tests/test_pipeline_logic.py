"""Pure logic in the pipeline: no models, no ffmpeg, no real audio."""
from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from helpers import FakeEmbedder, blobs, seg, voices
from voiceextract import pipeline as p

# ------------------------------------------------------------------ frame masks


def frames(mask) -> list[int]:
    return np.flatnonzero(mask).tolist()


def test_to_mask_without_segments_is_empty():
    assert not p.to_mask([], 10).any()


def test_to_mask_of_zero_frames():
    assert p.to_mask([(0.0, 1.0)], 0).shape == (0,)


def test_to_mask_marks_the_covered_frames():
    assert frames(p.to_mask([(0.25, 0.75)], 100)) == list(range(25, 75))


def test_to_mask_rounds_the_end_up():
    assert frames(p.to_mask([(0.0, 0.251)], 100)) == list(range(26))


def test_to_mask_clips_a_start_before_zero():
    assert frames(p.to_mask([(-1.0, 0.5)], 100)) == list(range(50))


def test_to_mask_clips_an_end_past_the_recording():
    assert frames(p.to_mask([(0.5, 99.0)], 100)) == list(range(50, 100))


def test_to_mask_ignores_a_segment_past_the_recording():
    assert not p.to_mask([(2.0, 3.0)], 100).any()


def test_to_mask_ignores_a_reversed_segment():
    assert not p.to_mask([(0.75, 0.25)], 100).any()


def test_to_mask_merges_overlapping_segments():
    assert frames(p.to_mask([(0.0, 0.5), (0.25, 0.75)], 100)) == list(range(75))


@pytest.mark.xfail(strict=True, reason="已知问题 #7：结束时间为负时，负切片会把大半 mask 置真")
def test_to_mask_ignores_a_segment_entirely_before_zero():
    assert not p.to_mask([(-1.0, -0.5)], 100).any()


@pytest.mark.parametrize(
    "mask, expected",
    [
        ([], []),
        ([False, False], []),
        ([True, True, True], [(0, 3)]),
        ([True, False, True], [(0, 1), (2, 3)]),
        ([False, True, True, False], [(1, 3)]),
    ],
)
def test_runs(mask, expected):
    assert p.runs(np.array(mask, bool)) == expected


def test_runs_inverts_to_mask():
    segs = [(10, 25), (40, 41), (90, 100)]
    assert p.runs(p.to_mask([(a / 100, b / 100) for a, b in segs], 100)) == segs


def test_dilate_by_zero_changes_nothing():
    mask = np.array([False, True, False])
    assert p.dilate(mask, 0) is mask


def test_dilate_grows_both_ways_and_stops_at_the_edges():
    mask = np.zeros(6, bool)
    mask[1] = True
    assert frames(p.dilate(mask, 2)) == [0, 1, 2, 3]


def test_dilate_of_an_empty_mask():
    assert p.dilate(np.zeros(0, bool), 3).shape == (0,)


# --------------------------------------------------------------- vectors / audio


def test_normalize_gives_unit_rows():
    out = p.normalize(np.array([[3.0, 4.0], [0.0, 2.0]]))
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_normalize_leaves_a_zero_vector_without_nans():
    assert np.array_equal(p.normalize(np.zeros(3)), np.zeros(3))


def test_fade_of_nothing():
    assert p.fade(np.zeros(0, np.float32), p.SR).shape == (0,)


def test_fade_leaves_a_single_sample_alone():
    assert p.fade(np.ones(1, np.float32), p.SR).tolist() == [1.0]


def test_fade_silences_both_ends_only():
    y = p.fade(np.ones(p.SR, np.float32), p.SR)
    assert y[0] == 0.0 and y[-1] == 0.0
    assert np.all(y[p.SR // 4 : 3 * p.SR // 4] == 1.0)


def test_fade_ramps_do_not_overlap_on_short_clips():
    assert p.fade(np.ones(3, np.float32), p.SR).tolist() == [0.0, 1.0, 0.0]


def test_fade_does_not_modify_its_input():
    y = np.ones(p.SR, np.float32)
    p.fade(y, p.SR)
    assert np.all(y == 1.0)


def test_wav_bytes_round_trip():
    y = np.linspace(-0.5, 0.5, 1000, dtype=np.float32)
    back, sr = sf.read(io.BytesIO(p._wav_bytes(y)), dtype="float32")
    assert sr == p.SR
    assert np.allclose(back, y, atol=1e-3)


def test_embedder_handles_no_chunks_without_loading_a_model():
    assert p.Embedder().many([]).shape == (0, 192)


# ------------------------------------------------------------------ window spans


@pytest.mark.parametrize("a, b", [(5, 5), (0, 10), (0, 34), (0, 35)])
def test_window_spans_keep_short_turns_whole(a, b):
    assert p._window_spans(a, b, 30, 10) == [(a, b)]


@pytest.mark.parametrize("b", [36, 100, 105, 1234])
def test_window_spans_cover_long_turns_with_equal_windows(b):
    spans = p._window_spans(0, b, 30, 10)
    assert spans[0][0] == 0 and spans[-1][1] == b
    assert all(e - s == 30 for s, e in spans)
    assert all(0 < n[0] - c[0] <= 10 for c, n in zip(spans, spans[1:]))


# -------------------------------------------------------------------- SRT times


@pytest.mark.parametrize(
    "t, expected",
    [
        (0, "00:00:00,000"),
        (0.001, "00:00:00,001"),
        (1.5, "00:00:01,500"),
        (61.25, "00:01:01,250"),
        (3661.5, "01:01:01,500"),
        (0.9996, "00:00:01,000"),
        (59.9999, "00:01:00,000"),
        (360000, "100:00:00,000"),
    ],
)
def test_srt_time(t, expected):
    assert p._srt_time(t) == expected


# -------------------------------------------------------------------- text search


@pytest.mark.parametrize(
    "cn, digits",
    [
        ("零", "0"),
        ("〇", "0"),
        ("十", "10"),
        ("十四", "14"),
        ("二十", "20"),
        ("二十一", "21"),
        ("五一", "51"),
        ("二零二四", "2024"),
        ("一百零五", "105"),
        ("一千零一", "1001"),
        ("两千", "2000"),
        ("三万", "30000"),
        ("一万二千", "12000"),
    ],
)
def test_cn_number(cn, digits):
    assert p._cn_number(cn) == digits


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", ""),
        ("，。！ \n", ""),
        ("Hello, World!", "helloworld"),
        ("a_b-c", "abc"),
        ("第十四集", "第14集"),
    ],
)
def test_norm_text(text, expected):
    assert p._norm_text(text) == expected


@pytest.mark.parametrize("query", ["", "   ", "，。！", "___"])
def test_search_ignores_queries_without_content(query):
    assert p.search_text({"segments": [seg()]}, query) == {"hits": [], "votes": []}


def test_search_of_an_empty_result():
    assert p.search_text({"segments": []}, "你好") == {"hits": [], "votes": []}


def test_search_finds_an_exact_phrase():
    out = p.search_text({"segments": [seg(id=7, spk="S2", start=1.0, end=2.0, text="今天天气不错")]}, "天气")
    assert out["hits"] == [{"segment": 7, "spk": "S2", "start": 1.0, "end": 2.0, "text": "今天天气不错", "score": 1.0}]
    assert out["votes"] == [{"id": "S2", "score": 1.0}]


@pytest.mark.parametrize(
    "query, text",
    [
        ("十四", "第14集"),
        ("14", "第十四集"),
        ("hello world", "Hello, World!"),
        ("你好吗", "你好，吗？"),
    ],
)
def test_search_ignores_case_punctuation_and_number_style(query, text):
    assert p.search_text({"segments": [seg(text=text)]}, query)["hits"][0]["score"] == 1.0


def test_search_accepts_a_close_match():
    hits = p.search_text({"segments": [seg(text="今天天气真好")]}, "今天天气不错")["hits"]
    assert hits[0]["score"] == pytest.approx(4 / 6, abs=0.001)


def test_search_rejects_a_distant_match():
    assert p.search_text({"segments": [seg(text="明天下雨")]}, "今天天气不错")["hits"] == []


def test_search_skips_segments_without_text():
    no_text = seg()
    del no_text["text"]
    assert p.search_text({"segments": [no_text, seg(text="")]}, "你好")["hits"] == []


def test_search_ranks_by_score_then_time():
    segs = [seg(id=0, start=9.0, text="今天天气真好"), seg(id=1, start=5.0, text="今天天气不错"), seg(id=2, start=1.0, text="今天天气不错")]
    assert [h["segment"] for h in p.search_text({"segments": segs}, "今天天气不错")["hits"]] == [2, 1, 0]


def test_search_limit_cuts_hits_but_not_votes():
    segs = [seg(id=i, start=float(i), text="你好") for i in range(5)]
    out = p.search_text({"segments": segs}, "你好", limit=2)
    assert len(out["hits"]) == 2
    assert out["votes"] == [{"id": "S1", "score": 5.0}]


def test_search_votes_rank_speakers_by_total_score():
    segs = [seg(id=0, spk="S1"), seg(id=1, spk="S2"), seg(id=2, spk="S2")]
    assert [v["id"] for v in p.search_text({"segments": segs}, "你好")["votes"]] == ["S2", "S1"]


# --------------------------------------------------------------- segment selection


def ids(segs) -> list[int]:
    return [s["id"] for s in segs]


def test_select_with_nothing_to_choose_from():
    assert p.select_segments({"segments": []}, ["S1"], {}) == []
    assert p.select_segments({"segments": [seg()]}, [], {}) == []


def test_select_keeps_only_the_chosen_speakers_in_time_order():
    segs = [
        seg(id=0, start=9.0, end=10.0),
        seg(id=1, spk="S2", start=5.0, end=6.0),
        seg(id=2, spk="S3", start=1.0, end=2.0),
        seg(id=3, start=2.0, end=3.0),
    ]
    assert ids(p.select_segments({"segments": segs}, ["S1", "S2"], {})) == [3, 1, 0]


@pytest.mark.parametrize("opts", [{"exclude": [0]}, {"min_sim": 0.7}, {"min_margin": 0.3}, {"min_segment": 2.0}])
def test_select_filters(opts):
    result = {"segments": [seg(id=0, end=1.0, sim=0.5, margin=0.1), seg(id=1, start=2.0, end=5.0)]}
    assert ids(p.select_segments(result, ["S1"], {})) == [0, 1]
    assert ids(p.select_segments(result, ["S1"], opts)) == [1]


def test_select_drops_non_speech_events_unless_told_otherwise():
    segs = [seg(id=0, event="Speech"), seg(id=1, event="Laughter"), seg(id=2)]
    assert ids(p.select_segments({"segments": segs}, ["S1"], {})) == [0, 2]
    assert ids(p.select_segments({"segments": segs}, ["S1"], {"skip_events": False})) == [0, 1, 2]


def test_select_drops_empty_text_unless_told_otherwise():
    segs = [seg(id=0, text=""), seg(id=1)]
    assert ids(p.select_segments({"segments": segs}, ["S1"], {})) == [1]
    assert ids(p.select_segments({"segments": segs}, ["S1"], {"skip_empty_text": False})) == [0, 1]


def test_preview_of_nothing():
    assert p.pick_preview_segments([]) == []


def test_preview_falls_back_to_the_longest_clips():
    segs = [seg(id=i, start=float(i * 10), end=float(i * 10) + d) for i, d in enumerate([1.0, 2.0, 2.5, 1.5])]
    assert ids(p.pick_preview_segments(segs)) == [1, 2, 3]


def test_preview_skips_clips_that_are_not_clean_speech():
    good = [seg(id=i, start=i * 10.0, end=i * 10.0 + 5) for i in range(3)]
    bad = [seg(id=8, start=50.0, end=55.0, event="Laughter"), seg(id=9, start=60.0, end=65.0, text="")]
    assert ids(p.pick_preview_segments(good + bad)) == [0, 1, 2]


def test_preview_prefers_the_purest_clips():
    segs = [seg(id=i, start=i * 10.0, end=i * 10.0 + 5, sim=0.5) for i in range(8)]
    for s in segs[2:5]:
        s["sim"] = 0.9
    assert ids(p.pick_preview_segments(segs)) == [2, 3, 4]


def test_preview_stops_at_the_target_length():
    segs = [seg(id=i, start=i * 20.0, end=i * 20.0 + 10) for i in range(3)]
    assert ids(p.pick_preview_segments(segs, target=15.0)) == [0, 1]


# ------------------------------------------------------------------ stage progress


def drive(active: list[str]) -> list[float]:
    seen = []
    make = p._stage_reporter(lambda stage, frac, msg: seen.append(frac), active)
    for stage in active:
        r = make(stage, "label")
        r(0.0)
        r(0.5)
        r(1.0)
    return seen


@pytest.mark.parametrize("skip", [[], ["separate"]])
def test_stage_progress_runs_from_zero_to_one(skip):
    seen = drive([k for k, _, _ in p.STAGES if k not in skip])
    assert seen[0] == 0.0
    assert seen[-1] == pytest.approx(1.0)
    assert seen == sorted(seen)


def test_stage_progress_stays_inside_the_stage():
    seen = []
    active = [k for k, _, _ in p.STAGES]
    r = p._stage_reporter(lambda stage, frac, msg: seen.append(frac), active)("diarize", "label")
    r(-1.0)
    r(2.0)
    r(0.0)
    r(1.0)
    assert seen[0] == seen[2]
    assert seen[1] == seen[3]


def test_stage_message_defaults_to_the_label():
    seen = []
    r = p._stage_reporter(lambda stage, frac, msg: seen.append((stage, msg)), ["extract"])("extract", "label")
    r(0.5)
    r(0.5, "custom")
    assert seen == [("extract", "label"), ("extract", "custom")]


# ---------------------------------------------------------------------- clustering


@pytest.mark.parametrize("n", [0, 1, 2])
def test_cluster_with_too_few_points_is_one_group(n):
    assert p.spectral_cluster(np.eye(4)[:n]).tolist() == [0] * n


def test_cluster_with_three_points_does_not_crash():
    assert len(p.spectral_cluster(np.eye(4)[:3])) == 3


def same_grouping(labels, truth) -> bool:
    return len({(a, b) for a, b in zip(labels.tolist(), truth.tolist())}) == len(set(truth.tolist()))


def test_cluster_finds_two_voices():
    X, truth = blobs(30, 30)
    assert same_grouping(p.spectral_cluster(X), truth)


def test_cluster_finds_three_unequal_voices():
    X, truth = blobs(60, 30, 15)
    assert same_grouping(p.spectral_cluster(X), truth)


def test_cluster_of_identical_points_is_one_group():
    assert len(np.unique(p.spectral_cluster(np.tile(np.eye(4)[0], (20, 1))))) == 1


def test_cluster_honours_a_requested_count():
    X, truth = blobs(30, 30)
    assert same_grouping(p.spectral_cluster(X, oracle=2), truth)


def test_cluster_clamps_a_request_for_more_groups_than_points():
    X, _ = blobs(5, 5)
    assert len(p.spectral_cluster(X, oracle=50)) == 10


def test_cluster_labels_every_point_when_it_subsamples():
    X, truth = blobs(60, 60)
    labels = p.spectral_cluster(X, max_points=50)
    assert len(labels) == 120
    assert same_grouping(labels, truth)


# ------------------------------------------------------------ voice change splitting


def test_split_of_no_segments():
    assert p.split_on_change(np.zeros(0, np.float32), [], FakeEmbedder(), 0.4, lambda f: None) == []


def test_split_leaves_short_segments_alone():
    class NoEmbedder:
        def many(self, *args):
            raise AssertionError("segments under 2 s never need voiceprints")

    assert p.split_on_change(voices((0, 3)), [(0, 150)], NoEmbedder(), 0.4, lambda f: None) == [(0, 150)]


def test_split_leaves_one_voice_alone():
    assert p.split_on_change(voices((0, 10)), [(0, 1000)], FakeEmbedder(), 0.4, lambda f: None) == [(0, 1000)]


def test_split_cuts_exactly_where_the_voice_changes():
    x = voices((0, 5), (1, 5))
    assert p.split_on_change(x, [(0, 1000)], FakeEmbedder(), 0.4, lambda f: None) == [(0, 500), (500, 1000)]


def test_split_handles_several_changes_and_keeps_other_segments():
    x = voices((0, 5), (1, 5), (0, 5), (2, 1))
    segs = [(0, 1500), (1500, 1600)]
    out = p.split_on_change(x, segs, FakeEmbedder(), 0.4, lambda f: None)
    assert out == [(0, 500), (500, 1000), (1000, 1500), (1500, 1600)]


def test_split_threshold_of_zero_never_cuts():
    x = voices((0, 5), (1, 5))
    assert p.split_on_change(x, [(0, 1000)], FakeEmbedder(), 0.0, lambda f: None) == [(0, 1000)]


@pytest.mark.parametrize("change", [0.5, 1.0, 1.25, 2.0, 2.5])
def test_split_never_leaves_a_piece_under_one_second(change):
    x = voices((0, change), (1, 3.0 - change))
    pieces = p.split_on_change(x, [(0, 300)], FakeEmbedder(), 0.4, lambda f: None)
    assert pieces[0][0] == 0 and pieces[-1][1] == 300
    assert all(b - a >= 100 for a, b in pieces)


# ------------------------------------------------------------------ ffmpeg parsing


def ffmpeg_says(monkeypatch, text):
    monkeypatch.setattr(p.subprocess, "run", lambda *a, **k: SimpleNamespace(stderr=text))


@pytest.mark.parametrize(
    "stderr, expected",
    [
        (
            "  Duration: 01:02:03.50, start: 0.0\n  Stream #0:0: Video: h264\n  Stream #0:1: Audio: aac",
            {"duration": 3723.5, "has_video": True, "has_audio": True},
        ),
        (
            "  Duration: 00:00:10.00\n  Stream #0:0: Audio: mp3",
            {"duration": 10.0, "has_video": False, "has_audio": True},
        ),
        (
            "  Duration: 00:00:10.00\n  Stream #0:0: Audio: mp3\n  Stream #0:1: Video: mjpeg (Baseline)",
            {"duration": 10.0, "has_video": False, "has_audio": True},
        ),
        (
            "  Duration: 00:00:10.00\n  Stream #0:0: Audio: flac\n  Stream #0:1: Video: png",
            {"duration": 10.0, "has_video": False, "has_audio": True},
        ),
        (
            "  Duration: 00:00:10.00\n  Stream #0:0: Video: h264",
            {"duration": 10.0, "has_video": True, "has_audio": False},
        ),
        ("  Duration: N/A\n  Stream #0:0: Audio: opus", {"duration": 0.0, "has_video": False, "has_audio": True}),
        ("x.mp4: Invalid data found when processing input", {"duration": 0.0, "has_video": False, "has_audio": False}),
        ("", {"duration": 0.0, "has_video": False, "has_audio": False}),
    ],
)
def test_probe_media_reads_ffmpeg_output(monkeypatch, tmp_path, stderr, expected):
    ffmpeg_says(monkeypatch, stderr)
    assert p.probe_media(tmp_path / "x.mp4") == expected


def test_loudnorm_filter_uses_the_measured_values(monkeypatch):
    stats = '{"input_i": "-24.50", "input_tp": "-3.00", "input_lra": "5.00", "input_thresh": "-35.00", "target_offset": "0.10"}'
    ffmpeg_says(monkeypatch, f'noise {{"early": "block"}}\n{stats}\n'.encode())
    af = p.loudnorm_filter(np.zeros(10, np.float32), p.SR, target=-20.0)
    assert af.startswith("loudnorm=I=-20.0:")
    assert "linear=true" in af
    for field in ("measured_I=-24.50", "measured_TP=-3.00", "measured_LRA=5.00", "measured_thresh=-35.00", "offset=0.10"):
        assert field in af
