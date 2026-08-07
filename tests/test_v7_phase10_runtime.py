"""GAIRA V7 — Phase 10 tests: contracts, adapters, validation, service, reporting."""
from __future__ import annotations

import json

import numpy as np
import pytest

from gaira.v7.adapters import PLANNED_FORMATS, load as load_spectrum
from gaira.v7.contracts import (InferenceOptions, InferenceRequest, Modality, SampleType,
                                Severity, SpectrumInput, SpectrumMetadata)
from gaira.v7.io import frozen_root
from gaira.v7.reporting import SCOPE_NOTES, render
from gaira.v7.runtime import freeze as FREEZE
from gaira.v7.runtime.service import GAIRAService, SpectrumRejected
from gaira.v7.validation import validate

FIXTURES = frozen_root().parents[1] / "tests" / "fixtures" / "v7_phase10"


@pytest.fixture(scope="module")
def svc():
    return GAIRAService.instance()


@pytest.fixture(scope="module")
def corpus():
    z = np.load(frozen_root() / "phase01/artifacts/balanced_references_v1.npz",
                allow_pickle=True)
    return np.asarray(z["X"], float), np.asarray(z["grid"], float)


def _req(grid, y, **opts):
    return InferenceRequest(
        spectrum=SpectrumInput(wavenumber=grid.tolist(), intensity=y.tolist()),
        options=InferenceOptions(already_preprocessed=True, **opts))


# ── frozen assets ────────────────────────────────────────────────────────────
def test_ten_frozen_assets_are_pinned():
    assert len(FREEZE.FROZEN_ASSETS) == 10
    assert all(FREEZE.EXPECTED_DIGESTS[a] for a in FREEZE.FROZEN_ASSETS)


def test_frozen_assets_verify_against_the_committed_tree():
    s = FREEZE.summary()
    assert s["all_present"] and s["all_match"]
    assert s["n_pinned"] == 10


def test_a_changed_asset_raises(monkeypatch):
    bad = dict(FREEZE.EXPECTED_DIGESTS)
    bad["phase02/artifacts/csm_dictionary_v1.npz"] = "0" * 32
    monkeypatch.setattr(FREEZE, "EXPECTED_DIGESTS", bad)
    with pytest.raises(FREEZE.FrozenAssetError):
        FREEZE.verify(strict=True)


def test_pinned_digests_match_a_fresh_recomputation():
    assert FREEZE.pin() == {k: v for k, v in FREEZE.EXPECTED_DIGESTS.items()}


# ── adapters ─────────────────────────────────────────────────────────────────
def test_csv_with_header():
    p = load_spectrum(b"Wavenumber,Intensity\n450,1\n452,2\n454,3\n456,4\n", "a.csv")
    assert p.ok and p.wavenumber.tolist() == [450, 452, 454, 456]
    assert any(d.code == "input.header" for d in p.diagnostics)


def test_tsv_and_semicolon_and_space():
    for payload, name in ((b"450\t1\n452\t2\n454\t3\n", "a.tsv"),
                          (b"450;1\n452;2\n454;3\n", "a.txt"),
                          (b"450 1\n452 2\n454 3\n", "a.dat")):
        p = load_spectrum(payload, name)
        assert p.ok, name
        assert p.intensity.tolist() == [1, 2, 3]


def test_descending_axis_is_reversed_and_reported():
    p = load_spectrum(b"1800,5\n1798,6\n1796,7\n1794,8\n", "d.csv")
    assert p.wavenumber.tolist() == [1794, 1796, 1798, 1800]
    assert p.intensity.tolist() == [8, 7, 6, 5], "intensities must follow their wavenumbers"
    assert any(d.code == "input.descending" for d in p.diagnostics)


def test_nan_rows_are_dropped_with_a_diagnostic():
    p = load_spectrum(b"450,1\n452,nan\n454,3\n456,4\n458,5\n", "n.csv")
    assert any(d.code == "input.non_finite" for d in p.diagnostics)
    assert np.isfinite(p.intensity).all()


def test_duplicate_wavenumbers_are_averaged_not_dropped_silently():
    p = load_spectrum(b"450,1\n450,3\n452,2\n454,9\n", "dup.csv")
    d = [x for x in p.diagnostics if x.code == "input.duplicate_wavenumbers"]
    assert d, "a duplicate must be reported"
    assert p.wavenumber.tolist() == [450, 452, 454]
    assert p.intensity[0] == 2.0


def test_a_bad_intensity_never_desynchronises_the_columns():
    """The worst possible adapter bug: a spectrum that still looks like a spectrum.

    A row whose wavenumber parses and whose intensity does not once appended the wavenumber and
    then failed, pairing every subsequent intensity with the wrong wavenumber.
    """
    p = load_spectrum(b"".join(f"{450 + 2 * i},{'BAD' if i == 2 else 10 + i}\n".encode()
                               for i in range(12)), "align.csv")
    assert p.ok
    for w, v in zip(p.wavenumber.tolist(), p.intensity.tolist()):
        assert v == 10 + (w - 450) / 2, f"{w} paired with {v}"


def test_a_parse_failure_is_reported_not_disguised(monkeypatch):
    """A raising `parse` once surfaced as 'unrecognised format', hiding a real defect."""
    from gaira.v7.adapters import ADAPTERS
    victim = ADAPTERS[1]
    monkeypatch.setattr(type(victim), "parse",
                        lambda self, payload, filename=None: (_ for _ in ()).throw(
                            RuntimeError("boom")))
    p = load_spectrum(b"450,1\n452,2\n454,3\n", "x.csv")
    assert not p.ok
    assert any(d.code == "input.parse_failed" for d in p.diagnostics)


def test_malformed_rows_beyond_ten_percent_are_an_error():
    rows = b"".join(f"{450 + 2 * i},{'x' if i % 2 else i}\n".encode() for i in range(20))
    p = load_spectrum(rows, "m.csv")
    assert not p.ok
    assert any(d.code == "input.malformed_rows" and d.severity is Severity.ERROR
               for d in p.diagnostics)


def test_binary_and_planned_formats_are_refused_not_guessed():
    assert not load_spectrum(b"\x00\x01\x02binary\x00", "x.csv").ok
    for ext in PLANNED_FORMATS:
        p = load_spectrum(b"whatever", f"s{ext}")
        assert not p.ok
        assert any(d.code == "input.format_not_supported" for d in p.diagnostics)


def test_array_adapter_round_trips():
    p = load_spectrum((np.linspace(450, 1800, 100), np.ones(100)))
    assert p.ok and p.wavenumber.size == 100


def test_adapter_never_repairs_a_length_mismatch():
    p = load_spectrum((np.arange(10.0), np.arange(9.0)))
    assert not p.ok
    assert any(d.code == "input.length_mismatch" for d in p.diagnostics)


# ── validation ───────────────────────────────────────────────────────────────
def test_good_spectrum_passes(corpus):
    X, g = corpus
    assert validate(g, X[0]).can_run


@pytest.mark.parametrize("code,x,y", [
    ("input.too_few_points", np.linspace(450, 1800, 10), np.ones(10)),
    ("coverage.insufficient", np.linspace(1000, 1100, 200), np.linspace(1, 2, 200)),
    ("intensity.all_zero", np.linspace(450, 1800, 676), np.zeros(676)),
    ("intensity.constant", np.linspace(450, 1800, 676), np.ones(676)),
])
def test_error_conditions(code, x, y):
    v = validate(x, y)
    assert not v.can_run
    assert any(d.code == code for d in v.diagnostics), [d.code for d in v.diagnostics]


def test_unsupported_modality_is_an_error_not_a_warning(corpus):
    X, g = corpus
    v = validate(g, X[0], modality=Modality.AG_SERS)
    assert not v.can_run
    d = next(x for x in v.diagnostics if x.code == "scope.modality_unsupported")
    assert d.severity is Severity.ERROR


def test_unvalidated_sample_type_warns_but_still_runs(corpus):
    X, g = corpus
    v = validate(g, X[0], sample_type=SampleType.SERUM)
    assert v.can_run
    d = next(x for x in v.diagnostics if x.code == "scope.sample_type_unvalidated")
    assert d.severity is Severity.WARNING


def test_sample_type_does_not_change_a_single_number(svc, corpus):
    """The scope warning is metadata. If it moved a number, the contract would be a lie."""
    X, g = corpus
    digests = set()
    for st in (SampleType.PURE, SampleType.SERUM, SampleType.EV, SampleType.TISSUE):
        req = InferenceRequest(
            spectrum=SpectrumInput(wavenumber=g.tolist(), intensity=X[3].tolist()),
            metadata=SpectrumMetadata(sample_type=st),
            options=InferenceOptions(already_preprocessed=True))
        digests.add(svc.infer(req).result_digest)
    assert len(digests) == 1


def test_partial_coverage_warns_and_still_runs():
    x = np.linspace(900, 1500, 400)          # 44% of 450-1800, above the 10% error floor
    y = np.abs(np.sin(x / 40)) + 0.1
    v = validate(x, y)
    assert v.can_run
    assert any(d.code == "coverage.partial" for d in v.diagnostics)


# ── service ──────────────────────────────────────────────────────────────────
def test_health_and_engine_info(svc):
    h = svc.health()
    assert h.status == "ok" and h.frozen_assets_verified and h.n_frozen_assets == 10
    i = svc.engine_info()
    assert (i.n_lsms, i.n_csms, i.n_molecules, i.n_chemistry_axes) == (50, 49, 154, 16)
    assert i.supported_modalities == ["raman"]
    assert i.validated_sample_types == ["pure"]
    assert i.validated_performance["molecule_top1"] == 0.6053
    assert i.validated_performance["chemistry_top1_heldout"] == 0.8507
    assert any("open-set" in lim for lim in i.known_limitations)


def test_inference_is_deterministic(svc, corpus):
    X, g = corpus
    a, b = svc.infer(_req(g, X[7])), svc.infer(_req(g, X[7]))
    assert a.result_digest == b.result_digest
    assert a.model_dump(exclude={"engine"}) == b.model_dump(exclude={"engine"})


def test_results_are_immutable(svc, corpus):
    X, g = corpus
    r = svc.infer(_req(g, X[2]))
    with pytest.raises(Exception):
        r.chemistry = None


def test_every_score_reconciles(svc, corpus):
    X, g = corpus
    r = svc.infer(_req(g, X[11]))
    assert all(h.reconciles for h in r.retrieval.top)
    assert r.audit.all_scores_reconcile
    for h in r.retrieval.top:
        assert abs(h.contribution_sum - h.similarity) < 1e-9


def test_non_negativity_holds_at_every_layer(svc, corpus):
    X, g = corpus
    r = svc.infer(_req(g, X[13]))
    assert min(r.csm.activation) >= 0.0
    assert min(r.lsm.activation) >= 0.0
    assert min(r.chemistry.evidence) >= 0.0


def test_rejected_spectrum_raises_with_the_full_validation():
    svc = GAIRAService.instance()
    with pytest.raises(SpectrumRejected) as e:
        svc.infer(InferenceRequest(
            spectrum=SpectrumInput(wavenumber=[1.0, 2.0], intensity=[1.0, 2.0])))
    assert not e.value.validation.can_run
    assert e.value.validation.diagnostics


def test_options_narrow_the_view_without_changing_the_science(svc, corpus):
    X, g = corpus
    full = svc.infer(_req(g, X[4]))
    lean = svc.infer(_req(g, X[4], include_lsm=False, include_provenance=False,
                          include_audit=False))
    assert lean.lsm is None and lean.provenance is None and lean.audit is None
    assert lean.result_digest == full.result_digest


def test_compare_runs_both_independently(svc, corpus):
    from gaira.v7.contracts import CompareRequest
    X, g = corpus
    c = svc.compare(CompareRequest(a=_req(g, X[0]), b=_req(g, X[120]),
                                   label_a="one", label_b="two"))
    assert 0.0 <= c.csm_cosine <= 1.0 and 0.0 <= c.chemistry_cosine <= 1.0
    assert len(c.chemistry_delta) == 16
    assert c.a.result_digest == svc.infer(_req(g, X[0])).result_digest
    assert "biological state change" in c.scope_note


# ── golden regression ────────────────────────────────────────────────────────
def test_golden_fixtures_reproduce(svc, corpus):
    X, g = corpus
    golden = json.loads((FIXTURES / "golden_inference_v1.json").read_text())
    assert golden["fingerprints"]["atlas"] == "09ed804a40836f4a05a91ba10900cded"
    for name, case in golden["cases"].items():
        if case["spectrum_index"] is None:
            continue
        r = svc.infer(_req(g, X[case["spectrum_index"]]))
        assert r.chemistry.predicted_class == case["predicted_class"], name
        assert abs(r.csm.explained_variance - case["csm_explained_variance"]) < 1e-12, name
        for got, want in zip(r.csm.activation, case["csm_activation"]):
            assert abs(got - want) < 1e-12, name
        for hit, want in zip(r.retrieval.top, case["top_molecules"]):
            assert hit.molecule == want["molecule"], name
            assert abs(hit.similarity - want["similarity"]) < 1e-12, name


def test_noise_control_matches_the_recorded_behaviour(svc):
    """Phase 09 audit C5b, frozen as a fixture: noise is NOT flagged by explained variance."""
    golden = json.loads((FIXTURES / "golden_inference_v1.json").read_text())
    case = golden["cases"]["synthetic_noise_control"]
    g = np.asarray(golden["grid"], float)
    y = np.asarray(case["spectrum"], float)
    r = svc.infer(_req(g, y))
    assert abs(r.csm.explained_variance - case["csm_explained_variance"]) < 1e-12
    assert r.csm.explained_variance > 0.50, (
        "white noise reconstructing ABOVE the 0.50 unknown floor is the documented limitation; "
        "if this ever changes the audit text must change with it")
    assert r.confidence.overall < 0.60


# ── reporting ────────────────────────────────────────────────────────────────
def test_report_formats(svc, corpus):
    X, g = corpus
    r = svc.infer(_req(g, X[9], include_reconstruction=True))
    j = render(r, "json")
    assert json.loads(j)["result"]["result_digest"] == r.result_digest
    h = render(r, "html")
    assert "<html" in h and "not a concentration" in h
    p = render(r, "pdf")
    assert p[:5] == b"%PDF-" and len(p) > 50_000


def test_report_is_reproducible(svc, corpus):
    """Identical apart from the generation timestamp, which is isolated in the metadata."""
    X, g = corpus
    r = svc.infer(_req(g, X[9]))
    a, b = json.loads(render(r, "json")), json.loads(render(r, "json"))
    a["metadata"].pop("generated_utc"); b["metadata"].pop("generated_utc")
    assert a == b


def test_report_carries_fingerprints_and_scope(svc, corpus):
    X, g = corpus
    r = svc.infer(_req(g, X[9]))
    d = json.loads(render(r, "json"))
    assert d["metadata"]["atlas_fingerprint"] == r.engine.atlas_fingerprint
    assert d["result"]["engine"]["fingerprints"]["atlas"] == \
        "09ed804a40836f4a05a91ba10900cded"
    assert len(d["scope_and_limitations"]) == len(SCOPE_NOTES)
    assert any("open-set" in s for s in d["scope_and_limitations"])


def test_report_rejects_an_unknown_format(svc, corpus):
    X, g = corpus
    with pytest.raises(ValueError):
        render(svc.infer(_req(g, X[0])), "docx")


def test_interpretation_is_deterministic_and_bounded(svc, corpus):
    X, g = corpus
    r = svc.infer(_req(g, X[30]))
    assert r.interpretation == svc.infer(_req(g, X[30])).interpretation
    assert "reference analogue" in r.interpretation
    for banned in ("diagnos", "disease", "concentration of", "identified as"):
        assert banned not in r.interpretation.lower()
