# GAIRA Demo Reasoning v1 — Freeze Manifest

**Date:** 2026-07-15
**Purpose:** certify `gaira_demo_reasoning_v1` as the **frozen, audited reference build**, and prove that **0 v1 files were modified** during v2 development.

---

## Repository state

| Field | Value |
| --- | --- |
| Branch | `migration-safety-gaira-2026-07-05` |
| Commit | `fafc390d0ea2d7fdeeb98c9fddabc09b832b9054` (`fafc390`) |
| v1 absolute path | `/Users/surajpg/projects/GAIRA/gaira_demo_reasoning_v1` |
| v1 tracked files (git) | 16 |
| v1 diff vs HEAD | **empty** (no unstaged, no staged changes) |
| v1 `git status` | clean (no modified, no untracked non-transient files) |

---

## Included files (16) — SHA-256

Checksums are stored in the companion file
[`gaira_demo_reasoning_v1_sha256_2026-07-15.txt`](gaira_demo_reasoning_v1_sha256_2026-07-15.txt)
and were re-verified with `shasum -a 256 -c` at report time — **all 16 OK**.

| File | SHA-256 |
| --- | --- |
| `AUDIT_NEXT_DEMO.md` | `2c94e1723207b8c50d5670114da58f88e521e18a142505f6c4de19288754bbba` |
| `BIOLOGICAL_PILOT_BSV_AUDIT.md` | `e20bcfa78bc08a1af646cb537db78ffc60f1ec5b9f29a16a159239c01c21cbd9` |
| `README.md` | `4f4255dd4711852f95a25ea91f0c289b063227f8ecc2d0505a5dc09405c874d0` |
| `app.py` | `f3a328ad13d552909a5d801b0cf7f82c1fdd7befb4895b6abdb4f8b971b1227f` |
| `gaira_core/__init__.py` | `bbfff7d2be5b5f8b59d56549875badc160d968781798af485f90efc83f439522` |
| `gaira_core/bsv_projection.py` | `1faea5805359e4f5a05115039e2853af6d139df77e782955dfaf0af2d918bf5c` |
| `gaira_core/config.py` | `e599fc86db96c987b5f106e928d75be48d249755bfdf395b513aea2156b0dcc8` |
| `gaira_core/data_loader.py` | `85e94892a53547d272562b6915c0c837b890aee72af18dbe16ad832f325f75a6` |
| `gaira_core/evidence_synthesis.py` | `4fa23ef511c637d0aaabbf8df16cec5ce31b09569ef11acbd111fc85fd723767` |
| `gaira_core/motif_scoring.py` | `46197e15ca5d06b701e099d26f47f5c7b25a5166462c7356927750bdd87269ed` |
| `gaira_core/mss_scoring.py` | `24b3a2f307d96bf985aa89decd9c544ba67b3a0b5ebc4951285650f01b683fb5` |
| `gaira_core/plotting.py` | `62cd613b92c1e9b50d55dc8a5a59dc9804ec92f900c07ea4d79d17f3fc291716` |
| `gaira_core/preprocessing.py` | `d8b5ff4ab84474c9b1a350a54875def32b7e37133da38fdc70e184b83875f5fd` |
| `gaira_core/primitive_extraction.py` | `539f24b1eb320eb4f4d9f99dcef284da3135dd183e149292d8d1438e46a8486e` |
| `gaira_core/report_builder.py` | `0b3bf45506d3aab8e32cfc6e711af8d5a5cef22a9d89b1a405f6271a6e3c7f4f` |
| `gaira_core/substrate_physics.py` | `cc94a5fec3e27b0b1c41c13720b072bc0049e1660b52841f6aa31ba272a75c8b` |

**Total included files: 16** (3 markdown + 1 app.py + 12 `gaira_core/*.py`).

---

## Excluded transient files (12)

Byte-compiled caches are excluded from the freeze (regenerated on any import,
git-ignored, not part of the source of truth):

```
gaira_demo_reasoning_v1/gaira_core/__pycache__/*.cpython-312.pyc   (12 files)
```

No `.DS_Store` present under v1. The `data/{calibration,pilots,grounding,cached}`
subdirectories are empty in v1 (contain no files) and therefore contribute no
checksummed entries.

---

## Comparison against git and pre-copy evidence

- **git:** `git diff HEAD -- gaira_demo_reasoning_v1/` is empty (unstaged and staged); `git status` for v1 is clean. v1 has 16 tracked files, matching the checksummed set exactly.
- **pre-copy evidence:** v2 was created by `rsync -a --exclude=__pycache__` from v1. The scientific engine files copied into v2 are byte-identical to v1 (proven independently by the numerical regression — all outputs identical to floating-point tolerance; see `GAIRA_DEMO_V1_V2_NUMERICAL_REGRESSION_2026-07-15.md`). v2 edits were confined to v2's own tree.
- **post-work re-verification:** `shasum -a 256 -c` against this manifest returned **OK for all 16 files** after all v2 development and after processes that import v1 (regression harness). Only excluded `.pyc` caches may have regenerated.

---

## Certification

> **0 v1 files were modified during v2 development.**
> `gaira_demo_reasoning_v1` is the **frozen, audited reference build** of the GAIRA
> Scientific Reasoning Demo. Any future change to the demo must occur in `v2`
> (or a later version), never in `v1`.

No restoration was required (no v1 file changed).
