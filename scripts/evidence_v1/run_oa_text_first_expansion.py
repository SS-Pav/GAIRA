from __future__ import annotations

from gaira.evidence_v1.oa_text_first_expansion import run_oa_text_first_expansion


def main() -> None:
    summary = run_oa_text_first_expansion()
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
