# `GAIRA_v7_rebuild/archive/`

Superseded V7 material retained for provenance. **Currently empty.**

## What belongs here

- superseded V7 documents, when a revision materially changes a specification
- abandoned V7 experimental branches worth recording
- earlier drafts of decision rules, kept so that a rule change is visible as a change

## What must not be stored here

- **V5, V6, V6.2, or V6.3 material.** Those live in their own trees and are not V7's to move.
- **Anything still in use.** Archiving is for superseded material only.
- **Raw data.**
- **Large binaries.**

## Why archive rather than delete

Scientific provenance. If a decision rule changed between Phase 02 and Phase 03, the record of
what it used to be — and when it changed — is part of the evidence. Deleting the old version
makes the change invisible, and an invisible rule change is indistinguishable from a post-hoc
one (principle P-12).

## Rules

1. Every archived item carries a note: what superseded it, when, and why.
2. Archived material is never referenced as current.
3. Archiving is not a substitute for git history — it is for material whose *presence* matters
   to a reader, not just to `git log`.
