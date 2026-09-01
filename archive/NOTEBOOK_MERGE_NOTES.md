# Notebook provenance — how `cot_em_analysis_full.ipynb` was assembled

Written 2026-09-01, during the split into subprojects.

The project's notebook existed in two diverged copies:

| copy | cells | had | lacked |
|---|---|---|---|
| repo `cot_em_analysis.ipynb` (now archived here) | 80 | the 2026-08-26/30 patches: preflight cell 2, the `\n\n</think>` whitespace fix and SUPERSEDED banner in cells 53–56/58 | **all experiment cells after §18i** |
| Drive `cot_em_analysis.ipynb` (id `1tHZJFF_y5J-HswDa1sfYKGQPcIb_uuCs`, exported 2026-09-01) | 109 | cells 80–108: 18k/18l/18m/18o/18o(b)/18l(b)/18p/18p(b)/18r/18t and the §18v S0–S6 session cells, 25 of 29 with outputs | the repo-side patches (its cells 2, 53–56, 58 are the pre-patch versions) |

The patches went into git; the new experiments went into Drive. Neither copy was
complete.

**`cot_em_analysis_full.ipynb` is the canonical merge**: repo cells 0–79
(patched versions) + Drive cells 80–105, 107, 108 appended as canonical indices
80–107. Drive cell 106 was dropped (byte-identical duplicate of repo cell 58).
Drive's older cells 2 and 53–56/58 were dropped in favour of the repo's patched
versions. One mechanical fix: Colab writes an illegal `metadata` key into stream
outputs; it was stripped so the file validates under nbformat v4.

Because canonical indices 0–79 equal the old repo notebook's indices, every
cell number cited in `narrative_master.md` and `REBUILD_RUNBOOK.md` still
resolves against the full notebook.

Note: the Drive copy's cell 2 records a security incident — an earlier version
of the preflight cell ended with a bare `preflight()` call, whose return value
(the verified HF token) was written in plaintext into the saved notebook and
auto-saved to Drive. That token was revoked and rotated at the time
(2026-08-26). The 2026-09-01 export was scanned for token-shaped strings before
being committed: none found.

The subproject notebooks (`00_foundation/pipeline.ipynb`,
`01_cot_monitoring/monitoring.ipynb`, `02_cot_swapping/swapping.ipynb`,
`03_linear_probe/linear_probe.ipynb`) are verbatim subsets of the canonical merge,
each with one added markdown header cell. Coverage was verified: every canonical
cell appears in exactly one subproject notebook.
