# TODO / noted issues

Out-of-scope issues spotted while working on a task. Each entry: what, where, and when noticed.

- (Phase 1, open, needs a paid F1TV account) Run the FastF1 half of
  `tests/network/test_units.py` (units at Suzuka and Monza, plus the OpenF1-vs-FastF1 axis
  comparison) and record the results in `docs/data.md`. Until then `FASTF1_POS_TO_M` is
  documented but unverified, and the two sources must not be mixed in one track build.
- (Phase 1, open) OpenF1 lacks deleted-lap flags (`Deleted` is missing). Deletions for track
  limits appear in OpenF1's `race_control` messages; parsing them would fill the flag.
- (Phase 1, open) No live FastF1 session has been exported yet, so the FastF1 code paths are
  tested only against fakes built from the installed 3.9.2 API. First real `apex data fetch` may
  surface dtype quirks (e.g. `Deleted` as object, string `nGear`).
- (Phase 1, open) Add a trimmed FastF1 lap fixture once FastF1 data is accessible.
- (Phase 1) pandas 2.3.3 + NumPy 2.5 warn on `pd.Timedelta(seconds=...)`; tests use
  `pd.Timedelta(x, unit="s")`. FastF1 itself may emit this warning on real data.
- (Phase 2, open) Replace Suzuka's assumed 12 m width (ADR 0003) with a measured width (survey
  data, aerial imagery, or a source with real lateral position information).
- (Phase 2, open) Explain the systematic ~0.5 % shortfall of the reconstructed length against
  the official length (Suzuka −0.502 %, Monza −0.544 %; `docs/data.md`). The Phase 2 Suzuka
  length check (±0.5 %) fails by 0.002 percentage points.
- (Phase 2) `apex track build --fixture synthetic_*` (mentioned in CLAUDE.md) is not
  implemented: the synthetic generator lives in `tests/fixtures/`, which `src/` must not import.
