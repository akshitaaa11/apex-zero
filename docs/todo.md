# TODO / noted issues

Out-of-scope issues spotted while working on a task. Each entry: what, where, and when noticed.

- (Phase 1, open) Run the FastF1 half of `tests/network/test_units.py` (units at Suzuka and
  Monza, plus the OpenF1-vs-FastF1 axis comparison) with an F1 account, record the results in
  `docs/data.md`, then tick P1.4. Until then `FASTF1_POS_TO_M` is documented but unverified.
- (Phase 1, open) No live FastF1 session has been exported yet, so the FastF1 code paths are
  tested only against fakes built from the installed 3.9.2 API. First real `apex data fetch` may
  surface dtype quirks (e.g. `Deleted` as object, string `nGear`).
- (Phase 1, open) Add a trimmed FastF1 lap fixture once FastF1 data is accessible.
- (Phase 1) pandas 2.3.3 + NumPy 2.5 warn on `pd.Timedelta(seconds=...)`; tests use
  `pd.Timedelta(x, unit="s")`. FastF1 itself may emit this warning on real data.
