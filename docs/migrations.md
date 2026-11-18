# Migrations

Every change to a versioned format (`TrackAsset`, `Ghost`, `CarParams`, replay bundle
`REPLAY_FORMAT_VERSION`, cache layout/keys) gets an entry here: the version bump, what changed,
why, and how to migrate existing data (or that it must be regenerated).

## TrackAsset v1 (Phase 2, P2.1), new

`track.asset.ASSET_VERSION = 1`. First version; nothing to migrate. Written to
`APEX_PROCESSED_DIR/tracks/<name>/`: `asset.npz` (float64 channels `s, x, y, z, psi, kappa,
grade, bank, w_left, w_right` on a uniform ~1 m grid, `s = 0` at the timing line) and
`meta.json` (`name, length, n_points, source_sessions, build_hash, params, report,
asset_sha256`). `report` holds the build's validation results, including `width_source` and
`bank_source` (ADR 0003). `track.glb` (glTF 2.0, y-up) sits next to them. `TrackAsset.load`
refuses another version or a checksum mismatch.

## Cache layout v2 (Phase 1, OpenF1 as primary source, ADR 0002)

`CACHE_LAYOUT_VERSION = 2`. Migration: none needed by hand. A v1 session no longer counts as
cached (`is_complete` checks the version) and is re-fetched on the next `apex data fetch`.

- `session_meta.json`: `round_number` may be `null` (OpenF1 publishes no round numbers);
  `source` may be `"openf1"` (`source_version` `"api-v1"`).
- `aliases.json` keys gain the source: `"<source>|<year>|<event query>|<session>"`. Old keys are
  ignored.
- `POSITION.Status` is exempt from the NaN budget (absent throughout for OpenF1).
- Laps whose lap window holds no position or car samples are rejected (listed in
  `rejected_laps.json`) instead of being written as empty files.
- For OpenF1 exports, `SessionTime` counts from the session's scheduled `date_start`; see
  `data/openf1_loader.py` for how lap tables are derived.

## Cache layout v1 (Phase 1, P1.1), superseded by v2

`data.cache.CACHE_LAYOUT_VERSION = 1`. First version; nothing to migrate.

- `APEX_CACHE_DIR/fastf1/`: FastF1's own cache. `APEX_CACHE_DIR/openf1/<h[:2]>/<h>.json`: OpenF1
  responses keyed by sha256 of the canonical request URL.
- `APEX_PROCESSED_DIR/<year>/<event_slug>/<session>/`: `laps.parquet`,
  `<DRIVER>/pos_lap<NNN>.parquet`, `<DRIVER>/car_lap<NNN>.parquet`, optional
  `rejected_laps.json`, and `session_meta.json` (written last; lists every file with its sha256).
- `APEX_PROCESSED_DIR/aliases.json`: `"<year>|<event query>|<session>"` → session directory, so a
  repeated fetch needs no schedule lookup.
- Table contracts (SI units): `data.schemas.POSITION`, `CAR`, `LAPS`.

A session whose `layout_version` differs from the current one counts as not cached and is
re-fetched.
