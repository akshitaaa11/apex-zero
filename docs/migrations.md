# Migrations

Every change to a versioned format (`TrackAsset`, `Ghost`, `CarParams`, replay bundle
`REPLAY_FORMAT_VERSION`, cache layout/keys) gets an entry here: the version bump, what changed,
why, and how to migrate existing data (or that it must be regenerated).

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
