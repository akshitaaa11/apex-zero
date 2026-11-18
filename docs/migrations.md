# Migrations

Every change to a versioned format (`TrackAsset`, `Ghost`, `CarParams`, replay bundle
`REPLAY_FORMAT_VERSION`, cache layout/keys) gets an entry here: the version bump, what changed,
why, and how to migrate existing data (or that it must be regenerated).

## Cache layout v1 (Phase 1, P1.1), new

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
