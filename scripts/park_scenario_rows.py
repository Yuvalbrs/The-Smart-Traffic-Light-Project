"""Park (or restore) SCN-03's campaign rows so the scene can be demonstrated from empty.

SCN-03 is one of the five CONFIRMATORY scenarios. Its rows are pre-registered evidence, so they
are exported in full before anything is deleted and can be put back byte-for-byte afterwards.

    python -m scripts.park_scenario_rows park      -> export to a JSON file, then delete
    python -m scripts.park_scenario_rows restore   -> re-insert exactly what was exported
"""
import json
import sqlite3
import sys
from pathlib import Path

# Resolved from this file, never hardcoded. The absolute dev-machine paths that used to sit here
# crashed the script on every other computer - a poor look in a repository whose central claim is
# that its results reproduce somewhere other than the machine that produced them.
_REPO_ROOT = Path(__file__).resolve().parent.parent
DB = _REPO_ROOT / "data" / "traffic.db"
PARK = _REPO_ROOT / "data" / "scn03_parked.json"
SCEN = "SCN-03"

TABLES = ("experiment_run", "episode", "episode_kpi", "observation")


def cols(c, t):
    return [r[1] for r in c.execute("PRAGMA table_info(%s)" % t)]


def park():
    c = sqlite3.connect(DB)
    run_ids = [r[0] for r in c.execute(
        "select distinct r.id from experiment_run r join episode e on e.run_id_fk=r.id "
        "where e.scenario=? and r.mode='eval'", (SCEN,))]
    if not run_ids:
        print("nothing to park - %s already has no eval rows" % SCEN)
        return
    q = ",".join("?" * len(run_ids))
    ep_ids = [r[0] for r in c.execute(
        "select id from episode where run_id_fk in (%s) and scenario=?" % q, run_ids + [SCEN])]

    dump = {"scenario": SCEN, "tables": {}}
    dump["tables"]["experiment_run"] = [
        dict(zip(cols(c, "experiment_run"), row))
        for row in c.execute("select * from experiment_run where id in (%s)" % q, run_ids)]
    if ep_ids:
        eq = ",".join("?" * len(ep_ids))
        dump["tables"]["episode"] = [
            dict(zip(cols(c, "episode"), row))
            for row in c.execute("select * from episode where id in (%s)" % eq, ep_ids)]
        dump["tables"]["episode_kpi"] = [
            dict(zip(cols(c, "episode_kpi"), row))
            for row in c.execute(
                "select * from episode_kpi where episode_id_fk in (%s)" % eq, ep_ids)]
        dump["tables"]["observation"] = [
            dict(zip(cols(c, "observation"), row))
            for row in c.execute(
                "select * from observation where episode_id_fk in (%s)" % eq, ep_ids)]
    else:
        for t in ("episode", "episode_kpi", "observation"):
            dump["tables"][t] = []

    PARK.write_text(json.dumps(dump), encoding="utf-8")
    print("exported to %s" % PARK.name)
    for t in TABLES:
        print("   %-16s %d rows" % (t, len(dump["tables"][t])))

    if ep_ids:
        eq = ",".join("?" * len(ep_ids))
        c.execute("delete from observation where episode_id_fk in (%s)" % eq, ep_ids)
        c.execute("delete from episode_kpi where episode_id_fk in (%s)" % eq, ep_ids)
        c.execute("delete from episode where id in (%s)" % eq, ep_ids)
    # NOT the experiment_run rows. One run holds episodes for EVERY scenario in the campaign,
    # so deleting the parents orphans the other four scenarios' episodes and they vanish from
    # every query that joins through the run. That is exactly what happened the first time this
    # ran: SCN-05 dropped from 246 episodes to 6.
    c.commit()
    left = c.execute(
        "select count(*) from episode e join experiment_run r on e.run_id_fk=r.id "
        "where e.scenario=? and r.mode='eval'", (SCEN,)).fetchone()[0]
    print("deleted. %s eval episodes remaining: %d" % (SCEN, left))


def restore():
    if not PARK.exists():
        raise SystemExit("no parked file at %s" % PARK)
    dump = json.loads(PARK.read_text(encoding="utf-8"))
    c = sqlite3.connect(DB)
    total = 0
    # experiment_run is included for the case where a run really did belong to this scenario
    # alone; "insert or replace" makes re-inserting a row that still exists harmless.
    for t in TABLES:                      # parents before children
        rows = dump["tables"].get(t) or []
        for row in rows:
            keys = list(row)
            c.execute(
                "insert or replace into %s (%s) values (%s)"
                % (t, ",".join('"%s"' % k for k in keys), ",".join("?" * len(keys))),
                [row[k] for k in keys])
        total += len(rows)
        print("   %-16s %d rows restored" % (t, len(rows)))
    c.commit()
    n = c.execute(
        "select count(*) from episode e join experiment_run r on e.run_id_fk=r.id "
        "where e.scenario=? and r.mode='eval'", (SCEN,)).fetchone()[0]
    print("restored %d rows. %s eval episodes now: %d" % (total, SCEN, n))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "park":
        park()
    elif mode == "restore":
        restore()
    else:
        raise SystemExit(__doc__)
