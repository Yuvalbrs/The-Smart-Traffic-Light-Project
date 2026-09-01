"""Load the forecaster's training corpus into the database, so training reads from a database.

The forecaster has always trained off 100 CSV files that this project's own simulator wrote. The
database held results and nothing else. This script puts the corpus where the schema always
expected it - ``experiment_run`` -> ``episode`` -> ``observation``, a table that has existed since
T-01-03 and has been empty ever since - so that ``src/ml/lstm_data.py`` can build exactly the same
training tensors from a query instead of from a directory listing.

**Nothing about the model changes.** The point is provenance, not new data: one run row per
ingested corpus carrying the ``data_version`` the checkpoints are pinned to, one episode row per
(scenario, seed), and one observation row per decision step. A number in a report can then be
traced from the checkpoint, to the corpus version, to the episode, to the row.

The corpus is small - 100 files x ~360 rows = ~36k observations, about 2.5 MB as CSV - so this is
a straightforward bulk insert rather than anything clever.

Run::

    python -m scripts.ingest_lstm_corpus                # data/lstm -> data/traffic.db
    python -m scripts.ingest_lstm_corpus --replace      # re-ingest after regenerating the corpus
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.engine import create_db_engine, init_db
from src.db.models import Episode, ExperimentRun, Observation

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DB = _REPO_ROOT / "data" / "traffic.db"
_DEFAULT_CORPUS = _REPO_ROOT / "data" / "lstm"

#: Marks the run rows this script owns, so re-ingesting cannot disturb eval or live runs.
CORPUS_MODE = "corpus"
CORPUS_NAME = "lstm-corpus"

N_MOVEMENTS = 12


def read_corpus_csv(path: Path) -> list[tuple[int, float, list[float], list[float]]]:
    """One CSV -> ``[(step, sim_time, queues[12], counts[12]), ...]``.

    Column order is the T-01-05 header: ``step, sim_time, q_M0..q_M11, c_M0..c_M11``. Parsed
    positionally, exactly as ``src/ml/lstm_data._load_features`` does, so the two paths cannot
    disagree about which column is which.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for line in lines[1:]:  # drop the header
        if not line.strip():
            continue
        cells = [float(c) for c in line.split(",")]
        queues = cells[2 : 2 + N_MOVEMENTS]
        counts = cells[2 + N_MOVEMENTS : 2 + 2 * N_MOVEMENTS]
        rows.append((int(cells[0]), cells[1], queues, counts))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--corpus", type=Path, default=_DEFAULT_CORPUS)
    parser.add_argument("--db", type=Path, default=_DEFAULT_DB)
    parser.add_argument("--replace", action="store_true",
                        help="delete any previously ingested corpus first")
    args = parser.parse_args()

    manifest_path = args.corpus / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"{manifest_path} not found. Generate the corpus first:\n"
            f"  python -m scripts.generate_lstm_data"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # The manifest's `data_version` is PER FILE - one per CSV, derived from that file's own
    # inputs - so there are 100 of them and none is the corpus's identity. The corpus id is the
    # AGGREGATE over every file's content hash, and it is computed by the same function the
    # checkpoint filenames are built from, so the run row records exactly the id the deployed
    # forecaster is pinned to rather than a second opinion about it.
    from scripts.train_lstm import _dataset_data_version

    data_version = _dataset_data_version(args.corpus)

    engine = create_db_engine(args.db)
    init_db(engine)

    with Session(engine) as session:
        existing = session.scalars(
            select(ExperimentRun).where(ExperimentRun.mode == CORPUS_MODE)
        ).all()
        if existing and not args.replace:
            raise SystemExit(
                f"{len(existing)} corpus run(s) already ingested "
                f"(data_version {existing[0].data_version}). Pass --replace to re-ingest."
            )
        for run in existing:
            session.delete(run)  # cascades to episodes and their observations
        session.flush()

        run = ExperimentRun(
            name=CORPUS_NAME,
            mode=CORPUS_MODE,
            controller="webster",  # the corpus is generated under the fixed-time baseline
            config={
                "corpus_dir": str(args.corpus.relative_to(_REPO_ROOT)),
                "n_files": len(manifest),
                "note": "forecaster training corpus; see scripts/generate_lstm_data.py",
            },
            data_version=data_version,
            git_sha=manifest[0].get("generator_git_sha"),
            sumo_version=manifest[0].get("sumo_version"),
        )
        session.add(run)
        session.flush()

        n_obs = 0
        for entry in sorted(manifest, key=lambda e: e["file"]):
            csv_path = args.corpus / entry["file"]
            if not csv_path.exists():
                raise SystemExit(f"{csv_path} is in the manifest but not on disk")
            rows = read_corpus_csv(csv_path)
            if rows and entry.get("n_rows") not in (None, len(rows)):
                raise SystemExit(
                    f"{entry['file']}: manifest says {entry['n_rows']} rows, file has {len(rows)}"
                )

            episode = Episode(
                run_id_fk=run.id,
                index_in_run=len(session.new),  # position is not meaningful here; seed identifies it
                seed=int(entry["seed"]),
                scenario=entry["scenario"],
                # The file's OWN data_version, kept per episode - the run carries the aggregate.
                done_reason=f"corpus:{entry['data_version']}",
            )
            session.add(episode)
            session.flush()

            session.bulk_save_objects([
                Observation(
                    episode_id_fk=episode.id,
                    step=step,
                    sim_time=sim_time,
                    # Stored under the same names the CSV header uses, so the DB-backed loader
                    # reads them positionally in the identical order.
                    state={"q": queues, "c": counts},
                )
                for step, sim_time, queues, counts in rows
            ])
            n_obs += len(rows)

        session.commit()

    print(f"[corpus] {len(manifest)} files, data_version {data_version}")
    print(f"[corpus] wrote 1 run, {len(manifest)} episodes, {n_obs} observations")
    print(f"[corpus] -> {args.db}")


if __name__ == "__main__":
    main()
