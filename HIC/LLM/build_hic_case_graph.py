"""CLI for writing a local graph JSON from completed HIC scored-case JSONL."""

import argparse
import json
import os
import tempfile
from pathlib import Path

from LLM.hic_case_graph import build_case_graph


def _read_records(path):
    records = []
    seen = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on line {line_number}: {error.msg}") from None
        known = record.get("known_at_forecast", {}) if isinstance(record, dict) else {}
        case_id = known.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"line {line_number} does not contain a case_id")
        if case_id in seen:
            raise ValueError(f"duplicate case_id in input: {case_id}")
        seen.add(case_id)
        records.append(record)
    if not records:
        raise ValueError("input JSONL contains no scored HIC records")
    return records


def _write_json(path, value, *, overwrite=False):
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.stem}.",
            suffix=".tmp",
            delete=False,
            newline="\n",
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        if overwrite:
            os.replace(temporary, destination)
        elif os.name == "nt":
            os.rename(temporary, destination)
        else:
            os.link(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="completed scored-case JSONL snapshot")
    parser.add_argument("--query-id", required=True, help="case ID to use as the query node")
    parser.add_argument("--semantic-top-k", type=int, default=10)
    parser.add_argument("--allow-cross-series", action="store_true")
    parser.add_argument("--output", required=True, help="new local graph JSON path")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output")
    args = parser.parse_args(argv)

    records = _read_records(args.input)
    queries = [
        record for record in records
        if record["known_at_forecast"]["case_id"] == args.query_id
    ]
    if len(queries) != 1:
        raise ValueError(f"expected one query case {args.query_id!r}, found {len(queries)}")
    query = queries[0]
    memories = [record for record in records if record is not query]
    graph = build_case_graph(
        memories,
        query,
        semantic_top_k=args.semantic_top_k,
        same_series_only=not args.allow_cross_series,
    )
    _write_json(args.output, graph, overwrite=args.overwrite)
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "query_id": args.query_id,
        "nodes": len(graph["node_ids"]),
        "edges": len(graph["edge_type"]),
        "semantic_top_k": args.semantic_top_k,
        "same_series_only": not args.allow_cross_series,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
