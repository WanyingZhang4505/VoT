"""Append the sealed outer-test preflight checkpoint to the Process Book."""

import hashlib
import json
import shutil
from pathlib import Path

from docx import Document
from docx.shared import RGBColor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"C:\Users\admin\Desktop\Research GNN\VoT_HIC_样本与流程记录.docx")
BACKUP = PROJECT_ROOT / "process_book_backup_20261018_sealed_preflight.docx"
OUTPUT = PROJECT_ROOT / "tmp" / "process_book_sealed_preflight_updated.docx"
MANIFEST = PROJECT_ROOT / "tmp" / "process_book_sealed_preflight_manifest.json"


def _snapshot(document):
    paragraphs = [paragraph.text for paragraph in document.paragraphs]
    tables = [
        [[cell.text for cell in row.cells] for row in table.rows]
        for table in document.tables
    ]
    return paragraphs, tables


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    if BACKUP.exists() or OUTPUT.exists() or MANIFEST.exists():
        raise FileExistsError("Refusing to overwrite an existing backup/output/manifest.")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE, BACKUP)

    document = Document(SOURCE)
    old_paragraphs, old_tables = _snapshot(document)
    document.add_heading("2026-10-18 Sealed outer-test preflight", level=2)
    additions = [
        "The original monthly dry-run used the full split loader, which parsed all OT values and built outer-test truth arrays even though the manifest did not display them. That was not strict enough for the sealed-test rule, so I did not run that preflight before correcting it.",
        "The new include_outer_test_targets=False path counts CSV records to establish split boundaries, then parses only the fit and validation prefix. It does not convert outer-test OT strings into numeric inputs, construct outer-test windows, or emit outer-test IDs, truths, counts, or call estimates. A regression test places nonnumeric sentinel values in the outer-test OT column and verifies the development-only loader still succeeds without returning test windows.",
        "The safe preflight covers the five datasets supported by the current monthly 8-history/12-target adapter: Agriculture 294 fit memories and 6 validation windows; Climate 298 and 6; Economy_sort 251 and 5; Security 171 and 3; SocialGood 198 and 11. Total: 1,212 fit memories, 31 validation windows, and 3,888 estimated logical calls through validation, including four missing or invalid template-generation calls. This is not the entire VoT corpus and is not a forecast-accuracy result.",
        "Verification: py_compile passed for the loader, planner, and tests; focused monthly data/planner tests passed 9/9; the 25-module offline regression suite passed 158/158 in a fresh workspace temp directory. An earlier full-suite attempt hit two Windows PermissionErrors while replacing temporary checkpoints in a reused temp directory; the affected HIC memory-builder tests passed in a fresh directory before the full rerun.",
        "No DeepSeek request or Qdrant write occurred, and the prior API call authorization is consumed. Outer-test work remains deferred until a validation configuration is frozen and a new exact run is approved. The production graph model, vertex/edge semantics, and topology remain open; GraphBundle and separate adapters are the intended plug-in boundary.",
    ]
    for text in additions:
        document.add_paragraph(text)

    heading = document.paragraphs[len(old_paragraphs)]
    for run in heading.runs:
        run.font.color.rgb = RGBColor(0, 0, 0)
    document.save(OUTPUT)

    updated = Document(OUTPUT)
    new_paragraphs, new_tables = _snapshot(updated)
    if new_paragraphs[: len(old_paragraphs)] != old_paragraphs:
        raise AssertionError("Original paragraph text changed during append.")
    if new_tables != old_tables:
        raise AssertionError("Original table content changed during append.")
    if len(new_paragraphs) != len(old_paragraphs) + 6:
        raise AssertionError("Unexpected paragraph count after append.")
    if len(new_tables) != len(old_tables):
        raise AssertionError("Unexpected table count after append.")
    if new_paragraphs[-5:] != additions:
        raise AssertionError("One or more appended paragraphs are missing.")

    manifest = {
        "source": str(SOURCE),
        "backup": str(BACKUP),
        "output": str(OUTPUT),
        "source_sha256": _sha256(BACKUP),
        "output_sha256": _sha256(OUTPUT),
        "source_paragraphs": len(old_paragraphs),
        "output_paragraphs": len(new_paragraphs),
        "source_tables": len(old_tables),
        "output_tables": len(new_tables),
        "original_paragraphs_preserved": True,
        "original_tables_preserved": True,
        "appended_paragraphs": 6,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
