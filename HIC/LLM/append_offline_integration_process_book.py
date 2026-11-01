"""Append the 2026-10-18 offline regression/integration checkpoint to the Process Book."""

import hashlib
import json
import shutil
from pathlib import Path

from docx import Document
from docx.shared import RGBColor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"C:\Users\admin\Desktop\Research GNN\VoT_HIC_样本与流程记录.docx")
BACKUP = PROJECT_ROOT / "process_book_backup_20261018_offline_integration.docx"
OUTPUT = PROJECT_ROOT / "tmp" / "process_book_offline_integration_updated.docx"
MANIFEST = PROJECT_ROOT / "tmp" / "process_book_offline_integration_manifest.json"


def _snapshot(doc):
    paragraphs = [paragraph.text for paragraph in doc.paragraphs]
    tables = [
        [[cell.text for cell in row.cells] for row in table.rows]
        for table in doc.tables
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
    original_paragraphs, original_tables = _snapshot(document)
    document.add_heading(
        "2026-10-18 Offline regression and graph interface checkpoint", level=2
    )
    paragraphs = [
        "The workspace virtual environment was completed with project-declared dependencies. With TEMP and TMP redirected to the writable project directory, 25 focused unittest modules passed 157 of 157 tests. The live Qdrant integration test was not included in that batch and was checked separately using read-only operations. No DeepSeek request was made.",
        "In offline mode, the cached BAAI/bge-m3 encoder produced a 1,024-dimensional query vector from an existing training summary. A count and top-1 search against the existing hic_bge_m3_train_100 collection returned 100 points and retrieved the same case, socialgood_existing_train_2001-08, with cosine score 1.0. This verifies local embedding/index compatibility only; it does not measure retrieval relevance or forecast accuracy.",
        "The Qdrant check performed no insert, update, delete, or collection change, and did not access test or outer-test data. No new forecast batch or validation optimization was run because the previously approved DeepSeek call budget is consumed. Any new paid evaluation needs a fresh exact run approval.",
        "The production graph design remains open. GraphBundle is the shared, open-vocabulary boundary; the case-memory graph, hierarchy graph, and GraphSAGE forecaster are examples for testing data plumbing, not selected production semantics or models. Vertex identity, edge meaning, topology, features, task, and GNN will be chosen later from the research question and verified data, with adapters added around the stable contract.",
    ]
    for text in paragraphs:
        document.add_paragraph(text)

    heading = document.paragraphs[len(original_paragraphs)]
    for run in heading.runs:
        run.font.color.rgb = RGBColor(0, 0, 0)
    document.save(OUTPUT)

    reopened = Document(OUTPUT)
    new_paragraphs, new_tables = _snapshot(reopened)
    if new_paragraphs[: len(original_paragraphs)] != original_paragraphs:
        raise AssertionError("Original paragraph text changed during append.")
    if new_tables != original_tables:
        raise AssertionError("Original table content changed during append.")
    if len(new_paragraphs) != len(original_paragraphs) + 5:
        raise AssertionError("Unexpected paragraph count after append.")
    if len(new_tables) != len(original_tables):
        raise AssertionError("Unexpected table count after append.")
    if any(text not in new_paragraphs[-4:] for text in paragraphs):
        raise AssertionError("One or more appended paragraphs are missing.")

    manifest = {
        "source": str(SOURCE),
        "backup": str(BACKUP),
        "output": str(OUTPUT),
        "source_sha256": _sha256(BACKUP),
        "output_sha256": _sha256(OUTPUT),
        "source_paragraphs": len(original_paragraphs),
        "output_paragraphs": len(new_paragraphs),
        "source_tables": len(original_tables),
        "output_tables": len(new_tables),
        "original_paragraphs_preserved": True,
        "original_tables_preserved": True,
        "appended_paragraphs": 5,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
