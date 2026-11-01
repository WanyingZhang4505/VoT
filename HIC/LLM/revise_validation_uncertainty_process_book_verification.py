"""Update the validation-method Process Book entry with final test results."""

from pathlib import Path
import hashlib
import os
import shutil

from docx import Document


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = Path(r"C:\Users\admin\Desktop\Research GNN\VoT_HIC_样本与流程记录.docx")
BACKUP_PATH = WORKSPACE_ROOT / "process_book_backup_20261018_validation_uncertainty_final_suite.docx"
WORKING_PATH = WORKSPACE_ROOT / "tmp" / "validation_uncertainty_process_book.final_suite.tmp.docx"
DESKTOP_TEMP_PATH = DOC_PATH.with_name("VoT_HIC_样本与流程记录.validation_uncertainty.final_suite.tmp.docx")
OLD_PREFIX = "最终验证：新增配对区块 bootstrap/融合报告测试通过。"
UPDATED = (
    "最终验证：新增配对区块 bootstrap/融合报告测试通过。两个相关 unittest 模块 25/25 通过；"
    "将 Python 临时目录设为项目内可写的 tmp 后，活动 tests/ 全套 pytest 为 180 passed、"
    "1 skipped、12 subtests passed。默认 Windows 沙箱 TEMP 不可写导致的 4 项测试已在隔离目录复跑通过。"
    "对仓库根目录运行 pytest 会额外收集 archive/ 下两个历史测试文件，但其实现模块已迁移，"
    "因此全仓库收集阶段有 2 个 ImportError；这不属于活动 tests/ 套件。py_compile 也通过。"
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def table_text(table):
    return [[cell.text for cell in row.cells] for row in table.rows]


def main():
    for path in (BACKUP_PATH, WORKING_PATH, DESKTOP_TEMP_PATH):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing path: {path}")
    source_hash = sha256(DOC_PATH)
    document = Document(DOC_PATH)
    old_paragraphs = [paragraph.text for paragraph in document.paragraphs]
    old_tables = [table_text(table) for table in document.tables]
    matches = [paragraph for paragraph in document.paragraphs if paragraph.text.startswith(OLD_PREFIX)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one validation paragraph, found {len(matches)}")
    target = matches[0]
    target.clear()
    target.add_run(UPDATED)

    WORKING_PATH.parent.mkdir(parents=True, exist_ok=True)
    document.save(WORKING_PATH)
    staged = Document(WORKING_PATH)
    new_paragraphs = [paragraph.text for paragraph in staged.paragraphs]
    changed = [index for index, pair in enumerate(zip(old_paragraphs, new_paragraphs)) if pair[0] != pair[1]]
    if changed != [old_paragraphs.index(next(text for text in old_paragraphs if text.startswith(OLD_PREFIX)))]:
        raise RuntimeError("Unexpected paragraph changes during verification update")
    if [table_text(table) for table in staged.tables] != old_tables:
        raise RuntimeError("Existing table content changed during verification update")
    if UPDATED not in new_paragraphs:
        raise RuntimeError("Updated verification paragraph is missing")
    if sha256(DOC_PATH) != source_hash:
        raise RuntimeError("The Desktop Process Book changed during editing; refusing replacement")

    shutil.copy2(DOC_PATH, BACKUP_PATH)
    shutil.copy2(WORKING_PATH, DESKTOP_TEMP_PATH)
    os.replace(DESKTOP_TEMP_PATH, DOC_PATH)
    final = Document(DOC_PATH)
    if [paragraph.text for paragraph in final.paragraphs] != new_paragraphs:
        raise RuntimeError("Post-replacement paragraph verification failed")
    if [table_text(table) for table in final.tables] != old_tables:
        raise RuntimeError("Post-replacement table verification failed")
    print({
        "backup": str(BACKUP_PATH),
        "paragraphs": len(final.paragraphs),
        "tables": len(final.tables),
        "changed_paragraph_count": len(changed),
        "existing_tables_preserved": True,
        "desktop_sha256_before": source_hash,
        "desktop_sha256_after": sha256(DOC_PATH),
    })


if __name__ == "__main__":
    main()
