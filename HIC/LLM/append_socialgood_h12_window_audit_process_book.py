"""Append the SocialGood H12 per-window error audit to the Process Book."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor


DOC_PATH = Path(r"C:\Users\admin\Desktop\Research GNN\VoT_HIC_样本与流程记录.docx")
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
BACKUP_PATH = WORKSPACE_ROOT / "process_book_backup_20261018_socialgood_h12_window_audit.docx"
TEMP_PATH = DOC_PATH.with_name("VoT_HIC_样本与流程记录.socialgood_h12_window_audit.tmp.docx")
SECTION_TITLE = "2026 10 18 SocialGood H12 per window audit"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def table_text(document):
    return [[[cell.text for cell in row.cells] for row in table.rows] for table in document.tables]


def set_run(run, *, size=9.5, bold=False):
    run.font.name = "Arial"
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(size)
    run.bold = bold
    run.font.color.rgb = RGBColor.from_string("1F1F1F")


def append_paragraph(document, text):
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(5)
    paragraph.paragraph_format.line_spacing = 1.0
    set_run(paragraph.add_run(text))


def main():
    if not DOC_PATH.is_file():
        raise FileNotFoundError(DOC_PATH)
    for path in (BACKUP_PATH, TEMP_PATH):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing path: {path}")

    source_hash = sha256(DOC_PATH)
    document = Document(DOC_PATH)
    old_paragraphs = [p.text for p in document.paragraphs]
    old_tables = table_text(document)
    if any(p.text.strip() == SECTION_TITLE for p in document.paragraphs):
        raise RuntimeError(f"Section already exists: {SECTION_TITLE}")

    heading = document.add_paragraph(style="Heading 1")
    heading.paragraph_format.keep_with_next = True
    heading.paragraph_format.space_before = Pt(10)
    heading.paragraph_format.space_after = Pt(6)
    set_run(heading.add_run(SECTION_TITLE), size=14, bold=True)
    append_paragraph(
        document,
        "对同一 11 个 SocialGood H12 validation window 做逐窗复核后，K=1、K=10、K=16 各自在 11 个窗口中的 5 个窗口上低于 K=0 的 MAE，在 4 个窗口上低于 K=0 的 RMSE。这个逐窗结果与整体 MAE/RMSE 的排名不同，说明不同时间段的检索迁移效果不稳定；它不是额外的独立测试集。",
    )
    append_paragraph(
        document,
        "2001 validation window 中，K=0 的 MAE 为 0.3667，而 K=1、K=10、K=16 分别为 1.0583、1.0500、1.1250；三个检索臂都取回 1998 validation case。此观察能指出一个值得检查的错配，但单凭这个窗口不能确定错因，也不能据此断言文本相似度造成误差。",
    )
    append_paragraph(
        document,
        "下一步可先离线比较已保存的文本向量相似度与数值轨迹相似度，并检查错误窗口的检索案例和预测差异，再决定是否值得为替代检索或 prompt 方案申请新的 API 预算。本次逐窗审计没有新 API 调用、Qdrant 写入或 outer-test 访问。",
    )

    if sha256(DOC_PATH) != source_hash:
        raise RuntimeError("Process Book changed during edit; refusing to install")
    shutil.copy2(DOC_PATH, BACKUP_PATH)
    document.save(TEMP_PATH)
    staged = Document(TEMP_PATH)
    if [p.text for p in staged.paragraphs[:len(old_paragraphs)]] != old_paragraphs:
        raise RuntimeError("Existing paragraphs changed in staged Process Book")
    if table_text(staged)[:len(old_tables)] != old_tables:
        raise RuntimeError("Existing tables changed in staged Process Book")
    if not any(p.text.strip() == SECTION_TITLE for p in staged.paragraphs):
        raise RuntimeError("New per-window section missing from staged Process Book")

    os.replace(TEMP_PATH, DOC_PATH)
    installed = Document(DOC_PATH)
    if [p.text for p in installed.paragraphs[:len(old_paragraphs)]] != old_paragraphs:
        raise RuntimeError("Installed Process Book did not preserve old paragraphs")
    if table_text(installed)[:len(old_tables)] != old_tables:
        raise RuntimeError("Installed Process Book did not preserve old tables")
    print({
        "path": str(DOC_PATH),
        "backup": str(BACKUP_PATH),
        "paragraphs_before": len(old_paragraphs),
        "paragraphs_after": len(installed.paragraphs),
        "tables_before": len(old_tables),
        "tables_after": len(installed.tables),
        "sha256": sha256(DOC_PATH),
        "section": SECTION_TITLE,
    })


if __name__ == "__main__":
    main()
