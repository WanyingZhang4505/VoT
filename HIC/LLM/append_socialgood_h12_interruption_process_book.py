"""Append the interrupted SocialGood H12 run and approved resume budget."""

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
BACKUP_PATH = WORKSPACE_ROOT / "process_book_backup_20261018_socialgood_h12_interrupted.docx"
TEMP_PATH = DOC_PATH.with_name("VoT_HIC_样本与流程记录.socialgood_h12_interrupted.tmp.docx")
SECTION_TITLE = "2026 10 18 SocialGood H12 interrupted run and resume authorization"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def table_text(document):
    return [[cell.text for cell in row.cells] for table in document.tables for row in table.rows]


def add_run(paragraph, text, *, size=9.5, bold=False):
    run = paragraph.add_run(text)
    run.font.name = "Arial"
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(size)
    run.bold = bold
    run.font.color.rgb = RGBColor.from_string("1F1F1F")
    return run


def append_body(document, text):
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(5)
    paragraph.paragraph_format.line_spacing = 1.0
    add_run(paragraph, text)


def main():
    if not DOC_PATH.is_file():
        raise FileNotFoundError(DOC_PATH)
    for path in (BACKUP_PATH, TEMP_PATH):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing path: {path}")

    source_hash = sha256(DOC_PATH)
    source = Document(DOC_PATH)
    original_paragraphs = [p.text for p in source.paragraphs]
    original_tables = table_text(source)
    if any(p.text.strip() == SECTION_TITLE for p in source.paragraphs):
        raise RuntimeError(f"Section already exists: {SECTION_TITLE}")

    heading = source.add_paragraph(style="Heading 1")
    heading.paragraph_format.keep_with_next = True
    heading.paragraph_format.space_before = Pt(10)
    heading.paragraph_format.space_after = Pt(6)
    add_run(heading, SECTION_TITLE, size=14, bold=True)
    append_body(
        source,
        "本次 SocialGood H12 运行使用 16 个 fit case、11 个按时间排序的 validation window，比较 K=0/1/10/16。原计划需要 136 次逻辑调用、最多 272 次 HTTP 尝试。第一次请求在本机 DNS 阶段失败，没有到达 DeepSeek；该物理尝试仍保留在 HTTP ledger 中，逻辑调用计数已在备份后修正。",
    )
    append_body(
        source,
        "同一 run ID 随后完成了前两个训练记忆。运行中断时，HTTP ledger 为 8/272，训练 checkpoint 记录 7 次逻辑调用；第三个 case 的 summary 请求已返回，但 Windows 在 os.replace checkpoint 时抛出 PermissionError，响应未能保存，因此该 summary 仍待重请求。此时 K=16 集合已有 2 条；K=1 和 K=10 要等训练集构建完成后再从 K=16 按前缀播种。尚无 validation forecast 或指标报告，outer-test 未读取。",
    )
    append_body(
        source,
        "为降低瞬时文件锁导致的再次丢失，LLM/native_hic_pilot.py 的 _atomic_json() 现在对 PermissionError 最多重试五次；若仍失败，会保留完整临时 checkpoint 并报告其路径。新增的替换锁测试已通过。用户已授权在原 272 次 HTTP 上限内，将本次逻辑调用上限提高到 137，以补做未保存的 summary 并继续同一个 run。后续仍只在 validation 上比较方案，outer-test 继续封存。",
    )

    if sha256(DOC_PATH) != source_hash:
        raise RuntimeError("Process Book changed during the edit; refusing to install")
    shutil.copy2(DOC_PATH, BACKUP_PATH)
    source.save(TEMP_PATH)

    staged = Document(TEMP_PATH)
    if [p.text for p in staged.paragraphs[:len(original_paragraphs)]] != original_paragraphs:
        raise RuntimeError("Existing paragraphs changed in staged Process Book")
    if table_text(staged)[:len(original_tables)] != original_tables:
        raise RuntimeError("Existing tables changed in staged Process Book")
    if not any(p.text.strip() == SECTION_TITLE for p in staged.paragraphs):
        raise RuntimeError("New section missing from staged Process Book")

    os.replace(TEMP_PATH, DOC_PATH)
    installed = Document(DOC_PATH)
    if [p.text for p in installed.paragraphs[:len(original_paragraphs)]] != original_paragraphs:
        raise RuntimeError("Installed Process Book did not preserve existing paragraphs")
    if table_text(installed)[:len(original_tables)] != original_tables:
        raise RuntimeError("Installed Process Book did not preserve existing tables")
    print({
        "path": str(DOC_PATH),
        "backup": str(BACKUP_PATH),
        "paragraphs_before": len(original_paragraphs),
        "paragraphs_after": len(installed.paragraphs),
        "tables_before": len(original_tables),
        "tables_after": len(installed.tables),
        "sha256": sha256(DOC_PATH),
        "section": SECTION_TITLE,
    })


if __name__ == "__main__":
    main()
