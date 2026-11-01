"""Append the cutoff-safe native HIC dry-run plan to the Process Book."""

from pathlib import Path
import os
import shutil

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

from LLM.append_qdrant_integrity_process_book import (
    DOC_PATH,
    append_paragraph,
    set_borders,
    set_cell,
    set_run,
    sha256,
    table_text,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
BACKUP_PATH = WORKSPACE_ROOT / "process_book_backup_20261018_native_cutoff_plan.docx"
WORKING_PATH = WORKSPACE_ROOT / "tmp" / "native_cutoff_plan_process_book.tmp.docx"
DESKTOP_TEMP_PATH = DOC_PATH.with_name("VoT_HIC_样本与流程记录.native_cutoff.tmp.docx")
SECTION_TITLE = "2026 10 18 native HIC outer-test 延后读取与 SocialGood H12 dry-run"
ROWS = [
    ("阶段", "规模 / 配置", "说明"),
    ("Fit memory", "16 cases / 48 calls", "seed 20261018；每例 summary、forecast、correction"),
    ("Validation", "11 windows / 88 calls", "按时间顺序；K=0、1、10、16；真值仅在预测后评分/纠错"),
    ("总计划", "136 logical calls", "现有模板；0 template call；HTTP 上限按每逻辑调用2次为272"),
    ("Outer test", "deferred / null", "不构造窗口；不输出 ID、样本数或调用估计"),
]


def main():
    if not DOC_PATH.is_file():
        raise FileNotFoundError(DOC_PATH)
    for path in (BACKUP_PATH, WORKING_PATH, DESKTOP_TEMP_PATH):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing path: {path}")

    WORKING_PATH.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(DOC_PATH)
    document = Document(DOC_PATH)
    if any(paragraph.text.strip() == SECTION_TITLE for paragraph in document.paragraphs):
        raise RuntimeError(f"Section already exists: {SECTION_TITLE}")
    original_paragraphs = [paragraph.text for paragraph in document.paragraphs]
    original_tables = [table_text(table) for table in document.tables]

    document.add_page_break()
    heading = document.add_paragraph(style="Heading 1")
    heading.paragraph_format.space_after = Pt(6)
    set_run(heading.add_run(SECTION_TITLE), size=15, bold=True, color="1F1F1F")
    append_paragraph(
        document,
        "发现并修正 native HIC 的开发期读取边界：旧路径先完整解析 CSV，因而会把 outer-test 的 OT 数值载入内存。新路径只用日期列确定时间切分，再调用带 validation_end 截止的行解析器；截止后的 OT 在数值转换前被跳过。计划和执行准备只获得 fit/validation 数据，outer-test window、truth、ID、数量与调用预算均延后到方案冻结后。",
    )
    append_paragraph(
        document,
        "回归测试将 outer-test 期间的一个 OT 替换成非数字 sentinel。安全模式仍成功构造 fit/validation，且 invalid-target 计数不增加、test_windows 为空、outer_test_start 为 null。这表明 sentinel 没有进入目标值转换路径；用于最终评估的完整 split 接口仍保留，但当前不调用。",
    )
    table = document.add_table(rows=1, cols=3)
    table.style = "Table Grid"
    table.autofit = False
    widths = (1.0, 1.55, 4.0)
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    table.rows[0]._tr.get_or_add_trPr().append(repeat)
    for column, value in enumerate(ROWS[0]):
        set_cell(table.rows[0].cells[column], value, header=True)
    for row_index, values in enumerate(ROWS[1:]):
        cells = table.add_row().cells
        for column, value in enumerate(values):
            set_cell(cells[column], value)
            if row_index % 2:
                shade = OxmlElement("w:shd")
                shade.set(qn("w:fill"), "EAF2F8")
                cells[column]._tc.get_or_add_tcPr().append(shade)
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            cell.width = Inches(width)
    set_borders(table)
    append_paragraph(
        document,
        "SocialGood H12 的无副作用 dry-run 共有 16 个 fit case、11 个 chronological validation window，使用已有 template 和 K={1,10,16}，并保留 K=0 对照；估算 48 次训练建库调用 + 88 次验证调用 = 136 次逻辑调用，若每次最多两次 HTTP 尝试，则预算上限为 272。此处只是预算计划，未调用 API、未写 Qdrant、未读取 outer-test 目标。",
    )
    append_paragraph(
        document,
        "Graph ML 仍不预设 vertex/edge schema、拓扑、预测任务或 GNN 家族。之后只需把研究组最终确认的图结构和模型接到通用 GraphBundle adapter；当前 HIC 数据审计不替团队作这些研究决定。",
        size=9,
        after=3,
    )

    document.save(WORKING_PATH)
    check = Document(WORKING_PATH)
    if [p.text for p in check.paragraphs[: len(original_paragraphs)]] != original_paragraphs:
        raise RuntimeError("Existing paragraphs changed during append.")
    if [table_text(t) for t in check.tables[: len(original_tables)]] != original_tables:
        raise RuntimeError("Existing tables changed during append.")
    if not any(p.text.strip() == SECTION_TITLE for p in check.paragraphs):
        raise RuntimeError("New section heading is missing.")
    if table_text(check.tables[-1]) != [list(row) for row in ROWS]:
        raise RuntimeError("New plan table does not match expected rows.")
    if sha256(DOC_PATH) != source_hash:
        raise RuntimeError("The source document changed during editing; refusing replacement.")

    shutil.copy2(DOC_PATH, BACKUP_PATH)
    shutil.copy2(WORKING_PATH, DESKTOP_TEMP_PATH)
    os.replace(DESKTOP_TEMP_PATH, DOC_PATH)
    final = Document(DOC_PATH)
    if [p.text for p in final.paragraphs[: len(original_paragraphs)]] != original_paragraphs:
        raise RuntimeError("Post-replacement paragraph preservation check failed.")
    if [table_text(t) for t in final.tables[: len(original_tables)]] != original_tables:
        raise RuntimeError("Post-replacement table preservation check failed.")
    if table_text(final.tables[-1]) != [list(row) for row in ROWS]:
        raise RuntimeError("Post-replacement plan table verification failed.")
    print({
        "backup": str(BACKUP_PATH),
        "paragraphs_before": len(original_paragraphs),
        "paragraphs_after": len(final.paragraphs),
        "tables_before": len(original_tables),
        "tables_after": len(final.tables),
        "existing_content_preserved": True,
        "sha256_before": source_hash,
        "sha256_after": sha256(DOC_PATH),
    })


if __name__ == "__main__":
    main()
