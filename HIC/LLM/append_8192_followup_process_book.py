"""Append the bounded 8,192-token continuation result to the process book."""

from pathlib import Path
import shutil

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = ROOT / "VoT_HIC_样本与流程记录_100case_stage.docx"
BACKUP_PATH = ROOT / "_research_gnn_working" / "process_book_backup_20261018_8192_followup.docx"
SECTION_TITLE = "2026-10-18 长上下文 cap8192 续跑记录"


def _run_style(run, *, size=9, bold=False, color=None):
    run.font.name = "Arial"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(size)
    run.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def _cell_border(cell):
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = qn(f"w:{edge}")
        element = borders.find(tag)
        if element is None:
            element = OxmlElement(f"w:{edge}")
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), "6")
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), "D9D9D9")


def _set_cell(cell, text, *, header=False):
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_after = Pt(2)
    _run_style(paragraph.add_run(text), size=9 if header else 8.5,
               bold=header, color="FFFFFF" if header else None)
    if header:
        tc_pr = cell._tc.get_or_add_tcPr()
        shade = OxmlElement("w:shd")
        shade.set(qn("w:fill"), "1F4E78")
        tc_pr.append(shade)
    _cell_border(cell)


def _repeat_header(row):
    row_properties = row._tr.get_or_add_trPr()
    marker = OxmlElement("w:tblHeader")
    marker.set(qn("w:val"), "true")
    row_properties.append(marker)


def main():
    if not DOC_PATH.is_file():
        raise FileNotFoundError(DOC_PATH)
    if BACKUP_PATH.exists():
        raise FileExistsError(BACKUP_PATH)
    document = Document(DOC_PATH)
    if any(p.text.strip() == SECTION_TITLE for p in document.paragraphs):
        raise RuntimeError("The cap8192 follow-up section already exists.")
    old_paragraphs = len(document.paragraphs)
    old_tables = len(document.tables)
    shutil.copy2(DOC_PATH, BACKUP_PATH)

    document.add_page_break()
    heading = document.add_paragraph(style="Heading 1")
    heading.paragraph_format.space_after = Pt(6)
    _run_style(heading.add_run(SECTION_TITLE), size=15, bold=True, color="1F1F1F")

    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(6)
    _run_style(paragraph.add_run(
        "按 bounded-search 原则补查 8,192-token：保留已有结果；当继续计算的成本明显超过预期且没有进度时停止，不强行追求每一轮都运行到底。"
    ), size=10)

    data = [
        ("尝试", "设置与观察", "结果"),
        ("cap8192-only 续跑", "从旧三 cap 检查点迁移已完成的 28 个网格和 cap8192 缓存，继续 `new_metr::48`。程序报告需编码 202 条新文本。", "在首格完成前停止；没有新网格分数或新缓存写入。检查点保留 28 结果 + 4 skips。"),
        ("2048+8192 配对续跑", "修正续跑设置，同时载入 2,048 与 8,192 缓存，以便 ≤2,048 的输入复用向量。token 审计显示 H48 的 202 条唯一文本全都超过 2,048（p50 3,555、p90 4,552、最大 5,681）。", "仍需重新编码 202 条长文本；约 70 秒未到首个 40 条进度回报，遂在计算网格分数前停止。检查点保留 28 结果 + 4 skips；缓存没有新增向量。"),
        ("已完成的长上下文证据", "旧三 cap 运行已在前 28 个可用网格评估 cap512/2048/8192。", "cap8192 没有一个网格成为严格 MAE 第一；后续 4 个 new_metr 网格未有 cap8192 分数。"),
        ("当前决策", "保留完整 32-grid 的 cap512/2048 结果作为本轮 authoritative screen；保存两个可恢复的 cap8192 尝试检查点，不覆盖任何旧结果。", "不宣称 8,192 已完成全样本比较；继续扩算的收益不足以抵消当前本地 CPU 成本。若之后需要，可单独挑选少量长文本/候选做配对试验。"),
    ]
    table = document.add_table(rows=1, cols=3)
    table.style = "Table Grid"
    for cell, text in zip(table.rows[0].cells, data[0]):
        _set_cell(cell, text, header=True)
    _repeat_header(table.rows[0])
    for row in data[1:]:
        cells = table.add_row().cells
        for cell, text in zip(cells, row):
            _set_cell(cell, text)
    for row in table.rows:
        row.cells[0].width = Inches(1.15)
        row.cells[1].width = Inches(3.65)
        row.cells[2].width = Inches(1.7)

    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(5)
    _run_style(paragraph.add_run(
        "保留检查点：`tmp/rawfact_full_8192_20261018_01/` 与 `tmp/rawfact_full_2048_8192_20261018_01/`。两者为 in_progress，只有旧 28 个完整网格和 4 个 Environment skips；没有新的验证分数。"
    ), size=9)

    document.save(DOC_PATH)
    verified = Document(DOC_PATH)
    if len(verified.paragraphs) <= old_paragraphs or len(verified.tables) != old_tables + 1:
        raise RuntimeError("Append-only structural verification failed.")
    if len(verified.tables[-1].rows) != len(data) or len(verified.tables[-1].columns) != 3:
        raise RuntimeError("Unexpected table dimensions after append.")
    if not any(p.text.strip() == SECTION_TITLE for p in verified.paragraphs):
        raise RuntimeError("Saved follow-up heading is missing.")
    print({
        "document": str(DOC_PATH), "backup": str(BACKUP_PATH),
        "paragraphs_before": old_paragraphs, "paragraphs_after": len(verified.paragraphs),
        "tables_before": old_tables, "tables_after": len(verified.tables),
        "new_table_rows": len(verified.tables[-1].rows),
        "new_table_columns": len(verified.tables[-1].columns),
    })


if __name__ == "__main__":
    main()
