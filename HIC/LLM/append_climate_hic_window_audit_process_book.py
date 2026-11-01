"""Append the paired Climate H12 per-window audit to the process book."""

from pathlib import Path
import hashlib
import os
import shutil

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = Path(r"C:\Users\admin\Desktop\Research GNN\VoT_HIC_样本与流程记录.docx")
BACKUP_PATH = WORKSPACE_ROOT / "_research_gnn_working" / "process_book_backup_20261018_climate_window_audit.docx"
WORKING_PATH = WORKSPACE_ROOT / "_research_gnn_working" / "tmp" / "climate_window_audit_process_book.tmp.docx"
DESKTOP_TEMP_PATH = DOC_PATH.with_name("VoT_HIC_样本与流程记录.climate_window_audit.tmp.docx")
SECTION_TITLE = "2026 10 18 Climate H12 逐窗口审计"

ROWS = [
    ("Validation 起始月", "HIC K=0", "HIC K=1", "HIC K=10", "HIC K=25", "Persistence", "History mean"),
    ("2009-09", "0.629167", "0.697500", "0.608333", "0.588333", "0.573333", "0.561667"),
    ("2010-09", "0.467500", "0.530000", "0.473333", "0.477500", "0.423333", "0.489167"),
    ("2011-09", "0.387500", "0.336667", "0.304167", "0.238333", "0.208333", "0.297083"),
    ("2012-09", "0.512500", "0.534167", "0.505000", "0.548333", "0.532500", "0.536875"),
    ("2013-09", "0.503333", "0.440000", "0.264167", "0.454167", "0.466667", "0.482917"),
    ("2014-09", "0.690000", "0.730833", "0.980833", "0.740000", "0.775833", "0.560000"),
]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _table_text(table):
    return [[cell.text for cell in row.cells] for row in table.rows]


def _set_run(run, *, size=9, bold=False, color=None):
    run.font.name = "Arial"
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(size)
    run.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def _set_cell(cell, text, *, header=False, center=False):
    cell.text = ""
    cell.vertical_alignment = 1
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.paragraph_format.line_spacing = 1.0
    if center:
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _set_run(paragraph.add_run(text), size=8.5 if header else 8, bold=header, color="FFFFFF" if header else None)
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = qn(f"w:{edge}")
        border = borders.find(tag)
        if border is None:
            border = OxmlElement(f"w:{edge}")
            borders.append(border)
        border.set(qn("w:val"), "single")
        border.set(qn("w:sz"), "6")
        border.set(qn("w:space"), "0")
        border.set(qn("w:color"), "D9D9D9")
    if header:
        shade = OxmlElement("w:shd")
        shade.set(qn("w:fill"), "1F4E78")
        tc_pr.append(shade)


def _append_table(document):
    table = document.add_table(rows=1, cols=7)
    table.style = "Table Grid"
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    table.rows[0]._tr.get_or_add_trPr().append(repeat)
    for index, text in enumerate(ROWS[0]):
        _set_cell(table.rows[0].cells[index], text, header=True, center=index != 0)
    for values in ROWS[1:]:
        cells = table.add_row().cells
        for index, value in enumerate(values):
            _set_cell(cells[index], value, center=index != 0)
    widths = (Inches(1.05), Inches(0.78), Inches(0.78), Inches(0.82), Inches(0.82), Inches(0.95), Inches(0.95))
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            cell.width = width
    return table


def main():
    if not DOC_PATH.is_file():
        raise FileNotFoundError(DOC_PATH)
    for path in (BACKUP_PATH, WORKING_PATH, DESKTOP_TEMP_PATH):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing path: {path}")

    source_hash = _sha256(DOC_PATH)
    document = Document(DOC_PATH)
    if any(p.text.strip() == SECTION_TITLE for p in document.paragraphs):
        raise RuntimeError("The per-window audit is already present.")
    original_paragraphs = [p.text for p in document.paragraphs]
    original_tables = [_table_text(table) for table in document.tables]

    document.add_page_break()
    heading = document.add_paragraph(style="Heading 1")
    heading.paragraph_format.space_after = Pt(6)
    _set_run(heading.add_run(SECTION_TITLE), size=15, bold=True, color="1F1F1F")

    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(6)
    _set_run(paragraph.add_run(
        "对 Climate H12 的 6 个 validation window 做了配对逐窗复核：直接使用已保存的 12 步预测和对应真值重新计算 HIC MAE，并按同一窗口对齐 persistence 与 history mean 的基线分数。下表给出每个窗口的 MAE，数值越低越好。"
    ), size=10)

    table = _append_table(document)

    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(7)
    paragraph.paragraph_format.space_after = Pt(6)
    _set_run(paragraph.add_run(
        "K=1、K=10、K=25 分别在 6 个窗口中的 2、4、3 个优于 K=0。逐窗最低 MAE 的方案分别由 history mean、persistence、HIC K=10 各取得 2 次；K=25 虽然是汇总指标最好的 HIC 检索臂，却没有在任何单个窗口取得最低 MAE。这说明检索纠错有时有帮助，但效果随窗口变化，不能说检索越多就越准。"
    ), size=10)

    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(6)
    _set_run(paragraph.add_run(
        "汇总上，K=25 相比 K=0 的 MAE 低 4.493%、RMSE 低 5.233%；但相比 history mean，MAE 高 4.063%、RMSE 高 1.736%，也未胜过 persistence。该结论只描述这 6 个 validation window，不代表显著性或未来泛化。"
    ), size=10)

    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(6)
    _set_run(paragraph.add_run(
        "另检查了 18 条 validation 检索记录（6 个窗口 × K=1/10/25）：每条均包含有内容的结构化摘要、三个预期顶层部分、四个 predictive-factor 字段、四个 forecast-outlook 字段和非空 correction。之前 PowerShell 中显示的 `@{...}` 是对象格式化展示，不表示摘要为空。"
    ), size=10)

    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(4)
    _set_run(paragraph.add_run(
        "本次只复核已保存结果，没有新增 DeepSeek 请求、embedding 推理、Qdrant 写入或读取任何 outer-test 真值。运行报告：LLM/evaluations/native_hic_pilot_native_climate_h12_20261018_02.md 与同名 JSON。"
    ), size=9)

    document.save(WORKING_PATH)
    checked = Document(WORKING_PATH)
    if [p.text for p in checked.paragraphs[:len(original_paragraphs)]] != original_paragraphs:
        raise RuntimeError("Existing paragraph content changed during append.")
    if [_table_text(t) for t in checked.tables[:len(original_tables)]] != original_tables:
        raise RuntimeError("Existing table content changed during append.")
    if len(checked.tables) != len(original_tables) + 1:
        raise RuntimeError("Unexpected number of tables after append.")
    if _table_text(checked.tables[-1]) != [list(row) for row in ROWS]:
        raise RuntimeError("The appended per-window table does not match the audited values.")
    if not any(p.text.strip() == SECTION_TITLE for p in checked.paragraphs):
        raise RuntimeError("Appended section heading is missing.")
    if _sha256(DOC_PATH) != source_hash:
        raise RuntimeError("The Desktop process book changed during editing; refusing replacement.")

    shutil.copy2(DOC_PATH, BACKUP_PATH)
    shutil.copy2(WORKING_PATH, DESKTOP_TEMP_PATH)
    os.replace(DESKTOP_TEMP_PATH, DOC_PATH)

    final = Document(DOC_PATH)
    if [p.text for p in final.paragraphs[:len(original_paragraphs)]] != original_paragraphs:
        raise RuntimeError("Post-replacement paragraph preservation check failed.")
    if [_table_text(t) for t in final.tables[:len(original_tables)]] != original_tables:
        raise RuntimeError("Post-replacement table preservation check failed.")
    if _table_text(final.tables[-1]) != [list(row) for row in ROWS]:
        raise RuntimeError("Post-replacement audit table differs from the audited values.")

    print({
        "backup": str(BACKUP_PATH),
        "paragraphs_before": len(original_paragraphs),
        "paragraphs_after": len(final.paragraphs),
        "tables_before": len(original_tables),
        "tables_after": len(final.tables),
        "new_table_dimensions": [len(final.tables[-1].rows), len(final.tables[-1].columns)],
        "existing_content_preserved": True,
        "desktop_sha256_before": source_hash,
        "desktop_sha256_after": _sha256(DOC_PATH),
    })


if __name__ == "__main__":
    main()
