"""Append a same-split HIC validation comparison to the process book."""

from pathlib import Path
import os
import shutil

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = ROOT / "VoT_HIC_样本与流程记录_100case_stage.docx"
BACKUP_PATH = ROOT / "_research_gnn_working" / "process_book_backup_20261018_hic_validation.docx"
TEMP_PATH = DOC_PATH.with_name("VoT_HIC_样本与流程记录_100case_stage.validation.tmp.docx")
SECTION_TITLE = "2026 10 18 HIC 验证集候选对照"


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
    paragraph.paragraph_format.space_after = Pt(3)
    paragraph.paragraph_format.line_spacing = 1.05
    if center:
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _set_run(paragraph.add_run(text), size=9 if header else 8.5, bold=header,
             color="FFFFFF" if header else None)
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


def _table_text(table):
    return [[cell.text for cell in row.cells] for row in table.rows]


def main():
    if not DOC_PATH.is_file():
        raise FileNotFoundError(DOC_PATH)
    if BACKUP_PATH.exists() or TEMP_PATH.exists():
        raise FileExistsError("Refusing to overwrite an existing backup or temporary output.")

    document = Document(DOC_PATH)
    if any(p.text.strip() == SECTION_TITLE for p in document.paragraphs):
        raise RuntimeError("The validation comparison section already exists.")
    old_paragraphs = [p.text for p in document.paragraphs]
    old_tables = [_table_text(table) for table in document.tables]

    document.add_page_break()
    heading = document.add_paragraph(style="Heading 1")
    heading.paragraph_format.space_after = Pt(6)
    _set_run(heading.add_run(SECTION_TITLE), size=15, bold=True, color="1F1F1F")

    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(6)
    _set_run(paragraph.add_run(
        "以下是对已保存报告 native_hic_pilot_native_agri_h12_k24_20261018_01.json 的复核，不是本轮新调用 API。固定配置为 Agriculture 月度序列、历史 8 个月、预测 12 个月、24 个训练案例、6 个按时间排列的验证窗口（72 个目标点）；另有 7 个 outer-test 窗口未用于本次选择。"
    ), size=10)

    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(7)
    _set_run(paragraph.add_run(
        "HIC 固定使用已生成模板（SHA-256 06b1366d39e0ca7db051849a9a8187553431af6dd4586af3ad06c235fa943046）、BAAI/bge-m3（1,024 维、CLS、L2、8,192-token 上限、无 query instruction）及本地 embedded Qdrant cosine 检索；只改变历史记忆规模 K。K=24 是 HIC arms 中 MAE 和 RMSE 最低的配置，但 persistence baseline 仍更好，因此该验证结果不支持宣称 HIC 检索已稳定增益。"
    ), size=10)

    rows = [
        ("候选", "MAE", "RMSE", "同一验证切分上的结论"),
        ("HIC K=0", "5.254558", "6.583344", "不检索历史纠错案例"),
        ("HIC K=1", "4.318975", "5.544496", "优于 K=0；仍略差于 persistence"),
        ("HIC K=10", "4.725255", "6.153195", "未优于 K=1 或 persistence"),
        ("HIC K=24", "4.237805", "5.372687", "HIC 检索候选中最佳；仍未胜 persistence"),
        ("Persistence", "4.118798", "5.249086", "本切分总体最佳；使用最近历史值作为预测"),
        ("History mean", "5.230565", "6.632223", "数值基线"),
        ("Recent mean 3", "4.492625", "5.818344", "数值基线"),
        ("Linear trend", "6.252421", "8.007469", "数值基线"),
    ]
    table = document.add_table(rows=1, cols=4)
    table.style = "Table Grid"
    header_props = table.rows[0]._tr.get_or_add_trPr()
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    header_props.append(repeat)
    for cell, value in zip(table.rows[0].cells, rows[0]):
        _set_cell(cell, value, header=True, center=value in ("MAE", "RMSE"))
    for values in rows[1:]:
        cells = table.add_row().cells
        for idx, (cell, value) in enumerate(zip(cells, values)):
            _set_cell(cell, value, center=idx in (1, 2))
    widths = (Inches(1.25), Inches(0.9), Inches(0.9), Inches(3.35))
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            cell.width = width

    document.save(TEMP_PATH)
    checked = Document(TEMP_PATH)
    if [p.text for p in checked.paragraphs[:len(old_paragraphs)]] != old_paragraphs:
        raise RuntimeError("Existing paragraph content changed during append.")
    if [_table_text(t) for t in checked.tables[:len(old_tables)]] != old_tables:
        raise RuntimeError("Existing table content changed during append.")
    if len(checked.tables) != len(old_tables) + 1:
        raise RuntimeError("Unexpected number of tables after append.")
    if len(checked.tables[-1].rows) != len(rows) or len(checked.tables[-1].columns) != 4:
        raise RuntimeError("Unexpected dimensions for the validation table.")
    if not any(p.text.strip() == SECTION_TITLE for p in checked.paragraphs):
        raise RuntimeError("Appended section heading is missing.")

    shutil.copy2(DOC_PATH, BACKUP_PATH)
    os.replace(TEMP_PATH, DOC_PATH)
    final = Document(DOC_PATH)
    print({
        "backup": str(BACKUP_PATH),
        "paragraphs_before": len(old_paragraphs),
        "paragraphs_after": len(final.paragraphs),
        "tables_before": len(old_tables),
        "tables_after": len(final.tables),
        "new_table_dimensions": [len(final.tables[-1].rows), len(final.tables[-1].columns)],
    })


if __name__ == "__main__":
    main()
