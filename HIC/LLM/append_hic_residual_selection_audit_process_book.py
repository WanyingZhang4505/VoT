"""Append the baseline-inclusive selection caveat to the process book."""

from pathlib import Path
import hashlib
import os
import shutil

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


DOC_PATH = Path(r"C:\Users\admin\Desktop\Research GNN\VoT_HIC_样本与流程记录.docx")
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
BACKUP_PATH = WORKSPACE_ROOT / "process_book_backup_20261018_residual_transfer_audit.docx"
WORKING_PATH = WORKSPACE_ROOT / "tmp" / "hic_residual_transfer_audit_process_book.tmp.docx"
DESKTOP_TEMP_PATH = DOC_PATH.with_name("VoT_HIC_样本与流程记录.residual_transfer_audit.tmp.docx")
SECTION_TITLE = "2026 10 18 Residual transfer 选择切片复核"
ROWS = [
    ("切片", "Residual transfer K=10 MAE / RMSE", "Persistence MAE / RMSE"),
    ("选择集：前三个 validation 窗口", "0.415444 / 0.550566", "0.401667 / 0.530131"),
    ("前向集：后三个 validation 窗口", "0.427944 / 0.538709", "0.591667 / 0.707032"),
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


def _borders(table):
    borders = table._tbl.tblPr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        table._tbl.tblPr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        node = borders.find(qn(f"w:{edge}"))
        if node is None:
            node = OxmlElement(f"w:{edge}")
            borders.append(node)
        node.set(qn("w:val"), "single")
        node.set(qn("w:sz"), "6")
        node.set(qn("w:space"), "0")
        node.set(qn("w:color"), "D9D9D9")


def _set_cell(cell, text, header=False):
    cell.text = ""
    cell.vertical_alignment = 1
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_after = Pt(2)
    _set_run(paragraph.add_run(text), size=8.5, bold=header, color="FFFFFF" if header else None)
    if header:
        shade = OxmlElement("w:shd")
        shade.set(qn("w:fill"), "1F4E78")
        cell._tc.get_or_add_tcPr().append(shade)


def main():
    if not DOC_PATH.is_file():
        raise FileNotFoundError(DOC_PATH)
    for path in (BACKUP_PATH, WORKING_PATH, DESKTOP_TEMP_PATH):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing path: {path}")
    before_hash = _sha256(DOC_PATH)
    document = Document(DOC_PATH)
    if any(p.text.strip() == SECTION_TITLE for p in document.paragraphs):
        raise RuntimeError(f"Section already exists: {SECTION_TITLE}")
    old_paragraphs = [p.text for p in document.paragraphs]
    old_tables = [_table_text(table) for table in document.tables]

    document.add_page_break()
    heading = document.add_paragraph(style="Heading 1")
    heading.paragraph_format.space_after = Pt(6)
    _set_run(heading.add_run(SECTION_TITLE), size=15, bold=True, color="1F1F1F")
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(5)
    paragraph.paragraph_format.line_spacing = 1.0
    _set_run(paragraph.add_run(
        "补充核验：前三个选择窗口中，persistence 的 MAE/RMSE（0.401667/0.530131）优于残差迁移网格选出的 K=10（0.415444/0.550566）。当时参数只在残差迁移候选内部选择，没有把数值基线纳入同一选择器。后三个已复用窗口上残差迁移胜过 persistence，但这种排序反转说明三窗选择很不稳定；不能将 K=10称为跨所有方法选出的稳定最优方案。后续应在新 validation 区间把残差迁移、K=0 与简单数值基线放进同一个预先规定的选择流程，并继续保留 outer-test。"
    ), size=9.5)

    table = document.add_table(rows=1, cols=3)
    table.style = "Table Grid"
    table.autofit = False
    widths = (Inches(2.2), Inches(2.15), Inches(2.15))
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    table.rows[0]._tr.get_or_add_trPr().append(repeat)
    for index, value in enumerate(ROWS[0]):
        _set_cell(table.rows[0].cells[index], value, header=True)
    for row_values in ROWS[1:]:
        cells = table.add_row().cells
        for index, value in enumerate(row_values):
            _set_cell(cells[index], value)
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            cell.width = width
    _borders(table)

    document.save(WORKING_PATH)
    checked = Document(WORKING_PATH)
    if [p.text for p in checked.paragraphs[:len(old_paragraphs)]] != old_paragraphs:
        raise RuntimeError("Existing paragraphs changed during append.")
    if [_table_text(t) for t in checked.tables[:len(old_tables)]] != old_tables:
        raise RuntimeError("Existing tables changed during append.")
    if _table_text(checked.tables[-1]) != [list(row) for row in ROWS]:
        raise RuntimeError("New selection-audit table differs from expected content.")
    if _sha256(DOC_PATH) != before_hash:
        raise RuntimeError("Source document changed while preparing append.")

    shutil.copy2(DOC_PATH, BACKUP_PATH)
    shutil.copy2(WORKING_PATH, DESKTOP_TEMP_PATH)
    os.replace(DESKTOP_TEMP_PATH, DOC_PATH)
    final = Document(DOC_PATH)
    if [p.text for p in final.paragraphs[:len(old_paragraphs)]] != old_paragraphs:
        raise RuntimeError("Post-replacement paragraph preservation check failed.")
    if [_table_text(t) for t in final.tables[:len(old_tables)]] != old_tables:
        raise RuntimeError("Post-replacement table preservation check failed.")
    if _table_text(final.tables[-1]) != [list(row) for row in ROWS]:
        raise RuntimeError("Post-replacement table verification failed.")
    print({
        "backup": str(BACKUP_PATH),
        "paragraphs_before": len(old_paragraphs),
        "paragraphs_after": len(final.paragraphs),
        "tables_before": len(old_tables),
        "tables_after": len(final.tables),
        "new_table_dimensions": [len(final.tables[-1].rows), len(final.tables[-1].columns)],
        "existing_content_preserved": True,
        "sha256_before": before_hash,
        "sha256_after": _sha256(DOC_PATH),
    })


if __name__ == "__main__":
    main()
