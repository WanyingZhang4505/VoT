"""Append the offline Climate H12 ten-memory-to-eleventh-case audit."""

from pathlib import Path
import hashlib
import os
import shutil

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


DOC_PATH = Path(r"C:\Users\admin\Desktop\Research GNN\VoT_HIC_样本与流程记录.docx")
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
BACKUP_PATH = WORKSPACE_ROOT / "process_book_backup_20261018_10_to_11_replay.docx"
WORKING_PATH = WORKSPACE_ROOT / "tmp" / "hic_10_to_11_process_book.tmp.docx"
DESKTOP_TEMP_PATH = DOC_PATH.with_name("VoT_HIC_样本与流程记录.10_to_11.tmp.docx")
SECTION_TITLE = "2026 10 18 Climate H12 十条记忆预测第十一例"

ROWS = [
    ("方案", "MAE", "RMSE", "说明"),
    ("HIC K=0", "0.629167", "0.818897", "不检索历史纠错"),
    ("HIC K=10", "0.608333", "0.679620", "10 条训练记忆，top-1 检索"),
    ("HIC K=25", "0.588333", "0.744357", "25 条训练记忆"),
    ("历史均值", "0.561667", "0.728558", "数值基线"),
    ("Persistence", "0.573333", "0.708720", "数值基线"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def table_text(table):
    return [[cell.text for cell in row.cells] for row in table.rows]


def set_run(run, *, size=9.5, bold=False, color=None):
    run.font.name = "Arial"
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(size)
    run.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def append_paragraph(document, text, *, size=9.5, after=5):
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(after)
    paragraph.paragraph_format.line_spacing = 1.0
    set_run(paragraph.add_run(text), size=size)


def set_borders(table, color="D9D9D9"):
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
        node.set(qn("w:color"), color)


def set_cell(cell, text, *, header=False):
    cell.text = ""
    cell.vertical_alignment = 1
    paragraph = cell.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT if header or len(text) > 12 else WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.paragraph_format.line_spacing = 1.0
    set_run(paragraph.add_run(text), size=8.5, bold=header, color="FFFFFF" if header else None)
    if header:
        shading = OxmlElement("w:shd")
        shading.set(qn("w:fill"), "1F4E78")
        cell._tc.get_or_add_tcPr().append(shading)


def main():
    if not DOC_PATH.is_file():
        raise FileNotFoundError(DOC_PATH)
    for path in (BACKUP_PATH, WORKING_PATH, DESKTOP_TEMP_PATH):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing path: {path}")

    WORKING_PATH.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(DOC_PATH)
    document = Document(DOC_PATH)
    if any(p.text.strip() == SECTION_TITLE for p in document.paragraphs):
        raise RuntimeError(f"Section already exists: {SECTION_TITLE}")
    original_paragraphs = [p.text for p in document.paragraphs]
    original_tables = [table_text(table) for table in document.tables]

    document.add_page_break()
    heading = document.add_paragraph(style="Heading 1")
    heading.paragraph_format.space_after = Pt(6)
    set_run(heading.add_run(SECTION_TITLE), size=15, bold=True, color="1F1F1F")
    append_paragraph(
        document,
        "本节复核已保存的 native_climate_h12_20261018_02 运行：从 25 个 fit case 中按 seed 选出的 10 条训练记忆建库，再预测第一个按时间顺序的 validation case。该 query 为 climate_validation_series_20090101T000000，12 个月目标期为 2009-09 至 2010-08。",
    )
    append_paragraph(
        document,
        "用 checkpoint 中保存的向量重新计算 cosine 后，top-1 记忆为 climate_fit_series_19990101T000000，分数 0.933256，与 K=10 预测记录里的检索 ID 一致。随后用已保存的预测和该 validation case 的 12 个月真值重新计算误差；没有重新调用 DeepSeek、生成 embedding 或写入 Qdrant。",
    )

    table = document.add_table(rows=1, cols=4)
    table.style = "Table Grid"
    table.autofit = False
    widths = (Inches(1.35), Inches(0.9), Inches(0.9), Inches(3.0))
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    table.rows[0]._tr.get_or_add_trPr().append(repeat)
    for col, value in enumerate(ROWS[0]):
        set_cell(table.rows[0].cells[col], value, header=True)
    for row_index, values in enumerate(ROWS[1:]):
        cells = table.add_row().cells
        for col, value in enumerate(values):
            set_cell(cells[col], value)
            if row_index % 2:
                shade = OxmlElement("w:shd")
                shade.set(qn("w:fill"), "EAF2F8")
                cells[col]._tc.get_or_add_tcPr().append(shade)
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            cell.width = width
    set_borders(table)

    append_paragraph(
        document,
        "在这个单独窗口中，K=10 的 MAE/RMSE 都低于 K=0，但历史均值仍然更好。扩展到同一运行的六个 validation windows（72 个目标月）后，K=0 为 0.531667/0.667576，K=10 为 0.522639/0.668818：MAE 降低 1.70%，RMSE 增加 0.19%；K=10 的窗口 MAE 在 6 个窗口中胜出 4 个，RMSE 胜出 5 个。历史均值总体仍领先，因此不能据此宣称 HIC 已稳定提高准确率。",
    )
    append_paragraph(
        document,
        "这是已用于开发的回顾性 validation，不是新的盲测；outer-test 数据没有读取。OT 的含义和单位仍未由数据源说明，误差只按原始数值尺度报告。详细复算和逐窗口结果见 _research_gnn_working/LLM/evaluations/native_climate_h12_10_to_11_replay_audit_20261018.md。GraphBundle 继续保持模型中立，不在本节确定生产 vertex、edge 或 GNN。",
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
        raise RuntimeError("New comparison table does not match expected rows.")
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
        raise RuntimeError("Post-replacement comparison table verification failed.")
    print({
        "backup": str(BACKUP_PATH),
        "paragraphs_before": len(original_paragraphs),
        "paragraphs_after": len(final.paragraphs),
        "tables_before": len(original_tables),
        "tables_after": len(final.tables),
        "new_table_dimensions": [len(final.tables[-1].rows), len(final.tables[-1].columns)],
        "existing_content_preserved": True,
        "sha256_before": source_hash,
        "sha256_after": sha256(DOC_PATH),
    })


if __name__ == "__main__":
    main()
