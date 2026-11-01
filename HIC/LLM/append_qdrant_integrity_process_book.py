"""Append the read-only local Qdrant integrity audit to the Process Book."""

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
BACKUP_PATH = WORKSPACE_ROOT / "process_book_backup_20261018_qdrant_integrity.docx"
WORKING_PATH = WORKSPACE_ROOT / "tmp" / "qdrant_integrity_process_book.tmp.docx"
DESKTOP_TEMP_PATH = DOC_PATH.with_name("VoT_HIC_样本与流程记录.qdrant_audit.tmp.docx")
SECTION_TITLE = "2026 10 18 本地 Qdrant 训练向量库完整性核验"
ROWS = [
    ("集合", "向量数", "状态", "维度", "完成样本 / 观测目标"),
    ("hic_bge_m3_train_1", "1", "green", "1,024", "1 / 12"),
    ("hic_bge_m3_train_10", "10", "green", "1,024", "10 / 每例12"),
    ("hic_bge_m3_train_100", "100", "green", "1,024", "100 / 每例12"),
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
    run._element.get_or_add_rPr().get_or_add_rFonts().set(
        qn("w:eastAsia"), "Microsoft YaHei"
    )
    run.font.size = Pt(size)
    run.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def append_paragraph(document, text, *, size=9.5, after=5):
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(after)
    paragraph.paragraph_format.line_spacing = 1.0
    set_run(paragraph.add_run(text), size=size)


def set_cell(cell, text, *, header=False):
    cell.text = ""
    cell.vertical_alignment = 1
    paragraph = cell.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT if header else WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.paragraph_format.line_spacing = 1.0
    set_run(
        paragraph.add_run(text),
        size=8.5,
        bold=header,
        color="FFFFFF" if header else None,
    )
    if header:
        shading = OxmlElement("w:shd")
        shading.set(qn("w:fill"), "1F4E78")
        cell._tc.get_or_add_tcPr().append(shading)


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
        "本次只读核验本机 Qdrant 的 BGE-M3 训练记忆集合；没有调用 LLM API、没有写入向量，也没有读取 evaluation/test 集合中的点。三个训练集合分别包含 1、10、100 条记录，向量维度均为 1,024，集合状态为 green，所有记录都标记为 completed 且有 12 个已观测目标值。",
    )
    table = document.add_table(rows=1, cols=5)
    table.style = "Table Grid"
    table.autofit = False
    widths = (Inches(2.0), Inches(0.65), Inches(0.7), Inches(0.7), Inches(2.1))
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
            cell.width = width
    set_borders(table)
    append_paragraph(
        document,
        "1/10 的 case ID 都包含在 100 条集合中；100 条 case ID 与本地 kb_socialgood_bge_m3_100train.json 完全一致，摘要逐条一致，向量逐维比较的最大绝对误差为 2.9802322387695312e-08（浮点序列化差异）。这证明的是向量库与源快照的存储一致性，不代表检索纠错正确，也不代表预测误差下降。",
    )
    append_paragraph(
        document,
        "Graph ML 接口继续保持可插拔：GraphBundle 不预设最终模型、vertex 类型、edge 语义、拓扑或预测任务。等研究数据和任务由团队确定后，再将相应 schema 与模型接入；本次核验不替研究组决定图结构。完整细节见 _research_gnn_working/LLM/evaluations/qdrant_train_1_10_100_integrity_audit_20261018.md。",
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
        raise RuntimeError("New Qdrant table does not match expected rows.")
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
        raise RuntimeError("Post-replacement Qdrant table verification failed.")
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
