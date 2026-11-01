"""Append this implementation checkpoint to the Desktop HIC process book."""

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
BACKUP_PATH = WORKSPACE_ROOT / "process_book_backup_20261018_graph_implementation.docx"
WORKING_PATH = WORKSPACE_ROOT / "tmp" / "hic_graph_implementation_process_book.tmp.docx"
DESKTOP_TEMP_PATH = DOC_PATH.with_name("VoT_HIC_样本与流程记录.graph_implementation.tmp.docx")
SECTION_TITLE = "2026 10 18 HIC Graph ML 实现进展"
ROWS = [
    ("模块", "实现内容", "验证与限制"),
    (
        "Scored-case record",
        "将预测时可见的输入和预测，与评分后才知道的 truth、metrics、correction 分区保存。JSONL 快照原子写入并检查 case ID。",
        "7 项测试通过；forecast_features 不返回任何事后结果。",
    ),
    (
        "HIC case-memory graph",
        "已完成 case 作为历史节点；仅连接 target_end 严格早于 query target_start 的记录。支持同序列筛选和独立命名的 semantic_top_k。",
        "6 项测试通过；合成 CLI 输出 3 节点、4 条边。语义相似不代表因果。",
    ),
    (
        "Hierarchy snapshot",
        "同一预测起点的 entity-series 节点，通过显式 parent_by_entity 建立 parent-to-child 边。",
        "6 项测试通过；拒绝缺父节点、不同 cutoff、重复实体和循环。",
    ),
    (
        "PyTorch GraphSAGE",
        "两层、按关系类型做加权 mean aggregation，以 edge_index/index_add_ 传递信息并输出预测 horizon。",
        "5 项测试通过并能反向传播；尚未用真实 HIC 数据拟合或评估。",
    ),
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


def _set_cell(cell, text, *, header=False, shade=None):
    cell.text = ""
    cell.vertical_alignment = 1
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.paragraph_format.line_spacing = 1.05
    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    _set_run(paragraph.add_run(text), size=8.5 if header else 8.2, bold=header,
             color="FFFFFF" if header else None)

    tc_pr = cell._tc.get_or_add_tcPr()
    if shade:
        shading = OxmlElement("w:shd")
        shading.set(qn("w:fill"), shade)
        tc_pr.append(shading)
    margins = OxmlElement("w:tcMar")
    for side, width in (("top", "90"), ("start", "110"), ("bottom", "90"), ("end", "110")):
        edge = OxmlElement(f"w:{side}")
        edge.set(qn("w:w"), width)
        edge.set(qn("w:type"), "dxa")
        margins.append(edge)
    tc_pr.append(margins)

    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border = OxmlElement(f"w:{side}")
        border.set(qn("w:val"), "single")
        border.set(qn("w:sz"), "4")
        border.set(qn("w:space"), "0")
        border.set(qn("w:color"), "D9D9D9")
        borders.append(border)


def _append_table(document):
    table = document.add_table(rows=1, cols=3)
    table.style = "Table Grid"
    header_props = OxmlElement("w:tblHeader")
    header_props.set(qn("w:val"), "true")
    table.rows[0]._tr.get_or_add_trPr().append(header_props)
    for index, value in enumerate(ROWS[0]):
        _set_cell(table.rows[0].cells[index], value, header=True, shade="1F4E78")
    for row_index, values in enumerate(ROWS[1:]):
        cells = table.add_row().cells
        for index, value in enumerate(values):
            _set_cell(cells[index], value, shade="F2F2F2" if row_index % 2 else "FFFFFF")
    widths = (Inches(1.25), Inches(2.65), Inches(3.10))
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            cell.width = width
    return table


def _append_paragraph(document, text, *, size=9.5, after=5):
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(after)
    paragraph.paragraph_format.line_spacing = 1.05
    _set_run(paragraph.add_run(text), size=size)
    return paragraph


def main():
    if not DOC_PATH.is_file():
        raise FileNotFoundError(DOC_PATH)
    for path in (BACKUP_PATH, WORKING_PATH, DESKTOP_TEMP_PATH):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing path: {path}")

    original_hash = _sha256(DOC_PATH)
    original = Document(DOC_PATH)
    if any(paragraph.text.strip() == SECTION_TITLE for paragraph in original.paragraphs):
        raise RuntimeError("The graph implementation section already exists.")
    paragraphs_before = [paragraph.text for paragraph in original.paragraphs]
    tables_before = [_table_text(table) for table in original.tables]

    original.add_page_break()
    heading = original.add_paragraph(style="Heading 1")
    heading.paragraph_format.space_after = Pt(6)
    _set_run(heading.add_run(SECTION_TITLE), size=14, bold=True, color="000000")
    _append_paragraph(
        original,
        "本轮把 HIC 完成案例转换为可检查的图数据接口，并实现了两种用途不同的图。所有实现都在本地运行，不改变 HIC 提示词、DeepSeek 设置、BGE-M3 参数或 Qdrant 数据。",
    )
    _append_table(original)
    _append_paragraph(
        original,
        "重要边界：Climate H12 目前只有一个 series，因此只能检查数据和模型管线，不能据此声称 GNN 学到了可泛化的层级关系。当前代码没有在真实 HIC 数据上训练或比较 MAE/RMSE；后续需要确认真实 parent-child 关系、生成足够的同起点 snapshots，并按时间顺序做验证。",
    )
    _append_paragraph(
        original,
        "四个新增测试模块共 24 项测试通过，Python 语法编译检查通过。此次未调用 DeepSeek、未运行 embedding、未读写 Qdrant、未访问 outer-test，也未提交或推送 Git。",
    )

    WORKING_PATH.parent.mkdir(parents=True, exist_ok=True)
    original.save(WORKING_PATH)
    staged = Document(WORKING_PATH)
    if [paragraph.text for paragraph in staged.paragraphs[:len(paragraphs_before)]] != paragraphs_before:
        raise RuntimeError("Existing paragraph content changed during append.")
    if [_table_text(table) for table in staged.tables[:len(tables_before)]] != tables_before:
        raise RuntimeError("Existing table content changed during append.")
    if _table_text(staged.tables[-1]) != [list(row) for row in ROWS]:
        raise RuntimeError("The appended implementation table differs from the expected content.")
    if _sha256(DOC_PATH) != original_hash:
        raise RuntimeError("The Desktop process book changed during editing; refusing replacement.")

    shutil.copy2(DOC_PATH, BACKUP_PATH)
    shutil.copy2(WORKING_PATH, DESKTOP_TEMP_PATH)
    os.replace(DESKTOP_TEMP_PATH, DOC_PATH)

    final = Document(DOC_PATH)
    if [paragraph.text for paragraph in final.paragraphs[:len(paragraphs_before)]] != paragraphs_before:
        raise RuntimeError("Post-replacement paragraph preservation check failed.")
    if [_table_text(table) for table in final.tables[:len(tables_before)]] != tables_before:
        raise RuntimeError("Post-replacement table preservation check failed.")
    if _table_text(final.tables[-1]) != [list(row) for row in ROWS]:
        raise RuntimeError("Post-replacement implementation table differs from the expected content.")
    if not any(paragraph.text.strip() == SECTION_TITLE for paragraph in final.paragraphs):
        raise RuntimeError("Appended section heading is missing.")

    print({
        "backup": str(BACKUP_PATH),
        "paragraphs_before": len(paragraphs_before),
        "paragraphs_after": len(final.paragraphs),
        "tables_before": len(tables_before),
        "tables_after": len(final.tables),
        "new_table_dimensions": [len(final.tables[-1].rows), len(final.tables[-1].columns)],
        "existing_content_preserved": True,
        "sha256_before": original_hash,
        "sha256_after": _sha256(DOC_PATH),
    })


if __name__ == "__main__":
    main()
