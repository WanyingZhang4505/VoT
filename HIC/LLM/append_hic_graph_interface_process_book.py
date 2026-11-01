"""Append the HIC-to-Graph-ML interface audit to the Word process book."""

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
BACKUP_PATH = WORKSPACE_ROOT / "process_book_backup_20261018_hic_graph_interface.docx"
WORKING_PATH = WORKSPACE_ROOT / "tmp" / "hic_graph_interface_process_book.tmp.docx"
DESKTOP_TEMP_PATH = DOC_PATH.with_name("VoT_HIC_样本与流程记录.hic_graph_interface.tmp.docx")
SECTION_TITLE = "2026 10 18 HIC 到 Graph ML 接口审计"

ROWS = [
    ("图表示", "节点", "关系边", "可学习目标"),
    (
        "HIC 案例记忆图",
        "已完成评分的 forecast window/case",
        "严格过去的时间边 + BGE 摘要相似边",
        "可作检索上下文；若学习纠错效用，需有配对反事实评分",
    ),
    (
        "层级时间序列图",
        "同一 forecast origin 下的实体/series",
        "仅用显式验证过的 parent-child/业务关系",
        "未来连续时间序列值；用 MAE/RMSE 评估",
    ),
    (
        "当前 HIC/Qdrant",
        "Qdrant 每条已完成 case 一个向量",
        "`K=1/10/25` 是 bank size；每次实际 top-1",
        "Qdrant 不保存 truth/prediction；不能直接充当 GNN 标签库",
    ),
    (
        "防泄漏边界",
        "query 只包含预测 cutoff 时可见信息",
        "来源 case 必须在 query 目标期开始前完成",
        "truth、metrics、correction utility 只在评分后进入训练标签",
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


def _set_cell(cell, text, *, header=False):
    cell.text = ""
    cell.vertical_alignment = 1
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.paragraph_format.line_spacing = 1.0
    _set_run(
        paragraph.add_run(text),
        size=8.5 if header else 8,
        bold=header,
        color="FFFFFF" if header else None,
    )
    tc_pr = cell._tc.get_or_add_tcPr()
    shade = OxmlElement("w:shd") if header else None
    if shade is not None:
        shade.set(qn("w:fill"), "1F4E78")
        tc_pr.append(shade)


def _append_table(document):
    table = document.add_table(rows=1, cols=4)
    table.style = "Table Grid"
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    table.rows[0]._tr.get_or_add_trPr().append(repeat)
    for index, text in enumerate(ROWS[0]):
        _set_cell(table.rows[0].cells[index], text, header=True)
    for values in ROWS[1:]:
        cells = table.add_row().cells
        for index, value in enumerate(values):
            _set_cell(cells[index], value)
    widths = (Inches(1.22), Inches(1.78), Inches(2.08), Inches(2.10))
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            cell.width = width
    return table


def _append_paragraph(document, text, *, size=9.5, after=5):
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(after)
    paragraph.paragraph_format.line_spacing = 1.0
    _set_run(paragraph.add_run(text), size=size)
    return paragraph


def main():
    if not DOC_PATH.is_file():
        raise FileNotFoundError(DOC_PATH)
    for path in (BACKUP_PATH, WORKING_PATH, DESKTOP_TEMP_PATH):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing path: {path}")

    source_hash = _sha256(DOC_PATH)
    document = Document(DOC_PATH)
    if any(p.text.strip() == SECTION_TITLE for p in document.paragraphs):
        raise RuntimeError("The HIC-to-Graph-ML audit is already present.")
    original_paragraphs = [p.text for p in document.paragraphs]
    original_tables = [_table_text(table) for table in document.tables]

    document.add_page_break()
    heading = document.add_paragraph(style="Heading 1")
    heading.paragraph_format.space_after = Pt(6)
    _set_run(heading.add_run(SECTION_TITLE), size=15, bold=True, color="1F1F1F")

    _append_paragraph(
        document,
        "本次只做本地代码与已保存产物审计，没有调用 DeepSeek、运行 embedding、写入 Qdrant、训练模型或读取 outer-test 真值。实际代码中，case 先由 HIC 生成结构化 summary 和无检索 forecast；完成 outcome correction 后才生成 BGE-M3 向量并 upsert 到 Qdrant。",
    )
    _append_paragraph(
        document,
        "检索参数需要分开命名：K=1/10/25 表示各个嵌套 collection 的记忆库大小；Qdrant 的 search limit 仍为 1。未来图中的 semantic_top_k 才表示 query node 连到多少个语义邻居。语义相似边只表示文本相近，不代表因果关系。",
    )
    _append_paragraph(
        document,
        "持久化边界也很重要：Qdrant payload 保存 summary、correction、embedding 和完成状态，不保存 truth 或 forecast 数组；当前训练 checkpoint 也不含 truth。图模型所需的 future target、误差和 correction utility 必须写入独立的 scored-case artifact，并与预测时可见的输入特征分开。",
    )
    _append_table(document)
    _append_paragraph(
        document,
        "因此要区分两类图：HIC 案例记忆图连接已完成的历史 forecast window；hierarchical time-series 图则在同一预测起点连接不同实体 series。现有 Climate H12 只有一个 series、25 个 fit case 和 6 个 validation window，只适合检查数据管线；数据集里的 group_columns 也只是分组字段，不能单独证明 parent-child 聚合关系。",
    )
    _append_paragraph(
        document,
        "已将后续实现步骤记入 docs/superpowers/plans/2026-10-18-hic-graph-interface.md：先建分离输入/结局的 scored-case 格式，再构建确定性的时间/语义边，之后只有在真实层级映射和对齐 snapshots 得到验证后，才训练 PyTorch GraphSAGE 回归基线。当前环境是 PyTorch 2.6.0 CPU；PyG 未安装，计划首版用 edge_index + index_add_，不添加依赖。",
        size=9,
        after=3,
    )

    document.save(WORKING_PATH)
    checked = Document(WORKING_PATH)
    if [p.text for p in checked.paragraphs[:len(original_paragraphs)]] != original_paragraphs:
        raise RuntimeError("Existing paragraph content changed during append.")
    if [_table_text(t) for t in checked.tables[:len(original_tables)]] != original_tables:
        raise RuntimeError("Existing table content changed during append.")
    if len(checked.tables) != len(original_tables) + 1:
        raise RuntimeError("Unexpected number of tables after append.")
    if _table_text(checked.tables[-1]) != [list(row) for row in ROWS]:
        raise RuntimeError("The appended graph interface table does not match the audit.")
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
        raise RuntimeError("Post-replacement graph interface table differs from the audit.")

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
