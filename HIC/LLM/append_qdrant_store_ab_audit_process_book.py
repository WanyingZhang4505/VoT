"""Append the live Qdrant and saved HIC A/B audit to the Word process book."""

from pathlib import Path
import hashlib
import os
import shutil

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = Path(r"C:\Users\admin\Desktop\Research GNN\VoT_HIC_样本与流程记录.docx")
BACKUP_PATH = WORKSPACE_ROOT / "process_book_backup_20261018_qdrant_audit.docx"
WORKING_PATH = WORKSPACE_ROOT / "tmp" / "qdrant_audit_process_book.tmp.docx"
DESKTOP_TEMP_PATH = DOC_PATH.with_name("VoT_HIC_样本与流程记录.qdrant_audit.tmp.docx")
SECTION_TITLE = "2026 10 18 Live Qdrant 与 HIC 保存结果复核"

ROWS = [
    ("运行与记忆数", "检索 K", "MAE", "RMSE"),
    ("BGE-small", "0", "0.752500", "1.688120"),
    ("BGE-small", "1", "0.749167", "1.724058"),
    ("BGE-small", "10", "0.785833", "1.694181"),
    ("BGE-small", "100", "0.818083", "1.724837"),
    ("BGE-M3", "0", "0.756250", "1.719308"),
    ("BGE-M3", "1", "0.728750", "1.635740"),
    ("BGE-M3", "10", "0.730833", "1.661413"),
    ("BGE-M3", "100", "0.685833", "1.641722"),
    ("Persistence", "n/a", "1.294167", "2.166122"),
]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _table_text(table):
    return [[cell.text for cell in row.cells] for row in table.rows]


def _set_run(run, *, size=9.5, bold=False, color=None):
    run.font.name = "Arial"
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(size)
    run.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def _set_cell(cell, text, *, header=False, row_index=0):
    cell.text = ""
    cell.vertical_alignment = 1
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.paragraph_format.line_spacing = 1.0
    _set_run(
        paragraph.add_run(text),
        size=8.5 if header else 8.25,
        bold=header,
        color="FFFFFF" if header else None,
    )
    tc_pr = cell._tc.get_or_add_tcPr()
    shade = OxmlElement("w:shd")
    if header:
        shade.set(qn("w:fill"), "1F4E78")
        tc_pr.append(shade)
    elif row_index % 2 == 0:
        shade.set(qn("w:fill"), "F2F6FA")
        tc_pr.append(shade)
    margins = OxmlElement("w:tcMar")
    for side, value in (("top", "90"), ("start", "100"), ("bottom", "90"), ("end", "100")):
        node = OxmlElement(f"w:{side}")
        node.set(qn("w:w"), value)
        node.set(qn("w:type"), "dxa")
        margins.append(node)
    tc_pr.append(margins)


def _append_table(document):
    table = document.add_table(rows=1, cols=4)
    table.style = "Table Grid"
    table.autofit = False
    header_props = OxmlElement("w:tblHeader")
    header_props.set(qn("w:val"), "true")
    table.rows[0]._tr.get_or_add_trPr().append(header_props)
    widths = (Inches(2.15), Inches(0.85), Inches(1.25), Inches(1.25))
    for row_index, values in enumerate(ROWS):
        if row_index:
            cells = table.add_row().cells
        else:
            cells = table.rows[0].cells
        for cell_index, value in enumerate(values):
            _set_cell(cell= cells[cell_index], text=value, header=(row_index == 0), row_index=row_index)
            cells[cell_index].width = widths[cell_index]
    tbl_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border = OxmlElement(f"w:{edge}")
        border.set(qn("w:val"), "single")
        border.set(qn("w:sz"), "4")
        border.set(qn("w:space"), "0")
        border.set(qn("w:color"), "D9D9D9")
        borders.append(border)
    tbl_pr.append(borders)
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
        raise RuntimeError("The live Qdrant audit is already present.")
    original_paragraphs = [p.text for p in document.paragraphs]
    original_tables = [_table_text(table) for table in document.tables]

    document.add_page_break()
    heading = document.add_paragraph(style="Heading 1")
    heading.paragraph_format.space_after = Pt(6)
    _set_run(heading.add_run(SECTION_TITLE), size=15, bold=True, color="1F1F1F")
    _append_paragraph(
        document,
        "本次只读核对了已经运行的本地 Qdrant、保存的向量清单和两份已完成的 HIC 结果。没有发起新的 DeepSeek 请求、运行 embedding、写入或删除 collection、训练 GNN，也没有读取 outer-test 真值。",
    )
    _append_paragraph(
        document,
        "127.0.0.1:6333 实际由桌面 Research GNN Compose 项目的 researchgnn-qdrant-1 提供，使用 researchgnn_qdrant_storage 卷。当前工作目录中的另一个 Compose 项目没有运行容器；本次没有启动第二套服务，也没有改动卷。在线服务共有 12 个 collection。",
    )
    _append_paragraph(
        document,
        "hic_bge_m3_train_100 含 100 个唯一、已完成且观测到 12 个目标月的案例。case ID 与本地 BGE-M3 清单完全相同；按 case ID 比较的 100 个 1,024 维向量，最大坐标差为 3.00158e-08，属于浮点序列化差异。K=100 的滚动评估库为 100 个初始记忆、3 个 warm-up 和 10 个评估窗口，共 113 条；K=1 和 K=10 评估库也与运行清单的随机初始样本及相同 13 个窗口一致。旧的 train_1/train_10 collection 不是这次运行抽取的随机子集。",
    )
    _append_paragraph(
        document,
        "按每个预测时点当时已经完成的记忆重放 10 次 top-1 检索，返回的 case ID 与保存报告 10/10 一致，cosine 分数的最大差为 5.69931e-08。第一条查询命中 socialgood_rolling_eval_2011-12，cosine 为 0.922298。",
    )
    _append_paragraph(
        document,
        "下面的 MAE/RMSE 是从两份保存结果中的原始预测和真值重新计算的。两次运行使用相同的 10 个测试窗口和 120 个目标值；表中 K 表示初始检索记忆库大小，HIC 每次仍只检索 top-1。",
        after=3,
    )
    _append_table(document)
    _append_paragraph(
        document,
        "在这次 BGE-M3 运行内，K=100 的 MAE 比 K=0 低 9.31%，RMSE 低 4.51%；K=1 的 RMSE 最低。检索并非每个窗口都带来改善。BGE-small 与 BGE-M3 来自不同的 DeepSeek 调用序列，因此不能把两次结果差异解释为 embedding 模型的因果效果。两组窗口都来自同一条 SocialGood 月度序列且历史区间有重叠；这 10 个已查看窗口已用于开发，不再作为后续调参集。",
    )
    _append_paragraph(
        document,
        "存储格式也并不完全统一：100 条初始摘要是自由文本，另 13 条滚动摘要是结构化对象。现有 summary_to_embedding_text 能处理两种输入；预测提示词传入的是检索到的 correction，而不是摘要本身。",
    )
    _append_paragraph(
        document,
        "图模型边界保持开放：继续以 GraphBundle 作为模型无关接口，不预设业务 vertex/edge 语义，也不把 case-memory 图、显式层级图或 GraphSAGE 示例当成最终方案。后续应按已验证的研究定义添加图构建器和模型适配器，同时保留各特征的可用时间与预测/评分后的标签隔离。",
    )
    _append_paragraph(
        document,
        "证据文件：LLM/evaluations/hic_qdrant_store_ab_audit_20261018.md；LLM/evaluations/vot_hic_qdrant_hic_20261017_random10.json；LLM/evaluations/vot_hic_qdrant_hic_m3only_20261017.json。",
        size=8.75,
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
        raise RuntimeError("The appended metrics table does not match the audit.")
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
        raise RuntimeError("Post-replacement metrics table differs from the audit.")

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
