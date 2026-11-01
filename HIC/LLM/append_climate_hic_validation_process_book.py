"""Append the completed Climate H12 validation result to the process book."""

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
BACKUP_PATH = WORKSPACE_ROOT / "_research_gnn_working" / "process_book_backup_20261018_climate_h12.docx"
WORKING_PATH = WORKSPACE_ROOT / "_research_gnn_working" / "tmp" / "climate_h12_process_book.tmp.docx"
DESKTOP_TEMP_PATH = DOC_PATH.with_name("VoT_HIC_样本与流程记录.climate_h12.tmp.docx")
SECTION_TITLE = "2026 10 18 Climate H12 实际 HIC 验证"


def _set_run(run, *, size=9, bold=False, color=None):
    run.font.name = "Arial"
    run._element.get_or_add_rPr().get_or_add_rFonts().set(
        qn("w:eastAsia"), "Microsoft YaHei"
    )
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
    _set_run(
        paragraph.add_run(text),
        size=9 if header else 8.5,
        bold=header,
        color="FFFFFF" if header else None,
    )
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


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    if not DOC_PATH.is_file():
        raise FileNotFoundError(DOC_PATH)
    for path in (BACKUP_PATH, WORKING_PATH, DESKTOP_TEMP_PATH):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing path: {path}")

    original_hash = _sha256(DOC_PATH)
    document = Document(DOC_PATH)
    if any(p.text.strip() == SECTION_TITLE for p in document.paragraphs):
        raise RuntimeError("The Climate H12 section already exists.")
    old_paragraphs = [p.text for p in document.paragraphs]
    old_tables = [_table_text(table) for table in document.tables]

    document.add_page_break()
    heading = document.add_paragraph(style="Heading 1")
    heading.paragraph_format.space_after = Pt(6)
    _set_run(heading.add_run(SECTION_TITLE), size=15, bold=True, color="1F1F1F")

    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(6)
    _set_run(paragraph.add_run(
        "本次在 Climate 月度数据上执行了真实 HIC 流程：历史 8 个月、预测 12 个月、25 个 fit case 和 6 个按时间排列的 validation window，共 72 个目标点。Climate 源表 496 行；本轮审计未发现无效时间戳、无效 OT、重复时间戳或 cadence gap，事实文本均非空；唯一可用数值协变量为 Anomaly。OT 的业务含义和单位没有数据字典说明，因此误差只按该数据集原尺度解释。"
    ), size=10)

    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(6)
    _set_run(paragraph.add_run(
        "DeepSeek Flash 按本轮新生成的 HIC template 处理 fit 与 validation 输入；输入限于每个 case 的 8 个月 OT 历史、截止时点前的事实文本和可用协变量。fit 的 12 个月历史真值用于生成 correction memory；validation 目标在 K=0/1/10/25 的预测完成后才揭示，再用于评分、纠错和增量入库。BAAI/bge-m3 使用 1,024 维 CLS 向量、L2 归一化、8,192 token 上限；Qdrant 为本地 embedded cosine。7 个 outer-test 窗口未进入 HIC 提示、检索、纠错或评分。"
    ), size=10)

    rows = [
        ("候选", "MAE", "RMSE", "结论（同一 validation）"),
        ("HIC K=0", "0.531667", "0.667576", "不检索历史 correction"),
        ("HIC K=1", "0.544861", "0.688126", "未优于 K=0 或数值基线"),
        ("HIC K=10", "0.522639", "0.668818", "较 K=0 的 MAE 略低；RMSE 略高"),
        ("HIC K=25", "0.507778", "0.632640", "检索组中最佳；仍落后 history mean 与 persistence"),
        ("Persistence", "0.496667", "0.624873", "简单数值基线"),
        ("History mean", "0.487951", "0.621843", "本次总体最佳数值基线"),
        ("Recent mean 3", "0.578611", "0.709162", "简单数值基线"),
        ("Linear trend", "1.447113", "1.652522", "简单数值基线"),
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
        for index, (cell, value) in enumerate(zip(cells, values)):
            _set_cell(cell, value, center=index in (1, 2))
    widths = (Inches(1.2), Inches(0.85), Inches(0.85), Inches(3.5))
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            cell.width = width

    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(7)
    paragraph.paragraph_format.space_after = Pt(5)
    _set_run(paragraph.add_run(
        "解释：K=25 是本次已测 HIC 检索设置中最好的，但 history mean 的 MAE 和 RMSE 都更低；因此本结果不支持“检索越多越准”或 HIC 已超过数值基线。三组本地 Qdrant collection 最终分别含 7、16、31 个 1,024 维向量，即 1/10/25 个 fit memory 加上 6 个已评分 validation case；检查 payload 后确认未发现 outer-test case。"
    ), size=10)

    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(5)
    _set_run(paragraph.add_run(
        "调用记录：1 次 template 生成 + 75 次 fit 调用 + 48 次 validation 调用，共 124 次逻辑调用；成功运行账本记录 124 次 HTTP 尝试。此前第一次运行在本地 DNS 解析处失败，单独保留 1 次 HTTP 尝试记录；两次合计 125/248，未超批准上限。运行未记录 token 用量或费用，不能据此推算成本。"
    ), size=10)

    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(4)
    _set_run(paragraph.add_run(
        "运行 ID：native_climate_h12_20261018_02。详细预测、逐窗评分、生成模板 hash、Qdrant 集合名和调用账本见工作区 LLM/evaluations/native_hic_pilot_native_climate_h12_20261018_02.json。该结果是 6 个验证窗口上的候选对照，不是 outer-test 结果，也不是全局最优结论。"
    ), size=9)

    document.save(WORKING_PATH)
    checked = Document(WORKING_PATH)
    if [p.text for p in checked.paragraphs[:len(old_paragraphs)]] != old_paragraphs:
        raise RuntimeError("Existing paragraph content changed during append.")
    if [_table_text(t) for t in checked.tables[:len(old_tables)]] != old_tables:
        raise RuntimeError("Existing table content changed during append.")
    if len(checked.tables) != len(old_tables) + 1:
        raise RuntimeError("Unexpected number of tables after append.")
    if len(checked.tables[-1].rows) != len(rows) or len(checked.tables[-1].columns) != 4:
        raise RuntimeError("Unexpected dimensions for the Climate comparison table.")
    if not any(p.text.strip() == SECTION_TITLE for p in checked.paragraphs):
        raise RuntimeError("Appended section heading is missing.")
    if _sha256(DOC_PATH) != original_hash:
        raise RuntimeError("The Desktop process book changed during editing; refusing replacement.")

    shutil.copy2(DOC_PATH, BACKUP_PATH)
    shutil.copy2(WORKING_PATH, DESKTOP_TEMP_PATH)
    os.replace(DESKTOP_TEMP_PATH, DOC_PATH)
    final = Document(DOC_PATH)
    if [p.text for p in final.paragraphs[:len(old_paragraphs)]] != old_paragraphs:
        raise RuntimeError("Post-replacement paragraph preservation check failed.")
    if [_table_text(t) for t in final.tables[:len(old_tables)]] != old_tables:
        raise RuntimeError("Post-replacement table preservation check failed.")
    print({
        "backup": str(BACKUP_PATH),
        "paragraphs_before": len(old_paragraphs),
        "paragraphs_after": len(final.paragraphs),
        "tables_before": len(old_tables),
        "tables_after": len(final.tables),
        "new_table_dimensions": [len(final.tables[-1].rows), len(final.tables[-1].columns)],
        "desktop_sha256_before": original_hash,
        "desktop_sha256_after": _sha256(DOC_PATH),
        "existing_content_preserved": True,
    })


if __name__ == "__main__":
    main()
