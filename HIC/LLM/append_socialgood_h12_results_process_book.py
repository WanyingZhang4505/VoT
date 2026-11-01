"""Append final SocialGood H12 validation results to the Process Book."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


DOC_PATH = Path(r"C:\Users\admin\Desktop\Research GNN\VoT_HIC_样本与流程记录.docx")
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
BACKUP_PATH = WORKSPACE_ROOT / "process_book_backup_20261018_socialgood_h12_results.docx"
TEMP_PATH = DOC_PATH.with_name("VoT_HIC_样本与流程记录.socialgood_h12_results.tmp.docx")
SECTION_TITLE = "2026 10 18 SocialGood H12 validation results"
ROWS = [
    ("方案", "MAE", "RMSE", "说明"),
    ("HIC K=0", "0.385985", "0.488989", "不检索案例"),
    ("HIC K=1", "0.472727", "0.625197", "1 条训练记忆"),
    ("HIC K=10", "0.463636", "0.597638", "10 条训练记忆"),
    ("HIC K=16", "0.474091", "0.628387", "16 条训练记忆"),
    ("Persistence", "0.383333", "0.487883", "沿用最近观测值"),
    ("History mean", "0.518182", "0.627114", "8 个月历史均值"),
    ("Recent mean 3", "0.464141", "0.561698", "最近 3 个月均值"),
    ("Linear trend", "0.604473", "0.758809", "线性趋势基线"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def table_text(document):
    return [[[cell.text for cell in row.cells] for row in table.rows] for table in document.tables]


def set_run(run, *, size=9, bold=False, color="1F1F1F"):
    run.font.name = "Arial"
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(size)
    run.bold = bold
    run.font.color.rgb = RGBColor.from_string(color)


def append_paragraph(document, text, *, size=9.5):
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(5)
    paragraph.paragraph_format.line_spacing = 1.0
    set_run(paragraph.add_run(text), size=size)


def set_cell(cell, text, *, header=False):
    cell.text = ""
    cell.vertical_alignment = 1
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.paragraph_format.line_spacing = 1.0
    set_run(paragraph.add_run(text), size=8.5, bold=header, color="FFFFFF" if header else "1F1F1F")
    if header:
        shade = OxmlElement("w:shd")
        shade.set(qn("w:fill"), "1F4E78")
        cell._tc.get_or_add_tcPr().append(shade)
    elif len(cell._tc.getparent().getparent().tr_lst) % 2 == 0:
        shade = OxmlElement("w:shd")
        shade.set(qn("w:fill"), "EAF2F8")
        cell._tc.get_or_add_tcPr().append(shade)


def set_table_borders(table):
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


def main():
    if not DOC_PATH.is_file():
        raise FileNotFoundError(DOC_PATH)
    for path in (BACKUP_PATH, TEMP_PATH):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing path: {path}")

    source_hash = sha256(DOC_PATH)
    document = Document(DOC_PATH)
    old_paragraphs = [p.text for p in document.paragraphs]
    old_tables = table_text(document)
    if any(p.text.strip() == SECTION_TITLE for p in document.paragraphs):
        raise RuntimeError(f"Section already exists: {SECTION_TITLE}")

    heading = document.add_paragraph(style="Heading 1")
    heading.paragraph_format.keep_with_next = True
    heading.paragraph_format.space_before = Pt(10)
    heading.paragraph_format.space_after = Pt(6)
    set_run(heading.add_run(SECTION_TITLE), size=14, bold=True)
    append_paragraph(
        document,
        "同一 run ID socialgood_h12_validation_20261018_01 已完成 16 个 fit case 的训练记忆构建，并按时间顺序评估 11 个 validation window；每个窗口包含 8 个月历史和 12 个月目标，共 132 个目标值。每个窗口先用当时可用的历史记忆预测，揭示该窗口真值后再将其追加到 Qdrant，供后续窗口检索。比较 HIC K=0/1/10/16 和四个数值基线。",
    )

    table = document.add_table(rows=1, cols=4)
    table.style = "Table Grid"
    table.autofit = False
    widths = (Inches(1.35), Inches(0.9), Inches(0.9), Inches(3.0))
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    table.rows[0]._tr.get_or_add_trPr().append(repeat)
    for column, value in enumerate(ROWS[0]):
        set_cell(table.rows[0].cells[column], value, header=True)
    for values in ROWS[1:]:
        cells = table.add_row().cells
        for column, value in enumerate(values):
            set_cell(cells[column], value)
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            cell.width = width
    set_table_borders(table)

    append_paragraph(
        document,
        "Persistence 是本次 validation 的最低误差方案；HIC K=10 是检索臂中误差最低的方案，但 MAE/RMSE 均高于 K=0，也高于 Persistence。K 值增加没有带来单调改善。因此，这组数据和当前模板、摘要、检索与预测流程下，历史事件检索没有提升 12 个月预测精度。",
    )
    append_paragraph(
        document,
        "逻辑调用共 137 次，实际 HTTP 尝试 139/272；其中两次 DNS 失败未到达 DeepSeek。Qdrant 最终 K=1/10/16 集合分别有 12/21/27 条记录。只读核验确认每个集合都恰好包含其训练前缀和 11 个 validation case，ID 无缺失或额外项。outer-test 仍延后读取，报告未包含 outer-test ID 或目标。",
    )
    append_paragraph(
        document,
        "这些数字是 11 个已用于开发的 validation window 上的结果，不是盲测或泛化证明；OT 的单位和业务含义仍未从数据源确认。保存报告为 _research_gnn_working/LLM/evaluations/native_hic_pilot_socialgood_h12_validation_20261018_01.json。GraphBundle 仍保持可插拔，不在本次实验中固定 vertex、edge 语义或 GNN。",
    )

    if sha256(DOC_PATH) != source_hash:
        raise RuntimeError("Process Book changed during edit; refusing to install")
    shutil.copy2(DOC_PATH, BACKUP_PATH)
    document.save(TEMP_PATH)
    staged = Document(TEMP_PATH)
    if [p.text for p in staged.paragraphs[:len(old_paragraphs)]] != old_paragraphs:
        raise RuntimeError("Existing paragraphs changed in staged Process Book")
    if table_text(staged)[:len(old_tables)] != old_tables:
        raise RuntimeError("Existing tables changed in staged Process Book")
    if not any(p.text.strip() == SECTION_TITLE for p in staged.paragraphs):
        raise RuntimeError("New results section missing from staged Process Book")

    os.replace(TEMP_PATH, DOC_PATH)
    installed = Document(DOC_PATH)
    if [p.text for p in installed.paragraphs[:len(old_paragraphs)]] != old_paragraphs:
        raise RuntimeError("Installed Process Book did not preserve old paragraphs")
    if table_text(installed)[:len(old_tables)] != old_tables:
        raise RuntimeError("Installed Process Book did not preserve old tables")
    print({
        "path": str(DOC_PATH),
        "backup": str(BACKUP_PATH),
        "paragraphs_before": len(old_paragraphs),
        "paragraphs_after": len(installed.paragraphs),
        "tables_before": len(old_tables),
        "tables_after": len(installed.tables),
        "sha256": sha256(DOC_PATH),
        "section": SECTION_TITLE,
    })


if __name__ == "__main__":
    main()
