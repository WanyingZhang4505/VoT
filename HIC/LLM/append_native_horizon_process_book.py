"""Append the native-horizon Qdrant fix and integration check to the process book."""

from pathlib import Path
import os
import shutil

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = ROOT / "VoT_HIC_样本与流程记录_100case_stage.docx"
BACKUP_PATH = ROOT / "_research_gnn_working" / "process_book_backup_20261018_native_horizon.docx"
TEMP_PATH = DOC_PATH.with_name("VoT_HIC_样本与流程记录_100case_stage.append.tmp.docx")
SECTION_TITLE = "2026 10 18 HIC 原生 horizon 与 Qdrant 验证记录"


def _style_run(run, *, size=9, bold=False, color=None):
    run.font.name = "Arial"
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(size)
    run.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def _style_cell(cell, text, *, header=False):
    cell.text = ""
    cell.vertical_alignment = 1
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_after = Pt(3)
    paragraph.paragraph_format.line_spacing = 1.08
    _style_run(
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


def main():
    if not DOC_PATH.is_file():
        raise FileNotFoundError(DOC_PATH)
    if BACKUP_PATH.exists() or TEMP_PATH.exists():
        raise FileExistsError("Refusing to overwrite the existing backup or temporary output.")

    document = Document(DOC_PATH)
    if any(paragraph.text.strip() == SECTION_TITLE for paragraph in document.paragraphs):
        raise RuntimeError("The native-horizon process-book section already exists.")
    original_paragraphs = [paragraph.text for paragraph in document.paragraphs]
    original_tables = [_table_text(table) for table in document.tables]

    document.add_page_break()
    heading = document.add_paragraph(style="Heading 1")
    heading.paragraph_format.space_after = Pt(6)
    _style_run(heading.add_run(SECTION_TITLE), size=15, bold=True, color="1F1F1F")

    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(6)
    _style_run(paragraph.add_run(
        "本次按 bounded-search 原则记录一项真实流程修复和一次本地集成检查。候选方案不要求连续提升；每次应记录具体改动、验证集结果、成本与失败情况，最后只称为当前测试范围内观测到的最佳方案。外层 test 保持封存，直到验证集选择完成。"
    ), size=10)

    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(7)
    _style_run(paragraph.add_run(
        "本次改动属于存储层兼容性修复，不是新的预测准确率实验；它没有改变 template、输入摘要、embedding 参数或预测规则。"
    ), size=10)

    rows = [
        ("事项", "具体变更或操作", "验证结果与边界"),
        (
            "原生 horizon 校验",
            "在 `LLM/qdrant_store.py` 移除 observed targets 必须恰好为 12 个的限制；改为要求非空、有限数值，并在提供 prediction 时校验两者长度一致。",
            "支持数据集实际配置的不同预测步数；不再因 H 不等于 12 而拒绝完整 case。",
        ),
        (
            "回归测试",
            "新增 H=6 正常写入、prediction 长度不匹配、空 targets 三种覆盖；运行 Qdrant store、native data/experiment/pilot、memory builder 与 rolling evaluation 六个测试模块。",
            "43/43 项通过。PowerShell TEMP/TMP 指向工作区 `tmp/test_temp`，避开默认临时目录权限问题。",
        ),
        (
            "本地 Qdrant 往返",
            "复用已运行的本机服务，在唯一临时集合中写入一条合成 H=6 case，用相同的 1,024 维单位向量检索，再删除该临时集合。",
            "写入成功，count=1，检索回相同 case，cosine score=1.0；临时集合已删除。没有触碰其他集合。",
        ),
        (
            "结果范围",
            "该探测使用合成 payload 与单位向量，只检查服务连接、写入、检索和 horizon 长度处理。",
            "不是实际 BGE embedding、真实 HIC case 或准确率证据；本次 0 次 DeepSeek/API 调用、0 次 BGE 推理、0 个预测分数。此前 offline raw-fact validation 结果不变，也不代表完整 HIC 最优。",
        ),
    ]

    table = document.add_table(rows=1, cols=3)
    table.style = "Table Grid"
    header_row = table.rows[0]
    header_properties = header_row._tr.get_or_add_trPr()
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    header_properties.append(repeat)
    for cell, text in zip(header_row.cells, rows[0]):
        _style_cell(cell, text, header=True)
    for row in rows[1:]:
        cells = table.add_row().cells
        for cell, text in zip(cells, row):
            _style_cell(cell, text)
    widths = (Inches(1.0), Inches(2.75), Inches(2.7))
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            cell.width = width

    document.save(TEMP_PATH)
    checked = Document(TEMP_PATH)
    if [p.text for p in checked.paragraphs[: len(original_paragraphs)]] != original_paragraphs:
        raise RuntimeError("Existing paragraph content changed during append.")
    if [_table_text(table) for table in checked.tables[: len(original_tables)]] != original_tables:
        raise RuntimeError("Existing table content changed during append.")
    if len(checked.tables) != len(original_tables) + 1:
        raise RuntimeError("Unexpected table count after append.")
    if len(checked.tables[-1].rows) != 5 or len(checked.tables[-1].columns) != 3:
        raise RuntimeError("Unexpected dimensions for the new 5-by-3 table.")
    if not any(paragraph.text.strip() == SECTION_TITLE for paragraph in checked.paragraphs):
        raise RuntimeError("Appended heading is missing.")

    shutil.copy2(DOC_PATH, BACKUP_PATH)
    os.replace(TEMP_PATH, DOC_PATH)
    final = Document(DOC_PATH)
    if len(final.paragraphs) <= len(original_paragraphs) or len(final.tables) != len(original_tables) + 1:
        raise RuntimeError("Final append-only verification failed.")
    print({
        "document": str(DOC_PATH),
        "backup": str(BACKUP_PATH),
        "paragraphs_before": len(original_paragraphs),
        "paragraphs_after": len(final.paragraphs),
        "tables_before": len(original_tables),
        "tables_after": len(final.tables),
        "new_table_dimensions": [len(final.tables[-1].rows), len(final.tables[-1].columns)],
    })


if __name__ == "__main__":
    main()
