"""Append the framework-neutral GraphBundle checkpoint to the HIC process book."""

import hashlib
import os
import shutil
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = Path(r"C:\Users\admin\Desktop\Research GNN\VoT_HIC_样本与流程记录.docx")
BACKUP_PATH = WORKSPACE_ROOT / "process_book_backup_20261018_graph_contract.docx"
WORKING_PATH = WORKSPACE_ROOT / "tmp" / "graph_contract_process_book.tmp.docx"
DESKTOP_TEMP_PATH = DOC_PATH.with_name("VoT_HIC_样本与流程记录.graph_contract.tmp.docx")
SECTION_TITLE = "2026 10 18 通用 GraphBundle 接口"

TEXT = [
    "本次把 Graph ML 部分收敛为可插拔的数据接口。节点代表什么、边有哪些业务含义、最终选哪一种 GNN 目前都没有定；之前的 case-memory 图、层级图和 GraphSAGE 只作为独立的数据流示例，不视为最终模型。",
    "新增 LLM/graph_contract.py。GraphBundle 用稳定 ID 描述节点和边，节点类型及关系名开放扩展；特征按命名 block 保留，并记录 available_at。监督目标单独保存及标注可见时间。校验器检查节点引用、索引一致性和 as_of 截止时间，不负责替研究者决定图拓扑、特征编码或模型。",
    "验证时使用已保存的 31 条 Climate H12 记录，按时间顺序构建 6 个验证查询的示例图。节点数依次为 26 至 31，边数为 36 至 41；每个查询只暴露此前已完成案例的 25 至 30 个监督目标，当前 query 标签均未暴露。该步骤验证接口转换和时间边界，不是模型训练，也不代表准确率提升。",
    "包含 GraphBundle 的 61 项相关回归测试全部通过。此次没有调用 DeepSeek、运行新的 embedding、改动 Qdrant、训练 GNN 或读取 outer-test 标签。后续确定业务节点/边定义后，只需新增对应的 source builder 和模型特征 adapter，并保持本接口的时间可见性检查。",
]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _table_text(table):
    return [[cell.text for cell in row.cells] for row in table.rows]


def _add_text(document, text):
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(6)
    paragraph.paragraph_format.line_spacing = 1.08
    run = paragraph.add_run(text)
    run.font.name = "Arial"
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(10)


def main():
    if not DOC_PATH.is_file():
        raise FileNotFoundError(DOC_PATH)
    for path in (BACKUP_PATH, WORKING_PATH, DESKTOP_TEMP_PATH):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing path: {path}")

    original_hash = _sha256(DOC_PATH)
    document = Document(DOC_PATH)
    if any(paragraph.text.strip() == SECTION_TITLE for paragraph in document.paragraphs):
        raise RuntimeError("GraphBundle process-book section already exists")
    original_paragraphs = [paragraph.text for paragraph in document.paragraphs]
    original_tables = [_table_text(table) for table in document.tables]

    document.add_page_break()
    heading = document.add_paragraph(style="Heading 1")
    heading.paragraph_format.space_after = Pt(8)
    run = heading.add_run(SECTION_TITLE)
    run.font.name = "Arial"
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.color.rgb = RGBColor(31, 31, 31)
    for text in TEXT:
        _add_text(document, text)

    document.save(WORKING_PATH)
    candidate = Document(WORKING_PATH)
    if [paragraph.text for paragraph in candidate.paragraphs[:len(original_paragraphs)]] != original_paragraphs:
        raise RuntimeError("Existing process-book paragraphs changed")
    if [_table_text(table) for table in candidate.tables[:len(original_tables)]] != original_tables:
        raise RuntimeError("Existing process-book tables changed")
    if not any(paragraph.text.strip() == SECTION_TITLE for paragraph in candidate.paragraphs):
        raise RuntimeError("New section heading is missing")
    if _sha256(DOC_PATH) != original_hash:
        raise RuntimeError("Desktop process book changed during editing; refusing replacement")

    shutil.copy2(DOC_PATH, BACKUP_PATH)
    shutil.copy2(WORKING_PATH, DESKTOP_TEMP_PATH)
    os.replace(DESKTOP_TEMP_PATH, DOC_PATH)

    final = Document(DOC_PATH)
    if [paragraph.text for paragraph in final.paragraphs[:len(original_paragraphs)]] != original_paragraphs:
        raise RuntimeError("Post-replacement paragraph preservation check failed")
    if [_table_text(table) for table in final.tables[:len(original_tables)]] != original_tables:
        raise RuntimeError("Post-replacement table preservation check failed")
    if not any(paragraph.text.strip() == SECTION_TITLE for paragraph in final.paragraphs):
        raise RuntimeError("Post-replacement heading verification failed")

    print({
        "document": str(DOC_PATH),
        "backup": str(BACKUP_PATH),
        "paragraphs_before": len(original_paragraphs),
        "paragraphs_after": len(final.paragraphs),
        "tables_before": len(original_tables),
        "tables_after": len(final.tables),
        "existing_content_preserved": True,
        "sha256_before": original_hash,
        "sha256_after": _sha256(DOC_PATH),
    })


if __name__ == "__main__":
    main()
