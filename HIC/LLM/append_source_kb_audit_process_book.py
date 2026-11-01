"""Append a source corpus/vector-cache audit to the HIC process book."""

import hashlib
import os
import shutil
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = Path(r"C:\Users\admin\Desktop\Research GNN\VoT_HIC_样本与流程记录.docx")
BACKUP_PATH = WORKSPACE_ROOT / "process_book_backup_20261018_source_kb_audit.docx"
WORKING_PATH = WORKSPACE_ROOT / "tmp" / "source_kb_audit_process_book.tmp.docx"
DESKTOP_TEMP_PATH = DOC_PATH.with_name("VoT_HIC_样本与流程记录.source_kb_audit.tmp.docx")
SECTION_TITLE = "2026 10 18 原始案例与向量缓存核对"

TEXT = [
    "为避免把模型缓存误认为已经完成的 HIC 案例，本次只盘点了公开 VoT 仓库与本地知识库，没有生成新预测。公开仓库的 LLM/hic.py 目前写有一个训练样例（#611 OT，8 个历史值、12 个真值和一段固定新闻）以及一个测试样例（#719 OT，8 个历史值和一段固定新闻）。仓库中其他数据集文件和预测结果尚未构成 100 条新闻、纠错与结果齐全的 HIC 记忆库。",
    "本地 BAAI/bge-m3 缓存包含 100 个唯一 case ID、100 段文本摘要及 100 个 1,024 维向量，配置上限为 8,192 tokens，实测训练文本为 448 至 1,110 tokens。这个缓存没有附带真值或纠错。另一份 100 条记录仍使用 BAAI/bge-small-en-v1.5 的 384 维向量，并存有结构化摘要与纠错；两份文件摘要表示不同，尚未验证一一对应关系，因此不能按行号合并或当成已完成的 M3 案例入库。",
    "本地 Qdrant 地址 127.0.0.1:6333 当前无响应。Docker Desktop Linux engine 管道返回权限拒绝，因此本次未读取或写入任何 collection。没有调用 DeepSeek API、没有新跑 embedding、没有训练 GNN，也没有访问 outer-test。恢复本地 Docker 连接后，应仅把来源、向量、纠错和完整结果都能核验的案例导入独立 collection。",
    "本次再次确认 GraphBundle 的生产节点、边和模型选择都保持开放。现有 case-memory 图、显式层级图和 GraphSAGE 均为接口示例，不代表最终业务建模方案。graph_contract.py 的 10 项测试全部通过；后续只需按最终研究定义新增图构建器和模型适配器。",
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
        raise RuntimeError("Source corpus audit section already exists")
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
    if [p.text for p in candidate.paragraphs[:len(original_paragraphs)]] != original_paragraphs:
        raise RuntimeError("Existing process-book paragraphs changed")
    if [_table_text(t) for t in candidate.tables[:len(original_tables)]] != original_tables:
        raise RuntimeError("Existing process-book tables changed")
    if not any(p.text.strip() == SECTION_TITLE for p in candidate.paragraphs):
        raise RuntimeError("New section heading is missing")
    if _sha256(DOC_PATH) != original_hash:
        raise RuntimeError("Desktop process book changed during editing; refusing replacement")

    shutil.copy2(DOC_PATH, BACKUP_PATH)
    shutil.copy2(WORKING_PATH, DESKTOP_TEMP_PATH)
    os.replace(DESKTOP_TEMP_PATH, DOC_PATH)

    final = Document(DOC_PATH)
    if [p.text for p in final.paragraphs[:len(original_paragraphs)]] != original_paragraphs:
        raise RuntimeError("Post-replacement paragraph preservation check failed")
    if [_table_text(t) for t in final.tables[:len(original_tables)]] != original_tables:
        raise RuntimeError("Post-replacement table preservation check failed")
    if not any(p.text.strip() == SECTION_TITLE for p in final.paragraphs):
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
