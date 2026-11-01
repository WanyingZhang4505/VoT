"""Append the paired validation uncertainty implementation to the Process Book."""

from pathlib import Path
import hashlib
import os
import shutil

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = Path(r"C:\Users\admin\Desktop\Research GNN\VoT_HIC_样本与流程记录.docx")
BACKUP_PATH = WORKSPACE_ROOT / "process_book_backup_20261018_validation_uncertainty.docx"
WORKING_PATH = WORKSPACE_ROOT / "tmp" / "validation_uncertainty_process_book.tmp.docx"
DESKTOP_TEMP_PATH = DOC_PATH.with_name("VoT_HIC_样本与流程记录.validation_uncertainty.tmp.docx")
SECTION_TITLE = "2026-10-18 Paired Validation Uncertainty Implementation"

PARAGRAPHS = [
    "本次实现用户批准的配对移动区块 bootstrap。对同一验证窗口中的预测臂保持配对，每次抽取连续时间窗口区块，计算 ΔMAE 与 ΔRMSE，并报告 95% 区间；固定随机种子以便复现。窗口少于 5 个时只作描述性结果；当前每个配置通常只有一个 series、2–40 个窗口且滚动目标区间可能重叠，因此即使区间也只能称探索性，不作为显著性结论。",
    "新增跨数据集汇总规则：先在每个数据集家族内平均不同 horizon 的相对误差改善，再对数据集家族等权平均，避免原始量纲或配置数量较多的数据集主导整体结论。该函数已实现，但尚未有完整 32 配置的有效 HIC 预测结果可供汇总。",
    "Climate H12 validation report 现在会在验证集内选择数值基线与融合权重 alpha，生成一个派生的 HIC/数值融合预测，并计算配对不确定性。由于基线和 alpha 使用同一组 validation windows 选择，融合分数明确标为 tuning-biased；任何测试期运行前必须冻结配置。原先的 full_reasoning 主比较保持不变。",
    "回查了 8 个 native development pilot checkpoint 和 Climate H12 report：未发现可验证的完整 request_identity，因此不能把这些旧记录认证为精确 API 缓存命中。仅有 prompt hash 不足以复用响应。全语料计划当前是 32 个候选配置、1,150 个 fit windows、309 个 validation windows；缺少的模板调用数为 5。以上仍是离线计划，不代表已跑完或节省了 API 调用。",
    "验证：phase-two 与 full-reasoning 两个测试模块共 24 项通过；两个 Python 模块及测试文件的 py_compile 通过。全量 unittest discovery 未完成：Windows 沙箱默认临时目录出现 PermissionError，4 个观察到的失败都发生在测试写入 TemporaryDirectory 文件时；这次未改动相关测试或临时目录配置。",
    "本次只改本地评估代码与记录，没有调用 DeepSeek、运行 embedding、读写 Qdrant、访问 outer-test 真值或启动全语料评估。新实现尚未在完整数据集上产生预测结果，因此不能据此声称 HIC 准确率提高。",
]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _table_text(table):
    return [[cell.text for cell in row.cells] for row in table.rows]


def _style_run(run, *, size=9.5, bold=False):
    run.font.name = "Arial"
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(size)
    run.bold = bold
    run.font.color.rgb = RGBColor.from_string("1F1F1F")


def main():
    if not DOC_PATH.is_file():
        raise FileNotFoundError(DOC_PATH)
    for path in (BACKUP_PATH, WORKING_PATH, DESKTOP_TEMP_PATH):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing path: {path}")

    source_hash = _sha256(DOC_PATH)
    document = Document(DOC_PATH)
    if any(p.text.strip() == SECTION_TITLE for p in document.paragraphs):
        raise RuntimeError("The paired uncertainty section is already present.")
    original_paragraphs = [paragraph.text for paragraph in document.paragraphs]
    original_tables = [_table_text(table) for table in document.tables]

    document.add_page_break()
    heading = document.add_paragraph(style="Heading 2")
    heading.paragraph_format.space_before = Pt(5)
    heading.paragraph_format.space_after = Pt(6)
    _style_run(heading.add_run(SECTION_TITLE), size=13, bold=True)
    for content in PARAGRAPHS:
        paragraph = document.add_paragraph()
        paragraph.paragraph_format.space_after = Pt(5)
        paragraph.paragraph_format.line_spacing = 1.0
        _style_run(paragraph.add_run(content))

    WORKING_PATH.parent.mkdir(parents=True, exist_ok=True)
    document.save(WORKING_PATH)
    staged = Document(WORKING_PATH)
    if [paragraph.text for paragraph in staged.paragraphs[:len(original_paragraphs)]] != original_paragraphs:
        raise RuntimeError("Existing paragraph content changed during append.")
    if [_table_text(table) for table in staged.tables[:len(original_tables)]] != original_tables:
        raise RuntimeError("Existing table content changed during append.")
    if not any(paragraph.text.strip() == SECTION_TITLE for paragraph in staged.paragraphs):
        raise RuntimeError("Appended section heading is missing.")
    if _sha256(DOC_PATH) != source_hash:
        raise RuntimeError("The Desktop Process Book changed during editing; refusing replacement.")

    shutil.copy2(DOC_PATH, BACKUP_PATH)
    shutil.copy2(WORKING_PATH, DESKTOP_TEMP_PATH)
    os.replace(DESKTOP_TEMP_PATH, DOC_PATH)
    final = Document(DOC_PATH)
    if [paragraph.text for paragraph in final.paragraphs[:len(original_paragraphs)]] != original_paragraphs:
        raise RuntimeError("Post-replacement paragraph preservation check failed.")
    if [_table_text(table) for table in final.tables[:len(original_tables)]] != original_tables:
        raise RuntimeError("Post-replacement table preservation check failed.")
    if not any(paragraph.text.strip() == SECTION_TITLE for paragraph in final.paragraphs):
        raise RuntimeError("Post-replacement section heading is missing.")

    print({
        "backup": str(BACKUP_PATH),
        "paragraphs_before": len(original_paragraphs),
        "paragraphs_after": len(final.paragraphs),
        "tables_before": len(original_tables),
        "tables_after": len(final.tables),
        "existing_content_preserved": True,
        "desktop_sha256_before": source_hash,
        "desktop_sha256_after": _sha256(DOC_PATH),
    })


if __name__ == "__main__":
    main()
