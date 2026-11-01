"""Append the generic HIC development-runner record to the canonical Process Book."""

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"C:\Users\admin\Desktop\Research GNN\VoT_HIC_样本与流程记录.docx")
BACKUP = PROJECT_ROOT / "tmp" / "process_book_backup_20261018_hic_generic_runner.docx"
STAGED = PROJECT_ROOT / "tmp" / "VoT_HIC_样本与流程记录.hic_generic_runner.tmp.docx"
DESKTOP_STAGED = SOURCE.with_name("VoT_HIC_样本与流程记录.hic_generic_runner.tmp.docx")
SECTION_TITLE = "2026-10-18 Generic HIC Development Runner"

PARAGRAPHS = [
    "本次新增离线可测、可恢复的 HIC 开发阶段执行框架。RequestCache 以请求设置和完整 prompt 的 SHA-256 作为精确身份，不把 prompt、原始 call_id 或凭据写入缓存；超时后留下未决状态，只有显式授权重试才会再次调用。Windows 文件替换遇到短暂锁冲突时会有限重试，最终失败则保留完整恢复文件。",
    "单配置 runner 固定使用 native HIC 的 fit/validation 划分，并强制调用 load_native_hic_splits(include_outer_test_targets=False)。每个 fit case 依次完成 summary、初始预测、结果纠错、embedding，只有完整纠错记录才进入专属 fit-memory。检查点绑定配置、模板、source SHA-256 和可预测输入，恢复时会拒绝不匹配的来源或模板。",
    "validation 阶段固定 fit-memory，不把 validation case 写入向量库。每个窗口共用同一份 summary 和 top-1 fit 历史案例，成对生成 no_retrieval、summary_retrieval 与 full_reasoning 三路预测，并计算 persistence、history mean、recent mean 和 linear trend 数值基线。只有三路预测都写入检查点后，runner 才读取该窗口的 validation truth 并评分。",
    "报告还会在 validation windows 上选择数值基线及 HIC/数值融合权重，并标出同一验证集选择带来的 tuning bias。配对移动区块 bootstrap 沿用既有不确定性实现；窗口少于 5 个时只作描述性结果，当前单序列区间仍属于探索性证据。跨数据集汇总先计算各系列的相对改善，再对数据集家族等权汇总。",
    "真实离线 planner 本次核验到 32 个 development-ready dataset/horizon 配置、1,150 个 fit case 和 309 个 validation window；plan 标记 outer_test_targets_read=false、provider_requests_made=0。Agriculture H6、Energy H12 和 SocialGood H6 使用合成 fixture、假 provider、假 embedder 与内存 store 完成了 runner smoke test。这只验证流程和隔离契约，不代表真实 HIC 预测结果。",
    "验证结果：新增 request-cache 与 grid-runner 共 20 项 focused unittest 通过；活动 tests/ 全套 pytest 为 200 passed、1 skipped、12 subtests passed。运行命令为 .venv\\Scripts\\python.exe -B -m pytest -p no:cacheprovider -q tests，并将 TEMP/TMP 指向项目内 tmp，以避开 Windows 默认临时目录权限限制。本次没有新增第三方依赖，因此未安装软件包。",
    "本次没有调用 DeepSeek、运行 BGE embedding、读写 Qdrant、计算 outer-test 指标或启动全语料真实预测。32 个配置仍只是可执行的开发清单，尚未得到真实 HIC 预测及准确性结果；启动真实开发运行前仍须确认该次的 API logical-call 与 HTTP-attempt 上限。封存 outer-test 继续保持未读取。",
]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _table_snapshot(document):
    return [[cell.text for cell in row.cells] for table in document.tables for row in table.rows]


def _style_run(run, *, size=9.5, bold=False):
    run.font.name = "Arial"
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(size)
    run.bold = bold
    run.font.color.rgb = RGBColor.from_string("1F1F1F")


def _verify_preserved(document, paragraphs, tables, inline_shapes, sections):
    if [paragraph.text for paragraph in document.paragraphs[:len(paragraphs)]] != paragraphs:
        raise RuntimeError("existing Process Book paragraphs changed")
    if _table_snapshot(document)[:len(tables)] != tables:
        raise RuntimeError("existing Process Book tables changed")
    if len(document.inline_shapes) != inline_shapes or len(document.sections) != sections:
        raise RuntimeError("existing Process Book images or section structure changed")


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    if not PROJECT_ROOT.joinpath("tmp").is_dir():
        raise FileNotFoundError(PROJECT_ROOT / "tmp")
    for path in (BACKUP, STAGED, DESKTOP_STAGED):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {path}")

    original_hash = _sha256(SOURCE)
    document = Document(SOURCE)
    if any(paragraph.text.strip() == SECTION_TITLE for paragraph in document.paragraphs):
        raise RuntimeError("generic-runner section already exists")
    original_paragraphs = [paragraph.text for paragraph in document.paragraphs]
    original_tables = _table_snapshot(document)
    original_inline_shapes = len(document.inline_shapes)
    original_sections = len(document.sections)

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

    document.save(STAGED)
    _verify_preserved(Document(STAGED), original_paragraphs, original_tables, original_inline_shapes, original_sections)
    staged_document = Document(STAGED)
    if not any(paragraph.text.strip() == SECTION_TITLE for paragraph in staged_document.paragraphs):
        raise RuntimeError("staged Process Book section is missing")
    if _sha256(SOURCE) != original_hash:
        raise RuntimeError("canonical Process Book changed during staging")

    shutil.copy2(SOURCE, BACKUP)
    shutil.copy2(STAGED, DESKTOP_STAGED)
    try:
        if _sha256(SOURCE) != original_hash:
            raise RuntimeError("canonical Process Book changed before replacement")
        for attempt in range(5):
            try:
                os.replace(DESKTOP_STAGED, SOURCE)
                break
            except PermissionError as error:
                if attempt == 4:
                    raise PermissionError("could not replace Process Book after five attempts") from error
                time.sleep(0.1 * (attempt + 1))
    finally:
        if DESKTOP_STAGED.exists():
            DESKTOP_STAGED.unlink()

    final_document = Document(SOURCE)
    _verify_preserved(final_document, original_paragraphs, original_tables, original_inline_shapes, original_sections)
    if not any(paragraph.text.strip() == SECTION_TITLE for paragraph in final_document.paragraphs):
        raise RuntimeError("new section is missing after installation")
    if len(final_document.paragraphs) != len(original_paragraphs) + len(PARAGRAPHS) + 2:
        raise RuntimeError("unexpected paragraph count after installation")
    print(json.dumps({
        "backup": str(BACKUP),
        "paragraphs_before": len(original_paragraphs),
        "paragraphs_after": len(final_document.paragraphs),
        "tables_before": len(document.tables),
        "tables_after": len(final_document.tables),
        "inline_shapes_preserved": len(final_document.inline_shapes) == original_inline_shapes,
        "existing_content_preserved": True,
        "sha256_before": original_hash,
        "sha256_after": _sha256(SOURCE),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
