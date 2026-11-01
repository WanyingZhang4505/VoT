# Agent Test 8 — 五个 MAE 退化例的事后迭代

本报告是已解封 validation truth 后的定向调优，不是盲测或独立 benchmark；原始 Test 8 15 例报告不变。每例以原始 A 固定比较，最多五轮；首次达到 MAE 改善或持平即停止。

最终方向（相对原始 A）：改善 3，持平 2，退化 0；总修订轮次 5。

| Case | Horizon | 原始 A MAE | 原始 B0 MAE | 修订轮数 | 最终方向 |
|---|---:|---:|---:|---:|---|
| `security_validation_series_20150701T000000` | 10 | 2.8544058e+08 | 2.8844058e+08 | 1 | improved |
| `security_validation_series_20160501T000000` | 10 | 2.4305552e+08 | 2.4413594e+08 | 1 | improved |
| `security_validation_series_20170301T000000` | 10 | 5.0575693e+08 | 5.9158603e+08 | 1 | improved |
| `new_metr_validation_series_20200129T060000` | 192 | 3.953125 | 4.5885417 | 1 | tied |
| `new_metr_validation_series_20200605T060000` | 192 | 2.5677083 | 2.859375 | 1 | tied |

每轮 MAE/MSE/RMSE、correction/forecast 哈希和 worker 记录见同目录 `final_report.json`、`scores/` 与 `cases/`。原始 Qdrant collections 均未写入；outer-test 未读取。


执行审计：5例均一次有效修订，无技术失败或重试。第11例响应由主 agent 上下文按冻结的后验任务生成；request/response hash 已记录。该轮显式使用 validation truth，结果属于后验调参，不是盲测 benchmark。未写入 Qdrant，未读取 outer-test。


**方法范围说明：**本报告中的首轮后验试验直接修订预测，未在生成新预测前注入修订后的 correction。因此它不能回答 correction 更新是否带来改善；该问题由 `../posthoc_correction_driven_degraded5_iter5_20261101/reports/final_report.md` 单独评估。原试验数据保留供审计。
