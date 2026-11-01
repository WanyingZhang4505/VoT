# Agent Test 8：五个退化 case 的 correction 驱动后验实验

本实验纠正前一轮的操作解释：每轮先根据该 case 已解封的 validation 误差更新原 Top-1 correction，再把修订版 correction 注入原 forecast-time 输入，生成一次新 B。预测任务只含原预测时点可见输入和修订 correction，不含 validation truth。最多5轮；达到相对原始 A 的 MAE 改善或持平即停止。该实验使用同一 case 的验证反馈修订 correction，属于后验开发调参，不是盲测 benchmark。

5/5 个 case 在第1轮即达到 MAE 改善或持平：改善3例、持平2例、退化0例。每例的 MAE/MSE/RMSE、修订 correction、任务与响应 hash 见 `final_report.json`、`corrections/`、`tasks/` 和 `scores/`。

| Ordinal | Case | A MAE | 原始 B0 MAE | 修订 correction 后 B MAE | 对 A |
|---:|---|---:|---:|---:|---|
| 4 | `security_validation_series_20150701T000000` | 2.854406e+08 | 2.884406e+08 | 2.083653e+08 | improved |
| 5 | `security_validation_series_20160501T000000` | 2.430555e+08 | 2.441359e+08 | 2.283962e+08 | improved |
| 6 | `security_validation_series_20170301T000000` | 5.057569e+08 | 5.91586e+08 | 3.732934e+08 | improved |
| 9 | `new_metr_validation_series_20200129T060000` | 3.953125 | 4.588542 | 3.953125 | tied |
| 11 | `new_metr_validation_series_20200605T060000` | 2.567708 | 2.859375 | 2.567708 | tied |

审计限制：原始 validation truth 数组没有进入 reforecast prompt；修订 correction 本身包含由该 truth 计算出的定性误差反馈，这是本实验明确的后验调参机制。outer-test 未读取，Qdrant 未写入。修订预测由主 agent 上下文生成，未调用独立推理 worker；因此数值作为后验人工执行结果解读。Test 8 原始预测锁和报告保留未改。
