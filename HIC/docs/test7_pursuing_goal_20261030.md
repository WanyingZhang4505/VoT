# Agent Test 7 持续目标

基于 Qdrant `agent_test_7` 中从 `agent_test_6` 完整复制的 608 条记录，顺序完成 Health H12/H24/H48、Security H6/H8/H12 和 Weather H48/H96/H336 的 95 个开发 case。使用固定的 BAAI/bge-m3 生成当前 case 向量，在 Test 7 全部现有记录中检索最相近的一条，并直接把该记录的 correction 放入 Full HIC；不再叠加数据集、目标、单位、频率、实体或 horizon 资格筛选。No Retrieval A 与 Full HIC B 使用相同当前输入，分别独立生成，B 不读取 A 的预测数组或理由。

每个 case 先保存并锁定 A/B，再揭示该开发 case 真值并计算 MAE/MSE。若 B 的 MAE 严格优于固定 A，则结束该 case；若持平或更差，诊断失败原因，修订本轮命中的 correction，并生成下一轮 B。最多进行三轮 Full HIC 预测，改善即提前停止；三轮后仍未改善则记录终态，不按真值从多份预测中择优。技术失败单独标记，不计为持平。

完成评分后，将该 case 的摘要向量及 outcome-blind correction 写入 Test 7，供后续 case 检索；不写入当前 case 的答案或预测数组。每个 case 独立记录检索命中、revision、prompt、请求/响应、评分与诊断。95 例及其迭代均属于开发评估，不宣称为独立测试。保留失败与退化结果，逐 3 个完成 case 汇报进度，并报告整体、分家族、分配置的 MAE/MSE/RMSE、改善/持平/退化比例及严重退化。
