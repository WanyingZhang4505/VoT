# Agent Test 8 — 一次性冻结评估

状态：complete_15_cases_terminal；评分 15/15；技术失败事件 6；重试/恢复动作 6；终态未解决失败 0。

本结果是基于 Test 7 开发记忆库的一次性冻结评估，不是未接触开发经验的外部 benchmark。Health/Security/new_metr 的 target/unit 保持未知口径，按原始数值计算。

## 分组汇总

| 分组 | n | A MAE / MSE / RMSE | B MAE / MSE / RMSE | 改善 / 持平 / 退化 |
|---|---:|---:|---:|---:|
| 总体 | 15 | 68950208 / 6.3052349e+16 / 1.1094175e+08 | 74944176 / 7.2404686e+16 / 1.1740931e+08 | 7 / 3 / 5 |
| Health | 3 | 0.75785635 / 1.2146899 / 1.0108271 | 0.57350059 / 0.65422978 / 0.75586494 | 3 / 0 / 0 |
| Security | 3 | 3.4475101e+08 / 3.1526175e+17 / 5.5470873e+08 | 3.7472085e+08 / 3.6202343e+17 / 5.8704653e+08 | 0 / 0 / 3 |
| new_metr | 9 | 9.1174769 / 180.01331 / 12.064569 | 8.5949074 / 159.41667 / 11.365089 | 4 / 3 / 2 |
| Health/H36 | 3 | 0.75785635 / 1.2146899 / 1.0108271 | 0.57350059 / 0.65422978 / 0.75586494 | 3 / 0 / 0 |
| Security/H10 | 3 | 3.4475101e+08 / 3.1526175e+17 / 5.5470873e+08 | 3.7472085e+08 / 3.6202343e+17 / 5.8704653e+08 | 0 / 0 / 3 |
| new_metr/H192 | 9 | 9.1174769 / 180.01331 / 12.064569 | 8.5949074 / 159.41667 / 11.365089 | 4 / 3 / 2 |

## 逐例结果

| # | Case | A MAE/MSE/RMSE | B MAE/MSE/RMSE | MAE方向 | Top-1 point / source case | 时间资格 |
|---:|---|---:|---:|---|---|---|
| 1 | `health_validation_National|X_20150126T000000` | 0.9886556/1.46939/1.212184 | 0.8010233/1.0296/1.014692 | improved | `ebf17d8d-4908-5f8e-9501-02f6baf971a3` / `health_validation_National|X_20150420T000000` | False |
| 2 | `health_validation_National|X_20151005T000000` | 0.995768/2.013671/1.419039 | 0.656343/0.8076094/0.8986709 | improved | `6ca316e5-2cb6-54ce-a93c-e05bfcd8dc32` / `health_validation_National|X_20160905T000000` | False |
| 3 | `health_validation_National|X_20160613T000000` | 0.2891455/0.1610084/0.4012585 | 0.2631354/0.1254803/0.354232 | improved | `0df4ef5b-8157-5b23-b840-9cc6636132dc` / `health_validation_National|X_20160613T000000` | False |
| 4 | `security_validation_series_20150701T000000` | 2.854406e+08/2.455166e+17/4.954964e+08 | 2.884406e+08/2.453751e+17/4.953535e+08 | worsened | `04d90392-e9c9-5a99-9562-84cfe94411f5` / `security_validation_series_20150901T000000` | False |
| 5 | `security_validation_series_20160501T000000` | 2.430555e+08/2.410668e+17/4.909855e+08 | 2.441359e+08/2.422628e+17/4.922019e+08 | worsened | `cd6e2e57-6583-5701-a735-bd817caaeb69` / `security_validation_series_20160501T000000` | False |
| 6 | `security_validation_series_20170301T000000` | 5.057569e+08/4.592018e+17/6.776443e+08 | 5.91586e+08/5.984325e+17/7.735842e+08 | worsened | `adb35c6f-9e55-59d4-9b59-7bcb7a97ff33` / `security_validation_series_20170901T000000` | False |
| 7 | `new_metr_validation_series_20190923T060000` | 11.19792/230.9896/15.19834 | 10.33854/203.0052/14.24799 | improved | `be9f18ad-d8d0-5ba8-adab-2f8f01036c22` / `new_metr_validation_series_20210216T060000` | False |
| 8 | `new_metr_validation_series_20191126T060000` | 9.286458/121.0885/11.00402 | 7.880208/93.65104/9.677347 | improved | `9d411475-7f5f-5280-bd0d-267e2c003049` / `new_metr_validation_series_20191126T060000` | False |
| 9 | `new_metr_validation_series_20200129T060000` | 3.953125/47.01562/6.856794 | 4.588542/38.25521/6.185079 | worsened | `4ff01674-3cd3-5198-8c09-39f67f221d20` / `new_metr_validation_series_20200129T060000` | False |
| 10 | `new_metr_validation_series_20200402T060000` | 4.734375/68.83854/8.2969 | 4.734375/68.83854/8.2969 | tied | `69686ae9-d3d6-5509-b53b-5ae12a3826ed` / `new_metr_validation_series_20200113T060000` | True |
| 11 | `new_metr_validation_series_20200605T060000` | 2.567708/18.79688/4.335536 | 2.859375/16.18229/4.022722 | worsened | `5ed59414-3f8c-55a4-aae1-7dffe91c46b3` / `new_metr_validation_series_20201027T060000` | False |
| 12 | `new_metr_validation_series_20200808T060000` | 10.20833/285.1562/16.88657 | 10.20833/285.1562/16.88657 | tied | `6af9d4a6-2f7c-57e6-a5da-e986df427292` / `new_metr_validation_series_20200520T060000` | True |
| 13 | `new_metr_validation_series_20201011T060000` | 14.72396/300.9115/17.3468 | 13.14062/256.8802/16.02748 | improved | `458e1bcc-57e8-5326-84e1-964451303e25` / `new_metr_validation_series_20190923T060000` | True |
| 14 | `new_metr_validation_series_20201214T060000` | 20.90625/510.6042/22.59655 | 19.125/436.0625/20.88211 | improved | `1afe15b9-0951-557d-a63f-b54263930d62` / `new_metr_validation_series_20200707T060000` | True |
| 15 | `new_metr_validation_series_20210216T060000` | 4.479167/36.71875/6.0596 | 4.479167/36.71875/6.0596 | tied | `458e1bcc-57e8-5326-84e1-964451303e25` / `new_metr_validation_series_20190923T060000` | True |

Truth 解封逐例记录见 `truth_audit/`；outer-test targets 未读取。评估期间 Qdrant 写入数为 0。

## 技术失败、恢复与协议偏差

详细记录见 [postrun_audit.json](postrun_audit.json)。两次本机 Qdrant socket 权限失败均以相同冻结配置恢复。第 7 例 A 初次响应为 149/192；恢复期间出现两次超时重排和一次 max_output_tokens 响应传输失败。随后同一冻结请求生成了两份完整 A 响应：runner 首先记录并锁定 case07_a_recovery3.json；较晚到达的 case07_a_recovery.json 保留但未评分。第二份完整 A 在锁定前已生成，偏离每例仅一个有效 A 的协议。

## 检索与隔离审计

15 个 Top-1 命中全部原样保留；5/15 严格早于查询历史起点，10/15 时间资格不合格，4/15 为同来源 case/历史起点。truth 审计 15/15 与对应预测锁 hash 匹配且解封晚于锁定；outer-test 未读取。Test 8 final live collection 为 green、703 points、1024D Cosine；评估期间写入 0。

