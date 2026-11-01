# Agent Test 8 评估后历史经验写入记录

## 写入结果

已将15个case的历史经验追加到 `agent_test_8`。collection 从703点增至718点，所有15条都使用独立 namespace `agent_test_8_postscore_validation_experience_v1`；原有703点未覆盖。写入前 collection 为 green，向量配置1024维 Cosine。

写入前快照：`snapshot/agent_test_8-8008848820407063-2026-11-01-02-42-53.snapshot`，大小 15311360 bytes，SHA256 `fb2f62775d5d285ef2073dc4079913c1b8b7a050a5a1fca4cdd839dcf6db7734`。完整 point IDs 和逐点 hash 见 `audit/write_receipt.json`；写入计划及来源见 `audit/ingest_plan.json`。

## 存入内容

每点包含 forecast-time summary、case-specific qualitative correction，以及 validation split、来源、MAE 指标、原始预测锁 hash、truth hash、Top-1 来源与时间资格审计等 metadata。`upsert_completed_case` 只用真值和预测数组做长度/数值校验；payload 不存储原始 truth、A/B 预测数组。Top-1 同源、相同 case 与时间不合格记录按原样保留在 metadata，不做过滤。

5个原先退化的 case 使用后验 correction 驱动试验的修订经验，并标记 `posthoc_validation_feedback_correction_revision`；其余10个记录原始冻结 Test 8 结果。此 collection 现在包含 validation-derived memory，后续检索会接触这些经验，不能再用于声称独立盲测。原始冻结评估的报告、预测锁和703点快照仍保留。现有 `run_agent_test8.py` 有 703 点冻结计数保护；此后不能用已扩展的718点 collection 恢复原 run，应建立新的 run/基线。

| 序号 | Case | 最终经验方向 | 新 point ID | 写入 |
|---:|---|---|---|---|
| 1 | `health_validation_National|X_20150126T000000` | improved | `b4286de1-adad-5e0d-97f2-9613a409e6c0` | inserted |
| 2 | `health_validation_National|X_20151005T000000` | improved | `05e604d6-5fae-5461-995a-31bcc74ebae6` | inserted |
| 3 | `health_validation_National|X_20160613T000000` | improved | `edd7deb3-aa57-5e6f-be9f-42785961a86f` | inserted |
| 4 | `security_validation_series_20150701T000000` | improved | `c8687b92-9a08-5413-9971-2e4146033dfb` | inserted |
| 5 | `security_validation_series_20160501T000000` | improved | `a53d0f84-8c86-5ee7-88a4-8fea84e6de6c` | inserted |
| 6 | `security_validation_series_20170301T000000` | improved | `de223897-ab7d-5a2d-9d61-e18f92ddfdd6` | inserted |
| 7 | `new_metr_validation_series_20190923T060000` | improved | `02ba478f-560c-59be-9f54-587f94cb731d` | inserted |
| 8 | `new_metr_validation_series_20191126T060000` | improved | `545e7b1c-5781-527d-9482-053083b78965` | inserted |
| 9 | `new_metr_validation_series_20200129T060000` | tied | `51c85847-6fbe-5f54-9b69-c0917480ff33` | inserted |
| 10 | `new_metr_validation_series_20200402T060000` | tied | `cf797671-4cbd-57f7-bd95-5aae3f879c69` | inserted |
| 11 | `new_metr_validation_series_20200605T060000` | tied | `734e9663-d99c-576c-845c-6552132f4438` | inserted |
| 12 | `new_metr_validation_series_20200808T060000` | tied | `20f8037e-a43c-5b1d-9082-c3aab6e24f34` | inserted |
| 13 | `new_metr_validation_series_20201011T060000` | improved | `657eede5-def0-57f3-897c-c2b4d046cb03` | inserted |
| 14 | `new_metr_validation_series_20201214T060000` | improved | `96c97a6c-7361-5954-8267-3005a795cf00` | inserted |
| 15 | `new_metr_validation_series_20210216T060000` | tied | `e536e76c-52c5-59ae-9e73-33e873d3cef5` | inserted |

outer-test 未读取。Qdrant 写入仅指向 `agent_test_8`。
