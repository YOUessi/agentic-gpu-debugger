# 定向复测费用统计

`scripts/rerun_failed_public_units.py` 现在保存逐次调用 ID、完整 usage 和 accounting
汇总。可选 `--accounting-rates rates.json` 只用于估算，不影响执行权限、次数或费用上限。
没有费率时总费用仍为 null；已知部分单独记录，未知调用不会当作免费。

费率文件字段：model、source、input_usd_per_million、output_usd_per_million，
以及可选 cached_input_usd_per_million。输入/输出价格按每百万 token 的美元填写。
有缓存 token 而没有缓存价格时，不会擅自套用普通输入价格。

历史复测无需重跑。只读工具核对公开调用产物的路径、大小、哈希和调用 ID，
同一调用的 STARTED 与终态只计一次；输出 JSON，不修改旧实验目录：

```bash
PYTHONPATH=src python tools/retest_accounting.py /path/to/retest --rates /path/to/rates.json
```

2026-09-26 的八项复测共 35 次调用，输入 99609、输出 8233 token。
沿用项目历史记录的输入 1.32、输出 3.96 美元/百万 token，且明确不计缓存优惠时，
估算合计 0.16408656 美元（格式两项 0.03946932，超时六项 0.12461724）。
这不是账单，也不是当前价格声明；真实扣费由供应商账单确定。
本次补算没有查询余额或新调用 API，没有覆盖原 cost_usd=null 的历史文件。
