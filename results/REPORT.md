# aiXcompact 评测与成本报告

本报告汇总内部评测的最终结果。aiXcompact 的平均上下文保留率为 **25.89%**，即减少 **74.11%**。评测脚本、测试集与逐条真实输出不随本项目发布。

数据来自内部保留评测集。数据集、划分方式和逐条结果不公开；本报告只发布聚合指标。

## 原始模型效果


| 指标                  | DeepSeek 历史参考 | aiXcompact 发布权重 |
| ------------------- | ------------- | ------------ |
| 规则综合分               | 0.7332        | 0.7012       |
| 格式有效率               | 1.0000        | 0.9939       |
| 严格格式有效率             | 0.9817        | 0.9573       |
| 决策覆盖                | 1.0000        | 0.9936       |
| 信息保留规则分             | 0.4968        | 0.4981       |
| 替换质量规则分             | 0.4556        | 0.4011       |
| 平均保留率（全部样本；失败按原文保留） | 18.46%        | 25.89%       |
| 成功样本目标偏差（百分点）       | 12.60         | 15.07        |


规则分不是任务成功率；教师输出经过数据构造筛选。独立 thinking 不计入，协议中的 analysis、decisions、replacement、summary 全部保留。质量、格式、成功率与全样本保留率均使用完整内部评测集分母。目标偏差等成功条件统计单列；逐条原始计数不随本项目发布。

## GPU 成本对比：aiXcompact vs DeepSeek V4 Pro

币种 USD，公开租价情景，独立 thinking 排除。aiXcompact 用实测请求时间；DeepSeek 为下述公开吞吐代理下的参考预算，尚未核验同负载的 prefill/decode 成本。


| 情景       | aiXcompact / 次 | DeepSeek GPU 参考情景 / 次 | 估算节省   |
| -------- | ----------- | ---------------------- | ------ |
| 利用率 100% | $0.030897   | $0.141652              | 78.19% |
| 利用率 50%  | $0.061794   | $0.283303              | 78.19% |
| 利用率 25%  | $0.123588   | $0.566606              | 78.19% |


aiXcompact 计价为 1×A100 SXM 40GB、$1.99/GPU-hour。

### DeepSeek GPU 参考情景

采用 NVIDIA Dynamo [固定版本的](https://github.com/ai-dynamo/dynamo/blob/1278ebef752b6795c2d61caa90d9e3bd5d64764d/recipes/deepseek-v4/deepseek-v4-pro/README.md) `agg-b200-agentic` [基准](https://github.com/ai-dynamo/dynamo/blob/1278ebef752b6795c2d61caa90d9e3bd5d64764d/recipes/deepseek-v4/deepseek-v4-pro/README.md)：`nvidia/DeepSeek-V4-Pro-NVFP4`，Dynamo + vLLM、8×B200、TP8 + EP、MTP-2 投机解码、并发 13，系统输出吞吐为 72.79 tokens/s/GPU。按 [Lambda 公开租价](https://lambda.ai/pricing) $6.69/B200-hour 计价。

本批 164 条历史 DeepSeek 参考的完整协议输出累计 909,941 tokens，平均每次 5,548.42 tokens。独立 thinking 排除，协议内 analysis、decisions、replacement 和 summary 全部保留；未记录的 thinking 长度不作为零长度观测。

**假设本压缩负载维持公开基准的系统吞吐不变**，按整组吞吐分摊费用：

```text
平均协议输出 O = 909,941 / 164 ≈ 5,548.420732 tokens
整组输出吞吐 Q = 8 × 72.79 = 582.32 tokens/s
整组小时租价 H = 8 × 6.69 = $53.52/hour
每次参考 GPU 成本 = O / Q × H / 3600 / U ≈ $0.141652 / U
```

U=100% 时约为 **$0.141652/次、$141.652/千次**。`U` 仅分摊基准运行之外的额外闲置租卡时间，不是显存利用率；已通过整组吞吐分摊并发费用，不再额外乘并发数 13。估算不包含冷启动、预热、服务后处理、额外重试、税费、存储和运维成本。

该[公开负载](https://github.com/ai-dynamo/dynamo/blob/1278ebef752b6795c2d61caa90d9e3bd5d64764d/recipes/deepseek-v4/README.md)的中位输入为 64K、输出 400 tokens、90% KV reuse；本批输入平均约 59.5K、协议输出约 5.5K tokens，且尚未取得相同缓存、并发、输出范围下的性能测量。**上述数字是吞吐代理情景，不是同负载实测成本或实际成本上下界。** 表中百分比按 `1 − aiXcompact / DeepSeek` 计算，三档利用率同为 78.19%，因为两边费用都按 `1/U` 放大；它只比较该代理情景下的每次费用，不是同负载实测节省。 本任务的 prefill/decode 用时、投机接受率及排除独立 thinking 后的性能仍未核验，不能由 GPU 数量比、报价比或历史 API 耗时替代。

## 发布边界

公开仓库保留本聚合报告和质量图表；逐条评测输出、价格复核明细、运行清单和内部日志不随项目发布。

## 运行与解码边界

观测显存包含 vLLM 预分配的 KV cache，不是权重自身大小。
发布运行中有请求达到输出上限；该失败保留在聚合分母中，未删除或挑选重跑。

合成样例在 temperature=0、max_tokens=4096 时，两种模式均正常结束，但普通/ngram 文本不一致：分别 1464 / 809 tokens。首次分歧在第 29 个 token；ngram 的两个候选 log-probability 并列，普通模式相差 0.125。未证明唯一成因，不承诺逐 token 等价。本报告的全量质量、性能与成本均使用 ngram k=32。
上述观察仅用于说明两种解码配置存在可见差异；本报告的全量结果统一使用 ngram k=32。
