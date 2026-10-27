# 服务接口与兜底行为

本服务保留 v2.2 压缩协议。默认端口为 12003，调用可配置的 OpenAI 兼容 vLLM 接口；不需要私有网关或 Apply 服务。

| 接口 | 说明 |
|---|---|
| GET `/health` | HTTP 服务存活检查，不代表模型已经就绪 |
| POST `/validate` | 只接收 `messages`，不执行推理 |
| POST `/compact` | 返回 `{status:"ok", data:{messages,stats}}` |
| POST `/cancel/{request_id}` | 取消在途请求；找到时返回 200，已完成/未注册时返回 404 |

`/compact` 接收 `messages`，以及可选的 `request_id`、`system_prompt`、`tool_defs`、`tool_names`、`target_ratio_pct` 和 `debug`。普通调用可以不传 `request_id`，但要支持取消就必须传入新的 UUID。重复的在途 ID 返回 409；顶层字段或类型错误返回 400；消息结构、工具 ID 或配对错误返回 422；取消时原请求返回 499；后端错误或无效模型协议返回 502。错误响应不包含替换后的 `messages`，OpenCode 会保留原历史。

`target_ratio_pct` 是**保留比例**，不是删除比例：25 表示目标保留约四分之一。默认值为 25，允许 1–99，建议使用 10–50 之间的 5 的倍数。

`tool_defs` 中每项包含 `name`、`description`，以及可选的 `parameters`（完整 JSON Schema 对象）和兼容的 `args` 参数说明。服务将完整描述与 JSON Schema 写入现有模型协议的 `<tool_description>`，保留根级约束、嵌套结构和 `$defs`，不截断工具描述。`parameters` 类型错误返回 400。

## 处理顺序

1. 校验消息结构、ID 类型与唯一性，将紧随同一 assistant 调用组的多条工具结果合为一个结果组；保留所有结果内容。
2. 将 agent 系统提示、工具描述和历史消息渲染为 XML-like 轨迹。
3. 流式调用模型，只把任务协议内容作为模型输出。
4. 按协议字段顺序解析；检查完整根节点、非空分析与摘要、逐项 ID 覆盖以及替换动作的适用对象。全部通过后才应用删除或替换决策。
5. 计算可见消息 token，并执行保留率兜底检查。

## 保留率兜底

目标值为 `t` 时，有效区间为 `[max(0,t−20)/100, min(100,t+20)/100)`。达到或超过上限表示压缩不足；低于下限表示压缩过度。输入少于 2000 token 时跳过下限检查。

触发兜底后，服务会丢弃模型的逐项决策，保留摘要，并从最近的连续消息单元开始追加，工具调用与结果始终成对保留。当下一个消息单元放不进预算时停止，不会跳过它去保留更早消息。

**预算边界：**如果真实摘要本身超过目标预算，服务仍会保留完整摘要，并通过 `summary_over_budget` 标记。因此最终 token 数并不保证始终严格小于目标。缺失摘要、重复或遗漏决策 ID、无效动作等协议错误直接返回 502，不会把原始分析或截断文本作为摘要返回。

重点响应字段：`actual_retention`、`retention_band`、`fallback_triggered`、`fallback_reason`、`schema`、`budget_tokens`，以及必要时的 `summary_over_budget`。成功响应中的 `has_structured_output` 为 `true`，表示协议已经通过校验。

## Token 计算与调试

模型输入中的长度提示直接读取 `$MODEL_PATH/tokenizer.json`，使用与导出模型相同的特殊 token。跨机器部署时可用 `LENGTH_TOKENIZER_PATH` 指定单独复制的 tokenizer。服务启动时加载并缓存 tokenizer，不加载模型权重；配置或文件缺失会直接启动失败。

长度提示、保留率与兜底预算统一使用本地模型 tokenizer；不调用 `/tokenize`，也不会因网络故障切换到字符估算。`/validate` 中的 `estimated_tokens` 仍是消息 JSON 的粗略字符估算，仅用于诊断，不参与保留率决策。默认不落盘保存原始 trace；仅在私有调试时设置 `TRACE_ENABLED=1` 和 `TRACE_DIR`。`debug:true` 会在响应中返回原始 prompt/output，但不会启用磁盘 trace。日志默认写入 stderr，也可用 `LOG_FILE` 开启轮转日志。

取消请求会关闭上游流式连接。取消注册表保存在单进程内存中，因此只支持一个 Uvicorn worker；多 worker 或负载均衡部署需要替换该注册表。
