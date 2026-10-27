# 模型输入、输出与数据构造

## 三层格式

业务层向 `/compact` 传入 `messages`、agent 的 `system_prompt`、工具定义 `tool_defs` 与 `target_ratio_pct`。服务将其渲染成模型输入；模型返回决策协议，服务再生成 coding agent 可消费的新 `messages`。

这些层次不能混用：模型原始输出并不是压缩后的聊天消息数组。

## 模型输入

模型 API 使用两条消息：`system` 为压缩任务指令，`user` 为目标保留率说明及轨迹。轨迹结构为：

```xml
<coding_agent_trajectory>
<coding_agent_system_prompt>原 coding agent 的系统提示</coding_agent_system_prompt>
<coding_agent_tools_define>工具名称、说明及参数定义</coding_agent_tools_define>
<coding_agent_messages>
<user_message><id>1</id><num_tokens>长度提示</num_tokens><content>用户需求</content></user_message>
<assistant_tool_pair>
<id>2</id><num_call_tokens>调用长度提示</num_call_tokens>
<content>调用前说明</content>
<tool_call><function_name>read_file</function_name><function_args>参数</function_args></tool_call>
<num_result_tokens>结果长度提示</num_result_tokens>
<tool_result_content>工具原始结果</tool_result_content>
</assistant_tool_pair>
<assistant_message><id>3</id><num_tokens>长度提示</num_tokens><content>助手消息</content></assistant_message>
</coding_agent_messages>
</coding_agent_trajectory>
```

完整格式由 `service/prompt.py` 定义，以上是结构示意。运行输入构造示例前，请设置 `MODEL_PATH`，服务会直接读取其中的 `tokenizer.json` 计算长度提示。一次 assistant 消息包含多个调用时，按工具调用展开为多个 item；ID 对应渲染后的 item，而不是原消息数组下标。工具调用与结果用非空字符串 `toolCallId` 关联，并匹配 `toolName`；同一组的结果可以分布在紧随该 assistant 消息的多条连续 `tool` 消息中，服务会统一组装。重复 ID、孤立或跨组结果会被拒绝。`examples/request.json` 完全由合成内容组成。

消息顶层可选的 `global_id` 仅用于当前压缩的决策定位。返回的 `messages` 会移除该字段，下一次压缩按新的轨迹重新编号，避免摘要插入和消息删减造成 ID 冲突。工具调用与结果的 `toolCallId`，以及工具输入、输出内部的业务字段保持不变。

```bash
python -m examples.render_model_input > /tmp/compact-model-request.json
curl --fail -H 'Content-Type: application/json' --data-binary @/tmp/compact-model-request.json http://127.0.0.1:8000/v1/chat/completions
```

## 模型输出

根节点为 `<aix_trajectory_compress>`。必须保留 `<aix_analysis>`、`<aix_decisions>`、各项 `<aix_decision_item>` 中的 `<id>`、`<item_analysis>`、`<action>`，以及 `<aix_summary>`。

动作语义：

| 动作 | 适用项 | 效果 |
|---|---|---|
| `keep` | 所有 item | 保留原项 |
| `remove` | 所有 item | 删除该项，摘要应保留仍重要的信息 |
| `keep_with_replaced_tool_result` | assistant_tool_pair | 保留调用，仅以 `<replacement_tool_result_content>` 替换工具结果 |

每个输入 item 应有且只有一个决策，按 ID 顺序输出。摘要至少覆盖用户意图、技术概念、错误与修复、待办事项、当前工作。示例见 `examples/response.xml`；它是人工构造的格式样例，不是质量评测结果。

这是 **XML-like** 协议：标签内代码、`<`、`&` 等按原文保存，并非标准转义 XML。解析器按规定的字段顺序读取，分析或替换文本中引用的其他协议标签不会被当作决策。如果正文包含它自身承载标签的结束标记，请使用 `<![CDATA[...]]>` 包裹该字段，避免边界歧义。不要直接套用通用 XML DOM。模型输出必须包含完整根节点、非空分析与摘要，以及按输入顺序排列的全部唯一 ID；无效协议返回 502，保留率兜底只处理已经通过协议校验的结果。

`<aix_analysis>` 和 `<item_analysis>` 是任务协议的一部分。DeepSeek API 的独立 `reasoning_content`/外层 `<think>` 才是本次评测排除的 thinking，二者不能混淆。

## 兼容数据格式

如果要接入新的应用，请将自己的对话历史渲染成上面定义的轨迹格式，保留角色、工具调用 ID、工具结果以及任务系统提示。一次部署内应固定 prompt 模板与 tokenizer。仓库只提供合成样例作为接口数据格式示例。

assistant 目标必须包含完整的协议根节点。推理时只发送 system 和 user 两条消息，不要把参考答案发送给模型。将模型输出交给组装器前，请先校验 ID、动作合法性、工具配对和摘要结构。

本项目不公开内部数据来源、教师生成流程、过滤、采样或训练/测试划分方式。
