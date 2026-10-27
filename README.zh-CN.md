<div align="center">

# aiXcompact

### 压缩历史上下文，让编程任务继续。

**aiXcoder 推出的 4B 编程智能体上下文压缩模型与接入工具。**

[English](README.md) · [快速开始](#快速开始) · [接入 OpenCode](#接入-opencode) · [评测结果](#评测结果) · [文档导航](#文档导航)

</div>

长时间运行的编程智能体会积累大量文件内容、工具输出和中间讨论。**aiXcompact** 将这些历史整理为更短的续接上下文：选择需要保留的对话项、精简工具结果，并生成覆盖任务进展与待办事项的摘要。

本项目提供模型部署方式、完整压缩服务、调用示例，以及固定版本的 OpenCode / vLLM 适配补丁。

## 核心能力

- **结构化压缩**：逐项输出“保留、删除、替换工具结果”决策，便于检查上下文如何被处理。
- **单卡部署**：基于 Qwen3.5-4B 的 BF16 合并文本模型，已在单张 A100 40GB 上验证。
- **智能体接入**：接收历史消息、系统提示和工具定义，返回后续对话可直接使用的消息列表。
- **运行控制**：支持目标保留率、确定性兜底、请求取消，并返回实际保留率和兜底状态。

## 工作原理

```mermaid
flowchart LR
    A[智能体历史消息] --> B[构造轨迹输入]
    B --> C[aiXcompact 4B · vLLM]
    C --> D[逐项决策 + 摘要]
    D --> E[组装消息 + 检查保留率]
    E --> F[后续对话上下文]
```

模型负责生成压缩决策；服务负责将决策应用到原始消息，并保持工具调用与结果的配对关系。保留率超出设定区间时，兜底策略会使用摘要与最近消息重新组装上下文。

完整格式见[输入输出规范](docs/protocol-and-data.md)。[服务说明](docs/service.md)介绍兜底行为，包括“摘要本身超过预算”的处理边界。

## 快速开始

**环境要求：**模型部署需要 Linux、Docker 和 NVIDIA Container Toolkit；服务使用 Python 3.12。以下命令均在项目根目录执行。

**模型权重：**[aiXcoder/aiXcompact-4B](https://huggingface.co/aiXcoder/aiXcompact-4B)。下载完整模型目录，并将 `MODEL_PATH` 指向其本地路径。权重已完成 LoRA 合并，无需再次合并；请保留配置和 tokenizer 文件，详见[部署指南](docs/deployment.md)。

### 1. 安装服务

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. 启动模型

```bash
export MODEL_PATH=/absolute/path/to/aiXcompact-4B

docker build -f deploy/Dockerfile.vllm -t aixcoder-compact-vllm:0.22.1 .
GPU_ID=0 DECODE_MODE=ngram bash deploy/start-vllm.sh
```

`docker build` 以官方镜像 `vllm/vllm-openai:v0.22.1` 为底，把本仓库的 vLLM 补丁打进镜像，得到 `aixcoder-compact-vllm:0.22.1`。首次启动需要加载权重并编译内核。等待 `curl --fail http://127.0.0.1:8000/health` 成功返回后继续。使用 `DECODE_MODE=standard` 可关闭投机解码；本项目公布的性能结果使用 ngram k=32。

**补丁与加速说明：**补丁适配 aiXcompact 的纯文本权重，并修复 GDN 在投机解码时的状态处理。ngram 复用上下文中的重复片段生成候选，无需额外草稿模型；aiXcompact 开启 ngram 投机解码后，相较普通解码的参考加速比约为 **1.39×**；具体收益会随硬件、输入输出内容和并发情况变化。

### 3. 压缩一段对话

在另一个终端激活环境，启动压缩服务：

```bash
source .venv/bin/activate
export MODEL_PATH=/absolute/path/to/aiXcompact-4B
COMPACTION_HOST=127.0.0.1 python -m service.main
```

然后使用项目提供的合成示例调用：

```bash
curl --fail http://127.0.0.1:12003/compact \
  -H 'Content-Type: application/json' \
  --data-binary @examples/request.json
```

响应中的 `data.messages` 是后续对话的消息列表，`data.stats` 包含保留率与兜底状态。也可以运行 Python 示例：

```bash
python -m examples.demo
```

鉴权、服务容器化、离线安装及停止服务的方式见[部署指南](docs/deployment.md)。

## 接入 OpenCode

按照[适配指南](docs/integration.md)，为 **OpenCode 1.14.40** 应用补丁，然后在 `opencode.json` 中加入：

```json
{
  "compaction": {
    "service_url": "http://127.0.0.1:12003",
    "target_ratio_pct": 25
  }
}
```

在修改后的 OpenCode 中使用 `/compact`。适配器会将压缩结果用于后续对话，并将中断请求传递给压缩服务。不配置 `service_url` 时保留 OpenCode 原有的压缩方式。

`target_ratio_pct: 25` 表示**期望保留约 25%**，不是删除 25%。推荐取值为 10–50 之间、5 的倍数。

## 评测结果

在内部保留评测集上，发布配置的**平均上下文保留率为 25.89%，即减少 74.11%**，详见[方法说明](docs/evaluation.md)。

| 指标 | DeepSeek V4 Pro 历史参考 | aiXcompact |
|---|---:|---:|
| 规则综合分 | 0.7332 | 0.7012 |
| 平均上下文保留率 | 18.46% | 25.89% |

![内部评测：质量与上下文保留率](results/quality.svg)

**GPU 成本对比：**aiXcompact 使用实测请求耗时与公开 GPU 租价计价；DeepSeek V4 Pro 使用公开吞吐代理下的参考预算。对比结果依赖报告中注明的负载与部署假设。

评测使用 ngram k=32。达到输出上限的请求仍计入总分母。数据集构造、划分和逐条结果不公开；普通解码与 ngram 在合成样例上未达到逐 token 等价。[完整报告](results/REPORT.md)给出了聚合结果、兜底行为与限制，[方法说明](docs/evaluation.md)解释指标和成本假设。

## 文档导航

| 文档 | 内容 |
|---|---|
| [部署指南](docs/deployment.md) | 模型启动、服务配置、离线安装 |
| [适配指南](docs/integration.md) | OpenCode 1.14.40 / vLLM 0.22.1 补丁、验证与回滚 |
| [输入输出与数据构造](docs/protocol-and-data.md) | 协议格式与兼容数据的构造流程 |
| [服务接口](docs/service.md) | 保留率、兜底、取消与调试 |
| [结果报告](results/REPORT.md) | 质量、运行性能和成本情景 |

## 开发与测试

```bash
pip install -e '.[test]'
python -m pytest -q
```

仓库内测试覆盖压缩服务及模型流式响应处理。OpenCode 补丁的测试和真实服务联调方法见适配指南。

## 许可

项目代码采用 [Apache-2.0](LICENSE)，OpenCode 补丁保留上游 [MIT 许可](patches/LICENSE.opencode)。第三方归属说明见 [NOTICE](NOTICE)。
