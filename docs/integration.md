# 固定版本集成指南

## OpenCode 1.14.40

上游项目：https://github.com/anomalyco/opencode

固定 commit：`277f1c71486ed4795875d09bb5c0bbe504f06dd5`。使用 Bun 1.3.13，与上游 `packageManager` 保持一致。本项目不分发 OpenCode 二进制。

```bash
git clone https://github.com/anomalyco/opencode.git
cd opencode
git checkout 277f1c71486ed4795875d09bb5c0bbe504f06dd5
git apply --check /path/to/aixcoder-compact/patches/opencode-1.14.40-compaction.patch
git apply /path/to/aixcoder-compact/patches/opencode-1.14.40-compaction.patch
bun install --frozen-lockfile
cd packages/opencode
bun ../../node_modules/@typescript/native-preview/bin/tsgo.js --noEmit
OPENCODE_EXPERIMENTAL_DISABLE_FILEWATCHER=1 bun test --timeout 30000 test/config/compaction.test.ts test/session/compaction-cancel.test.ts test/session/compaction.test.ts
```

文件监听关闭参数只用于受限 macOS 环境下的测试，不改变运行时压缩逻辑。如果 frozen install 因平台依赖解析失败，应保留原 lockfile、检查依赖变化，并使用 Bun 1.3.13；不要关闭完整性校验。

将以下配置合并到运行时 `opencode.json`，不要覆盖已有的 provider/model 配置：

```json
{
  "compaction": {
    "service_url": "http://127.0.0.1:12003",
    "service_timeout": 1800000,
    "target_ratio_pct": 25
  }
}
```

省略或留空 `service_url` 时使用 OpenCode 自带压缩逻辑。配置后，OpenCode 会发送新的 UUID、模型消息、coding agent 的系统提示和工具定义。工具定义包含完整描述和 `parameters` JSON Schema，保留嵌套参数、必填项、枚举等约束。外部服务路径保留完整文本工具结果，不在发送前截断为前 2000 字符。服务返回的消息会写入压缩元数据，并在后续轮次展开；再次压缩时会继续看到上一次保留的消息，而不只是摘要。

上下文溢出触发自动压缩时，OpenCode 会在压缩完成后重新追加暂存的最新用户请求，保留其 agent、model、system 和工具设置；媒体附件按上游逻辑转换为文本占位说明。即使未保留原始尾部消息，后续模型也能收到这条请求。

使用上游开发入口启动 checkout：`bun run --cwd packages/opencode --conditions=browser src/index.ts`。coding-model provider 仍需使用者自行配置，本项目不提供凭据。

用户中断或超时时，OpenCode 会中止主请求，并使用独立的 5 秒超时调用 `/cancel/{id}`。取消返回 404 表示请求已经完成或没有注册。错误不会静默生成一个伪造摘要替换原始历史。

真实服务联调：

```bash
COMPACT_INTEGRATION_URL=http://127.0.0.1:12003 OPENCODE_EXPERIMENTAL_DISABLE_FILEWATCHER=1 \
  bun test --timeout 300000 test/session/compaction.test.ts -t 'external compaction integration'
```

该测试会使用真实压缩服务和模型，验证 OpenCode 的 SessionCompaction、消息持久化及连续两轮压缩；coding-model provider 使用测试 fixture，因此不代表完整 coding 任务成功率。

## vLLM 0.22.1

上游项目：https://github.com/vllm-project/vllm

固定 commit：`0decac0d96c42b49572498019f0a0e3600f50398`。

```bash
git checkout 0decac0d96c42b49572498019f0a0e3600f50398
git apply --check /path/to/aixcoder-compact/patches/vllm-0.22.1-gdn-ngram.patch
git apply /path/to/aixcoder-compact/patches/vllm-0.22.1-gdn-ngram.patch
```

pip/Docker 部署请使用项目提供的 Dockerfile 和哈希校验安装脚本，不要猜测 site-packages 路径。补丁增加纯文本 Qwen3.5 注册、混合状态大小计算、文本 M-RoPE 位置，以及投机/非投机状态切换的 GDN 修复。不改变模型为多模态架构，也不增加视觉权重。

## 回滚

如果受影响文件没有叠加其他修改：

```bash
git apply --check -R /path/to/the.patch
git apply -R /path/to/the.patch
```

容器场景请从原始固定镜像重新构建，并使用新的容器名。不要直接修改正在运行的生产容器。两个补丁均附带文件 SHA-256 清单。
