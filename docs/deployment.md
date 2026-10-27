# 部署指南

## 模型文件

使用已完成 LoRA 合并的 BF16 纯文本权重：`Qwen3_5ForCausalLM`、safetensors 分片、配置、索引、tokenizer 和 chat template。**上传的是已经完成 LoRA 合并的模型，使用者不需要安装 LoRA、加载 adapter 或执行合并。**不要替换成未合并的 LoRA adapter 或单独的基座模型。

模型权重：[aiXcoder/aiXcompact-4B](https://huggingface.co/aiXcoder/aiXcompact-4B)。下载完整模型目录后，将 `MODEL_PATH` 指向本地路径，并保留配置、索引、tokenizer 和 chat template 文件。

## vLLM 部署

要求：Linux、兼容的 NVIDIA 驱动、Docker 和 NVIDIA Container Toolkit。参考配置为 1× A100 SXM4 40GB、BF16、TP=1。vLLM 基线为 **0.22.1**，commit 为 `0decac0d96c42b49572498019f0a0e3600f50398`。

```bash
docker build -f deploy/Dockerfile.vllm -t aixcoder-compact-vllm:0.22.1 .
export MODEL_PATH=/absolute/path/to/aiXcompact-4B
GPU_ID=0 PORT=8000 DECODE_MODE=ngram bash deploy/start-vllm.sh
```

`DECODE_MODE=standard` 关闭投机解码。ngram 使用 k=32、lookup min=5/max=10。本版本不使用 per-LoRA 调度。参考评测关闭 prefix caching；两种模式使用最大上下文 262144、最大序列数 1、批量 token 上限 8192、GPU 显存利用率 0.90。这些是可复现默认值，不代表所有 GPU 都能达到相同吞吐。KV cache 空间不足时会在启动阶段报错，不会静默截断输入。

补丁包含 6 个文件：4 个 GDN 投机状态修复，以及纯文本模型注册和 M-RoPE/状态接口。`apply_vllm_patch.py` 会检查上游文件哈希，发现未知或部分修改时拒绝继续。已验证的补丁可重复执行。可通过 `--build-arg VLLM_BASE_IMAGE=...` 指定镜像源，但其 vLLM 版本和源码哈希必须一致。

如需鉴权，在启动前设置 `VLLM_API_KEY`，并将相同值设置为服务的 `LLM_TOKEN`。不要把密钥提交到仓库。默认只绑定 `127.0.0.1`；需要远程访问时请明确设置 `BIND_HOST`。

就绪检查：`curl --fail http://127.0.0.1:8000/health`。首次加载权重和编译 CUDA/Triton 内核可能需要几分钟。只查看自己的容器日志：`docker logs aixcoder-compact-vllm`。

## 压缩服务

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
export MODEL_PATH=/absolute/path/to/aiXcompact-4B
COMPACTION_HOST=127.0.0.1 python -m service.main
```

服务直接读取 `$MODEL_PATH/tokenizer.json`，用于轨迹长度提示和目标预算计算；不会再次加载模型权重，也不需要下载 tokenizer。tokenizer 缺失或不可读时会阻止启动，而不是静默改用字符估算。

可将 `deploy/service.env.example` 复制为私有环境文件，并运行 `set -a; . ./.env; set +a` 加载。每个终端/进程都需要获得 `MODEL_PATH`。如果模型和服务位于不同机器，可单独复制模型目录中的 `tokenizer.json`，再设置 `LENGTH_TOKENIZER_PATH`；它会覆盖 `MODEL_PATH` 下的默认路径。

容器方式：

```bash
docker build -f deploy/Dockerfile.service -t aixcoder-compact-service:0.1.0 .
docker run -d --name aixcoder-compact-service --network host \
  -e COMPACTION_HOST=127.0.0.1 \
  -e LLM_ENDPOINT=http://127.0.0.1:8000/v1/chat/completions \
  -e MODEL_PATH=/model \
  --mount "type=bind,source=$MODEL_PATH,target=/model,readonly" \
  aixcoder-compact-service:0.1.0
```

该示例使用 Linux host network。服务本身不需要 GPU。`PYTHON_BASE_IMAGE` 可指定可用的 Python 3.12 镜像或离线镜像；无论基镜像是什么，服务依赖仍按锁定版本安装。

## 冒烟测试与停止

```bash
curl --fail http://127.0.0.1:12003/health
curl --fail -H 'Content-Type: application/json' --data-binary @examples/request.json http://127.0.0.1:12003/compact
python -m examples.demo
```

只停止本项目创建的容器：`docker stop aixcoder-compact-service aixcoder-compact-vllm`。脚本不会停止其他任务、自动抢占 GPU 或覆盖模型目录。

## 离线安装依赖

服务器无法解析 Docker Hub/PyPI 时，可在联网机器下载 Linux Python 3.12 wheel：

```bash
python -m pip download -r service/requirements.txt -d wheels --platform manylinux_2_28_x86_64 --platform manylinux2014_x86_64 --implementation cp --python-version 312 --abi cp312 --abi abi3 --abi none --only-binary=:all:
docker build -f deploy/Dockerfile.service.offline --build-arg PYTHON_BASE_IMAGE=YOUR_AVAILABLE_PYTHON312_IMAGE -t aixcoder-compact-service:0.1.0 .
```

离线镜像会把锁定版本的服务依赖安装到 `/opt/compact-deps`，与基镜像中已有的推理框架依赖隔离。wheel 只是安装材料，不是项目源码，不要提交到 Git。

## 解码模式说明

完整发布评测使用 ngram k=32。普通解码同样可用，并在真实 OpenCode 联调中使用过。温度为 0 的合成样例没有得到完全相同的文本：两种模式的候选 token 排名存在差异。这只证明两种配置存在观测差异，不能据此确定唯一原因，也不承诺逐 token 等价或无损输出。请在自己的业务负载上评估所选模式，不要混用两种模式的性能结果。
