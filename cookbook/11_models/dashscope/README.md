# DashScope Cookbook

This cookbook demonstrates Qwen model integration with Agno framework. Supports cloud deployment via DashScope API and local deployment via LM Studio, Ollama, or vLLM.

## Cloud Deployment (DashScope API)

### 1. Create and activate a virtual environment

```shell
python3 -m venv ~/.venvs/aienv
source ~/.venvs/aienv/bin/activate
```

### 2. Export your `DASHSCOPE_API_KEY` or `QWEN_API_KEY`

Get your API key from: https://modelstudio.console.alibabacloud.com/?tab=model#/api-key

```shell
export DASHSCOPE_API_KEY=***
```

### 3. Install libraries

```shell
pip install -U openai ddgs agno
```

### 4. Run basic Agent

- Streaming on

```shell
python cookbook/models/dashscope/basic_stream.py
```

- Streaming off

```shell
python cookbook/models/dashscope/basic.py
```

### 5. Run async Agent

- Async basic

```shell
python cookbook/models/dashscope/async_basic.py
```

- Async streaming

```shell
python cookbook/models/dashscope/async_basic_stream.py
```

### 6. Run Agent with Tools

- DuckDuckGo Search

```shell
python cookbook/models/dashscope/tool_use.py
```

- Async tool use

```shell
python cookbook/models/dashscope/async_tool_use.py
```

### 7. Run Agent that returns structured output

```shell
python cookbook/models/dashscope/structured_output.py
```

### 8. Run Agent that analyzes images

- Basic image analysis

```shell
python cookbook/models/dashscope/image_agent.py
```

- Image analysis with bytes

```shell
python cookbook/models/dashscope/image_agent_bytes.py
```

- Async image analysis

```shell
python cookbook/models/dashscope/async_image_agent.py
```

## Local Deployment

### Quick Start - LM Studio

#### 1. Download LM Studio

https://lmstudio.ai/

#### 2. Download Models

**Basic (text + tools):**
- Search: "qwen2.5-7b-instruct" → Download Q4_K_M

**Reasoning:**
- Search: "Qwen3-VL-4B-Thinking" → Download Q4_K_M

**Embeddings:**
- Search: "text-embedding-nomic-embed-text-v1.5"

#### 3. Start Server

Local Server → Load model → Start (port 1234)

#### 4. Run Examples

```bash
python cookbook/models/dashscope/qwen_local.py                  # Basic usage
python cookbook/models/dashscope/qwen_parallel_tools.py          # Parallel function calls
python cookbook/models/dashscope/qwen_reasoning_content.py       # Reasoning (thinking process)
python cookbook/models/dashscope/knowledge_tools.py              # Knowledge with local embeddings
```

### Model Comparison

| Feature | qwen2.5-7b | Qwen3-VL-4B-Thinking |
|---------|------------|----------------------|
| Size | 7B | 4B |
| Reasoning | No | Yes |
| Visual | No | Yes |
| Tools | Yes | Yes |
| Speed | Medium | Fast |

### Switch Models

```python
MODEL_ID = "qwen2.5-7b-instruct"              # Standard
MODEL_ID = "Qwen/Qwen3-VL-4B-Thinking-GGUF"   # Reasoning (lighter)
MODEL_ID = "Qwen/Qwen3-VL-8B-Thinking-GGUF"   # Reasoning (better)
```

### Switch Providers

**LM Studio:**
```python
BASE_URL = "http://localhost:1234/v1"
```

**Ollama:**
```python
MODEL_ID = "qwen2.5:7b"
BASE_URL = "http://localhost:11434/v1"
```

**vLLM:**
```python
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
BASE_URL = "http://localhost:8000/v1"
```

### Reasoning Models

**Text reasoning:**
- `qwq:32b` (Ollama)
- `Qwen/QwQ-32B-Preview` (HuggingFace)

**Visual reasoning:**
- `Qwen/Qwen3-VL-4B-Thinking-GGUF` (4B - lighter, faster)
- `Qwen/Qwen3-VL-8B-Thinking-GGUF` (8B - better quality)
- `Qwen/QVQ-72B` (72B - best, heavy)

