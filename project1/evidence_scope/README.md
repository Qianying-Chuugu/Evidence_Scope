# EvidenceScope

EvidenceScope 是一个面向中文企业与行业研究的学习型 Agent/RAG 项目。当前使用宁德时代 2025 年年度报告作为测试数据，已经实现基础 Tool Calling Agent、PDF 字符串搜索、PDF 分块、BM25、Dense Retrieval、基于 RRF 的 Hybrid Search 和初步检索评测。

## 当前目录

```text
evidence_scope/
├── main.py                       # DeepSeek Tool Calling Agent
├── project_context.md            # 项目背景、进度与协作约束
├── requirements.txt              # Python 依赖
├── .env.example                  # 环境变量示例
├── data/
│   ├── CATL_2025.pdf             # 宁德时代 2025 年年度报告
│   ├── chunks.json               # PDF 分块结果
│   ├── retrieval_eval.json       # 人工标注的检索评测数据
│   └── dense_vectors.npy         # 运行 Dense Retrieval 后生成的向量索引
├── tools/
│   └── pdf_search.py             # PDF 精确关键词搜索
└── retrieval/
    ├── __init__.py               # 将 retrieval 标记为 Python 包
    ├── chunking.py               # PDF 文本提取与字符分块
    ├── bm25_search.py            # BM25 稀疏检索
    ├── dense_search.py           # BGE 中文向量检索
    ├── hybrid_search.py          # 基于 RRF 的混合检索
    └── evaluate_retrieval.py     # 检索效果评测
```

Hybrid Search 尚未接入 `main.py` 的 Agent Loop；当前 Agent 仍使用早期的 PDF 精确关键词搜索工具。

## 环境准备

建议在仓库根目录执行以下命令：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\project1\evidence_scope\requirements.txt
```

复制环境变量示例，并填写自己的 DeepSeek API Key：

```powershell
Copy-Item .\project1\evidence_scope\.env.example .\project1\evidence_scope\.env
```

`.env`、`.venv`、IDE 配置、Python 缓存和生成的向量文件已经由仓库根目录的 `.gitignore` 排除，不应提交到 Git。

## 运行检索模块

进入项目目录：

```powershell
Set-Location .\project1\evidence_scope
```

重新生成 PDF 分块：

```powershell
python .\retrieval\chunking.py
```

当前按每块 500 个字符、相邻块重叠 150 个字符进行分块，最近一次生成 1378 个 Chunk。

运行 BM25 检索示例：

```powershell
python .\retrieval\bm25_search.py
```

运行 Dense Retrieval 示例：

```powershell
python .\retrieval\dense_search.py
```

Dense Retrieval 使用 `BAAI/bge-small-zh-v1.5`。程序会优先从 Hugging Face 本地缓存加载模型，不会在每次启动时访问网络；如果本机没有模型缓存，首次运行才会尝试下载。生成的向量保存到 `data/dense_vectors.npy`。

运行 Hybrid Search 和检索评测：

```powershell
python -m retrieval.hybrid_search
python -m retrieval.evaluate_retrieval
```

Hybrid Search 当前使用加权 RRF，默认权重为 BM25 `1.5`、Dense `1.0`。该权重来自现有 7 道问题的对照实验，随着评测集扩充仍需重新验证。

## 运行 Agent

确认 `.env` 已配置后，在项目目录执行：

```powershell
python .\main.py
```

当前 Agent 可调用：

- `search_annual_report`：在年报中执行精确关键词搜索并返回原文和 PDF 页码。

## 当前开发重点

当前已经建立包含 7 类问题的小规模评测集，并完成 BM25、Dense 和 Hybrid 的对比评测。下一步可以继续扩充评测问题，再针对 Dense 检索效果或 Reranker 做单变量实验。当前项目尚未引入 FastAPI、数据库或 Web 前端。
