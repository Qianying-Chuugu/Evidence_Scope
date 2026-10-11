# EvidenceScope

EvidenceScope 是一个面向中文企业与行业研究的学习型 Agent/RAG 项目。当前使用宁德时代 2025 年年度报告作为测试数据，已经实现基础 Tool Calling Agent、财务计算、PDF 字符串搜索、PDF 分块、BM25 检索和 Dense Retrieval。

## 当前目录

```text
evidence_scope/
├── main.py                       # DeepSeek Tool Calling Agent
├── project_context.md            # 项目背景、进度与协作约束
├── requirements.txt              # Python 依赖
├── .env.example                  # 环境变量示例
├── data/
│   ├── CATL_2025.pdf             # 宁德时代 2025 年年度报告
│   ├── financial_facts.json      # 已整理的结构化财务数据
│   ├── chunks.json               # PDF 分块结果
│   └── dense_vectors.npy         # 运行 Dense Retrieval 后生成的向量索引
├── tools/
│   ├── finance.py                # 财务数据查询与同比计算
│   └── pdf_search.py             # PDF 精确关键词搜索
└── retrieval/
    ├── chunking.py               # PDF 文本提取与字符分块
    ├── bm25_search.py            # BM25 稀疏检索
    └── dense_search.py           # BGE 中文向量检索
```

`hybrid_search.py` 尚未实现，BM25 和 Dense Retrieval 也尚未接入 `main.py` 的 Agent Loop。

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

运行 BM25 检索示例：

```powershell
python .\retrieval\bm25_search.py
```

运行 Dense Retrieval 示例：

```powershell
python .\retrieval\dense_search.py
```

Dense Retrieval 使用 `BAAI/bge-small-zh-v1.5`。程序会优先从 Hugging Face 本地缓存加载模型，不会在每次启动时访问网络；如果本机没有模型缓存，首次运行才会尝试下载。生成的向量保存到 `data/dense_vectors.npy`。

## 运行 Agent

确认 `.env` 已配置后，在项目目录执行：

```powershell
python .\main.py
```

当前 Agent 可调用：

- `compare_revenue`：查询宁德时代 2024、2025 年营业收入并计算同比增长率。
- `search_annual_report`：在年报中执行精确关键词搜索并返回原文和 PDF 页码。

## 当前开发重点

下一步计划实现基于 RRF 的 Hybrid Search，将 BM25 与 Dense Retrieval 的候选结果融合，再进行小规模检索效果对比。当前项目尚未引入 Reranker、FastAPI、数据库或 Web 前端。
