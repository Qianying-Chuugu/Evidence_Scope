# EvidenceScope

EvidenceScope 是一个面向中文企业与行业研究的学习型 Agent/RAG 项目。当前使用宁德时代 2025 年年度报告作为测试数据，已经实现基础 Tool Calling Agent、PDF 字符串搜索、PDF 分块、BM25、Dense Retrieval、基于 RRF 的 Hybrid Search 和初步检索评测。

## 当前目录

```text
evidence_scope/
├── main.py                       # 启动入口与组件组装
├── project_context.md            # 项目背景、进度与协作约束
├── requirements.txt              # Python 依赖
├── .env.example                  # 环境变量示例
├── agent/
│   ├── harness.py                # Agent Loop、模型调用与异常处理
│   └── tool_registry.py          # 工具 Schema 与函数调用映射
├── skills/
│   └── local_document_qa.py      # 本地文档问答系统提示词
├── data/
│   ├── CATL_2025.pdf             # 宁德时代 2025 年年度报告
│   ├── documents/                # 新增的本地 PDF 放在这里
│   ├── chunks.json               # PDF 分块结果
│   ├── retrieval_eval_dev.json   # 20 题开发集，可用于调参
│   ├── retrieval_eval_test.json  # 10 题冻结测试集，只用于最终检查
│   └── dense_vectors.npy         # 运行 Dense Retrieval 后生成的向量索引
├── tools/
│   ├── evidence_search.py        # Agent 使用的完整 RAG 检索工具
│   └── pdf_search.py             # 早期 PDF 精确关键词搜索
└── retrieval/
    ├── __init__.py               # 将 retrieval 标记为 Python 包
    ├── chunking.py               # PDF 文本提取与字符分块
    ├── bm25_search.py            # BM25 稀疏检索
    ├── dense_search.py           # BGE 中文向量检索
    ├── hybrid_search.py          # 基于 RRF 的混合检索
    ├── reranker.py               # Cross-Encoder 候选重排实验
    └── evaluate_retrieval.py     # 检索效果评测
```

项目按 Harness、Skills、Tools 和 Retrieval 分层：`main.py` 只负责组装；Harness 管理 Agent Loop；Skill 定义本地文档研究流程；Tool 向 Agent 暴露检索能力；Retrieval 实现底层算法。Agent 调用 `search_evidence` 后，工具内部会依次执行 BM25 与 Dense 召回、加权 RRF 融合和 Cross-Encoder 重排。

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

把需要检索的新 PDF 放入 `data/documents/`，然后重新生成统一分块：

```powershell
python -m retrieval.chunking
```

程序会扫描 `data/documents/` 下的所有 PDF，同时兼容现有的 `data/CATL_2025.pdf`。当前按每块 500 个字符、相邻块重叠 150 个字符分块，并为每个 Chunk 保存文件名、文件路径和 PDF 页码。分块完成后不需要手动生成向量；下一次启动 Dense 检索或 Agent 时会检测文件更新时间并自动重建向量。

运行 BM25 检索示例：

```powershell
python .\retrieval\bm25_search.py
```

运行 Dense Retrieval 示例：

```powershell
python .\retrieval\dense_search.py
```

Dense Retrieval 使用 `BAAI/bge-small-zh-v1.5`。程序会优先从 Hugging Face 本地缓存加载模型，不会在每次启动时访问网络；如果本机没有模型缓存，首次运行才会尝试下载。生成的向量保存到 `data/dense_vectors.npy`。

运行 Hybrid Search：

```powershell
python -m retrieval.hybrid_search
```

分别运行开发集、冻结测试集或全部评测：

```powershell
python -m retrieval.evaluate_retrieval --dataset dev
python -m retrieval.evaluate_retrieval --dataset test
python -m retrieval.evaluate_retrieval --dataset all
```

开发过程中只根据 `retrieval_eval_dev.json` 调整参数。`retrieval_eval_test.json` 应保持不变，只在完成一轮方案后运行，避免把测试集也调成“已见过的数据”。

Hybrid Search 当前使用加权 RRF，默认权重为 BM25 `1.5`、Dense `1.0`。该权重来自早期小规模实验，后续如需改权重，应只在 20 题开发集上比较，选定后再用冻结测试集验证一次。

运行单题 Reranker 实验：

```powershell
python -m retrieval.reranker
```

该实验使用本地缓存的 `BAAI/bge-reranker-base` 对 Hybrid Top-20 候选重新排序，并已作为第四条流程接入检索评测。最新结果如下（耗时不包含模型初始化）：

| 数据集 | 方法 | Hit@5 | MRR | Recall@5 | 平均耗时 |
| --- | --- | ---: | ---: | ---: | ---: |
| 开发集 20 题 | Hybrid | 0.85 | 0.5892 | 0.7812 | 14.17 ms |
| 开发集 20 题 | Hybrid + Reranker | 0.85 | 0.5492 | 0.7375 | 2681.93 ms |
| 冻结测试集 10 题 | Hybrid | 0.90 | 0.6533 | 0.85 | 14.67 ms |
| 冻结测试集 10 题 | Hybrid + Reranker | 0.90 | 0.75 | 0.90 | 2617.25 ms |

Reranker 在开发集上略降、在测试集上提高，且单题延迟约增加到 2.6 秒，结论还不够稳定。项目仍将完整的 Hybrid + Reranker 作为 Agent 的默认证据检索链路，后续重点在 Agent 层评估查询改写、多轮检索和最终回答质量；上表保留其离线检索效果与延迟取舍。

## 运行 Agent

确认 `.env` 已配置后，在项目目录执行：

```powershell
python .\main.py
```

当前 Agent 可调用：

- `search_evidence`：在全部本地 PDF 或指定文件中执行 BM25 + Dense + 加权 RRF + Reranker，返回原文、文件名、PDF 页码、Chunk ID 及各阶段排名。

Agent 会先构造包含年份、指标和统计口径的查询；如果第一轮证据不足，可以改写查询后再次调用工具。最终回答必须引用文件名、PDF 页码和 Chunk ID。

## 当前开发重点

当前已经建立 20 题开发集和 10 题冻结测试集，完成四路检索对比，并将完整 RAG 链路接入 Agent。下一步应建立 Agent 端到端评测，检查查询生成、证据引用、数值口径和最终答案是否正确；不要根据冻结检索测试集反复调参。当前项目尚未引入 FastAPI、数据库或 Web 前端。
