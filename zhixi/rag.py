"""维修知识库检索（RAG）。

默认后端：TF-IDF（字符 n-gram，无需下载模型、无需联网、对中文友好）。
可选后端：FAISS 向量索引（已安装 faiss-cpu 时自动启用，接口不变）。
生产环境可把 `_vectorize` 换成 bge-m3 / text-embedding-3 等嵌入模型，其余代码不动。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import KNOWLEDGE_DIR


@dataclass
class Evidence:
    doc: str
    section: str
    text: str
    score: float

    @property
    def citation(self) -> str:
        return f"{self.doc} · {self.section}"

    def as_dict(self) -> dict:
        return {
            "doc": self.doc,
            "section": self.section,
            "text": self.text,
            "score": round(float(self.score), 4),
            "citation": self.citation,
        }


def _split_markdown(text: str, doc_name: str) -> list[tuple[str, str]]:
    """按二级标题切片，返回 (章节标题, 正文) 列表。"""

    chunks: list[tuple[str, str]] = []
    title = doc_name
    buf: list[str] = []
    for line in text.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
        if line.startswith("## "):
            if buf and "".join(buf).strip():
                chunks.append((title, "\n".join(buf).strip()))
            title = line[3:].strip()
            buf = []
        else:
            buf.append(line)
    if "".join(buf).strip():
        chunks.append((title, "\n".join(buf).strip()))
    return [(t, b) for t, b in chunks if len(b) > 20]


class KnowledgeBase:
    def __init__(self, path: Path | str = KNOWLEDGE_DIR):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.path = Path(path)
        self.docs: list[dict] = []
        for f in sorted(self.path.glob("*.md")):
            raw = f.read_text(encoding="utf-8")
            for section, body in _split_markdown(raw, f.stem):
                self.docs.append({"doc": f.stem, "section": section, "text": body})
        if not self.docs:
            raise FileNotFoundError(f"知识库目录为空：{self.path}")

        corpus = [f"{d['doc']} {d['section']}\n{d['text']}" for d in self.docs]
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1)
        matrix = self.vectorizer.fit_transform(corpus)
        self.matrix = _l2_normalize(matrix)

        self.backend = "tfidf"
        self._faiss = None
        try:
            import faiss

            dense = np.asarray(self.matrix.todense(), dtype="float32")
            index = faiss.IndexFlatIP(dense.shape[1])
            index.add(dense)
            self._faiss = index
            self._dense = dense
            self.backend = "faiss"
        except Exception:
            self._faiss = None

    def search(self, query: str, k: int = 3) -> list[Evidence]:
        q = _l2_normalize(self.vectorizer.transform([query]))
        if self._faiss is not None:
            scores, idx = self._faiss.search(np.asarray(q.todense(), dtype="float32"), k)
            pairs = list(zip(idx[0].tolist(), scores[0].tolist()))
        else:
            sims = np.asarray((q @ self.matrix.T).todense()).ravel()
            top = np.argsort(-sims)[:k]
            pairs = [(int(i), float(sims[i])) for i in top]

        out: list[Evidence] = []
        for i, score in pairs:
            if i < 0 or score <= 0:
                continue
            d = self.docs[i]
            out.append(Evidence(doc=d["doc"], section=d["section"], text=d["text"], score=score))
        return out

    def keywords_for_groups(self, groups: list[str]) -> str:
        """把工序站名称翻译成检索语句，用于自动构造 RAG 查询。"""

        tail = "设备 过程参数 漂移 排查 处置 步骤 点检 校验"
        return " ".join(groups) + " " + tail

    def aspect_queries(self, groups: list[str], high_risk: bool = False) -> list[str]:
        """把一个诊断问题拆成多个「关注面」，避免所有批次都只命中同一份文档。

        返回三条查询，分别覆盖：怎么查（排查依据）、怎么修（处置与点检）、
        要不要停（高风险时查停机规范，否则查异常批次处置）。
        """

        st = " ".join(groups) if groups else "工序站"
        queries = [
            f"{st} 过程参数 漂移 排查 判定 阈值",
            f"{st} 点检 校验 处置 步骤 复测 放行",
        ]
        queries.append(
            "停机 决策 原则 安全 复机 检查 清单"
            if high_risk
            else "异常批次 判定 在制 返工 报废 追溯"
        )
        return queries

    def search_aspects(self, queries: list[str], k: int = 3) -> list[Evidence]:
        """多关注面检索：每个子查询各取各自的最佳命中，再按分数补齐。

        这样可以保证返回的维修依据覆盖不同的环节，而不是三条都来自同一份手册。
        """

        picked: dict[tuple[str, str], Evidence] = {}
        ordered: list[Evidence] = []
        for q in queries:
            for hit in self.search(q, k=3):
                key = (hit.doc, hit.section)
                if key in picked:
                    continue
                picked[key] = hit
                ordered.append(hit)
                break
        if len(ordered) < k:
            for q in queries:
                for hit in self.search(q, k=6):
                    key = (hit.doc, hit.section)
                    if key in picked:
                        continue
                    picked[key] = hit
                    ordered.append(hit)
                    if len(ordered) >= k:
                        break
                if len(ordered) >= k:
                    break
        return sorted(ordered[:k], key=lambda h: -h.score)


def _l2_normalize(matrix):
    from sklearn.preprocessing import normalize

    return normalize(matrix, norm="l2", axis=1, copy=True)


def strip_markdown(text: str, limit: int = 320) -> str:
    t = re.sub(r"[#*`>|-]+", " ", text)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:limit] + ("…" if len(t) > limit else "")
