"""检索评测工作台用例：候选检索、标注保存与评测集校验。"""
import hashlib
import json
from pathlib import Path


class RetrievalReviewUseCase:
    def __init__(self, retriever, *, review_path: Path, cases_path: Path):
        self._retriever = retriever
        self._review_path = review_path
        self._cases_path = cases_path

    @staticmethod
    def case_id(query: str) -> str:
        """同一问题始终生成相同的评测用例 ID。"""
        return "review-" + hashlib.md5(query.encode("utf-8")).hexdigest()[:16]

    def search(self, query: str, *, top_k: int = 5) -> dict:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 必须是非空字符串")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 100:
            raise ValueError("top_k 必须是 1 到 100 之间的整数")
        rows = self._retriever.retrieve(query.strip(), top_k=top_k, context=None)
        candidates = [{"rank": i, "text": text, "document_id": doc,
                       "chunk_id": cid, "score": score}
                      for i, (text, doc, cid, score) in enumerate(rows, 1)]
        return {"id": self.case_id(query.strip()), "query": query.strip(),
                "candidate_k": top_k, "candidates": candidates}

    def save(self, record: dict) -> dict:
        query = str(record.get("query", "")).strip()
        candidates = record.get("candidates") or []
        allowed = {item.get("chunk_id") or item.get("document_id") for item in candidates}
        relevant = list(dict.fromkeys(record.get("relevant_chunk_ids") or []))
        forbidden = list(dict.fromkeys(record.get("forbidden_chunk_ids") or []))
        if not query or not set(relevant) <= allowed or not set(forbidden) <= allowed:
            raise ValueError("审核 ID 无效")
        if set(relevant) & set(forbidden):
            raise ValueError("相关与噪声不能标记同一 ID")
        self._review_path.parent.mkdir(parents=True, exist_ok=True)
        existing = self._read(self._review_path)
        item = dict(record)
        item["id"] = self.case_id(query)
        item["query"] = query
        item["relevant_chunk_ids"] = relevant
        item["forbidden_chunk_ids"] = forbidden
        existing = [x for x in existing
                    if x.get("id") != item["id"] and x.get("query") != query]
        existing.append(item)
        self._atomic_write(self._review_path, existing)
        formal = [{k: x.get(k) for k in
                   ("id", "query", "relevant_chunk_ids", "expected_sections", "forbidden_chunk_ids")}
                  for x in existing if x.get("relevant_chunk_ids")]
        self._atomic_write(self._cases_path, formal)
        return {"saved": len(existing), "formal": len(formal)}

    def validate(self) -> dict:
        data = self._read(self._cases_path)
        issues = []
        for case in data:
            ids = list(dict.fromkeys((case.get("relevant_chunk_ids") or []) +
                                     (case.get("forbidden_chunk_ids") or [])))
            # 通过 retriever 的底层存储检查 chunk 是否存在；没有该能力时只报告空标注。
            stale = []
            vectors = getattr(self._retriever, "vectors", None)
            if ids and vectors is not None and hasattr(vectors, "chunk_exists"):
                status = vectors.chunk_exists(ids)
                stale = [cid for cid, ok in status.items() if not ok]
            issues.append({"id": case.get("id"), "query": case.get("query"),
                           "stale_ids": stale,
                           "empty_relevant": not bool(case.get("relevant_chunk_ids"))})
        return {
            "total": len(data),
            "stale_cases": sum(bool(x["stale_ids"]) for x in issues),
            "empty_relevant_cases": sum(x["empty_relevant"] for x in issues),
            "issues": issues,
        }

    @staticmethod
    def _read(path: Path) -> list:
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []

    @staticmethod
    def _atomic_write(path: Path, data: list) -> None:
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)
