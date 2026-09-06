#!/usr/bin/env python3
"""Search the component records three ways.

  python3 search.py --mode json  "relay"          # exact substring over fields (what query.py does)
  python3 search.py --mode bm25  "blue cube five legs"
  python3 search.py --mode embed "blue cube five legs"
  python3 search.py --mode hybrid "blue cube five legs"   # bm25 + embed, rank fusion
"""
import argparse
import json
import time

from common import INDEX_DIR, BM25, Embedder, load_components


class Searcher:
    def __init__(self, modes=("json", "bm25", "embed", "hybrid")):
        self.comps = load_components()
        self.chunks = json.loads((INDEX_DIR / "chunks.json").read_text())
        self.bm25 = BM25([c["text"] for c in self.chunks]) if ("bm25" in modes or "hybrid" in modes) else None
        if "embed" in modes or "hybrid" in modes:
            import numpy as np
            self.vecs = np.load(INDEX_DIR / "vectors.npy")
            self.emb = Embedder()

    def _rank_chunks(self, scores, k):
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        seen, out = set(), []
        for i in order:
            cid = self.chunks[i]["id"]
            if cid not in seen:
                seen.add(cid)
                out.append((cid, float(scores[i]), self.chunks[i]["chunk"]))
            if len(out) >= k:
                break
        return out

    def search(self, q, mode, k=5):
        if mode == "json":
            ql = q.lower()
            hits = []
            for c in self.comps:
                blob = " ".join(str(v) for kk, v in c.items() if kk != "details").lower()
                if ql in blob:
                    hits.append((c["id"], 1.0, "seed_fields"))
            return hits[:k]
        if mode == "bm25":
            return self._rank_chunks(self.bm25.scores(q), k)
        if mode == "embed":
            qv = self.emb.encode([q])[0]
            return self._rank_chunks(self.vecs @ qv, k)
        if mode == "hybrid":
            # reciprocal rank fusion of bm25 and embed, per part
            a = self._rank_chunks(self.bm25.scores(q), 10)
            b = self._rank_chunks(self.vecs @ self.emb.encode([q])[0], 10)
            fused = {}
            for lst in (a, b):
                for r, (cid, _, ch) in enumerate(lst):
                    fused[cid] = fused.get(cid, 0.0) + 1.0 / (60 + r)
            top = sorted(fused.items(), key=lambda x: -x[1])[:k]
            return [(cid, sc, "fused") for cid, sc in top]
        raise ValueError(mode)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["json", "bm25", "embed", "hybrid"], default="embed")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("query")
    a = p.parse_args()
    s = Searcher(modes=(a.mode,))
    t = time.time()
    hits = s.search(a.query, a.mode, a.k)
    ms = (time.time() - t) * 1000
    for cid, score, chunk in hits:
        print(f"{score:7.3f}  {cid:24s} {chunk}")
    print(f"({a.mode}, {ms:.1f} ms)")


if __name__ == "__main__":
    main()
