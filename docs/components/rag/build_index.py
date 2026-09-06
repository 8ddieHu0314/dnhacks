#!/usr/bin/env python3
"""Chunk components.json and embed every chunk with a local MiniLM model. Writes index/."""
import json
import time

import numpy as np

from common import INDEX_DIR, Embedder, chunk_record, load_components


def main():
    comps = load_components()
    chunks = [ch for c in comps for ch in chunk_record(c)]
    INDEX_DIR.mkdir(exist_ok=True)
    (INDEX_DIR / "chunks.json").write_text(json.dumps(chunks, indent=1))
    t = time.time()
    emb = Embedder()
    load_s = time.time() - t
    t = time.time()
    vecs = emb.encode([c["text"] for c in chunks])
    embed_s = time.time() - t
    np.save(INDEX_DIR / "vectors.npy", vecs)
    print(f"parts={len(comps)} chunks={len(chunks)} dim={vecs.shape[1]} "
          f"model_load={load_s:.1f}s embed={embed_s:.1f}s avg_chunk_chars={sum(len(c['text']) for c in chunks)//len(chunks)}")


if __name__ == "__main__":
    main()
