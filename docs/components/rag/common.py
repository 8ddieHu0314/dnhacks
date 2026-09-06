"""Shared pieces for the component RAG: chunking, embedding, BM25. Stdlib plus torch/transformers."""
import json
import math
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
COMPONENTS = HERE.parent / "components.json"
INDEX_DIR = HERE / "index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_components():
    return json.loads(COMPONENTS.read_text())["components"]


def _j(x):
    if isinstance(x, (list, tuple)):
        return "; ".join(_j(i) for i in x if i)
    if isinstance(x, dict):
        return "; ".join(f"{k}: {_j(v)}" for k, v in x.items() if v)
    return str(x)


def chunk_record(c):
    """Three chunks per part: identity+visual, pins+electrical+wiring, safety+troubleshooting.
    Every chunk is prefixed with name and MPN so retrieval works from either."""
    d = c["details"]
    ident, fn, vis = d["identity"], d["function"], d["visual_identification"]
    head = f"{c['canonical_name']} (part number {c['mpn'].split(' (')[0][:40]}; on the kit lid: {c['name_on_kit']})."
    a = " ".join([head, "Category:", ident.get("category", ""), "Aliases:", _j(ident.get("aliases")),
                  "Function:", fn.get("one_sentence_plain_english", ""), fn.get("what_it_is_used_for", ""),
                  "Looks like:", vis.get("shape_and_size", ""), vis.get("color_and_markings", ""),
                  "Printed text:", vis.get("printed_text_to_look_for", ""),
                  "Easily confused with:", _j(vis.get("easily_confused_with"))])
    p, e, w = d["pins"], d["electrical"], d["wiring_to_uno"]
    b = " ".join([head, f"Pin count: {p.get('pin_count')}.", "Pinout:", _j(p.get("pinout")),
                  "Pin 1:", p.get("pin_1_identification", ""),
                  f"Polarity sensitive: {p.get('polarity_sensitive')}.", "Electrical:", _j(e),
                  "Key specs:", _j(d.get("key_specs")), "Wiring to UNO:", _j(w)])
    s, t = d["safety"], d["troubleshooting"]
    cc = " ".join([head, "Safety hazards:", _j(s.get("hazards")), "Handling:", _j(s.get("handling_notes")),
                   "Common mistakes:", _j(s.get("common_mistakes")), "Troubleshooting:",
                   _j(t.get("symptoms_and_causes")), "Multimeter test:", t.get("how_to_test_with_a_multimeter", "")])
    return [{"id": c["id"], "chunk": "identity_visual", "text": a},
            {"id": c["id"], "chunk": "pins_electrical", "text": b},
            {"id": c["id"], "chunk": "safety_troubleshooting", "text": cc}]


_TOK = re.compile(r"[a-z0-9]+")


def tokenize(s):
    return _TOK.findall(s.lower())


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in docs]
        self.N = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(1, self.N)
        self.tf = [Counter(d) for d in self.docs]
        df = Counter()
        for d in self.docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def scores(self, q):
        qt = tokenize(q)
        out = []
        for i, d in enumerate(self.docs):
            s, dl = 0.0, len(d)
            for t in qt:
                if t in self.tf[i]:
                    f = self.tf[i][t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out


class Embedder:
    def __init__(self, name=MODEL_NAME):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).eval()

    def encode(self, texts, batch=32):
        import numpy as np
        outs = []
        for i in range(0, len(texts), batch):
            x = self.tok(texts[i:i + batch], padding=True, truncation=True, max_length=384, return_tensors="pt")
            with self.torch.no_grad():
                h = self.model(**x).last_hidden_state
            m = x["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1)
            v = v / v.norm(dim=1, keepdim=True)
            outs.append(v.numpy())
        return np.concatenate(outs)
