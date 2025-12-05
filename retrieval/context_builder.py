import math
from transformers import AutoTokenizer

def estimate_tokens(text, tokenizer=None):
    if tokenizer:
        return len(tokenizer.encode(text))
    # rough estimate: 1 token ~ 4 chars
    return max(1, math.ceil(len(text) / 4))

def build_context(nodes, tokenizer=None, token_budget=2048, prefer_levels=None):
    if not nodes:
        return "", []

    # if prefer_levels specified, sort accordingly; else sort by level desc then score
    if prefer_levels:
        nodes = sorted(nodes, key=lambda n: (prefer_levels.index(n["_level"]) if n["_level"] in prefer_levels else 99, -n["_score"]))
    else:
        nodes = sorted(nodes, key=lambda n: (n.get("_level",0), -n["_score"]), reverse=True)

    included = []
    token_count = 0
    parts = []

    for n in nodes:
        text = n.get("summary", "")  # high-level summary field
        est = estimate_tokens(text, tokenizer)
        if token_count + est > token_budget:
            continue
        parts.append(f"[L{n.get('_level')}] {n.get('title','')} \n{text}")
        included.append(n.get("node_id"))
        token_count += est
        # optionally expand with child nodes if they exist and budget allows
        children = n.get("child_ids", []) or []
        for ch in children[:2]:  # include up to 2 children
            # We assume the calling code can map child node ids to node dicts; here we expect nodes param contains expanded descendants
            # So check for presence of child node in nodes list
            child_node = next((x for x in nodes if x.get("node_id")==ch), None)
            if child_node:
                ctext = child_node.get("summary","")
                cest = estimate_tokens(ctext, tokenizer)
                if token_count + cest <= token_budget:
                    parts.append(f"[L{child_node.get('_level')}] {child_node.get('title','')} \n{ctext}")
                    included.append(child_node.get("node_id"))
                    token_count += cest
    context = "\n\n".join(parts)
    return context, included
