
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Promote entity-relevant subtrees from a nested Reddit discussion.

Requirements:
- Input CSV: reddit_comments_nested.csv (must include: comment_id, parent_id, comment_text)
  Optional columns: depth, author, score, created_utc, etc.
- Input JSON: keyword_extraction_test_results.json (expects "top_20_entities" or "top_entities")

Promotion Rule:
- A subtree rooted at a non-top-level comment is promoted IF:
  (1) The subtree contains at least one of the important entities (case-insensitive, regex word-boundary match), AND
  (2) The subtree has > 3 replies (i.e., descendants count >= 3).

Outputs:
- JSON: promoted_threads.json
  {
    "meta": {...},
    "original_threads": [nested threads with promoted branches removed],
    "promoted_threads": [nested threads of promoted subtrees],
  }

- CSV: promoted_threads_flat.csv
  Columns: [
    thread_id, is_promoted, root_id, comment_id, original_parent_id,
    parent_id_in_thread, original_depth, depth_in_thread, matched_entities_root
  ]

Usage (standalone):
    python promote_subtrees.py \
        --comments /path/to/reddit_comments_nested.csv \
        --entities /path/to/keyword_extraction_test_results.json \
        --json_out /path/to/promoted_threads.json \
        --csv_out /path/to/promoted_threads_flat.csv
"""

import argparse
import json
import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


# -----------------------------
# Data Structures
# -----------------------------

@dataclass
class CommentNode:
    comment_id: str
    parent_id: Optional[str]
    text: str
    author: Optional[str] = None
    score: Optional[float] = None
    created_utc: Optional[float] = None
    original_depth: Optional[int] = None
    # Computed
    children: List[str] = field(default_factory=list)


# -----------------------------
# Utilities
# -----------------------------

def normalize_id(x) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    return s

def load_entities(entities_json_path: str) -> Set[str]:
    with open(entities_json_path, "r") as f:
        data = json.load(f)
    # Prefer "top_20_entities" keys if present; fallback to any dict of entities
    if isinstance(data, dict):
        if "top_20_entities" in data and isinstance(data["top_20_entities"], dict):
            ents = list(data["top_20_entities"].keys())
        elif "top_entities" in data and isinstance(data["top_entities"], list):
            ents = data["top_entities"]
        else:
            # Fallback: gather all string-like keys from any mapping fields
            ents = []
            for k, v in data.items():
                if isinstance(v, dict):
                    ents.extend(list(v.keys()))
    else:
        ents = []
    # Lowercase unique
    ents = [str(e).strip().lower() for e in ents if str(e).strip()]
    return set(ents)

def compile_entity_patterns(entities: Set[str]) -> List[Tuple[str, re.Pattern]]:
    patterns = []
    for e in sorted(entities, key=lambda s: (-len(s), s)):  # longer first
        # Word-boundary regex; handle phrases with spaces and apostrophes
        # \b works reasonably well for alphanumerics; for hyphens, we allow literal match
        # Escape entity safely.
        escaped = re.escape(e)
        # Allow simple word boundaries; if phrase contains spaces, \b at ends is still okay
        pat = re.compile(rf"\b{escaped}\b", re.IGNORECASE)
        patterns.append((e, pat))
    return patterns

def text_matches_entities(text: str, compiled_patterns: List[Tuple[str, re.Pattern]]) -> Set[str]:
    if not text:
        return set()
    matches = set()
    for e, pat in compiled_patterns:
        if pat.search(text):
            matches.add(e)
    return matches

def build_forest(df: pd.DataFrame) -> Tuple[Dict[str, CommentNode], List[str]]:
    """
    Returns:
        nodes: comment_id -> CommentNode
        roots: list of root comment_ids (top-level, where parent not in comment_id set)
    """
    # Ensure required columns
    required_cols = {"comment_id", "parent_id", "comment_text"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    # Normalize ids and prepare nodes
    df = df.copy()
    df["comment_id"] = df["comment_id"].apply(normalize_id)
    df["parent_id"] = df["parent_id"].apply(normalize_id)
    # Optional fields
    author_col = "author" if "author" in df.columns else None
    score_col = "score" if "score" in df.columns else None
    created_col = "created_utc" if "created_utc" in df.columns else None
    depth_col = "depth" if "depth" in df.columns else None

    nodes: Dict[str, CommentNode] = {}
    for _, row in df.iterrows():
        cid = row["comment_id"]
        pid = row["parent_id"]
        if cid is None:
            # skip invalid rows
            continue
        text = str(row["comment_text"]) if "comment_text" in row else ""
        node = CommentNode(
            comment_id=cid,
            parent_id=pid,
            text=text or "",
            author=(str(row[author_col]) if author_col else None),
            score=(float(row[score_col]) if score_col and not pd.isna(row[score_col]) else None),
            created_utc=(float(row[created_col]) if created_col and not pd.isna(row[created_col]) else None),
            original_depth=(int(row[depth_col]) if depth_col and not pd.isna(row[depth_col]) else None),
        )
        nodes[cid] = node

    # Build children mapping
    id_set = set(nodes.keys())
    for cid, node in nodes.items():
        pid = node.parent_id
        if pid in id_set:
            nodes[pid].children.append(cid)

    # Determine roots: parent_id not present in id_set
    roots = []
    for cid, node in nodes.items():
        if node.parent_id not in id_set:
            roots.append(cid)

    # Optional: topologically sort children by created_utc or something
    for node in nodes.values():
        node.children.sort(key=lambda c: (nodes[c].created_utc if nodes[c].created_utc is not None else math.inf))

    return nodes, roots

def compute_depths(nodes: Dict[str, CommentNode], roots: List[str]) -> None:
    """If original_depth not provided, compute depth via BFS."""
    have_depth = any(n.original_depth is not None for n in nodes.values())
    if have_depth:
        return
    for r in roots:
        q = deque([(r, 0)])
        while q:
            cid, d = q.popleft()
            nodes[cid].original_depth = d
            for ch in nodes[cid].children:
                q.append((ch, d + 1))

def collect_subtree(nodes: Dict[str, CommentNode], root_id: str) -> List[str]:
    """Return list of all node ids in the subtree rooted at root_id (including root)."""
    out = []
    stack = [root_id]
    while stack:
        x = stack.pop()
        out.append(x)
        stack.extend(nodes[x].children)
    return out

def count_descendants(nodes: Dict[str, CommentNode], root_id: str) -> int:
    """Number of descendants (replies) under the root."""
    return len(collect_subtree(nodes, root_id)) - 1

def subtree_entities(nodes: Dict[str, CommentNode], root_id: str, compiled_patterns) -> Set[str]:
    ents = set()
    for cid in collect_subtree(nodes, root_id):
        ents |= text_matches_entities(nodes[cid].text, compiled_patterns)
    return ents

def identify_promotions(
    nodes: Dict[str, CommentNode],
    roots: List[str],
    compiled_patterns,
    min_replies: int = 3
) -> Dict[str, Set[str]]:
    """
    Return a dict: promoted_root_id -> matched_entities
    Only consider non-top-level comments for promotion.
    """
    root_set = set(roots)
    promotions: Dict[str, Set[str]] = {}
    for cid, node in nodes.items():
        if cid in root_set:
            continue  # never promote a top-level comment
        # Replies threshold
        replies = count_descendants(nodes, cid)
        if replies < min_replies:
            continue
        # Entity presence
        ents = subtree_entities(nodes, cid, compiled_patterns)
        if ents:
            promotions[cid] = ents
    return promotions

# -----------------------------
# Building Output Structures
# -----------------------------

def build_nested_thread(nodes: Dict[str, CommentNode], root_id: str, cut_children: Set[str]) -> dict:
    """
    Build a nested dict for a thread rooted at root_id.
    cut_children: a set of child ids that should be cut at their root (used to remove promoted subtrees from original threads).
    """
    def build_rec(cid: str) -> dict:
        n = nodes[cid]
        child_structs = []
        for ch in n.children:
            if ch in cut_children:
                # Skip including this subtree here; it will appear as a promoted thread.
                continue
            child_structs.append(build_rec(ch))
        return {
            "comment_id": n.comment_id,
            "parent_id": n.parent_id,
            "text": n.text,
            "author": n.author,
            "score": n.score,
            "created_utc": n.created_utc,
            "original_depth": n.original_depth,
            "children": child_structs,
        }

    return build_rec(root_id)

def build_nested_subtree(nodes: Dict[str, CommentNode], root_id: str, original_roots: Set[str], matched_entities: Set[str]) -> dict:
    """
    Build the full nested subtree for a promoted root, with metadata about original placement.
    """
    # Find original top-level ancestor
    anc = root_id
    while nodes[anc].parent_id is not None and nodes[anc].parent_id in nodes:
        parent = nodes[anc].parent_id
        if parent in original_roots:
            top_ancestor = parent
            break
        anc = parent
    else:
        top_ancestor = None

    def build_rec(cid: str) -> dict:
        n = nodes[cid]
        return {
            "comment_id": n.comment_id,
            "parent_id": n.parent_id,
            "text": n.text,
            "author": n.author,
            "score": n.score,
            "created_utc": n.created_utc,
            "original_depth": n.original_depth,
            "children": [build_rec(ch) for ch in n.children],
        }

    subtree = build_rec(root_id)
    meta = {
        "promoted_root_id": root_id,
        "matched_entities": sorted(list(matched_entities)),
        "original_parent_id": nodes[root_id].parent_id,
        "original_top_level_ancestor": top_ancestor,
    }
    return {"meta": meta, "thread": subtree}

def flatten_thread_for_csv(
    thread_dict: dict,
    thread_id: str,
    is_promoted: bool,
    matched_entities_root: Optional[List[str]] = None
) -> List[dict]:
    """
    Flatten a nested thread into rows for CSV matching reddit_comments_nested.csv structure.
    """
    rows = []

    def rec(node: dict, depth_in_thread: int):
        rows.append({
            "comment_text": node["text"],
            "depth": depth_in_thread,
            "author": node["author"],
            "score": node["score"],
            "created_utc": node["created_utc"],
            "comment_id": node["comment_id"],
            "parent_id": node["parent_id"],
        })
        for ch in node.get("children", []):
            rec(ch, depth_in_thread + 1)

    rec(thread_dict, depth_in_thread=0)
    return rows

# -----------------------------
# Main
# -----------------------------

def run(comments_csv: str, entities_json: str, json_out: str, csv_out: str, min_replies: int = 3):
    # Load inputs
    df = pd.read_csv(comments_csv)
    entities = load_entities(entities_json)
    compiled_patterns = compile_entity_patterns(entities)

    # Build forest
    nodes, roots = build_forest(df)
    compute_depths(nodes, roots)
    original_roots_set = set(roots)

    # Identify promotions
    promotions = identify_promotions(nodes, roots, compiled_patterns, min_replies=min_replies)
    promoted_roots = set(promotions.keys())

    # Build original threads with promoted subtrees removed
    original_threads = []
    for r in roots:
        nested = build_nested_thread(nodes, r, cut_children=promoted_roots)
        original_threads.append(nested)

    # Build promoted threads (full subtrees)
    promoted_threads = []
    for pr in promoted_roots:
        ents = promotions[pr]
        promoted_threads.append(build_nested_subtree(nodes, pr, original_roots_set, ents))

    # Prepare JSON output
    meta = {
        "num_comments": len(nodes),
        "num_original_threads": len(roots),
        "num_promoted_threads": len(promoted_threads),
        "promotion_threshold_replies": min_replies,
        "entities_source_size": len(entities),
    }
    json_payload = {
        "meta": meta,
        "original_threads": original_threads,
        "promoted_threads": promoted_threads,
    }

    with open(json_out, "w") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2)

    # Prepare CSV output
    flat_rows: List[dict] = []

    # Original threads rows
    for idx, t in enumerate(original_threads, start=1):
        thread_id = f"orig_{idx}"
        flat_rows.extend(flatten_thread_for_csv(t, thread_id=thread_id, is_promoted=False))

    # Promoted threads rows
    for idx, t in enumerate(promoted_threads, start=1):
        thread_id = f"promoted_{idx}"
        matched_entities_root = t["meta"]["matched_entities"]
        flat_rows.extend(
            flatten_thread_for_csv(
                t["thread"],
                thread_id=thread_id,
                is_promoted=True,
                matched_entities_root=matched_entities_root
            )
        )

    flat_df = pd.DataFrame(flat_rows)
    flat_df.to_csv(csv_out, index=False)

    print(f"[INFO] Loaded {len(nodes)} comments across {len(roots)} original top-level threads.")
    print(f"[INFO] Entities loaded: {len(entities)}. Promotions found: {len(promoted_roots)}")
    if promoted_roots:
        print(f"[INFO] Promoted roots (sample): {list(promoted_roots)[:5]}")
    print(f"[OK] Wrote JSON to: {json_out}")
    print(f"[OK] Wrote CSV to:  {csv_out}")


def main():
    # Fixed input paths
    comments_csv = "Scripts/reddit_comments_nested.csv"
    entities_json = "Scripts/keyword_extraction_test_results.json"
    json_out = "promoted_threads.json"
    csv_out = "promoted_threads_restructured.csv"
    min_replies = 3

    run(comments_csv, entities_json, json_out, csv_out, min_replies=min_replies)


if __name__ == "__main__":
    main()
