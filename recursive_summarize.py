import os
import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans
from openai import OpenAI
from Scripts.entity_extractor import EntityExtractor

# --- Load API key from environment ---
api_key = "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"
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Discussion topic will be passed as argument

# --- 1. Load CSV into DataFrame ---
def load_comments(path):
    df = pd.read_csv(path)
    return df

# --- 2. Build tree (forest) using parent_id ---
def build_forest(df):
    G = nx.DiGraph()
    valid_ids = set(df['comment_id'])
    for _, row in df.iterrows():
        cid = row['comment_id']
        G.add_node(cid, **row.to_dict())
    for _, row in df.iterrows():
        pid = row['parent_id']
        cid = row['comment_id']
        if pd.notna(pid) and pid in valid_ids:
            G.add_edge(pid, cid)
    roots = [n for n, d in G.in_degree() if d == 0]
    return G, roots

# --- 3. Recursive enrichment & pruning ---
DRIFT_THRESHOLD = 0.85

def embed(text):
    resp = client.embeddings.create(model="text-embedding-ada-002", input=text)
    # Access embedding attribute from CreateEmbeddingResponse
    return resp.data[0].embedding

def cosine(a, b):
    import numpy as np
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def summarize_leaf(text):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Summarize this comment in 1-2 sentences."},
            {"role": "user", "content": text}
        ]
    )
    return resp.choices[0].message.content.strip()

def summarize_aggregate(context, combined, discussion_topic):
    prompt = f"Discussion Topic: {discussion_topic}\nContext: {context}\nChild Summaries:\n{combined}\nProduce a concise summary incorporating on-topic replies that are relevant to the discussion topic."
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content.strip()

def get_text(node_data):
    # Each row has a single comment in 'comment_text'
    text = node_data.get('comment_text')
    if not text or pd.isna(text):
        raise ValueError(f"Missing 'comment_text' for node {node_data.get('comment_id')}")
    return str(text)

# --- Comment Filtering Functions ---
def is_topic_relevant(comment_text, discussion_topic, threshold=0.5):
    """Check if comment is directly relevant to main discussion topic"""
    try:
        comment_embedding = embed(comment_text)
        topic_embedding = embed(discussion_topic)
        similarity = cosine(comment_embedding, topic_embedding)
        return similarity >= threshold
    except Exception as e:
        print(f"[WARNING] Error checking topic relevance: {e}")
        return True  # Default to keeping comment if error occurs

def quality_filter(comment_text):
    """Filter out low-quality comments"""
    import re
    
    text = comment_text.strip()
    
    # Too short (less than 10 characters or 3 words)
    if len(text) < 10 or len(text.split()) < 3:
        return False
    
    # Just emoji or punctuation
    if not any(c.isalpha() for c in text):
        return False
    
    # Common low-value patterns
    low_value_patterns = [
        r'^this\s*$', r'^exactly\s*$', r'^lol\s*$', r'^same\s*$',
        r'^\+1\s*$', r'^upvoted\s*$', r'^came here to say this',
        r'^deleted\s*$', r'^\[deleted\]\s*$', r'^\[removed\]\s*$'
    ]
    
    for pattern in low_value_patterns:
        if re.match(pattern, text.lower()):
            return False
    
    return True

def filter_irrelevant_comments(df, discussion_topic, topic_threshold=0.5):
    """Remove comments not relevant to discussion topic and low-quality comments"""
    print(f"[INFO] Filtering comments for topic relevance (threshold: {topic_threshold})")
    original_count = len(df)
    
    filtered_rows = []
    for _, row in df.iterrows():
        comment_text = get_text(row)
        
        # Quality filter first (cheaper check)
        if not quality_filter(comment_text):
            continue
            
        # Topic relevance filter (more expensive)
        if is_topic_relevant(comment_text, discussion_topic, topic_threshold):
            filtered_rows.append(row)
    
    filtered_df = pd.DataFrame(filtered_rows)
    filtered_count = len(filtered_df)
    removed_count = original_count - filtered_count
    
    print(f"[INFO] Filtered out {removed_count}/{original_count} comments ({removed_count/original_count*100:.1f}%)")
    print(f"[INFO] Keeping {filtered_count} relevant comments")
    
    return filtered_df

def enrich_comment(G, node_id, discussion_topic):
    node = G.nodes[node_id]
    text = get_text(node)
    children = list(G.successors(node_id))
    if not children:
        return summarize_leaf(text)
    kept = []
    for c in children:
        child_sum = enrich_comment(G, c, discussion_topic)
        if cosine(embed(child_sum), embed(text)) >= DRIFT_THRESHOLD:
            kept.append(child_sum)
    combined = "\n".join(kept)
    return summarize_aggregate(text, combined, discussion_topic)

# --- 4. Enrich all comments ---
def enrich_all(df, G, discussion_topic):
    enriched = {}
    for cid in G.nodes:
        enriched[cid] = enrich_comment(G, cid, discussion_topic)
    df['enriched'] = df['comment_id'].map(enriched)
    return df

# --- 5. Clustering ---
def cluster_comments(df, n_clusters):
    embs = [embed(t) for t in df['enriched']]
    labels = KMeans(n_clusters=n_clusters).fit_predict(embs)
    df['cluster'] = labels
    return df

# --- 6. Cluster-level summary & statement extraction ---
def summarize_cluster(G, roots, discussion_topic):
    parts = []
    for r in roots:
        parts.append(enrich_comment(G, r, discussion_topic))
    return "\n".join(parts)

def generate_theme(summary_text):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Generate a concise theme (3-5 words) that captures the main topic of this cluster of comments. The theme should be descriptive and specific to the content."},
            {"role": "user", "content": f"Summary: {summary_text}"}
        ]
    )
    return resp.choices[0].message.content.strip()

def extract_statements_grounded(summary_text, original_comments, entity_targets=None, cluster_entities=None):
    """
    Extract statements with entity grounding for proportional coverage.
    
    Args:
        summary_text: Summary of the cluster
        original_comments: Original comment texts
        entity_targets: Target number of statements per entity
        cluster_entities: Top entities for this cluster
    """
    # Build grounding context if entities provided
    grounding_context = ""
    if entity_targets and cluster_entities:
        total_target = sum(entity_targets.values())
        grounding_context = f"\n\nEntity Grounding Instructions:\nGenerate approximately {total_target} statements total. Focus on these key topics with the specified emphasis:\n"
        for entity, target_count in entity_targets.items():
            if entity in cluster_entities and target_count > 0:
                grounding_context += f"- '{entity}': aim for {target_count} statements\n"
        grounding_context += "\nEnsure statements cover these entities proportionally while maintaining quality and relevance."
    
    system_prompt = f''' You create thesis statements for comments in a discussion. These statements should summarise the main claim of the discussion. When you create these statements, you should also provide excerpt from that comment that supports that statement. Each excerpt should only map to one statement, but one statement can be mapped by multiple excerpt. Each comment can have multiple highlight, hence multiple excerpt.  Below are requirements for the statements
Requirements:
- Statements should be less than 20 words.
- It should not be a question.
- Make statements emotionally provocative and reaction-worthy
- Each statement should have one argument.
- A statement should not be fact that people can't disagree
- Avoid redundancy while ensuring comprehensive coverage
- All statements should be relevant to the discussion topic. You can check this by reading the statement itself and determining whether it is relevant to the topic, without considering other context.
- Do NOT use em dashes (—) anywhere in the statements
- Use periods, commas, or other punctuation instead of em dashes
For some short comments (only a few words), it is fine that there is no statement and excerpts if no opinion can be summarised.{grounding_context}

Output format should be a clean JSON object (no markdown formatting) with this structure:
{{
  "statements": [
    {{
      "statement": "Your thesis statement here",
      "excerpts": ["excerpt 1", "excerpt 2"]
    }}
  ],
  "comments": [
    {{
      "comment_id": "id",  
      "comment": "original comment text"
    }}
  ]
}}

Use the original comments provided below for excerpts. Only output the JSON object, no markdown formatting or extra text.'''

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Summary for analysis: {summary_text}\n\nOriginal comments to use for excerpts:\n{original_comments}"}
        ]
    )
    
    # Parse the JSON response
    import json, re
    response_text = resp.choices[0].message.content.strip()
    
    # Remove markdown formatting if present
    response_text = re.sub(r'^```json\s*', '', response_text)
    response_text = re.sub(r'\s*```$', '', response_text)
    
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        return {
            "statements": [],
            "comments": [],
            "error": "Failed to parse LLM response"
        }

def extract_statements(summary_text, original_comments):
    """Legacy function for backwards compatibility."""
    return extract_statements_grounded(summary_text, original_comments)

# --- 7. Pipeline orchestration ---
def run_pipeline(input_csv, output_csv, discussion_topic, n_clusters=5, pruned_output=None, topic_threshold=0.5, use_entity_grounding=False, total_statements=20):
    print(f"[INFO] Starting with {input_csv}")
    df = load_comments(input_csv)
    print(f"[INFO] Loaded {len(df)} rows")
    
    # Filter out irrelevant comments BEFORE building forest and clustering
    df = filter_irrelevant_comments(df, discussion_topic, topic_threshold)
    
    # Rebuild forest with filtered data
    G, roots = build_forest(df)
    print(f"[INFO] Forest: {len(G.nodes)} nodes, {len(roots)} roots")
    df = enrich_all(df, G, discussion_topic)
    print("[INFO] Enrichment done")
    df = cluster_comments(df, n_clusters)
    print(f"[INFO] Clustering into {n_clusters} groups")

    # Entity extraction for grounding (if enabled)
    entity_extractor = None
    overall_entity_targets = {}
    cluster_entities = {}
    
    if use_entity_grounding:
        print("[INFO] Extracting entities for statement grounding using LLM analysis")
        entity_extractor = EntityExtractor(method="llm", openai_client=client)
        
        # Extract overall entities from all comments
        all_comments = [get_text(row) for _, row in df.iterrows()]
        overall_entities = entity_extractor.extract_entities_from_comments(all_comments)
        overall_frequencies = entity_extractor.calculate_entity_frequencies(overall_entities, top_n=30)
        overall_entity_targets = entity_extractor.get_statement_targets(overall_frequencies, total_statements)
        
        # Extract cluster-specific entities
        cluster_entities = entity_extractor.extract_entities_from_clusters(df)
        
        print(f"[INFO] Entity extraction complete. Top entities: {list(overall_entities.keys())[:5]}")

    results = []
    pruned = {}
    for cid in sorted(df['cluster'].unique()):
        sub = df[df['cluster']==cid]
        print(f"[INFO] Cluster {cid}: {len(sub)} comments")
        subG, subR = build_forest(sub)
        # prune tree
        pG = nx.DiGraph()
        def prune(node):
            txt = get_text(subG.nodes[node])
            pG.add_node(node, text=txt)
            kept = []
            for ch in subG.successors(node):
                cs = prune(ch)
                if cosine(embed(cs), embed(txt))>=DRIFT_THRESHOLD:
                    pG.add_edge(node,ch)
                    kept.append(cs)
            return summarize_leaf(txt) if not kept else summarize_aggregate(txt, "\n".join(kept), discussion_topic)
        summary_parts = [prune(r) for r in subR]
        # Build pruned trees per root since tree_data accepts a single root
        pruned_trees = {}
        for r in subR:
            # extract the subtree starting from root r to ensure it's a true tree
            subtree = nx.dfs_tree(pG, r)
            pruned_trees[r] = nx.tree_data(subtree, r)
        pruned[cid] = pruned_trees
        full = "\n".join(summary_parts)
        
        # Get original comments for this cluster
        original_comments = ""
        for _, row in sub.iterrows():
            original_comments += f"Comment ID: {row['comment_id']}\nComment: {get_text(row)}\n\n"
        
        # Generate theme for this cluster
        theme = generate_theme(full)
        
        # Extract statements with entity grounding if enabled
        if use_entity_grounding and cid in cluster_entities:
            statements_result = extract_statements_grounded(
                full, 
                original_comments, 
                overall_entity_targets, 
                cluster_entities[cid]
            )
        else:
            statements_result = extract_statements(full, original_comments)
        
        results.append({
            'cluster': int(cid), 
            'theme': theme,
            'statements': statements_result,
            'entity_info': {
                'top_entities': list(cluster_entities.get(cid, {}).keys())[:5] if use_entity_grounding else [],
                'entity_targets': {k: v for k, v in overall_entity_targets.items() if k in cluster_entities.get(cid, {})} if use_entity_grounding else {}
            }
        })

    # Save as JSON instead of CSV
    import json
    with open(output_csv, 'w') as f:
        json.dump(results, f, indent=2)
    if pruned_output:
        import json
        with open(pruned_output,'w') as f:
            json.dump(pruned, f, indent=2)
    
    # Export entity analysis if grounding was used
    if use_entity_grounding and entity_extractor:
        entity_output = output_csv.replace('.json', '_entities.json')
        entity_extractor.export_entity_analysis(entity_output, cluster_entities)
        print(f"[INFO] Entity analysis saved to {entity_output}")

if __name__=='__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--input',required=True, help='Input CSV file with Reddit comments')
    p.add_argument('--output',required=True, help='Output JSON file for analysis results')
    p.add_argument('--topic',required=True, help='Discussion topic for context and analysis')
    p.add_argument('--clusters',type=int,default=5, help='Number of clusters (default: 5)')
    p.add_argument('--topic-threshold',type=float,default=0.5, help='Topic relevance threshold (0.0-1.0, default: 0.5)')
    p.add_argument('--pruned-output', help='Optional output file for pruned tree data')
    p.add_argument('--use-entity-grounding', action='store_true', help='Enable entity-grounded statement generation')
    p.add_argument('--total-statements', type=int, default=20, help='Total number of statements to target (default: 20)')
    args = p.parse_args()
    run_pipeline(args.input, args.output, args.topic, args.clusters, args.pruned_output, args.topic_threshold, args.use_entity_grounding, args.total_statements)


