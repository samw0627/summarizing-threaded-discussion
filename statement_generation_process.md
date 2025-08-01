# Statement Generation Process in recursive_summarize.py

## Overview

The `recursive_summarize.py` script implements a sophisticated pipeline for analyzing threaded Reddit discussions and generating provocative thesis statements from clustered comments. This document explains the statement generation process in detail.

## Pipeline Architecture

### 1. Data Loading and Tree Construction
- **Input**: CSV file with nested Reddit comments (from `reddit_nested_comments_scraper.py`)
- **Process**: Builds a directed graph forest using NetworkX to preserve comment hierarchy
- **Key Function**: `build_forest(df)` creates nodes and edges based on parent-child relationships

### 2. Recursive Comment Enrichment
The enrichment process uses a recursive algorithm to summarize comment threads while preserving context:

#### Core Functions:
- **`enrich_comment(G, node_id)`**: Recursively processes each comment and its replies
- **`summarize_leaf(text)`**: Summarizes individual comments using GPT-4o-mini
- **`summarize_aggregate(context, combined, discussion_topic)`**: Combines parent context with child summaries

#### Drift Detection:
- Uses cosine similarity on text embeddings to measure topic drift
- **Threshold**: 0.6 (configurable via `DRIFT_THRESHOLD`)
- **Purpose**: Filters out off-topic replies to maintain coherent summaries

### 3. Comment Clustering
- **Algorithm**: K-means clustering on enriched comment embeddings
- **Embeddings**: OpenAI's `text-embedding-ada-002` model
- **Default**: 5 clusters (configurable)
- **Output**: Groups semantically similar comments together

### 4. Statement Generation Process

#### Theme Generation
```python
def generate_theme(summary_text):
    # Generates 3-5 word descriptive themes for each cluster
    # Uses GPT-4o-mini to create concise, specific themes
```

#### Statement Extraction
The `extract_statements()` function is the core of the thesis generation process:

##### Input Processing:
- **Summary Text**: Enriched summaries from the clustering process
- **Original Comments**: Raw comment text with IDs for accurate excerpting

##### LLM Prompt Engineering:
The system prompt enforces strict requirements for statement quality:

**Statement Requirements:**
- Maximum 20 words
- Must be declarative (not questions)
- Emotionally provocative and reaction-worthy
- Single, focused argument
- Debatable (not universally accepted facts)
- Relevant to the discussion topic
- No em dashes (—) for consistency

**Output Structure:**
```json
{
  "statements": [
    {
      "statement": "Thesis statement text",
      "excerpts": ["supporting quote 1", "supporting quote 2"]
    }
  ],
  "comments": [
    {
      "comment_id": "unique_id",
      "comment": "original comment text"
    }
  ]
}
```

##### Response Processing:
1. **Markdown Removal**: Strips ```json formatting from LLM responses
2. **JSON Parsing**: Converts string responses to structured objects
3. **Error Handling**: Provides fallback structure for parsing failures

### 5. Output Generation

#### Final Structure:
Each cluster produces:
- **Cluster ID**: Numeric identifier
- **Theme**: Descriptive 3-5 word summary
- **Statements**: Array of thesis statements with supporting excerpts
- **Comments**: Original comment data for reference

#### File Format:
- **Output**: JSON file (not CSV) for structured data
- **Encoding**: UTF-8 with proper JSON serialization
- **Indentation**: 2 spaces for readability

## Key Features

### Hierarchical Processing
- Preserves Reddit's threaded discussion structure
- Maintains parent-child relationships in analysis
- Enables context-aware summarization

### Topic Coherence
- Drift detection prevents off-topic contamination
- Embedding-based similarity ensures semantic clustering
- Theme generation provides cluster-level context

### Statement Quality Control
- Multi-layered prompt engineering for provocative statements
- Excerpt validation using original comment text
- Consistent formatting and structure

### Scalability
- Configurable cluster count
- Adjustable drift threshold
- Modular function design

## Usage Example

```bash
python recursive_summarize.py --input reddit_comments_nested.csv --output analysis_output.json --clusters 5
```


```bash
python reddit_nested_comments_scraper.py [https://www.reddit.com/r/Seattle/comments/1mbpzfu/the_challenges_of_navigating_seattle_as_a/] 
```

## Technical Dependencies

- **OpenAI API**: GPT-4o-mini for summarization and statement generation
- **NetworkX**: Graph-based comment tree management  
- **scikit-learn**: K-means clustering implementation
- **pandas**: Data manipulation and CSV processing
- **numpy**: Numerical operations for embeddings

## Discussion Topic Integration

The process is designed around a specific discussion topic (currently: "The challenges of navigating Seattle as a disabled person"). This topic:
- Guides summarization context
- Influences statement relevance filtering
- Ensures thematic coherence across clusters

## Quality Assurance

The statement generation process includes multiple quality checkpoints:
1. **Enrichment Quality**: Recursive summarization preserves key arguments
2. **Clustering Validation**: Semantic similarity ensures topical coherence
3. **Statement Standards**: Strict prompt requirements enforce provocative, debatable claims
4. **Excerpt Accuracy**: Original comment text prevents summarization artifacts
5. **Format Consistency**: JSON parsing ensures uniform structure

This multi-stage process transforms raw Reddit discussions into structured, analyzable thesis statements while maintaining the authenticity and context of the original conversations.