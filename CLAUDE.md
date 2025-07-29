# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based Reddit discussion analysis toolkit with two main components:

1. **Comment Scraper** (`reddit_nested_comments_scraper.py`): Extracts threaded discussions from Reddit and preserves their hierarchical nesting structure in CSV format
2. **Comment Clusterer** (`reddit_comment_clusterer.py`): Applies unsupervised machine learning to cluster scraped comments by topics and exports results with topic keywords

## Architecture

### Comment Scraper Architecture
Uses a single-file design with these components:

1. **Reddit API Integration**: Uses the `praw` (Python Reddit API Wrapper) library to authenticate and fetch data from Reddit
2. **Comment Traversal**: Implements recursive traversal to maintain parent-child relationships in threaded discussions  
3. **CSV Export**: Outputs structured data with nesting levels represented as separate columns

Key functions:
- `get_reddit_instance()`: Handles Reddit API authentication using environment variables
- `extract_nested_comments()`: Recursively traverses comment trees to preserve hierarchy
- `write_nested_csv()`: Exports comments to CSV with nesting represented by empty columns
- `traverse_comments()`: Internal recursive function that processes replies at each depth level

### Comment Clusterer Architecture
Uses object-oriented design centered around the `RedditCommentClusterer` class:

1. **Text Preprocessing**: Cleans Reddit-specific formatting, removes URLs, normalizes text
2. **Feature Extraction**: Uses TF-IDF vectorization with unigrams and bigrams  
3. **Clustering**: Applies K-means with automatic optimal cluster detection via elbow method
4. **Topic Analysis**: Extracts top keywords from cluster centroids to identify topics

Key methods:
- `preprocess_text()`: Cleans and normalizes comment text
- `vectorize_comments()`: Converts text to TF-IDF feature vectors
- `find_optimal_clusters()`: Uses elbow method and silhouette analysis for optimal K
- `cluster_comments()`: Performs K-means clustering
- `extract_topic_keywords()`: Identifies topic keywords from cluster centroids

## Dependencies

### Comment Scraper Dependencies
- `praw` - Reddit API wrapper
- `python-dotenv` - Environment variable management
- Standard library: `csv`, `sys`, `argparse`, `os`, `typing`

### Comment Clusterer Dependencies
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning (TF-IDF, K-means, PCA, metrics)
- `matplotlib` - Plotting (for potential visualization)
- `seaborn` - Statistical visualization
- Standard library: `re`, `argparse`, `sys`, `typing`, `collections`, `warnings`

## Environment Setup

The script requires Reddit API credentials stored in environment variables:
- `REDDIT_CLIENT_ID` - Reddit app client ID (required)
- `REDDIT_CLIENT_SECRET` - Reddit app client secret (required)  
- `REDDIT_USER_AGENT` - User agent string (optional, defaults to 'python:reddit-comment-scraper:v1.0.0')

These should be stored in a `.env` file (not tracked in git).

## Usage

### Comment Scraping
Basic usage:
```bash
python3 reddit_nested_comments_scraper.py <reddit_url>
```

With options:
```bash
python3 reddit_nested_comments_scraper.py <reddit_url> -o output.csv --max-depth 5
```

### Comment Clustering
Basic usage (processes CSV from scraper):
```bash
python3 reddit_comment_clusterer.py reddit_comments_nested.csv
```

With options:
```bash
python3 reddit_comment_clusterer.py input.csv -o clustered_output.csv -k 8 --keywords 15
```

Common parameters:
- `-k, --clusters`: Specify number of clusters (auto-detected if omitted)
- `--min-clusters`: Minimum clusters to try (default: 2)
- `--max-clusters`: Maximum clusters to try (default: 15)  
- `--keywords`: Number of topic keywords to extract per cluster (default: 10)

## CSV Output Formats

### Scraper Output
The scraper CSV uses a unique nesting structure:
- Each nesting level gets its own column (`level_0`, `level_1`, etc.)
- Comment content is placed in the column corresponding to its depth
- Additional metadata columns: `author`, `score`, `created_utc`, `comment_id`, `parent_id`

### Clusterer Output  
The clusterer adds topic analysis to the original data:
- `cluster_id`: Assigned cluster number (0, 1, 2, etc.)
- `topic_keywords`: Top keywords identifying the cluster's topic
- `preprocessed_text`: Cleaned version of comment text used for clustering
- All original columns from scraper output are preserved

## Development Notes

### Comment Scraper Notes
- The script handles deleted comments by showing '[deleted]' as the author
- All "MoreComments" objects are expanded to get complete discussion threads
- Error handling includes Reddit API credential validation
- The recursive traversal preserves Reddit's native comment hierarchy

### Comment Clusterer Notes
- Preprocessing removes Reddit-specific formatting (user mentions, subreddit links, markdown)
- TF-IDF vectorization uses unigrams and bigrams with English stop words removed
- Optimal cluster count determined by elbow method with silhouette score validation
- Empty or very short comments are filtered out before clustering
- Topic keywords extracted from cluster centroids using highest TF-IDF weights