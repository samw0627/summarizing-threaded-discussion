#!/usr/bin/env python3
"""
Reddit Comment Clusterer

This script takes a CSV file of Reddit comments (output from reddit_nested_comments_scraper.py)
and clusters them by topics using unsupervised machine learning. It exports the results
with cluster assignments to a new CSV file.
"""

import pandas as pd
import numpy as np
import re
import argparse
import sys
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class RedditCommentClusterer:
    def __init__(self, min_clusters: int = 2, max_clusters: int = 15, random_state: int = 42):
        """
        Initialize the comment clusterer.
        
        Args:
            min_clusters: Minimum number of clusters to try
            max_clusters: Maximum number of clusters to try
            random_state: Random state for reproducibility
        """
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.random_state = random_state
        self.vectorizer = None
        self.kmeans = None
        self.optimal_k = None
        self.feature_matrix = None
        self.cluster_labels = None
        self.topic_keywords = {}
        
    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess text for clustering.
        
        Args:
            text: Raw comment text
            
        Returns:
            Cleaned text
        """
        if pd.isna(text) or text == '':
            return ''
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove Reddit-specific formatting
        text = re.sub(r'/u/\w+', '', text)  # Remove user mentions
        text = re.sub(r'/r/\w+', '', text)  # Remove subreddit mentions
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Remove bold formatting
        text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Remove italic formatting
        
        # Remove special characters but keep spaces and basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def load_comments_from_csv(self, csv_file: str) -> pd.DataFrame:
        """
        Load comments from the nested CSV format created by the scraper.
        
        Args:
            csv_file: Path to the CSV file
            
        Returns:
            DataFrame with processed comments
        """
        df = pd.read_csv(csv_file)
        
        # Extract comment text from the level columns
        level_columns = [col for col in df.columns if col.startswith('level_')]
        
        # Combine all comment text from different nesting levels
        comments = []
        for idx, row in df.iterrows():
            comment_text = ''
            for col in level_columns:
                if pd.notna(row[col]) and row[col] != '':
                    comment_text = row[col]
                    break
            
            if comment_text and comment_text.strip():
                comment_data = {
                    'original_index': idx,
                    'comment_text': comment_text,
                    'author': row.get('author', 'unknown'),
                    'score': row.get('score', 0),
                    'created_utc': row.get('created_utc', 0),
                    'comment_id': row.get('comment_id', ''),
                    'parent_id': row.get('parent_id', ''),
                    'depth': self._get_comment_depth(row, level_columns)
                }
                comments.append(comment_data)
        
        return pd.DataFrame(comments)
    
    def _get_comment_depth(self, row: pd.Series, level_columns: List[str]) -> int:
        """Get the depth/nesting level of a comment."""
        for i, col in enumerate(level_columns):
            if pd.notna(row[col]) and row[col] != '':
                return i
        return 0
    
    def vectorize_comments(self, comments: List[str]) -> np.ndarray:
        """
        Convert comment text to TF-IDF feature vectors.
        
        Args:
            comments: List of preprocessed comment texts
            
        Returns:
            TF-IDF feature matrix
        """
        # Filter out empty comments
        valid_comments = [comment for comment in comments if comment and comment.strip()]
        
        if len(valid_comments) < 2:
            raise ValueError("Need at least 2 valid comments for clustering")
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=1000,  # Limit to top 1000 features
            stop_words='english',
            ngram_range=(1, 2),  # Use unigrams and bigrams
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.95  # Ignore terms that appear in more than 95% of documents
        )
        
        # Fit and transform the comments
        self.feature_matrix = self.vectorizer.fit_transform(valid_comments)
        
        return self.feature_matrix
    
    def find_optimal_clusters(self, feature_matrix: np.ndarray) -> int:
        """
        Find optimal number of clusters using elbow method and silhouette analysis.
        
        Args:
            feature_matrix: TF-IDF feature matrix
            
        Returns:
            Optimal number of clusters
        """
        max_k = min(self.max_clusters, feature_matrix.shape[0] - 1)
        
        if max_k < self.min_clusters:
            print(f"Warning: Not enough comments for clustering. Using {max_k} clusters.")
            return max_k
        
        inertias = []
        silhouette_scores = []
        k_range = range(self.min_clusters, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(feature_matrix)
            
            inertias.append(kmeans.inertia_)
            
            if len(set(cluster_labels)) > 1:  # Need at least 2 clusters for silhouette score
                sil_score = silhouette_score(feature_matrix, cluster_labels)
                silhouette_scores.append(sil_score)
            else:
                silhouette_scores.append(0)
        
        # Find elbow point
        optimal_k = self._find_elbow_point(k_range, inertias)
        
        # Validate with silhouette score
        if silhouette_scores:
            best_sil_k = k_range[np.argmax(silhouette_scores)]
            print(f"Elbow method suggests {optimal_k} clusters")
            print(f"Best silhouette score at {best_sil_k} clusters ({max(silhouette_scores):.3f})")
            
            # Use silhouette score if significantly better
            if max(silhouette_scores) > 0.3 and abs(best_sil_k - optimal_k) <= 2:
                optimal_k = best_sil_k
        
        return optimal_k
    
    def _find_elbow_point(self, k_range: range, inertias: List[float]) -> int:
        """Find the elbow point in the inertia curve."""
        if len(inertias) < 3:
            return k_range[0]
        
        # Calculate the rate of change
        changes = []
        for i in range(1, len(inertias)):
            changes.append(inertias[i-1] - inertias[i])
        
        # Find the point where the rate of change decreases most
        if len(changes) > 1:
            second_changes = []
            for i in range(1, len(changes)):
                second_changes.append(changes[i-1] - changes[i])
            
            if second_changes:
                elbow_idx = np.argmax(second_changes) + 2  # +2 because of indexing offset
                return list(k_range)[min(elbow_idx, len(k_range) - 1)]
        
        # Fallback: use middle of range
        return list(k_range)[len(k_range) // 2]
    
    def cluster_comments(self, feature_matrix: np.ndarray, n_clusters: Optional[int] = None) -> np.ndarray:
        """
        Perform K-means clustering on the feature matrix.
        
        Args:
            feature_matrix: TF-IDF feature matrix
            n_clusters: Number of clusters (auto-detected if None)
            
        Returns:
            Cluster labels
        """
        if n_clusters is None:
            n_clusters = self.find_optimal_clusters(feature_matrix)
        
        self.optimal_k = n_clusters
        
        # Perform K-means clustering
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )
        
        self.cluster_labels = self.kmeans.fit_predict(feature_matrix)
        
        return self.cluster_labels
    
    def extract_topic_keywords(self, n_keywords: int = 10) -> Dict[int, List[str]]:
        """
        Extract top keywords for each cluster to understand topics.
        
        Args:
            n_keywords: Number of top keywords per cluster
            
        Returns:
            Dictionary mapping cluster ID to list of keywords
        """
        if self.kmeans is None or self.vectorizer is None:
            raise ValueError("Must run clustering before extracting keywords")
        
        feature_names = self.vectorizer.get_feature_names_out()
        cluster_centers = self.kmeans.cluster_centers_
        
        for cluster_id in range(len(cluster_centers)):
            # Get top feature indices for this cluster
            top_indices = cluster_centers[cluster_id].argsort()[-n_keywords:][::-1]
            top_keywords = [feature_names[i] for i in top_indices]
            self.topic_keywords[cluster_id] = top_keywords
        
        return self.topic_keywords
    
    def create_results_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create results DataFrame with cluster assignments and topic keywords.
        
        Args:
            df: Original comments DataFrame
            
        Returns:
            DataFrame with cluster information
        """
        results_df = df.copy()
        
        # Add cluster assignments
        if len(self.cluster_labels) == len(df):
            results_df['cluster_id'] = self.cluster_labels
        else:
            # Handle case where some comments were filtered out
            results_df['cluster_id'] = -1
            valid_idx = 0
            for idx, row in results_df.iterrows():
                if row['comment_text'] and row['comment_text'].strip():
                    if valid_idx < len(self.cluster_labels):
                        results_df.loc[idx, 'cluster_id'] = self.cluster_labels[valid_idx]
                        valid_idx += 1
        
        # Add topic keywords
        results_df['topic_keywords'] = results_df['cluster_id'].apply(
            lambda x: ', '.join(self.topic_keywords.get(x, ['unknown'])) if x >= 0 else 'filtered'
        )
        
        # Add preprocessed text for reference
        results_df['preprocessed_text'] = results_df['comment_text'].apply(self.preprocess_text)
        
        return results_df
    
    def export_results(self, results_df: pd.DataFrame, output_file: str):
        """
        Export clustering results to CSV.
        
        Args:
            results_df: Results DataFrame
            output_file: Output CSV file path
        """
        # Reorder columns for better readability
        column_order = [
            'cluster_id', 'topic_keywords', 'comment_text', 'preprocessed_text',
            'author', 'score', 'depth', 'created_utc', 'comment_id', 'parent_id', 'original_index'
        ]
        
        # Only include columns that exist
        available_columns = [col for col in column_order if col in results_df.columns]
        export_df = results_df[available_columns]
        
        export_df.to_csv(output_file, index=False, encoding='utf-8')
        
        # Print summary statistics
        print(f"\nClustering Results Summary:")
        print(f"Total comments processed: {len(results_df)}")
        print(f"Number of clusters: {self.optimal_k}")
        print(f"Results exported to: {output_file}")
        
        cluster_counts = results_df['cluster_id'].value_counts().sort_index()
        print(f"\nComments per cluster:")
        for cluster_id, count in cluster_counts.items():
            if cluster_id >= 0:
                keywords = ', '.join(self.topic_keywords.get(cluster_id, ['unknown'])[:5])
                print(f"  Cluster {cluster_id}: {count} comments (Keywords: {keywords})")


def main():
    """Main function to handle command line arguments and execute clustering."""
    parser = argparse.ArgumentParser(
        description='Cluster Reddit comments by topics using unsupervised learning'
    )
    parser.add_argument('input_csv', help='Input CSV file from reddit_nested_comments_scraper.py')
    parser.add_argument('-o', '--output', default='reddit_comments_clustered.csv',
                       help='Output CSV file path (default: reddit_comments_clustered.csv)')
    parser.add_argument('-k', '--clusters', type=int,
                       help='Number of clusters (auto-detected if not specified)')
    parser.add_argument('--min-clusters', type=int, default=2,
                       help='Minimum number of clusters to try (default: 2)')
    parser.add_argument('--max-clusters', type=int, default=15,
                       help='Maximum number of clusters to try (default: 15)')
    parser.add_argument('--keywords', type=int, default=10,
                       help='Number of keywords to extract per topic (default: 10)')
    
    args = parser.parse_args()
    
    try:
        print(f"Loading comments from: {args.input_csv}")
        
        # Initialize clusterer
        clusterer = RedditCommentClusterer(
            min_clusters=args.min_clusters,
            max_clusters=args.max_clusters
        )
        
        # Load and preprocess comments
        df = clusterer.load_comments_from_csv(args.input_csv)
        
        if len(df) < 2:
            print("Error: Need at least 2 comments for clustering.")
            sys.exit(1)
        
        print(f"Loaded {len(df)} comments")
        
        # Preprocess and vectorize
        preprocessed_comments = [clusterer.preprocess_text(text) for text in df['comment_text']]
        feature_matrix = clusterer.vectorize_comments(preprocessed_comments)
        
        print(f"Created feature matrix with {feature_matrix.shape[1]} features")
        
        # Perform clustering
        cluster_labels = clusterer.cluster_comments(feature_matrix, args.clusters)
        
        # Extract topic keywords
        topic_keywords = clusterer.extract_topic_keywords(args.keywords)
        
        # Create and export results
        results_df = clusterer.create_results_dataframe(df)
        clusterer.export_results(results_df, args.output)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()