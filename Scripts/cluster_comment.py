#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cluster Reddit comments with depth <= 2 using k-means clustering.

This script filters comments from reddit_comments_nested.csv to include only
those with depth <= 2, then applies k-means clustering to group similar comments.

Requirements:
- Input CSV: reddit_comments_nested.csv (must include: comment_text, depth)
- Dependencies: pandas, numpy, scikit-learn, matplotlib, seaborn

Usage:
    python3 cluster_comment.py

Outputs:
- clustered_comments_depth2.csv: Original data with cluster assignments and topic keywords
"""

import pandas as pd
import numpy as np
import re
import warnings
from collections import Counter
from typing import List, Tuple, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class RedditCommentClusterer:
    """
    K-means clustering for Reddit comments with depth <= 2.
    """
    
    def __init__(self, min_clusters: int = 2, max_clusters: int = 15, n_keywords: int = 10):
        """
        Initialize the clusterer.
        
        Args:
            min_clusters: Minimum number of clusters to try
            max_clusters: Maximum number of clusters to try
            n_keywords: Number of topic keywords to extract per cluster
        """
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.n_keywords = n_keywords
        self.vectorizer = None
        self.kmeans = None
        self.feature_matrix = None
        self.optimal_k = None
        
    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess Reddit comment text.
        
        Args:
            text: Raw comment text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
            
        # Remove Reddit-specific formatting
        text = re.sub(r'/u/\w+', '', text)  # Remove user mentions
        text = re.sub(r'/r/\w+', '', text)  # Remove subreddit mentions
        text = re.sub(r'https?://\S+', '', text)  # Remove URLs
        text = re.sub(r'www\.\S+', '', text)  # Remove www URLs
        
        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)  # Italic
        text = re.sub(r'~~(.*?)~~', r'\1', text)  # Strikethrough
        text = re.sub(r'`(.*?)`', r'\1', text)  # Code
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = text.strip().lower()
        
        return text
        
    def vectorize_comments(self, comments: List[str]) -> np.ndarray:
        """
        Convert comment text to TF-IDF feature vectors.
        
        Args:
            comments: List of preprocessed comment texts
            
        Returns:
            TF-IDF feature matrix
        """
        # Filter out very short comments
        filtered_comments = [c for c in comments if len(c.strip()) > 10]
        
        if len(filtered_comments) < self.min_clusters:
            raise ValueError(f"Need at least {self.min_clusters} valid comments for clustering")
            
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=2,  # Ignore terms that appear in fewer than 2 documents
            max_df=0.8  # Ignore terms that appear in more than 80% of documents
        )
        
        self.feature_matrix = self.vectorizer.fit_transform(filtered_comments)
        return self.feature_matrix
        
    def find_optimal_clusters(self, max_k: Optional[int] = None) -> int:
        """
        Find optimal number of clusters using elbow method and silhouette analysis.
        
        Args:
            max_k: Maximum K to try (defaults to self.max_clusters)
            
        Returns:
            Optimal number of clusters
        """
        if max_k is None:
            max_k = min(self.max_clusters, self.feature_matrix.shape[0] - 1)
            
        if max_k < self.min_clusters:
            return self.min_clusters
            
        inertias = []
        silhouette_scores = []
        k_range = range(self.min_clusters, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.feature_matrix)
            
            inertias.append(kmeans.inertia_)
            
            # Calculate silhouette score
            if k > 1 and len(set(cluster_labels)) > 1:
                sil_score = silhouette_score(self.feature_matrix, cluster_labels)
                silhouette_scores.append(sil_score)
            else:
                silhouette_scores.append(0)
                
        # Find elbow point
        if len(inertias) >= 2:
            # Simple elbow detection: find point with maximum second derivative
            second_derivatives = []
            for i in range(1, len(inertias) - 1):
                second_deriv = inertias[i-1] - 2*inertias[i] + inertias[i+1]
                second_derivatives.append(second_deriv)
                
            if second_derivatives:
                elbow_idx = np.argmax(second_derivatives) + 1
                elbow_k = list(k_range)[elbow_idx]
            else:
                elbow_k = self.min_clusters
        else:
            elbow_k = self.min_clusters
            
        # Choose K based on silhouette score if available
        if silhouette_scores and max(silhouette_scores) > 0.3:
            best_sil_idx = np.argmax(silhouette_scores)
            best_sil_k = list(k_range)[best_sil_idx]
            self.optimal_k = best_sil_k
        else:
            self.optimal_k = elbow_k
            
        return self.optimal_k
        
    def cluster_comments(self, n_clusters: Optional[int] = None) -> np.ndarray:
        """
        Perform K-means clustering on the comments.
        
        Args:
            n_clusters: Number of clusters (auto-detected if None)
            
        Returns:
            Cluster labels for each comment
        """
        if n_clusters is None:
            n_clusters = self.find_optimal_clusters()
            
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans.fit_predict(self.feature_matrix)
        
        return cluster_labels
        
    def extract_topic_keywords(self, cluster_labels: np.ndarray) -> dict:
        """
        Extract topic keywords for each cluster.
        
        Args:
            cluster_labels: Array of cluster assignments
            
        Returns:
            Dictionary mapping cluster_id to list of keywords
        """
        if self.vectorizer is None or self.kmeans is None:
            raise ValueError("Must run clustering before extracting keywords")
            
        feature_names = self.vectorizer.get_feature_names_out()
        cluster_keywords = {}
        
        for cluster_id in range(self.kmeans.n_clusters):
            # Get centroid for this cluster
            centroid = self.kmeans.cluster_centers_[cluster_id]
            
            # Get top keywords by TF-IDF weight
            top_indices = centroid.argsort()[-self.n_keywords:][::-1]
            keywords = [feature_names[i] for i in top_indices]
            
            cluster_keywords[cluster_id] = keywords
            
        return cluster_keywords


def load_and_filter_comments(csv_path: str) -> pd.DataFrame:
    """
    Load comments CSV and filter for depth <= 2.
    
    Args:
        csv_path: Path to the reddit_comments_nested.csv file
        
    Returns:
        Filtered DataFrame
    """
    df = pd.read_csv(csv_path)
    
    # Check required columns
    required_cols = {'comment_text', 'depth'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
        
    # Filter for depth <= 2
    filtered_df = df[df['depth'] <= 2].copy()
    
    # Remove empty or very short comments
    filtered_df = filtered_df[
        (filtered_df['comment_text'].notna()) & 
        (filtered_df['comment_text'].str.len() > 10)
    ]
    
    print(f"[INFO] Loaded {len(df)} total comments")
    print(f"[INFO] Filtered to {len(filtered_df)} comments with depth <= 2")
    
    return filtered_df


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cluster Reddit comments with depth <= 2")
    parser.add_argument("--input", "-i", default="reddit_comments_nested.csv", help="Input CSV file")
    parser.add_argument("--output", "-o", default="clustered_comments_depth2.csv", help="Output CSV file")
    
    args = parser.parse_args()
    
    input_csv = args.input
    output_csv = args.output
    
    try:
        # Load and filter data
        df = load_and_filter_comments(input_csv)
        
        if len(df) < 2:
            print("[ERROR] Need at least 2 comments for clustering")
            return
            
        # Initialize clusterer
        clusterer = RedditCommentClusterer(
            min_clusters=2,
            max_clusters=min(15, len(df) // 2),
            n_keywords=10
        )
        
        # Preprocess text
        print("[INFO] Preprocessing comment text...")
        df['preprocessed_text'] = df['comment_text'].apply(clusterer.preprocess_text)
        
        # Filter out comments that became empty after preprocessing
        valid_mask = df['preprocessed_text'].str.len() > 10
        df = df[valid_mask].copy()
        
        if len(df) < clusterer.min_clusters:
            print(f"[ERROR] Only {len(df)} valid comments after preprocessing, need at least {clusterer.min_clusters}")
            return
            
        # Vectorize comments
        print("[INFO] Vectorizing comments...")
        clusterer.vectorize_comments(df['preprocessed_text'].tolist())
        
        # Perform clustering
        print("[INFO] Finding optimal number of clusters...")
        optimal_k = clusterer.find_optimal_clusters()
        print(f"[INFO] Optimal number of clusters: {optimal_k}")
        
        print("[INFO] Performing K-means clustering...")
        cluster_labels = clusterer.cluster_comments(optimal_k)
        
        # Extract topic keywords
        print("[INFO] Extracting topic keywords...")
        topic_keywords = clusterer.extract_topic_keywords(cluster_labels)
        
        # Add results to dataframe
        df = df.reset_index(drop=True)
        df['cluster_id'] = cluster_labels
        df['topic_keywords'] = df['cluster_id'].map(
            lambda x: ', '.join(topic_keywords[x])
        )
        
        # Save results
        df.to_csv(output_csv, index=False)
        
        # Print summary
        cluster_counts = Counter(cluster_labels)
        print(f"\n[RESULTS] Clustering completed!")
        print(f"[RESULTS] Total comments clustered: {len(df)}")
        print(f"[RESULTS] Number of clusters: {optimal_k}")
        print(f"[RESULTS] Cluster distribution: {dict(cluster_counts)}")
        
        print(f"\n[TOPICS] Topic keywords by cluster:")
        for cluster_id, keywords in topic_keywords.items():
            count = cluster_counts[cluster_id]
            print(f"  Cluster {cluster_id} ({count} comments): {', '.join(keywords[:5])}")
            
        print(f"\n[OK] Results saved to: {output_csv}")
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return


if __name__ == "__main__":
    main()