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
import logging
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

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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
        logging.debug(f"preprocess_text called with: {repr(text)} (type: {type(text)})")
        
        if text is None or pd.isna(text) or text == '':
            logging.debug("Text is None, NaN, or empty - returning empty string")
            return ''
        
        try:
            # Convert to lowercase
            logging.debug("Converting to lowercase")
            text = text.lower()
            
            # Remove URLs
            logging.debug("Removing URLs")
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            
            # Remove Reddit-specific formatting
            logging.debug("Removing Reddit formatting")
            text = re.sub(r'/u/\w+', '', text)  # Remove user mentions
            text = re.sub(r'/r/\w+', '', text)  # Remove subreddit mentions
            text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Remove bold formatting
            text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Remove italic formatting
            
            # Remove special characters but keep spaces and basic punctuation
            logging.debug("Removing special characters")
            text = re.sub(r'[^\w\s.,!?-]', ' ', text)
            
            # Remove extra whitespace
            logging.debug("Removing extra whitespace")
            text = re.sub(r'\s+', ' ', text).strip()
            
            logging.debug(f"Preprocessed text result: {repr(text)}")
            return text
            
        except Exception as e:
            logging.error(f"Error in preprocess_text: {e}, input was: {repr(text)} (type: {type(text)})")
            raise
    
    def load_comments_from_csv(self, csv_file: str) -> pd.DataFrame:
        """
        Load comments from the nested CSV format created by the scraper.
        
        Args:
            csv_file: Path to the CSV file
            
        Returns:
            DataFrame with processed comments
        """
        logging.info(f"Loading CSV file: {csv_file}")
        
        try:
            df = pd.read_csv(csv_file)
            logging.info(f"CSV loaded successfully. Shape: {df.shape}")
            logging.debug(f"CSV columns: {df.columns.tolist()}")
            
            # Extract comment text from the level columns
            level_columns = [col for col in df.columns if col.startswith('level_')]
            logging.info(f"Found level columns: {level_columns}")
            
            # Combine all comment text from different nesting levels
            comments = []
            for idx, row in df.iterrows():
                logging.debug(f"Processing row {idx}")
                comment_text = ''
                
                for col in level_columns:
                    cell_value = row[col]
                    logging.debug(f"  Checking column {col}: {repr(cell_value)} (type: {type(cell_value)})")
                    
                    if pd.notna(cell_value) and cell_value != '':
                        comment_text = cell_value
                        logging.debug(f"  Found comment text in {col}: {repr(comment_text)}")
                        break
                
                if comment_text and str(comment_text).strip():
                    logging.debug(f"  Adding comment: {repr(comment_text)}")
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
                else:
                    logging.debug(f"  Skipping row {idx} - no valid comment text")
            
            logging.info(f"Extracted {len(comments)} valid comments from {len(df)} rows")
            return pd.DataFrame(comments)
            
        except Exception as e:
            logging.error(f"Error loading CSV file: {e}")
            raise
    
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
        logging.info(f"vectorize_comments called with {len(comments)} comments")
        
        # Filter out empty comments
        logging.debug("Filtering out empty comments")
        valid_comments = []
        for i, comment in enumerate(comments):
            logging.debug(f"Comment {i}: {repr(comment)} (type: {type(comment)})")
            if comment and str(comment).strip():
                valid_comments.append(comment)
            else:
                logging.debug(f"  Filtered out comment {i}")
        
        logging.info(f"After filtering: {len(valid_comments)} valid comments")
        
        if len(valid_comments) < 2:
            raise ValueError("Need at least 2 valid comments for clustering")
        
        try:
            # Initialize TF-IDF vectorizer
            logging.info("Initializing TF-IDF vectorizer")
            self.vectorizer = TfidfVectorizer(
                max_features=1000,  # Limit to top 1000 features
                stop_words='english',
                ngram_range=(1, 2),  # Use unigrams and bigrams
                min_df=2,  # Ignore terms that appear in less than 2 documents
                max_df=0.95  # Ignore terms that appear in more than 95% of documents
            )
            
            # Fit and transform the comments
            logging.info("Fitting and transforming comments with TF-IDF")
            self.feature_matrix = self.vectorizer.fit_transform(valid_comments)
            logging.info(f"Feature matrix created with shape: {self.feature_matrix.shape}")
            
            return self.feature_matrix
            
        except Exception as e:
            logging.error(f"Error in vectorize_comments: {e}")
            logging.error(f"Valid comments sample: {valid_comments[:5]}")
            raise
    
    def find_optimal_clusters(self, feature_matrix: np.ndarray) -> int:
        """
        Find optimal number of clusters using elbow method and silhouette analysis.
        
        Args:
            feature_matrix: TF-IDF feature matrix
            
        Returns:
            Optimal number of clusters
        """
        logging.info(f"find_optimal_clusters called with feature_matrix shape: {feature_matrix.shape}")
        
        try:
            max_k = min(self.max_clusters, feature_matrix.shape[0] - 1)
            logging.info(f"max_k calculated as: {max_k}")
            
            if max_k < self.min_clusters:
                logging.warning(f"Not enough comments for clustering. Using {max_k} clusters.")
                print(f"Warning: Not enough comments for clustering. Using {max_k} clusters.")
                return max_k
            
            inertias = []
            silhouette_scores = []
            k_range = range(self.min_clusters, max_k + 1)
            logging.info(f"Testing k values: {list(k_range)}")
            
            for k in k_range:
                logging.debug(f"Testing k={k}")
                try:
                    kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                    logging.debug(f"Created KMeans object for k={k}")
                    
                    cluster_labels = kmeans.fit_predict(feature_matrix)
                    logging.debug(f"fit_predict completed for k={k}, labels shape: {cluster_labels.shape}")
                    
                    inertias.append(kmeans.inertia_)
                    logging.debug(f"Inertia for k={k}: {kmeans.inertia_}")
                    
                    if len(set(cluster_labels)) > 1:  # Need at least 2 clusters for silhouette score
                        logging.debug(f"Calculating silhouette score for k={k}")
                        sil_score = silhouette_score(feature_matrix, cluster_labels)
                        silhouette_scores.append(sil_score)
                        logging.debug(f"Silhouette score for k={k}: {sil_score}")
                    else:
                        silhouette_scores.append(0)
                        logging.debug(f"Only one cluster for k={k}, using silhouette score 0")
                        
                except Exception as e:
                    logging.error(f"Error testing k={k}: {e}")
                    raise
            
            logging.info("Finding elbow point")
            # Find elbow point
            optimal_k = self._find_elbow_point(k_range, inertias)
            logging.info(f"Elbow point found at k={optimal_k}")
            
            # Validate with silhouette score
            if silhouette_scores:
                best_sil_k = k_range[np.argmax(silhouette_scores)]
                logging.info(f"Best silhouette score at k={best_sil_k}")
                print(f"Elbow method suggests {optimal_k} clusters")
                print(f"Best silhouette score at {best_sil_k} clusters ({max(silhouette_scores):.3f})")
                
                # Use silhouette score if significantly better
                if max(silhouette_scores) > 0.3 and abs(best_sil_k - optimal_k) <= 2:
                    optimal_k = best_sil_k
                    logging.info(f"Using silhouette-based optimal k={optimal_k}")
            
            return optimal_k
            
        except Exception as e:
            logging.error(f"Error in find_optimal_clusters: {e}")
            raise
    
    def _find_elbow_point(self, k_range: range, inertias: List[float]) -> int:
        """Find the elbow point in the inertia curve."""
        logging.debug(f"_find_elbow_point called with k_range: {list(k_range)}, inertias: {inertias}")
        
        try:
            if len(inertias) < 3:
                logging.debug("Less than 3 inertias, returning first k value")
                return list(k_range)[0]
            
            # Calculate the rate of change
            logging.debug("Calculating rate of change")
            changes = []
            for i in range(1, len(inertias)):
                change = inertias[i-1] - inertias[i]
                changes.append(change)
                logging.debug(f"Change {i}: {change}")
            
            # Find the point where the rate of change decreases most
            if len(changes) > 1:
                logging.debug("Calculating second-order changes")
                second_changes = []
                for i in range(1, len(changes)):
                    second_change = changes[i-1] - changes[i]
                    second_changes.append(second_change)
                    logging.debug(f"Second change {i}: {second_change}")
                
                if second_changes:
                    elbow_idx = np.argmax(second_changes) + 2  # +2 because of indexing offset
                    k_range_list = list(k_range)
                    result_idx = min(elbow_idx, len(k_range_list) - 1)
                    result = k_range_list[result_idx]
                    logging.debug(f"Elbow point found at index {elbow_idx}, returning k={result}")
                    return result
            
            # Fallback: use middle of range
            k_range_list = list(k_range)
            fallback_idx = len(k_range_list) // 2
            result = k_range_list[fallback_idx]
            logging.debug(f"Using fallback middle value: k={result}")
            return result
            
        except Exception as e:
            logging.error(f"Error in _find_elbow_point: {e}")
            logging.error(f"k_range: {k_range} (type: {type(k_range)})")
            logging.error(f"inertias: {inertias} (type: {type(inertias)})")
            raise
    
    def cluster_comments(self, feature_matrix: np.ndarray, n_clusters: Optional[int] = None) -> np.ndarray:
        """
        Perform K-means clustering on the feature matrix.
        
        Args:
            feature_matrix: TF-IDF feature matrix
            n_clusters: Number of clusters (auto-detected if None)
            
        Returns:
            Cluster labels
        """
        logging.info(f"cluster_comments called with feature_matrix shape: {feature_matrix.shape}")
        
        try:
            if n_clusters is None:
                logging.info("Finding optimal number of clusters")
                n_clusters = self.find_optimal_clusters(feature_matrix)
            
            logging.info(f"Using {n_clusters} clusters")
            self.optimal_k = n_clusters
            
            # Perform K-means clustering
            logging.info("Initializing K-means")
            self.kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                n_init=10,
                max_iter=300
            )
            
            logging.info("Fitting K-means and predicting clusters")
            self.cluster_labels = self.kmeans.fit_predict(feature_matrix)
            logging.info(f"Clustering complete. Labels shape: {self.cluster_labels.shape}")
            
            return self.cluster_labels
            
        except Exception as e:
            logging.error(f"Error in cluster_comments: {e}")
            raise
    
    def extract_topic_keywords(self, n_keywords: int = 10) -> Dict[int, List[str]]:
        """
        Extract top keywords for each cluster to understand topics.
        
        Args:
            n_keywords: Number of top keywords per cluster
            
        Returns:
            Dictionary mapping cluster ID to list of keywords
        """
        logging.info(f"extract_topic_keywords called with n_keywords={n_keywords}")
        
        if self.kmeans is None or self.vectorizer is None:
            raise ValueError("Must run clustering before extracting keywords")
        
        try:
            logging.info("Getting feature names from vectorizer")
            feature_names = self.vectorizer.get_feature_names_out()
            logging.debug(f"Feature names type: {type(feature_names)}, sample: {feature_names[:10] if len(feature_names) > 0 else 'empty'}")
            
            logging.info("Getting cluster centers")
            cluster_centers = self.kmeans.cluster_centers_
            logging.info(f"Cluster centers shape: {cluster_centers.shape}")
            
            for cluster_id in range(len(cluster_centers)):
                logging.debug(f"Processing cluster {cluster_id}")
                # Get top feature indices for this cluster
                top_indices = cluster_centers[cluster_id].argsort()[-n_keywords:][::-1]
                logging.debug(f"Top indices for cluster {cluster_id}: {top_indices}")
                
                top_keywords = []
                for i in top_indices:
                    keyword = feature_names[i]
                    logging.debug(f"  Feature {i}: {repr(keyword)} (type: {type(keyword)})")
                    top_keywords.append(keyword)
                
                self.topic_keywords[cluster_id] = top_keywords
                logging.debug(f"Keywords for cluster {cluster_id}: {top_keywords}")
            
            logging.info(f"Topic keywords extraction complete: {len(self.topic_keywords)} clusters")
            return self.topic_keywords
            
        except Exception as e:
            logging.error(f"Error in extract_topic_keywords: {e}")
            logging.error(f"kmeans: {self.kmeans}")
            logging.error(f"vectorizer: {self.vectorizer}")
            raise
    
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
        logging.info("Starting text preprocessing")
        preprocessed_comments = []
        for i, text in enumerate(df['comment_text']):
            logging.debug(f"Preprocessing comment {i}: {repr(text)}")
            processed = clusterer.preprocess_text(text)
            preprocessed_comments.append(processed)
        
        logging.info(f"Preprocessing complete. Got {len(preprocessed_comments)} processed comments")
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