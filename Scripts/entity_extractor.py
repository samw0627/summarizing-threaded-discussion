#!/usr/bin/env python3
"""
Entity Extractor

This module extracts key entities and topics from Reddit discussions using TF-IDF.
These entities are used to ground statement generation, ensuring more common topics
receive proportionally more statements.
"""

import pandas as pd
import numpy as np
import re
import json
from typing import List, Dict, Tuple, Optional, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter, defaultdict
import logging
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EntityExtractor:
    def __init__(self, method: str = "llm", openai_client: Optional[OpenAI] = None,
                 max_features: int = 500, min_df: int = 2, max_df: float = 0.8, 
                 ngram_range: Tuple[int, int] = (1, 3)):
        """
        Initialize the entity extractor.
        
        Args:
            method: Extraction method - "llm" or "tfidf"
            openai_client: OpenAI client instance (required if method="llm")
            max_features: Maximum number of features to extract (for TF-IDF)
            min_df: Minimum document frequency for terms (for TF-IDF)
            max_df: Maximum document frequency for terms (for TF-IDF)
            ngram_range: Range of n-grams to consider (for TF-IDF)
        """
        self.method = method
        self.client = openai_client
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.vectorizer = None
        self.entity_scores = {}
        self.entity_frequencies = {}
        
        if method == "llm" and openai_client is None:
            # Try to create client with API key from environment or hardcoded
            try:
                api_key = "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"
                self.client = OpenAI(api_key=api_key)
                logging.info("Initialized OpenAI client for LLM-based entity extraction")
            except Exception as e:
                logging.error(f"Failed to initialize OpenAI client: {e}")
                raise ValueError("OpenAI client required for LLM-based extraction")
        
    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess text for entity extraction.
        
        Args:
            text: Raw comment text
            
        Returns:
            Cleaned text
        """
        if text is None or pd.isna(text) or text == '':
            return ''
        
        try:
            # Convert to lowercase
            text = str(text).lower()
            
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
            
        except Exception as e:
            logging.error(f"Error in preprocess_text: {e}")
            return ''
    
    def extract_entities_llm(self, comments: List[str]) -> Dict[str, float]:
        """
        Extract key entities from comments using LLM analysis.
        
        Args:
            comments: List of comment texts
            
        Returns:
            Dictionary mapping entities to their importance scores
        """
        logging.info(f"Extracting entities from {len(comments)} comments using LLM")
        
        # Combine comments into a single text for analysis
        combined_text = "\n\n".join(comments[:50])  # Limit to avoid token limits
        
        system_prompt = """You are an expert at analyzing discussion content and identifying key topics, themes, and entities. 
        
Your task is to analyze the provided discussion comments and extract the most important keywords, phrases, and topics that represent the main themes being discussed.

Requirements:
- Identify 20-30 key entities (words, phrases, topics, concepts)
- Focus on substantive content words, not function words
- Include both single words and meaningful phrases (2-3 words max)
- Consider the importance and frequency of concepts in the discussion
- Assign each entity an importance score from 1-10 based on how central it is to the discussion

Output your analysis as a JSON object with this exact structure:
{
  "entities": [
    {
      "entity": "entity name or phrase",
      "score": 8.5,
      "explanation": "brief explanation of why this is important"
    }
  ]
}

Only output the JSON object, no additional text or markdown formatting."""

        user_prompt = f"Analyze the following discussion comments and extract key entities:\n\n{combined_text}"
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Remove markdown formatting if present
            response_text = re.sub(r'^```json\s*', '', response_text)
            response_text = re.sub(r'\s*```$', '', response_text)
            
            # Parse JSON response
            entity_data = json.loads(response_text)
            
            # Convert to entity scores dictionary
            entity_scores = {}
            for item in entity_data.get('entities', []):
                entity = item.get('entity', '').lower().strip()
                score = float(item.get('score', 0))
                if entity and score > 0:
                    entity_scores[entity] = score
            
            # Sort by score
            self.entity_scores = dict(sorted(entity_scores.items(), key=lambda x: x[1], reverse=True))
            
            logging.info(f"LLM extracted {len(self.entity_scores)} entities")
            return self.entity_scores
            
        except Exception as e:
            logging.error(f"Error in LLM entity extraction: {e}")
            # Fallback to TF-IDF if LLM fails
            logging.info("Falling back to TF-IDF extraction")
            return self.extract_entities_tfidf(comments)
    
    def extract_entities_tfidf(self, comments: List[str]) -> Dict[str, float]:
        """
        Extract key entities from a list of comments using TF-IDF.
        
        Args:
            comments: List of comment texts
            
        Returns:
            Dictionary mapping entities to their TF-IDF scores
        """
        logging.info(f"Extracting entities from {len(comments)} comments using TF-IDF")
        
        # Preprocess comments
        preprocessed_comments = [self.preprocess_text(comment) for comment in comments]
        
        # Filter out empty comments
        valid_comments = [comment for comment in preprocessed_comments if comment.strip()]
        
        if len(valid_comments) < 2:
            logging.warning("Not enough valid comments for entity extraction")
            return {}
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english',
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df
        )
        
        # Fit and transform comments
        tfidf_matrix = self.vectorizer.fit_transform(valid_comments)
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Calculate entity scores by summing TF-IDF scores across all documents
        entity_scores = {}
        for i, feature in enumerate(feature_names):
            # Sum TF-IDF scores across all documents for this feature
            score = tfidf_matrix[:, i].sum()
            entity_scores[feature] = score
        
        # Sort entities by score
        self.entity_scores = dict(sorted(entity_scores.items(), key=lambda x: x[1], reverse=True))
        
        logging.info(f"Extracted {len(self.entity_scores)} entities")
        return self.entity_scores
    
    def extract_entities_from_comments(self, comments: List[str]) -> Dict[str, float]:
        """
        Extract key entities from comments using the configured method.
        
        Args:
            comments: List of comment texts
            
        Returns:
            Dictionary mapping entities to their scores
        """
        if self.method == "llm":
            return self.extract_entities_llm(comments)
        else:
            return self.extract_entities_tfidf(comments)
    
    def extract_entities_from_clusters(self, clustered_df: pd.DataFrame) -> Dict[int, Dict[str, float]]:
        """
        Extract entities for each cluster separately.
        
        Args:
            clustered_df: DataFrame with comments and cluster assignments
            
        Returns:
            Dictionary mapping cluster IDs to their entity scores
        """
        cluster_entities = {}
        
        for cluster_id in clustered_df['cluster'].unique():
            if cluster_id < 0:  # Skip filtered out comments
                continue
                
            cluster_comments = clustered_df[clustered_df['cluster'] == cluster_id]
            
            # Extract comment texts from the cluster
            comments = []
            for _, row in cluster_comments.iterrows():
                # Try to get comment text from various possible columns
                text = row.get('comment_text') or row.get('enriched') or row.get('preprocessed_text', '')
                if text and str(text).strip():
                    comments.append(str(text))
            
            if comments:
                entities = self.extract_entities_from_comments(comments)
                cluster_entities[cluster_id] = entities
                logging.info(f"Cluster {cluster_id}: extracted {len(entities)} entities from {len(comments)} comments")
            else:
                cluster_entities[cluster_id] = {}
                logging.warning(f"Cluster {cluster_id}: no valid comments found")
        
        return cluster_entities
    
    def calculate_entity_frequencies(self, entity_scores: Dict[str, float], 
                                   top_n: int = 50) -> Dict[str, float]:
        """
        Calculate normalized frequencies for top entities.
        
        Args:
            entity_scores: Dictionary of entity TF-IDF scores
            top_n: Number of top entities to consider
            
        Returns:
            Dictionary mapping entities to normalized frequencies (0-1)
        """
        # Get top N entities
        top_entities = dict(list(entity_scores.items())[:top_n])
        
        if not top_entities:
            return {}
        
        # Normalize scores to frequencies (0-1 range)
        max_score = max(top_entities.values())
        min_score = min(top_entities.values())
        
        if max_score == min_score:
            # All entities have same score, assign equal frequency
            normalized_freq = 1.0 / len(top_entities)
            self.entity_frequencies = {entity: normalized_freq for entity in top_entities}
        else:
            # Normalize to 0-1 range
            self.entity_frequencies = {
                entity: (score - min_score) / (max_score - min_score)
                for entity, score in top_entities.items()
            }
        
        return self.entity_frequencies
    
    def get_statement_targets(self, entity_frequencies: Dict[str, float], 
                            total_statements: int = 20) -> Dict[str, int]:
        """
        Calculate target number of statements for each entity based on frequencies.
        
        Args:
            entity_frequencies: Dictionary of normalized entity frequencies
            total_statements: Total number of statements to generate
            
        Returns:
            Dictionary mapping entities to target statement counts
        """
        if not entity_frequencies:
            return {}
        
        # Calculate weighted statement allocation
        total_weight = sum(entity_frequencies.values())
        statement_targets = {}
        
        for entity, frequency in entity_frequencies.items():
            # Calculate proportional allocation
            target_count = int((frequency / total_weight) * total_statements)
            # Ensure at least 1 statement for significant entities
            if frequency > 0.1 and target_count == 0:
                target_count = 1
            statement_targets[entity] = target_count
        
        # Adjust to ensure total doesn't exceed target
        current_total = sum(statement_targets.values())
        if current_total > total_statements:
            # Reduce from highest frequency entities first
            sorted_entities = sorted(statement_targets.items(), key=lambda x: entity_frequencies[x[0]], reverse=True)
            excess = current_total - total_statements
            for entity, count in sorted_entities:
                if excess <= 0:
                    break
                reduction = min(excess, count - 1)  # Don't reduce below 1
                statement_targets[entity] = count - reduction
                excess -= reduction
        
        return statement_targets
    
    def export_entity_analysis(self, output_file: str, cluster_entities: Optional[Dict[int, Dict[str, float]]] = None):
        """
        Export entity analysis results to JSON file.
        
        Args:
            output_file: Path to output JSON file
            cluster_entities: Optional cluster-specific entity data
        """
        analysis_data = {
            'overall_entities': {
                'top_entities': dict(list(self.entity_scores.items())[:20]) if self.entity_scores else {},
                'entity_frequencies': self.entity_frequencies,
                'total_entities_extracted': len(self.entity_scores)
            }
        }
        
        if cluster_entities:
            analysis_data['cluster_entities'] = {}
            for cluster_id, entities in cluster_entities.items():
                analysis_data['cluster_entities'][str(cluster_id)] = {
                    'top_entities': dict(list(entities.items())[:10]),
                    'total_entities': len(entities)
                }
        
        with open(output_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        logging.info(f"Entity analysis exported to {output_file}")
    
    def get_top_entities(self, n: int = 20) -> List[Tuple[str, float]]:
        """
        Get top N entities by TF-IDF score.
        
        Args:
            n: Number of top entities to return
            
        Returns:
            List of (entity, score) tuples
        """
        return list(self.entity_scores.items())[:n]


def extract_entities_from_csv(csv_file: str, output_file: str = None, 
                            top_n: int = 50, total_statements: int = 20, method: str = "llm") -> EntityExtractor:
    """
    Convenience function to extract entities from a CSV file.
    
    Args:
        csv_file: Path to input CSV file
        output_file: Optional output file for analysis results
        top_n: Number of top entities to consider
        total_statements: Total statements for allocation calculation
        method: Extraction method - "llm" or "tfidf"
        
    Returns:
        EntityExtractor instance with results
    """
    # Load CSV file
    df = pd.read_csv(csv_file)
    
    # Extract comments from the CSV (handle nested format)
    comments = []
    if 'comment_text' in df.columns:
        comments = df['comment_text'].dropna().tolist()
    else:
        # Handle nested CSV format from scraper
        level_columns = [col for col in df.columns if col.startswith('level_')]
        for _, row in df.iterrows():
            for col in level_columns:
                if pd.notna(row[col]) and row[col] != '':
                    comments.append(str(row[col]))
                    break
    
    # Initialize extractor and extract entities
    extractor = EntityExtractor(method=method)
    entity_scores = extractor.extract_entities_from_comments(comments)
    entity_frequencies = extractor.calculate_entity_frequencies(entity_scores, top_n)
    statement_targets = extractor.get_statement_targets(entity_frequencies, total_statements)
    
    print(f"\nEntity Extraction Results ({method.upper()}):")
    print(f"Total entities extracted: {len(entity_scores)}")
    print(f"Top {min(10, len(entity_scores))} entities:")
    for i, (entity, score) in enumerate(list(entity_scores.items())[:10]):
        frequency = entity_frequencies.get(entity, 0)
        target_statements = statement_targets.get(entity, 0)
        print(f"  {i+1}. {entity}: {score:.3f} (freq: {frequency:.3f}, statements: {target_statements})")
    
    if output_file:
        extractor.export_entity_analysis(output_file)
    
    return extractor


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract key entities from Reddit discussions')
    parser.add_argument('input_csv', help='Input CSV file with Reddit comments')
    parser.add_argument('-o', '--output', help='Output JSON file for entity analysis')
    parser.add_argument('--top-entities', type=int, default=50, 
                       help='Number of top entities to consider (default: 50)')
    parser.add_argument('--total-statements', type=int, default=20,
                       help='Total number of statements for allocation (default: 20)')
    parser.add_argument('--method', choices=['llm', 'tfidf'], default='llm',
                       help='Extraction method: llm or tfidf (default: llm)')
    
    args = parser.parse_args()
    
    extract_entities_from_csv(
        args.input_csv, 
        args.output, 
        args.top_entities, 
        args.total_statements,
        args.method
    )