#!/usr/bin/env python3
"""
Entity-based Comment Filter

Filters Reddit comments based on the top 20 entities from keyword extraction results.
Preserves comments that match any of the top 20 entities.

Usage:
python entity_comment_filter.py
"""

import pandas as pd
import json
import re
import sys
from typing import List, Set


def load_top_20_entities() -> List[str]:
    """Load the top 20 entities from keyword extraction results."""
    try:
        with open('Scripts/keyword_extraction_test_results.json', 'r') as f:
            data = json.load(f)
        return list(data['top_20_entities'].keys())
    except Exception as e:
        print(f"Error loading entities: {e}", file=sys.stderr)
        sys.exit(1)


def tfidf_match_entities(text: str, entities: List[str], threshold: float = 0.1) -> List[str]:
    """Use TF-IDF vectorization to find similar content."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    if not text or pd.isna(text):
        return []
    
    # Combine text and entities for vectorization
    documents = [text] + entities
    
    try:
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),  # Use 1-3 grams for better phrase matching
            stop_words='english',
            lowercase=True,
            max_features=10000,
            min_df=1,  # Include terms that appear at least once
            token_pattern=r'\b\w+\b'  # Word boundaries
        )
        
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        # Calculate similarities between text and each entity
        text_vector = tfidf_matrix[0:1]
        entity_vectors = tfidf_matrix[1:]
        
        similarities = cosine_similarity(text_vector, entity_vectors)[0]
        
        matched_entities = []
        for i, similarity in enumerate(similarities):
            if similarity >= threshold:
                matched_entities.append(entities[i])
        
        return matched_entities
        
    except Exception as e:
        # Fallback to simple string matching if TF-IDF fails
        print(f"TF-IDF matching failed: {e}, falling back to simple matching")
        text_lower = text.lower()
        matched_entities = []
        
        for entity in entities:
            entity_words = entity.lower().split()
            # Check if any significant word from entity appears in text
            for word in entity_words:
                if len(word) > 3 and word in text_lower:
                    matched_entities.append(entity)
                    break
        
        return matched_entities


def filter_comments_by_entities(threshold: float = 0.1):
    """Main function to filter comments based on top 20 entities using TF-IDF."""
    print("Loading top 20 entities...")
    entities = load_top_20_entities()
    print(f"Loaded {len(entities)} entities")
    
    print("Loading comments from Scripts/reddit_comments_nested.csv...")
    try:
        df = pd.read_csv('Scripts/reddit_comments_nested.csv')
        print(f"Loaded {len(df)} comments")
    except Exception as e:
        print(f"Error loading comments: {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Filtering comments using TF-IDF similarity (threshold: {threshold})...")
    filtered_comments = []
    matched_count = 0
    
    for idx, row in df.iterrows():
        comment_text = row['comment_text']
        matched_entities = tfidf_match_entities(comment_text, entities, threshold)
        
        if matched_entities:
            # Add matched entities info to the row
            row_dict = row.to_dict()
            row_dict['matched_entities'] = ', '.join(matched_entities)
            row_dict['entity_count'] = len(matched_entities)
            row_dict['similarity_threshold'] = threshold
            filtered_comments.append(row_dict)
            matched_count += 1
    
    if not filtered_comments:
        print("No comments matched any of the top 20 entities")
        print(f"Try lowering the threshold (current: {threshold})")
        return
    
    # Create filtered DataFrame
    filtered_df = pd.DataFrame(filtered_comments)
    
    # Save filtered comments
    output_file = 'Scripts/filtered_comments_entities_tfidf.csv'
    filtered_df.to_csv(output_file, index=False)
    
    print(f"\nFiltering completed:")
    print(f"Original comments: {len(df)}")
    print(f"Filtered comments: {len(filtered_df)}")
    print(f"Retention rate: {len(filtered_df)/len(df)*100:.1f}%")
    print(f"Filtered comments saved to {output_file}")
    
    # Show entity distribution
    entity_counts = {}
    for comment in filtered_comments:
        for entity in comment['matched_entities'].split(', '):
            entity_counts[entity] = entity_counts.get(entity, 0) + 1
    
    print(f"\nTop entities found in filtered comments:")
    for entity, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {entity}: {count} comments")
    
    # Show some similarity scores for debugging
    if filtered_comments:
        print(f"\nSample similarity analysis:")
        sample_comment = filtered_comments[0]['comment_text']
        sample_entities = tfidf_match_entities(sample_comment, entities, 0.0)  # Get all scores
        print(f"Comment: {sample_comment[:100]}...")
        
        # Get actual similarity scores for debugging
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        try:
            documents = [sample_comment] + entities
            vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words='english', lowercase=True)
            tfidf_matrix = vectorizer.fit_transform(documents)
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
            
            print("Top similarity scores:")
            entity_scores = list(zip(entities, similarities))
            entity_scores.sort(key=lambda x: x[1], reverse=True)
            for entity, score in entity_scores[:5]:
                print(f"  {entity}: {score:.3f}")
        except:
            pass


if __name__ == "__main__":
    # Use a reasonable default threshold
    # 0.05 = more inclusive (55% retention)
    # 0.1 = balanced (23% retention)  
    # 0.15 = more selective (11% retention)
    filter_comments_by_entities(threshold=0.05)