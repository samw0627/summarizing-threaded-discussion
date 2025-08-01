#!/usr/bin/env python3
"""
Simple test script for keyword/entity extraction functionality.
Tests the TF-IDF-based entity extraction on Reddit comment data.
"""

import pandas as pd
from Scripts.entity_extractor import EntityExtractor
import json

def test_keyword_extraction():
    """Test keyword extraction on sample Reddit data."""
    
    print("=== Testing Keyword Extraction ===\n")
    
    # Load sample data
    print("1. Loading sample data...")
    df = pd.read_csv('reddit_comments_nested.csv')
    print(f"   Loaded {len(df)} rows from CSV")
    
    # Extract comments from nested format
    print("2. Extracting comments from nested CSV format...")
    comments = []
    for _, row in df.iterrows():
        if pd.notna(row['comment_text']) and str(row['comment_text']).strip():
            comments.append(str(row['comment_text']))
    
    print(f"   Found {len(comments)} valid comments")
    print(f"   Sample comment: {comments[0][:100]}...")
    
    # Initialize entity extractor with LLM method
    print("\n3. Initializing entity extractor with LLM method...")
    extractor = EntityExtractor(
        method="llm",      # Use LLM instead of TF-IDF
        max_features=200,  # Smaller for testing (TF-IDF fallback)
        min_df=1,          # Allow single occurrences (TF-IDF fallback)
        max_df=0.9,        # Less restrictive (TF-IDF fallback)
        ngram_range=(1, 3) # Unigrams, bigrams, trigrams (TF-IDF fallback)
    )
    
    # Extract entities
    print("4. Extracting key entities using LLM analysis...")
    entity_scores = extractor.extract_entities_from_comments(comments)
    print(f"   Extracted {len(entity_scores)} total entities")
    
    # Calculate frequencies and statement targets
    print("5. Calculating entity frequencies and statement targets...")
    entity_frequencies = extractor.calculate_entity_frequencies(entity_scores, top_n=30)
    statement_targets = extractor.get_statement_targets(entity_frequencies, total_statements=20)
    
    # Display results
    print("\n=== RESULTS ===")
    print(f"\nTop 15 Keywords/Entities:")
    print("-" * 60)
    print(f"{'Rank':<4} {'Entity':<25} {'TF-IDF':<8} {'Freq':<6} {'Statements'}")
    print("-" * 60)
    
    for i, (entity, score) in enumerate(list(entity_scores.items())[:15]):
        frequency = entity_frequencies.get(entity, 0)
        target_statements = statement_targets.get(entity, 0)
        print(f"{i+1:<4} {entity:<25} {score:<8.3f} {frequency:<6.3f} {target_statements}")
    
    # Show keyword categories
    print(f"\n=== Keyword Analysis ===")
    
    # Separate different types of entities
    ai_related = [e for e in entity_scores.keys() if any(term in e.lower() for term in ['ai', 'artificial', 'machine', 'algorithm', 'chatgpt', 'gpt'])]
    education_related = [e for e in entity_scores.keys() if any(term in e.lower() for term in ['education', 'student', 'teach', 'learn', 'school', 'university', 'class'])]
    thinking_related = [e for e in entity_scores.keys() if any(term in e.lower() for term in ['think', 'critical', 'cognitive', 'reasoning', 'analysis', 'problem'])]
    
    print(f"AI-related terms ({len(ai_related)}): {ai_related[:8]}")
    print(f"Education-related terms ({len(education_related)}): {education_related[:8]}")
    print(f"Thinking-related terms ({len(thinking_related)}): {thinking_related[:8]}")
    
    # Calculate topic distribution
    print(f"\n=== Topic Distribution for Statement Generation ===")
    total_ai_score = sum(entity_scores.get(e, 0) for e in ai_related)
    total_edu_score = sum(entity_scores.get(e, 0) for e in education_related)
    total_think_score = sum(entity_scores.get(e, 0) for e in thinking_related)
    total_scores = total_ai_score + total_edu_score + total_think_score
    
    if total_scores > 0:
        ai_percent = (total_ai_score / total_scores) * 100
        edu_percent = (total_edu_score / total_scores) * 100
        think_percent = (total_think_score / total_scores) * 100
        
        print(f"AI topics: {ai_percent:.1f}% of discussion")
        print(f"Education topics: {edu_percent:.1f}% of discussion") 
        print(f"Critical thinking topics: {think_percent:.1f}% of discussion")
        
        # Suggest statement allocation
        total_statements = 20
        ai_statements = int((ai_percent / 100) * total_statements)
        edu_statements = int((edu_percent / 100) * total_statements)
        think_statements = int((think_percent / 100) * total_statements)
        
        print(f"\nSuggested statement allocation (out of {total_statements}):")
        print(f"  AI-focused statements: {ai_statements}")
        print(f"  Education-focused statements: {edu_statements}")
        print(f"  Critical thinking statements: {think_statements}")
        print(f"  Other/mixed statements: {total_statements - ai_statements - edu_statements - think_statements}")
    
    # Save results for inspection
    results = {
        'total_entities': len(entity_scores),
        'total_comments': len(comments),
        'top_20_entities': dict(list(entity_scores.items())[:20]),
        'entity_frequencies': entity_frequencies,
        'statement_targets': statement_targets,
        'topic_analysis': {
            'ai_related': ai_related[:10],
            'education_related': education_related[:10],
            'thinking_related': thinking_related[:10]
        }
    }
    
    with open('keyword_extraction_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Test Complete ===")
    print(f"Detailed results saved to: keyword_extraction_test_results.json")
    
    return extractor, entity_scores, entity_frequencies, statement_targets

if __name__ == "__main__":
    test_keyword_extraction()