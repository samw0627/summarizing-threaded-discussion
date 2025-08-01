#!/usr/bin/env python3
"""
Cluster Summarizer with LLM Integration

This script takes a CSV file of clustered Reddit comments (output from reddit_comment_clusterer.py)
and generates AI-powered summaries for each cluster using OpenAI's API, outputting the results as a JSON file.
"""

import pandas as pd
import json
import argparse
import sys
import os
from typing import Dict, List, Any, Optional
from collections import Counter
import re
from openai import OpenAI
from dotenv import load_dotenv
import time

class ClusterSummarizer:
    def __init__(self, use_llm: bool = True, model: str = "gpt-3.5-turbo"):
        """
        Initialize the cluster summarizer.
        
        Args:
            use_llm: Whether to use LLM for generating summaries
            model: OpenAI model to use for summarization
        """
        self.use_llm = use_llm
        self.model = model
        self.client = None
        
        if use_llm:
            # Load environment variables
            load_dotenv()
            
            # Initialize OpenAI client
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                print("Warning: OPENAI_API_KEY not found in environment variables.")
                print("Please set your OpenAI API key in .env file or environment.")
                print("Falling back to rule-based summarization.")
                self.use_llm = False
            else:
                self.client = OpenAI(api_key=api_key)
                print(f"LLM summarization enabled using model: {model}")
    
    def generate_llm_summary(self, comments: List[str], topic_keywords: str, cluster_id: int) -> Dict[str, str]:
        """
        Generate LLM-powered summary for a cluster of comments.
        
        Args:
            comments: List of comment texts
            topic_keywords: Topic keywords from clustering
            cluster_id: ID of the cluster
            
        Returns:
            Dictionary containing LLM-generated summaries
        """
        if not self.use_llm or not self.client:
            return {
                "summary": "LLM summarization not available",
                "key_themes": "Not generated",
                "sentiment": "Not analyzed",
                "actionable_insights": "Not generated"
            }
        
        try:
            # Prepare comments for analysis (limit to avoid token limits)
            sample_comments = comments[:20] if len(comments) > 20 else comments
            comments_text = "\n\n".join([f"Comment {i+1}: {comment}" for i, comment in enumerate(sample_comments)])
            
            # Create the prompt
            prompt = f"""
You are analyzing a cluster of Reddit comments from a discussion thread. Below are comments that have been grouped together because they share similar topics or themes.

CLUSTER ID: {cluster_id}
TOPIC KEYWORDS: {topic_keywords}
TOTAL COMMENTS IN CLUSTER: {len(comments)}

SAMPLE COMMENTS:
{comments_text}

Please provide a comprehensive analysis in the following format:

1. SUMMARY: Write a 2-3 sentence summary of what this cluster of comments is primarily discussing.

2. KEY THEMES: Identify 3-5 main themes or topics that emerge from these comments.

3. SENTIMENT: Describe the overall sentiment/tone of the comments (positive, negative, neutral, mixed) and explain why.

4. ACTIONABLE INSIGHTS: Provide 2-3 actionable insights or recommendations based on the discussion themes.

Keep your response focused and concise, but thorough in capturing the essence of the discussion.
"""

            # Make API call with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that analyzes and summarizes online discussions with expertise in accessibility, technology, and user experience topics."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=1000,
                        temperature=0.3
                    )
                    
                    full_response = response.choices[0].message.content
                    
                    # Parse the response into sections
                    sections = {
                        "summary": "",
                        "key_themes": "",
                        "sentiment": "",
                        "actionable_insights": ""
                    }
                    
                    current_section = None
                    lines = full_response.split('\n')
                    
                    for line in lines:
                        line = line.strip()
                        if line.upper().startswith('SUMMARY'):
                            current_section = "summary"
                        elif line.upper().startswith('KEY THEMES'):
                            current_section = "key_themes"
                        elif line.upper().startswith('SENTIMENT'):
                            current_section = "sentiment"
                        elif line.upper().startswith('ACTIONABLE INSIGHTS'):
                            current_section = "actionable_insights"
                        elif line and current_section:
                            if sections[current_section]:
                                sections[current_section] += " " + line
                            else:
                                sections[current_section] = line
                    
                    # Clean up sections
                    for key in sections:
                        sections[key] = re.sub(r'^\d+\.\s*', '', sections[key].strip())
                    
                    # If parsing failed, use full response as summary
                    if not any(sections.values()):
                        sections["summary"] = full_response
                        sections["key_themes"] = "See summary"
                        sections["sentiment"] = "See summary"
                        sections["actionable_insights"] = "See summary"
                    
                    return sections
                    
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed for cluster {cluster_id}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        raise
            
        except Exception as e:
            print(f"Error generating LLM summary for cluster {cluster_id}: {e}")
            return {
                "summary": f"Error generating summary: {str(e)}",
                "key_themes": "Error in analysis",
                "sentiment": "Could not analyze",
                "actionable_insights": "Not available due to error"
            }

    def load_clustered_data(self, csv_file: str) -> pd.DataFrame:
        """
        Load the clustered comments CSV file.
        
        Args:
            csv_file: Path to the clustered comments CSV file
            
        Returns:
            DataFrame with clustered comments
        """
        try:
            df = pd.read_csv(csv_file)
            print(f"Loaded {len(df)} clustered comments from {csv_file}")
            return df
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            raise
    
    
    def generate_cluster_summary(self, cluster_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a summary for a single cluster.
        
        Args:
            cluster_data: DataFrame containing comments for a single cluster
            
        Returns:
            Dictionary containing cluster summary with LLM analysis and comments
        """
        cluster_id = cluster_data['cluster_id'].iloc[0]
        comments = cluster_data['comment_text'].tolist()
        
        # Get topic keywords (from the clustering output)
        topic_keywords = cluster_data['topic_keywords'].iloc[0] if 'topic_keywords' in cluster_data.columns else "unknown"
        
        # Generate LLM summary
        llm_analysis = self.generate_llm_summary(comments, topic_keywords, cluster_id)
        
        # Create simplified summary with only LLM analysis and comments
        summary = {
            "cluster_id": int(cluster_id),
            "llm_analysis": llm_analysis,
            "comments": comments
        }
        
        return summary
    
    
    def summarize_all_clusters(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summaries for all clusters in the dataset.
        
        Args:
            df: DataFrame with clustered comments
            
        Returns:
            Dictionary containing summaries for all clusters
        """
        # Group by cluster_id
        clusters = df.groupby('cluster_id')
        
        summaries = {
            "cluster_summaries": []
        }
        
        total_clusters = df['cluster_id'].nunique()
        print(f"Processing {total_clusters} clusters...")
        
        for cluster_id, cluster_data in clusters:
            if cluster_id >= 0:  # Skip filtered comments (-1)
                print(f"Summarizing cluster {cluster_id}...")
                cluster_summary = self.generate_cluster_summary(cluster_data)
                summaries["cluster_summaries"].append(cluster_summary)
        
        # Sort by cluster_id for consistency
        summaries["cluster_summaries"].sort(key=lambda x: x["cluster_id"])
        
        return summaries
    
    def export_to_json(self, summaries: Dict[str, Any], output_file: str):
        """
        Export summaries to JSON file.
        
        Args:
            summaries: Dictionary containing all cluster summaries
            output_file: Path to output JSON file
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summaries, f, indent=2, ensure_ascii=False)
            
            print(f"\nCluster summaries exported to: {output_file}")
            print(f"Total clusters summarized: {len(summaries['cluster_summaries'])}")
            
        except Exception as e:
            print(f"Error exporting JSON: {e}")
            raise


def main():
    """Main function to handle command line arguments and execute summarization."""
    parser = argparse.ArgumentParser(
        description='Summarize clustered Reddit comments using LLM and export to JSON'
    )
    parser.add_argument('input_csv', help='Input CSV file from reddit_comment_clusterer.py')
    parser.add_argument('-o', '--output', default='cluster_summaries.json',
                       help='Output JSON file path (default: cluster_summaries.json)')
    parser.add_argument('--no-llm', action='store_true',
                       help='Disable LLM summarization and use rule-based approach only')
    parser.add_argument('--model', default='gpt-3.5-turbo',
                       help='OpenAI model to use for summarization (default: gpt-3.5-turbo)')
    
    args = parser.parse_args()
    
    try:
        print(f"Loading clustered data from: {args.input_csv}")
        
        # Initialize summarizer
        use_llm = not args.no_llm
        summarizer = ClusterSummarizer(use_llm=use_llm, model=args.model)
        
        # Load clustered data
        df = summarizer.load_clustered_data(args.input_csv)
        
        # Generate summaries for all clusters
        summaries = summarizer.summarize_all_clusters(df)
        
        # Export to JSON
        summarizer.export_to_json(summaries, args.output)
        
        print("Summarization completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()