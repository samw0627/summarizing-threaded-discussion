#!/usr/bin/env python3
"""
Reddit Nested Comments Scraper

This script takes a Reddit discussion thread URL and outputs a CSV file
with comments preserving their nesting structure. Each level of nesting
is represented by empty columns before the comment content.
"""

import praw
import csv
import sys
import argparse
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_reddit_instance():
    """Initialize and return a Reddit instance using environment variables."""
    client_id = os.getenv('REDDIT_CLIENT_ID')
    client_secret = os.getenv('REDDIT_CLIENT_SECRET')
    user_agent = os.getenv('REDDIT_USER_AGENT', 'python:reddit-comment-scraper:v1.0.0')
    
    if not client_id or not client_secret:
        print("Error: Reddit API credentials not found in environment variables.")
        print("Please set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in your .env file.")
        sys.exit(1)
    
    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent
    )

def extract_nested_comments_to_csv(post_url: str, output_file: str, max_depth: int = None) -> List[Dict[str, Any]]:
    """
    Extract comments from a Reddit post and directly convert to CSV preserving hierarchical structure.
    
    Args:
        post_url (str): The URL of the Reddit post
        output_file (str): Output CSV file path
        max_depth (int): Maximum nesting depth (auto-calculated if None)
        
    Returns:
        List[Dict]: List of comment dictionaries with nesting information
    """
    reddit = get_reddit_instance()
    submission = reddit.submission(url=post_url)
    
    # Expand all "MoreComments" objects
    submission.comments.replace_more(limit=None)
    
    comments_data = []
    
    def traverse_comments(comment, depth=0):
        """Recursively traverse comments and their replies."""
        # Extract comment data
        author_name = comment.author.name if comment.author else '[deleted]'
        
        comment_data = {
            'depth': depth,
            'author': author_name,
            'content': comment.body,
            'score': comment.score,
            'created_utc': comment.created_utc,
            'comment_id': comment.id,
            'parent_id': comment.parent_id.split('_', 1)[1] if comment.parent_id and '_' in comment.parent_id else comment.parent_id
        }
        
        comments_data.append(comment_data)
        
        # Process replies
        for reply in comment.replies:
            traverse_comments(reply, depth + 1)
    
    # Process all top-level comments
    for comment in submission.comments:
        traverse_comments(comment)
    
    # Directly convert to CSV
    if comments_data:
        # Calculate maximum depth if not provided
        if max_depth is None:
            max_depth = max(comment['depth'] for comment in comments_data)
        
        # Create column headers with single comment_text column
        headers = ['comment_text', 'depth', 'author', 'score', 'created_utc', 'comment_id', 'parent_id']
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write headers
            writer.writerow(headers)
            
            # Write comment data
            for comment in comments_data:
                row = [
                    comment['content'],
                    comment['depth'],
                    comment['author'],
                    comment['score'],
                    comment['created_utc'],
                    comment['comment_id'],
                    comment['parent_id']
                ]
                
                writer.writerow(row)
    
    return comments_data

def main():
    """Main function to handle command line arguments and execute scraping."""
    parser = argparse.ArgumentParser(
        description='Scrape Reddit comments preserving nesting structure in CSV format'
    )
    parser.add_argument('url', help='Reddit discussion thread URL')
    parser.add_argument('-o', '--output', default='reddit_comments_nested.csv',
                       help='Output CSV file path (default: reddit_comments_nested.csv)')
    parser.add_argument('--max-depth', type=int, 
                       help='Maximum nesting depth to include (auto-calculated if not specified)')
    
    args = parser.parse_args()
    
    try:
        print(f"Scraping comments from: {args.url}")
        comments_data = extract_nested_comments_to_csv(args.url, args.output, args.max_depth)
        
        if not comments_data:
            print("No comments found in the discussion.")
            return
        
        print(f"Extracted {len(comments_data)} comments")
        
        # Calculate depth statistics
        depths = [comment['depth'] for comment in comments_data]
        max_depth = max(depths)
        print(f"Maximum nesting depth: {max_depth}")
        print(f"Comments saved to: {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()