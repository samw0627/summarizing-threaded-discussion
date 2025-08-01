#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Complete Statement Generation Pipeline for Reddit Discussion Analysis

This script orchestrates the entire pipeline in the correct order:
1. Scraper: Extract Reddit comments (Scripts/reddit_nested_comments_scraper.py)
2. Entity Extractor: Extract key entities (Scripts/entity_extractor.py)
3. Flatten: Restructure discussion space (Scripts/flatten-discussin-space.py)
4. Filter: Filter entity-relevant comments (Scripts/entity_comment_filter.py)
5. Cluster: Group comments by depth ≤ 2 (Scripts/cluster_comment.py)
6. Generate Themes: Use LLM to generate themes for each cluster
7. Generate Statements: Create thesis statements with excerpts

Final output format:
{
  "statements": [
    {
      "statement": "Your thesis statement here",
      "excerpts": ["excerpt 1", "excerpt 2", ...]
    }
  ],
  "comments": [
    {
      "comment_id": "id",  
      "comment": "original comment text"
    }
  ]
}
"""

import os
import sys
import json
import pandas as pd
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from openai import OpenAI

# Add Scripts directory to path for imports
scripts_dir = Path(__file__).parent / "Scripts"
sys.path.insert(0, str(scripts_dir))

# Import pipeline components
try:
    from entity_extractor import EntityExtractor
    from cluster_comment import RedditCommentClusterer, load_and_filter_comments
except ImportError as e:
    print(f"[ERROR] Failed to import pipeline components: {e}")
    print(f"[ERROR] Make sure all Scripts files are present")
    sys.exit(1)


class StatementGenerationPipeline:
    """
    Complete pipeline for generating statements from Reddit discussions.
    """
    
    def __init__(self, openai_api_key: str, working_dir: str = "."):
        """
        Initialize the pipeline.
        
        Args:
            openai_api_key: OpenAI API key for LLM operations
            working_dir: Working directory for intermediate files
        """
        self.client = OpenAI(api_key=openai_api_key)
        self.working_dir = Path(working_dir)
        self.scripts_dir = self.working_dir / "Scripts"
        
        # Pipeline file paths
        self.files = {
            'original_csv': self.scripts_dir / "reddit_comments_nested.csv",
            'entities_json': self.scripts_dir / "keyword_extraction_test_results.json",
            'flattened_json': self.working_dir / "promoted_threads.json",
            'flattened_csv': self.working_dir / "promoted_threads_restructured.csv",
            'filtered_csv': self.working_dir / "filtered_comments.csv",
            'clustered_csv': self.working_dir / "clustered_comments_depth2.csv",
            'final_output': self.working_dir / "statement_generation_results.json"
        }
        
    def run_script(self, script_name: str, args: List[str] = None) -> bool:
        """
        Run a pipeline script and return success status.
        
        Args:
            script_name: Name of the script to run
            args: Additional command line arguments
            
        Returns:
            True if script ran successfully, False otherwise
        """
        if args is None:
            args = []
            
        script_path = self.scripts_dir / script_name
        if not script_path.exists():
            script_path = self.working_dir / script_name
            
        if not script_path.exists():
            print(f"[ERROR] Script not found: {script_name}")
            return False
            
        try:
            cmd = [sys.executable, str(script_path)] + args
            print(f"[INFO] Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.working_dir)
            
            if result.returncode == 0:
                print(f"[SUCCESS] {script_name} completed successfully")
                if result.stdout:
                    print(result.stdout)
                return True
            else:
                print(f"[ERROR] {script_name} failed with return code {result.returncode}")
                if result.stderr:
                    print(f"STDERR: {result.stderr}")
                if result.stdout:
                    print(f"STDOUT: {result.stdout}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Failed to run {script_name}: {e}")
            return False
            
    def step_1_scraper(self, reddit_url: str) -> bool:
        """
        Step 1: Run the Reddit comment scraper.
        
        Args:
            reddit_url: URL of the Reddit discussion to scrape
            
        Returns:
            True if successful
        """
        print("\n" + "="*60)
        print("STEP 1: SCRAPING REDDIT COMMENTS")
        print("="*60)
        
        output_path = str(self.files['original_csv'])
        return self.run_script("reddit_nested_comments_scraper.py", [reddit_url, "-o", output_path])
        
    def step_2_extract_entities(self) -> bool:
        """
        Step 2: Extract entities from comments.
        
        Returns:
            True if successful
        """
        print("\n" + "="*60)
        print("STEP 2: EXTRACTING ENTITIES")
        print("="*60)
        
        input_csv = str(self.files['original_csv'])
        output_json = str(self.files['entities_json'])
        return self.run_script("entity_extractor.py", [input_csv, "-o", output_json])
        
    def step_3_flatten_space(self) -> bool:
        """
        Step 3: Flatten discussion space by promoting entity-relevant subtrees.
        
        Returns:
            True if successful
        """
        print("\n" + "="*60)
        print("STEP 3: FLATTENING DISCUSSION SPACE")
        print("="*60)
        
        return self.run_script("flatten-discussin-space.py")
        
    def step_4_filter_comments(self) -> bool:
        """
        Step 4: Filter comments based on entity relevance.
        
        Returns:
            True if successful
        """
        print("\n" + "="*60)
        print("STEP 4: FILTERING ENTITY-RELEVANT COMMENTS")
        print("="*60)
        
        input_csv = str(self.files['flattened_csv'])
        entities_json = str(self.files['entities_json'])
        output_csv = str(self.files['filtered_csv'])
        return self.run_script("entity_comment_filter.py", [input_csv, entities_json, "-o", output_csv])
        
    def step_4_cluster_comments(self) -> bool:
        """
        Step 4: Cluster comments with depth ≤ 2.
        
        Returns:
            True if successful
        """
        print("\n" + "="*60)
        print("STEP 4: CLUSTERING COMMENTS (DEPTH ≤ 2)")
        print("="*60)
        
        # Use the flattened CSV as input for clustering (skip filtering step)
        input_csv = str(self.files['flattened_csv'])
        output_csv = str(self.files['clustered_csv'])
        
        # If flattened CSV doesn't exist, use original CSV
        if not Path(input_csv).exists():
            input_csv = str(self.files['original_csv'])
            
        return self.run_script("cluster_comment.py", ["--input", input_csv, "--output", output_csv])
        
    def generate_theme(self, summary_text: str) -> str:
        """
        Generate a theme for a cluster using LLM.
        
        Args:
            summary_text: Summary text of the cluster
            
        Returns:
            Generated theme
        """
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Generate a concise theme (3-5 words) that captures the main topic of this cluster of comments. The theme should be descriptive and specific to the content."},
                    {"role": "user", "content": f"Summary: {summary_text}"}
                ]
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[WARNING] Failed to generate theme: {e}")
            return "General Discussion"
            
    def generate_statements_and_excerpts(self, cluster_data: Dict[str, Any], original_comments: List[Dict]) -> Dict[str, Any]:
        """
        Generate statements and excerpts for a cluster using LLM.
        
        Args:
            cluster_data: Data about the cluster
            original_comments: Original comment data
            
        Returns:
            Generated statements and excerpts in the required format
        """
        # Build context from cluster comments
        cluster_comments = [c for c in original_comments if c.get('cluster_id') == cluster_data['cluster_id']]
        
        if not cluster_comments:
            return {"statements": [], "comments": []}
            
        # Calculate reply counts for each comment
        reply_counts = {}
        for comment in cluster_comments:
            parent_id = comment.get('parent_id')
            if parent_id:
                reply_counts[parent_id] = reply_counts.get(parent_id, 0) + 1
        
        # Sort comments by number of replies (descending), then by score (descending)
        def get_reply_count(comment):
            comment_id = comment.get('comment_id', '')
            return reply_counts.get(comment_id, 0)
        
        def get_score(comment):
            score = comment.get('score', 0)
            return score if score is not None else 0
            
        sorted_comments = sorted(cluster_comments, 
                               key=lambda c: (get_reply_count(c), get_score(c)), 
                               reverse=True)
        
        # Format comments for LLM
        formatted_comments = ""
        comment_refs = []
        
        for comment in sorted_comments:
            comment_id = comment.get('comment_id', 'unknown')
            comment_text = comment.get('comment_text', '')
            reply_count = reply_counts.get(comment_id, 0)
            score = get_score(comment)
            
            formatted_comments += f"Comment ID: {comment_id} (Replies: {reply_count}, Score: {score})\nComment: {comment_text}\n\n"
            comment_refs.append({
                "comment_id": comment_id,
                "comment": comment_text
            })
            
        # Generate cluster summary for context
        cluster_summary = f"Cluster Theme: {cluster_data.get('theme', 'General Discussion')}\n"
        cluster_summary += f"Topic Keywords: {', '.join(cluster_data.get('topic_keywords', []))}\n"
        cluster_summary += "This cluster contains comments related to the above theme and keywords."
        
        system_prompt = '''You create thesis statements for comments in a discussion. These statements should summarise the main claim of the discussion. When you create these statements, you should also provide excerpts from comments that support that statement.

CRITICAL: Your excerpts must provide COMPREHENSIVE COVERAGE of all relevant comments in this cluster. Each statement should synthesize multiple viewpoints and include excerpts from as many relevant comments as possible to ensure complete representation of the cluster's content.

Requirements for Statements:
- Statements should be less than 20 words.
- It should not be a question.
- Make statements emotionally provocative and reaction-worthy
- Each statement should have one argument.
- A statement should not be fact that people can't disagree
- Avoid redundancy while ensuring comprehensive coverage
- All statements should be relevant to the discussion topic. You can check this by reading the statement itself and determining whether it is relevant to the topic, without considering other context.
- Do NOT use em dashes (—) anywhere in the statements
- Use periods, commas, or other punctuation instead of em dashes

Requirements for Excerpts:
- MAXIMIZE COVERAGE: Each statement should include excerpts from multiple comments (aim for 4-8 excerpts per statement when possible)
- Include excerpts that show different perspectives, supporting arguments, examples, and counterpoints related to the statement
- Each excerpt should be a meaningful quote that directly supports or relates to the statement
- Prioritize excerpts that contain the most substantive content
- Ensure that ALL significant comments in the cluster are represented through excerpts across your statements
- If a comment contains multiple important points, you can create multiple excerpts from it for different statements
- Excerpts should be direct quotes from the original comments, not paraphrased

Coverage Goals:
- Aim to use excerpts from at least 80% of the comments provided
- If there are 10 comments, try to include excerpts from 8+ of them
- If there are 20 comments, try to include excerpts from 16+ of them
- Distribute excerpts across statements to ensure comprehensive representation

For some very short comments (only a few words), it is fine to skip them if they don't add meaningful content.

Output format should be a clean JSON object (no markdown formatting) with this structure:
{
  "statements": [
    {
      "statement": "Your thesis statement here",
      "excerpts": ["excerpt 1", "excerpt 2", "excerpt 3", "excerpt 4", "excerpt 5", ...]
    }
  ],
  "comments": [
    {
      "comment_id": "id",  
      "comment": "original comment text"
    }
  ]
}

Use the original comments provided below for excerpts. Only output the JSON object, no markdown formatting or extra text.'''
        
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Cluster summary for analysis: {cluster_summary}\n\nOriginal comments to use for excerpts:\n{formatted_comments}"}
                ]
            )
            
            # Parse the JSON response
            response_text = resp.choices[0].message.content.strip()
            
            # Remove markdown formatting if present
            import re
            response_text = re.sub(r'^```json\s*', '', response_text)
            response_text = re.sub(r'\s*```$', '', response_text)
            
            try:
                result = json.loads(response_text)
                # Ensure comments field includes all cluster comments
                result["comments"] = comment_refs
                return result
            except json.JSONDecodeError:
                print(f"[WARNING] Failed to parse LLM response for cluster {cluster_data['cluster_id']}")
                return {
                    "statements": [],
                    "comments": comment_refs
                }
                
        except Exception as e:
            print(f"[WARNING] Failed to generate statements for cluster {cluster_data['cluster_id']}: {e}")
            return {
                "statements": [],
                "comments": comment_refs
            }
            
    def step_5_generate_themes_and_statements(self) -> bool:
        """
        Step 5: Generate themes and statements from clustered comments.
        
        Returns:
            True if successful
        """
        print("\n" + "="*60)
        print("STEP 5: GENERATING THEMES AND STATEMENTS")
        print("="*60)
        
        try:
            # Load clustered data
            clustered_csv = self.files['clustered_csv']
            if not clustered_csv.exists():
                print(f"[ERROR] Clustered CSV not found: {clustered_csv}")
                return False
                
            df = pd.read_csv(clustered_csv)
            print(f"[INFO] Loaded {len(df)} clustered comments")
            
            # Group by cluster
            clusters = df.groupby('cluster_id')
            results = {
                "statements": [],
                "comments": []
            }
            
            all_comments = []
            
            for cluster_id, cluster_df in clusters:
                print(f"[INFO] Processing cluster {cluster_id} with {len(cluster_df)} comments")
                
                # Generate theme for cluster
                cluster_keywords = cluster_df['topic_keywords'].iloc[0] if 'topic_keywords' in cluster_df.columns else ""
                sample_comments = cluster_df['comment_text'].head(3).tolist()
                cluster_summary = f"Keywords: {cluster_keywords}\nSample comments: {' | '.join(sample_comments)}"
                
                theme = self.generate_theme(cluster_summary)
                
                # Prepare cluster data
                cluster_data = {
                    'cluster_id': cluster_id,
                    'theme': theme,
                    'topic_keywords': cluster_keywords.split(', ') if cluster_keywords else []
                }
                
                # Convert cluster DataFrame to list of dictionaries
                cluster_comments = []
                for _, row in cluster_df.iterrows():
                    comment_dict = row.to_dict()
                    comment_dict['cluster_id'] = cluster_id
                    cluster_comments.append(comment_dict)
                    all_comments.append(comment_dict)
                
                # Generate statements and excerpts
                cluster_results = self.generate_statements_and_excerpts(cluster_data, cluster_comments)
                
                # Add cluster results to overall results
                if cluster_results.get("statements"):
                    results["statements"].extend(cluster_results["statements"])
                    
                print(f"[INFO] Generated {len(cluster_results.get('statements', []))} statements for cluster {cluster_id}")
                
            # Add all comments to final results
            results["comments"] = [
                {
                    "comment_id": comment.get('comment_id', 'unknown'),
                    "comment": comment.get('comment_text', '')
                }
                for comment in all_comments
            ]
            
            # Save unfiltered results as intermediate output
            unfiltered_output = self.working_dir / "unfiltered_statements.json"
            with open(unfiltered_output, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                
            print(f"[SUCCESS] Generated {len(results['statements'])} total statements")
            print(f"[SUCCESS] Unfiltered results saved to: {unfiltered_output}")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to generate themes and statements: {e}")
            return False
            
    def filter_statements_with_llm(self, statements: List[Dict], entities: List[str]) -> List[Dict]:
        """
        Use LLM to filter statements based on relevance to key entities.
        
        Args:
            statements: List of statement objects with "statement" and "excerpts" keys
            entities: List of key entities for the discussion
            
        Returns:
            Filtered list of relevant statement objects
        """
        if not statements:
            return []
            
        try:
            # Prepare entities list for the prompt
            entities_text = "\n".join([f"- {entity}" for entity in entities[:20]])  # Limit to top 20
            
            # Prepare statements for evaluation (batch process for efficiency)
            statements_text = ""
            for i, stmt_obj in enumerate(statements):
                statement = stmt_obj.get("statement", "")
                excerpts = stmt_obj.get("excerpts", [])
                excerpts_text = " | ".join(excerpts[:6])  # Show more excerpts for better evaluation
                statements_text += f"{i+1}. Statement: {statement}\n   Excerpts: {excerpts_text}\n\n"
                
            system_prompt = """You are tasked with filtering thesis statements to keep only those that are relevant to the main discussion topic. 

You will be given:
1. A list of key entities/topics from the discussion
2. A list of thesis statements with their supporting excerpts

Your task:
- Keep statements that are directly relevant to the key entities/topics
- Remove statements that are off-topic, irrelevant, or about unrelated matters
- Focus on whether the EXCERPTS (not just the statements) contain content related to the key entities
- Be strict but fair - keep statements that genuinely contribute to the discussion

Key entities/topics for this discussion:
{entities}

Guidelines:
- If excerpts mention or discuss any of the key entities above, likely keep the statement
- If excerpts are completely unrelated to these topics (e.g., jokes, random comments, completely different subjects), remove the statement
- Consider the semantic meaning, not just keyword matching
- Edge cases should be kept rather than removed if there's any reasonable connection

Output format: Return only the numbers of statements to KEEP, separated by commas (e.g., "1,3,5,7,12").
If no statements should be kept, return "NONE".
If all statements should be kept, return "ALL"."""
            
            user_prompt = f"Please evaluate these statements and return the numbers of those that should be KEPT:\n\n{statements_text}"
            
            # Make LLM call
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt.format(entities=entities_text)},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1  # Low temperature for consistent filtering
            )
            
            result_text = response.choices[0].message.content.strip()
            print(f"[DEBUG] LLM filtering result: {result_text}")
            
            # Parse the result
            if result_text.upper() == "NONE":
                return []
            elif result_text.upper() == "ALL":
                return statements
            else:
                # Parse comma-separated numbers
                try:
                    keep_indices = [int(x.strip()) - 1 for x in result_text.split(",") if x.strip().isdigit()]
                    # Filter statements based on indices
                    filtered = [statements[i] for i in keep_indices if 0 <= i < len(statements)]
                    return filtered
                except (ValueError, IndexError) as e:
                    print(f"[WARNING] Failed to parse LLM filtering result: {e}")
                    print(f"[WARNING] Raw result: {result_text}")
                    # Fallback: keep all statements if parsing fails
                    return statements
                    
        except Exception as e:
            print(f"[ERROR] LLM filtering failed: {e}")
            # Fallback: return all statements if LLM filtering fails
            return statements
            
    def step_6_filter_statements(self) -> bool:
        """
        Step 6: Filter out irrelevant statements using entity relevance.
        
        Returns:
            True if successful
        """
        print("\n" + "="*60)
        print("STEP 6: FILTERING IRRELEVANT STATEMENTS")
        print("="*60)
        
        try:
            # Load unfiltered statements
            unfiltered_output = self.working_dir / "unfiltered_statements.json"
            if not unfiltered_output.exists():
                print(f"[ERROR] Unfiltered statements not found: {unfiltered_output}")
                return False
                
            with open(unfiltered_output, 'r', encoding='utf-8') as f:
                unfiltered_data = json.load(f)
                
            # Load entities for relevance checking
            entities_json = self.files['entities_json']
            if not entities_json.exists():
                print(f"[WARNING] Entities file not found: {entities_json}")
                # If no entities file, copy unfiltered to final output
                with open(self.files['final_output'], 'w', encoding='utf-8') as f:
                    json.dump(unfiltered_data, f, ensure_ascii=False, indent=2)
                return True
                
            with open(entities_json, 'r', encoding='utf-8') as f:
                entities_data = json.load(f)
                
            # Extract top entities from the correct structure
            top_entities = set()
            if isinstance(entities_data, dict):
                if "overall_entities" in entities_data and "top_entities" in entities_data["overall_entities"]:
                    top_entities = set(entities_data["overall_entities"]["top_entities"].keys())
                elif "top_20_entities" in entities_data:
                    top_entities = set(entities_data["top_20_entities"].keys())
                elif "top_entities" in entities_data:
                    top_entities = set(entities_data["top_entities"])
                    
            if not top_entities:
                print("[WARNING] No entities found for filtering, keeping all statements")
                with open(self.files['final_output'], 'w', encoding='utf-8') as f:
                    json.dump(unfiltered_data, f, ensure_ascii=False, indent=2)
                return True
                
            print(f"[INFO] Using LLM to filter statements based on {len(top_entities)} key entities")
            print(f"[INFO] Key entities: {list(top_entities)[:10]}...")
            
            # Use LLM to filter statements
            filtered_statements = self.filter_statements_with_llm(
                unfiltered_data.get("statements", []), 
                list(top_entities)
            )
            original_count = len(unfiltered_data.get("statements", []))
                    
            # Prepare final results
            filtered_data = {
                "statements": filtered_statements,
                "comments": unfiltered_data.get("comments", [])
            }
            
            # Save filtered results
            with open(self.files['final_output'], 'w', encoding='utf-8') as f:
                json.dump(filtered_data, f, ensure_ascii=False, indent=2)
                
            filtered_count = len(filtered_statements)
            removed_count = original_count - filtered_count
            
            print(f"[SUCCESS] Filtered {removed_count}/{original_count} statements ({removed_count/original_count*100:.1f}% removed)")
            print(f"[SUCCESS] Kept {filtered_count} relevant statements")
            print(f"[SUCCESS] Final results saved to: {self.files['final_output']}")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to filter statements: {e}")
            return False
            
    def run_full_pipeline(self, reddit_url: str, skip_scraping: bool = False) -> bool:
        """
        Run the complete pipeline.
        
        Args:
            reddit_url: URL of Reddit discussion to analyze
            skip_scraping: Skip scraping if data already exists
            
        Returns:
            True if pipeline completed successfully
        """
        print("STARTING STATEMENT GENERATION PIPELINE")
        print("="*60)
        
        success = True
        
        # Step 1: Scraper (optional if data exists)
        if not skip_scraping or not self.files['original_csv'].exists():
            success &= self.step_1_scraper(reddit_url)
        else:
            print(f"[SKIP] Using existing scraped data: {self.files['original_csv']}")
            
        # Step 2: Extract entities
        if success:
            success &= self.step_2_extract_entities()
            
        # Step 3: Flatten discussion space
        if success:
            success &= self.step_3_flatten_space()
            
        # Step 4: Cluster comments (skip comment filtering, work with all data)
        if success:
            success &= self.step_4_cluster_comments()
            
        # Step 5: Generate themes and statements
        if success:
            success &= self.step_5_generate_themes_and_statements()
            
        # Step 6: Filter irrelevant statements
        if success:
            success &= self.step_6_filter_statements()
            
        if success:
            print("\n" + "="*60)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"Final results saved to: {self.files['final_output']}")
        else:
            print("\n" + "="*60)
            print("PIPELINE FAILED!")
            print("="*60)
            
        return success


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Complete Statement Generation Pipeline")
    parser.add_argument("reddit_url", help="URL of Reddit discussion to analyze")
    parser.add_argument("--api-key", required=True, help="OpenAI API key")
    parser.add_argument("--working-dir", default=".", help="Working directory for files")
    parser.add_argument("--skip-scraping", action="store_true", help="Skip scraping if data exists")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = StatementGenerationPipeline(
        openai_api_key=args.api_key,
        working_dir=args.working_dir
    )
    
    # Run pipeline
    success = pipeline.run_full_pipeline(
        reddit_url=args.reddit_url,
        skip_scraping=args.skip_scraping
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()