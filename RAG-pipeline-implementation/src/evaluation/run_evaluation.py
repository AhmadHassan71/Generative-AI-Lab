#!/usr/bin/env python
# src/evaluation/run_evaluation.py

import sys
import os
import json
import argparse
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

# Add the parent directory to the path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import our modules - use correct relative imports
from src.rag_chain.chain import answer, describe_image
from src.evaluation.performance_metrics import RAGEvaluator

# Load environment variables
load_dotenv()

def load_test_set(file_path):
    """
    Load a test set from a JSON or CSV file.
    
    Args:
        file_path (str): Path to the test set file
        
    Returns:
        list: List of test items with query and reference
    """
    file_path = Path(file_path)
    if file_path.suffix.lower() == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif file_path.suffix.lower() == '.csv':
        df = pd.read_csv(file_path)
        return df.to_dict('records')
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

def save_results(results, output_path):
    """
    Save evaluation results to a file.
    
    Args:
        results (pd.DataFrame): Evaluation results
        output_path (str): Path to save the results
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix.lower() == '.json':
        results.to_json(output_path, orient='records', indent=2)
    elif output_path.suffix.lower() in ['.csv', '.txt']:
        results.to_csv(output_path, index=False)
    elif output_path.suffix.lower() == '.xlsx':
        results.to_excel(output_path, index=False)
    else:
        results.to_csv(output_path.with_suffix('.csv'), index=False)
    
    print(f"Results saved to {output_path}")

def create_sample_test_set(output_path):
    """
    Create a sample test set for evaluation.
    
    Args:
        output_path (str): Path to save the sample test set
    """
    test_data = [
        {
            "query_id": "q1",
            "query": "What is the vision of FAST-NUCES?",
            "reference": "To produce world-class professionals, who are responsible citizens and good human beings."
        },
        {
            "query_id": "q2",
            "query": "How many campuses does FAST-NUCES have?",
            "reference": "FAST-NUCES has five campuses located in Karachi, Lahore, Islamabad, Peshawar, and Chiniot-Faisalabad."
        },
        {
            "query_id": "q3",
            "query": "What are the undergraduate programs offered at FAST-NUCES?",
            "reference": "FAST-NUCES offers various undergraduate programs including Bachelor of Business Administration, Bachelor of Science in Accounting & Finance, Artificial Intelligence, Business Analytics, Civil Engineering, Computer Science, Cyber Security, Data Science, Electrical Engineering, Financial Technologies, and Software Engineering."
        },
        {
            "query_id": "q4",
            "query": "What is the mission of FAST-NUCES?",
            "reference": "The mission is to identify and attract promising students from diverse communities to shape into visionary leaders and professionals; to provide quality education regardless of financial background, ethnicity, gender, or religion; and to promote research and scholarly activities to generate knowledge."
        },
        {
            "query_id": "q5",
            "query": "When was the 76th Convocation of FAST-NUCES held?",
            "reference": "The 76th Convocation of the University was held on November 19, 2023 at the Chiniot-Faisalabad Campus."
        }
    ]
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix.lower() == '.json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2)
    else:
        df = pd.DataFrame(test_data)
        df.to_csv(output_path.with_suffix('.csv'), index=False)
    
    print(f"Sample test set created at {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG system using various performance metrics")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Create test set subcommand
    create_parser = subparsers.add_parser('create-test', help='Create a sample test set')
    create_parser.add_argument('--output', '-o', default='data/evaluation/sample_test_set.json',
                              help='Path to save the sample test set')
    
    # Evaluate subcommand
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the RAG system')
    eval_parser.add_argument('--test-set', '-t', required=True,
                           help='Path to the test set file (JSON or CSV)')
    eval_parser.add_argument('--output', '-o', default='data/evaluation/results.csv',
                           help='Path to save the evaluation results')
    eval_parser.add_argument('--model', '-m', default='all-mpnet-base-v2',
                           help='Sentence embedding model for cosine similarity')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == 'create-test':
        create_sample_test_set(args.output)
    
    elif args.command == 'evaluate':
        # Initialize evaluator
        evaluator = RAGEvaluator(model_name=args.model)
        
        # Load test set
        print(f"Loading test set from {args.test_set}")
        test_data = load_test_set(args.test_set)
        print(f"Loaded {len(test_data)} test items")
        
        # Define the RAG function to evaluate
        def rag_function(query):
            return answer(query)
        
        # Run evaluation
        print("Running evaluation...")
        results = evaluator.evaluate_batch(test_data, rag_function)
        
        # Save results
        save_results(results, args.output)
        
        # Generate and display report
        report = evaluator.generate_evaluation_report()
        print("\nEvaluation Report:")
        print(json.dumps(report, indent=2))
        
        # Save report
        report_path = Path(args.output).with_name('evaluation_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {report_path}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()