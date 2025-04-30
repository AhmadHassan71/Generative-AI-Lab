# src/evaluation/performance_metrics.py

import time
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd
import csv
import os
from typing import List, Dict, Tuple, Union, Optional
from pathlib import Path

try:
    import nltk
    nltk.download('punkt', quiet=True)
except ImportError:
    print("NLTK not found. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "nltk"])
    import nltk
    nltk.download('punkt', quiet=True)

try:
    from rouge import Rouge
except ImportError:
    print("Rouge not found. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "rouge"])
    from rouge import Rouge


class RAGEvaluator:
    """
    A comprehensive evaluator for RAG (Retrieval Augmented Generation) systems that measures
    performance using multiple metrics: BLEU, ROUGE, cosine similarity, human judgment, and latency.
    """

    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        """
        Initialize the evaluator with the specified sentence embedding model.
        
        Args:
            model_name (str): The name of the sentence transformer model to use for embeddings.
        """
        self.rouge = Rouge()
        self.smooth = SmoothingFunction().method1
        # Initialize the sentence transformer model for cosine similarity
        self.model = SentenceTransformer(model_name)
        
        # Create directories for storing evaluation results
        results_dir = Path('data/evaluation')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Paths for storing different evaluation results
        self.auto_metrics_path = results_dir / 'auto_metrics.csv'
        self.human_eval_path = results_dir / 'human_evaluation.csv'
        self.latency_metrics_path = results_dir / 'latency_metrics.csv'

    def calculate_bleu(self, reference: str, candidate: str) -> float:
        """
        Calculate BLEU score between reference and candidate texts.
        
        Args:
            reference (str): Reference text (ground truth)
            candidate (str): Generated text to evaluate
            
        Returns:
            float: BLEU score from 0 to 1
        """
        # Tokenize the texts into words
        reference_tokens = reference.lower().split()
        candidate_tokens = candidate.lower().split()
        
        # Calculate BLEU score with smoothing
        return sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=self.smooth)

    def calculate_rouge(self, reference: str, candidate: str) -> Dict[str, Dict[str, float]]:
        """
        Calculate ROUGE scores between reference and candidate texts.
        
        Args:
            reference (str): Reference text (ground truth)
            candidate (str): Generated text to evaluate
            
        Returns:
            Dict: Dictionary containing ROUGE-1, ROUGE-2, and ROUGE-L F1 scores
        """
        try:
            return self.rouge.get_scores(candidate, reference)[0]
        except Exception as e:
            print(f"Error calculating ROUGE: {e}")
            # Return zeros if ROUGE calculation fails
            return {
                'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}
            }

    def calculate_cosine_similarity(self, reference: str, candidate: str) -> float:
        """
        Calculate cosine similarity between reference and candidate text embeddings.
        
        Args:
            reference (str): Reference text (ground truth)
            candidate (str): Generated text to evaluate
            
        Returns:
            float: Cosine similarity from 0 to 1
        """
        try:
            # Generate embeddings
            ref_embedding = self.model.encode([reference])
            cand_embedding = self.model.encode([candidate])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(ref_embedding, cand_embedding)[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Error calculating cosine similarity: {e}")
            return 0.0

    def measure_latency(self, func, *args, **kwargs) -> Tuple[any, float]:
        """
        Measure the execution time of a function.
        
        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Tuple: (function result, execution time in seconds)
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        return result, end_time - start_time

    def record_human_evaluation(self, query_id: str, query: str, response: str, 
                              relevance: int, coherence: int, factuality: int, 
                              helpfulness: int, comments: str = "") -> None:
        """
        Record human evaluation of a response.
        
        Args:
            query_id (str): Unique identifier for the query
            query (str): The query text
            response (str): The system's response
            relevance (int): Score for relevance (1-5)
            coherence (int): Score for coherence (1-5)
            factuality (int): Score for factuality (1-5)
            helpfulness (int): Score for helpfulness (1-5)
            comments (str): Optional evaluator comments
        """
        # Create header if file doesn't exist
        file_exists = os.path.isfile(self.human_eval_path)
        
        with open(self.human_eval_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            
            if not file_exists:
                writer.writerow(['query_id', 'query', 'response', 'relevance', 
                                'coherence', 'factuality', 'helpfulness', 'comments', 
                                'timestamp'])
            
            writer.writerow([
                query_id,
                query,
                response,
                relevance,
                coherence,
                factuality,
                helpfulness,
                comments,
                time.strftime("%Y-%m-%d %H:%M:%S")
            ])

    def evaluate_response(self, query: str, response: str, reference: str, 
                         query_id: str = None) -> Dict[str, float]:
        """
        Evaluate a response using automatic metrics.
        
        Args:
            query (str): The query text
            response (str): The system's response
            reference (str): Reference ground truth answer
            query_id (str): Optional unique identifier for the query
            
        Returns:
            Dict: Dictionary of evaluation metrics
        """
        if query_id is None:
            query_id = str(int(time.time()))
        
        # Calculate metrics
        bleu = self.calculate_bleu(reference, response)
        rouge_scores = self.calculate_rouge(reference, response)
        cosine_sim = self.calculate_cosine_similarity(reference, response)
        
        # Extract ROUGE F1 scores
        rouge1 = rouge_scores['rouge-1']['f']
        rouge2 = rouge_scores['rouge-2']['f']
        rougeL = rouge_scores['rouge-l']['f']
        
        # Compile results
        metrics = {
            'query_id': query_id,
            'bleu': bleu,
            'rouge1': rouge1,
            'rouge2': rouge2, 
            'rougeL': rougeL,
            'cosine_similarity': cosine_sim
        }
        
        # Save to CSV
        file_exists = os.path.isfile(self.auto_metrics_path)
        with open(self.auto_metrics_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=metrics.keys())
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(metrics)
        
        return metrics

    def evaluate_batch(self, test_data: List[Dict[str, str]], rag_function) -> pd.DataFrame:
        """
        Evaluate a batch of queries using the provided RAG function.
        
        Args:
            test_data (List[Dict]): List of dictionaries containing 'query', 'reference' texts
            rag_function: Function that takes a query and returns a response
            
        Returns:
            pd.DataFrame: DataFrame with evaluation results for the batch
        """
        results = []
        
        for i, item in enumerate(test_data):
            query = item['query']
            reference = item['reference']
            query_id = item.get('query_id', f"q{i}")
            
            # Measure response time
            response, latency = self.measure_latency(rag_function, query)
            
            # Evaluate quality
            metrics = self.evaluate_response(query, response, reference, query_id)
            
            # Add latency
            metrics['latency'] = latency
            metrics['query'] = query
            metrics['reference'] = reference
            metrics['response'] = response
            
            results.append(metrics)
            
            # Also record latency separately
            with open(self.latency_metrics_path, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                
                if i == 0 and not os.path.isfile(self.latency_metrics_path):
                    writer.writerow(['query_id', 'query', 'latency', 'timestamp'])
                
                writer.writerow([
                    query_id,
                    query,
                    latency,
                    time.strftime("%Y-%m-%d %H:%M:%S")
                ])
        
        # Create DataFrame from results
        return pd.DataFrame(results)

    def generate_evaluation_report(self) -> Dict[str, any]:
        """
        Generate a comprehensive evaluation report from recorded metrics.
        
        Returns:
            Dict: Dictionary containing summary statistics for each metric
        """
        report = {
            'automatic_metrics': {},
            'human_evaluation': {},
            'latency': {}
        }
        
        # Automatic metrics analysis
        if os.path.isfile(self.auto_metrics_path):
            metrics_df = pd.read_csv(self.auto_metrics_path)
            
            # Calculate statistics
            for metric in ['bleu', 'rouge1', 'rouge2', 'rougeL', 'cosine_similarity']:
                if metric in metrics_df.columns:
                    report['automatic_metrics'][metric] = {
                        'mean': metrics_df[metric].mean(),
                        'median': metrics_df[metric].median(),
                        'std': metrics_df[metric].std(),
                        'min': metrics_df[metric].min(),
                        'max': metrics_df[metric].max()
                    }
        
        # Human evaluation analysis
        if os.path.isfile(self.human_eval_path):
            human_df = pd.read_csv(self.human_eval_path)
            
            for metric in ['relevance', 'coherence', 'factuality', 'helpfulness']:
                if metric in human_df.columns:
                    report['human_evaluation'][metric] = {
                        'mean': human_df[metric].mean(),
                        'median': human_df[metric].median(),
                        'std': human_df[metric].std(),
                        'min': human_df[metric].min(),
                        'max': human_df[metric].max()
                    }
            
            # Calculate overall human evaluation score (average of all metrics)
            metrics = ['relevance', 'coherence', 'factuality', 'helpfulness']
            metrics_present = [m for m in metrics if m in human_df.columns]
            
            if metrics_present:
                human_df['overall'] = human_df[metrics_present].mean(axis=1)
                report['human_evaluation']['overall'] = {
                    'mean': human_df['overall'].mean(),
                    'median': human_df['overall'].median(),
                    'std': human_df['overall'].std(),
                    'min': human_df['overall'].min(),
                    'max': human_df['overall'].max()
                }
        
        # Latency analysis
        if os.path.isfile(self.latency_metrics_path):
            latency_df = pd.read_csv(self.latency_metrics_path)
            
            if 'latency' in latency_df.columns:
                report['latency'] = {
                    'mean': latency_df['latency'].mean(),
                    'median': latency_df['latency'].median(),
                    'std': latency_df['latency'].std(),
                    'min': latency_df['latency'].min(),
                    'max': latency_df['latency'].max(),
                    '95th_percentile': np.percentile(latency_df['latency'], 95)
                }
        
        return report

    def create_test_set(self, file_path: str, queries: List[str], references: List[str], 
                      query_ids: Optional[List[str]] = None) -> None:
        """
        Create a test set file for evaluation.
        
        Args:
            file_path (str): Path to save the test set
            queries (List[str]): List of query texts
            references (List[str]): List of reference answers
            query_ids (List[str], optional): List of query identifiers
        """
        if query_ids is None:
            query_ids = [f"q{i}" for i in range(len(queries))]
        
        # Ensure all lists have the same length
        if not (len(queries) == len(references) == len(query_ids)):
            raise ValueError("queries, references, and query_ids must have the same length")
        
        test_data = []
        for qid, query, reference in zip(query_ids, queries, references):
            test_data.append({
                'query_id': qid,
                'query': query,
                'reference': reference
            })
        
        # Save to CSV
        df = pd.DataFrame(test_data)
        df.to_csv(file_path, index=False)
        print(f"Test set saved to {file_path}")


# Example usage
if __name__ == "__main__":
    # Initialize evaluator
    evaluator = RAGEvaluator()
    
    # Define example test data (typically you'd load this from a file)
    test_data = [
        {
            'query_id': 'q1',
            'query': 'What is the vision of FAST-NUCES?',
            'reference': 'To produce world-class professionals, who are responsible citizens and good human beings.'
        },
        {
            'query_id': 'q2',
            'query': 'How many campuses does FAST-NUCES have?',
            'reference': 'FAST-NUCES has five campuses located in Karachi, Lahore, Islamabad, Peshawar, and Chiniot-Faisalabad.'
        }
    ]
    
    # Sample mock RAG function for testing
    def mock_rag_function(query):
        # This is just a mock function; in practice you'd use your actual RAG function
        if 'vision' in query.lower():
            return "To produce world-class professionals who are responsible citizens."
        elif 'campuses' in query.lower():
            return "The university has five campuses in different cities of Pakistan."
        else:
            return "I don't have enough information to answer this question."
    
    # Evaluate batch of queries
    results = evaluator.evaluate_batch(test_data, mock_rag_function)
    print(results)
    
    # Example of recording human evaluation
    evaluator.record_human_evaluation(
        query_id='q1',
        query='What is the vision of FAST-NUCES?',
        response='To produce world-class professionals who are responsible citizens.',
        relevance=5,
        coherence=5,
        factuality=4,
        helpfulness=5,
        comments="The response is accurate but slightly incomplete."
    )
    
    # Generate evaluation report
    report = evaluator.generate_evaluation_report()
    print("\nEvaluation Report:")
    print(report)