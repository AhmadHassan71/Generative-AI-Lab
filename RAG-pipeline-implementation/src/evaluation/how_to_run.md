# How to Run the RAG Evaluation System

This document provides step-by-step instructions for evaluating your RAG (Retrieval Augmented Generation) system using various performance metrics.

## Prerequisites

Before running the evaluation, make sure you have:

1. Python 3.7+ installed
2. The required Python packages:
   - nltk
   - rouge
   - sentence-transformers
   - pandas
   - numpy
   - scikit-learn
   - dotenv

If you haven't installed these packages, you can run:
```
pip install nltk rouge sentence-transformers pandas numpy scikit-learn python-dotenv
```

## Setup

1. Make sure your environment variables are set up correctly in your .env file:
   - OPENAI_API_KEY - for using OpenAI models
   - QDRANT_COLLECTION - name of your vector database collection
   - OPENAI_MODEL - model name (default: gpt-4o-mini)
   - MAX_TOKENS - maximum tokens for response generation
   - RELEVANCE_THRESHOLD - threshold for determining relevance

2. Ensure your RAG system is properly configured and running.

## Running the Evaluation

### Step 1: Create a Test Set

First, create a test set with queries and reference answers:

```
python src/evaluation/run_evaluation.py create-test --output data/evaluation/my_test_set.json
```

This will generate a sample test set with questions about FAST-NUCES. You can modify this file to add your own test queries and reference answers.

### Step 2: Run the Evaluation

Next, run the evaluation using your test set:

```
python src/evaluation/run_evaluation.py evaluate --test-set data/evaluation/my_test_set.json
```

By default, this will save the results to `data/evaluation/results.csv` and a summary report to `data/evaluation/evaluation_report.json`.

### Optional Parameters

- `--model`: Specify a different sentence embedding model (default: all-mpnet-base-v2)
- `--output`: Specify a custom output path for results

Example with custom parameters:
```
python src/evaluation/run_evaluation.py evaluate --test-set data/evaluation/my_test_set.json --model sentence-transformers/all-MiniLM-L6-v2 --output data/evaluation/custom_results.csv
```

## Analyzing the Results

After running the evaluation, you'll find:

1. `results.csv`: Detailed metrics for each query
2. `evaluation_report.json`: Summary statistics for all metrics
3. `auto_metrics.csv`: Records of automatic evaluation metrics
4. `latency_metrics.csv`: Response time measurements

## Human Evaluation

For human evaluation of responses:

1. Review the generated responses in the results file
2. Use the `RAGEvaluator.record_human_evaluation()` method to record human judgments:

```python
evaluator = RAGEvaluator()
evaluator.record_human_evaluation(
    query_id='q1',
    query='What is the vision of FAST-NUCES?',
    response='Generated response...',
    relevance=5,  # Score from 1-5
    coherence=5,  # Score from 1-5
    factuality=4,  # Score from 1-5
    helpfulness=5,  # Score from 1-5
    comments='The response is accurate but slightly incomplete.'
)
```

3. Generate an updated report that includes human evaluations:
```python
report = evaluator.generate_evaluation_report()
print(report)
```

## Troubleshooting

If you encounter import errors, ensure:
- You're running the script from the project root directory
- The project structure remains intact
- All required packages are installed correctly

For other issues, check the error messages for specific details on what might be going wrong.