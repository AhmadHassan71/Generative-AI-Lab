# RAG Evaluation Testing Approach and Metrics

This document outlines our comprehensive approach to evaluating the RAG (Retrieval Augmented Generation) system, including the testing methodology and metrics used.

## Testing Approach

Our evaluation framework employs a multi-faceted approach to assess the quality, accuracy, and efficiency of RAG-generated responses:

### 1. Test Set Creation

We created a test set of domain-specific queries about FAST-NUCES, covering diverse information such as:
- Institutional vision and mission
- Campus locations
- Academic programs offered
- Recent events and milestones

For each query, we established ground-truth reference answers extracted directly from the university's annual report to serve as a basis for comparison.

### 2. Automated Evaluation

Each query was processed through our RAG pipeline, which:
1. Embeds the query using a transformer model
2. Retrieves relevant documents from our vector database
3. Checks relevance of retrieved documents using similarity scores
4. Generates a natural language response using the retrieved context

All responses were automatically evaluated using multiple metrics, and the timing of each response was recorded for performance benchmarking.

### 3. Human Assessment (Optional)

For a subset of queries, human evaluators rated the responses on multiple dimensions to capture aspects of quality that automated metrics might miss.

## Evaluation Metrics

We evaluated our RAG system using the following metrics:

### 1. Automated Metrics

#### BLEU Score
- **What it measures**: Word-level precision between generated and reference answers
- **Scale**: 0 to 1 (higher is better)
- **Implementation**: NLTK's sentence_bleu with smoothing function
- **Significance**: Captures lexical overlap and precision in word choice

#### ROUGE Scores
- **What they measure**: Recall of n-grams between generated and reference answers
- **Types used**:
  - ROUGE-1: Unigram overlap (individual words)
  - ROUGE-2: Bigram overlap (pairs of adjacent words)
  - ROUGE-L: Longest common subsequence
- **Scale**: 0 to 1 (higher is better)
- **Significance**: Assesses how much of the reference answer is captured in the generated response

#### Semantic Similarity (Cosine Similarity)
- **What it measures**: Semantic meaning similarity between generated and reference answers
- **Method**: Embedding sentences using SentenceTransformer and calculating cosine similarity
- **Model used**: all-mpnet-base-v2 (default)
- **Scale**: 0 to 1 (higher is better)
- **Significance**: Captures semantic correspondence beyond exact word matching

#### Latency
- **What it measures**: Time taken to generate a response
- **Units**: Seconds
- **Statistics collected**: Mean, median, standard deviation, min, max, 95th percentile
- **Significance**: Evaluates system performance and user experience

### 2. Human Evaluation Metrics

These subjective assessments are gathered on a 5-point Likert scale (1-5, where 5 is best):

#### Relevance
- **Definition**: How well the response addresses the query
- **Question**: "Does the response directly answer the question asked?"

#### Coherence
- **Definition**: Logical flow and clarity of the response
- **Question**: "Is the response well-structured, clear, and easy to understand?"

#### Factuality
- **Definition**: Accuracy of information provided
- **Question**: "Does the response contain factually correct information?"

#### Helpfulness
- **Definition**: Utility of the response to the user
- **Question**: "Would this response be helpful to someone asking this question?"

## Evaluation Output

The evaluation generates several outputs:

1. **Detailed CSV Results**: Individual metrics for each query-response pair
2. **Summary Report**: Statistical analysis of all metrics (mean, median, std, min, max)
3. **Latency Analysis**: Performance metrics for system optimization
4. **Human Evaluation Records**: Subjective assessments and comments (if performed)

## Interpretation Guidelines

- **BLEU/ROUGE < 0.3**: Poor lexical overlap, potentially off-topic or missing key information
- **BLEU/ROUGE 0.3-0.5**: Moderate overlap, captures some key points
- **BLEU/ROUGE > 0.5**: Good overlap, contains most key information
- **Cosine Similarity < 0.7**: Potentially different meaning from reference
- **Cosine Similarity 0.7-0.85**: Semantically similar but may miss nuances
- **Cosine Similarity > 0.85**: Very good semantic match
- **Human Scores < 3**: Needs significant improvement
- **Human Scores 3-4**: Acceptable but could be better
- **Human Scores > 4**: Excellent performance

This comprehensive evaluation framework allows us to identify strengths and weaknesses in our RAG system, guiding targeted improvements for better information retrieval and response generation.