# Fraud Classification with Graph ML

This project explores graph-based machine learning techniques to detect potential fraud among healthcare providers and physicians.
The `fraud_analysis_graphml.py` script downloads a healthcare provider fraud dataset from Kaggle, constructs a bipartite graph linking providers and attending physicians, and computes graph features such as degree, eigenvector centrality, PageRank, and community detection.
It then trains a Random Forest classifier using baseline features, graph features, and community-based features to evaluate model performance.

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the analysis script:

```bash
python fraud_analysis_graphml.py
```

The script will download data via `kagglehub`, generate network visualizations, compute graph metrics, and train a classifier.
