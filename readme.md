# Semantic Trends in Twitter Language

This project analyzes semantic and contextual shifts in Twitter language from 2015 to 2023 using graph-based methods. It provides both preprocessing pipelines and an interactive Streamlit application for visualizing and comparing linguistic patterns across time.


## Dataset

The data is sampled from the public dataset:  
[https://huggingface.co/datasets/enryu43/twitter100m_tweets](https://huggingface.co/datasets/enryu43/twitter100m_tweets)

A uniform sample is extracted across years to ensure balanced comparison.


## Project Components

### `semanticTrends_dataprep.ipynb`  
Processes and samples tweets from the source dataset to create a clean, year-wise dataset suitable for graph construction.

### `semanticTrends_graphs.ipynb`  
Constructs two types of word graphs:
- **Embedding-based graphs**: Edges connect semantically similar words based on word embeddings.
- **Co-occurrence graphs**: Edges connect words that frequently appear together in the same tweet.

Graphs are saved in GraphML format for downstream analysis.

### `semanticTrends_main.py`  
Interactive Streamlit application that allows users to:
- Explore top word communities using greedy modularity clustering
- Visualize the most central words in each graph
- Identify words emerging over time
- Compare semantic neighborhoods for any word of interest


## Getting Started

### Requirements
Ensure the following dependencies are installed:
- Python 3.8+
- `streamlit`
- `networkx`
- `matplotlib`
- `pandas`
- `tqdm`
- `nltk`
- `spacy`
- A preloaded word embedding model compatible with `gensim`

Install dependencies with:

```bash
pip install -r requirements.txt
```

Run with:
```bash
python -m streamlit run semanticTrends_main.py