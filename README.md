# RAPTOR RAG System

## 🚀 Overview

This project implements a Retrieval-Augmented Generation (RAG) system leveraging the innovative RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) framework. RAPTOR creates a multi-layered, hierarchical representation of documents, allowing for nuanced retrieval strategies that can retrieve information at varying levels of granularity – from highly specific details to broad, abstract summaries.

The system provides two powerful retrieval modes:

- **Tree Traversal Retrieval**: Intelligently navigates the document hierarchy to find the most relevant context, descending to finer details only when necessary.

- **Collapsed Tree Retrieval**: Flattens the entire hierarchy into a single vector space for direct similarity search, offering a straightforward and often highly effective retrieval approach.

This project aims to demonstrate the power of hierarchical document representation for robust and efficient RAG applications, enhancing the accuracy and relevance of LLM-generated answers.



## ✨ Features

- **RAPTOR Hierarchy Generation**: Automatically chunks documents, embeds them, and recursively generates multi-level summaries, creating a rich, tree-like knowledge structure.

- **Two Retrieval Strategies**:
  - **Tree Traversal**: Smart, depth-first search for context, ideal for queries requiring specific detail or broad overview depending on context sufficiency.
  - **Collapsed Tree**: Simple, efficient similarity search across all levels of the hierarchy, effective for a wide range of queries.

- **Persistent Storage**: Uses ChromaDB as the vector store for efficient indexing and retrieval of embeddings.

- **LLM Integration**: Seamlessly integrates with OpenAI's large language models (LLMs) for abstractive summarization and answer generation.

- **Asynchronous API**: Built with FastAPI for a high-performance, asynchronous web API.

- **Docker Support**: Easily containerize and deploy the application.

## 🧠 How RAPTOR Works (Simplified)

RAPTOR builds a "tree of summaries" from your documents:

- **Level 1 (Leaves)**: Your original document is broken down into small chunks.

- **Level 2 onwards (Branches)**:
  - Chunks are grouped into themes (clusters).
  - Each group is summarized by an LLM.
  - These summaries then become the "chunks" for the next level up.
  - This process repeats for `N_LEVELS`, creating progressively more abstract summaries.

- **Top Level (Root)**: The highest level contains very broad summaries of the entire document, potentially condensing into a single "root summary" if the content allows.

When you ask a question, the system can either:

- **Traverse the Tree**: Start at the top, find relevant broad summaries, and only "zoom in" to more detailed summaries or original chunks if needed.

- **Collapse the Tree**: Treat all chunks and summaries (from every level) as individual searchable items in one big pool and find the most relevant ones directly.

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- `pip` package manager
- Docker (optional, for containerized deployment)

### Steps

**1. Clone the repository:**

```bash
git clone https://github.com/your-github-username/your-repo-name.git
cd your-repo-name
