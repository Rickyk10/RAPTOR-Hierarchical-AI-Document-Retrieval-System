# RAPTOR RAG System

## 🚀 Overview

This project implements a Retrieval-Augmented Generation (RAG) system leveraging the innovative RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) framework. RAPTOR creates a multi-layered, hierarchical representation of documents, allowing for nuanced retrieval strategies that can retrieve information at varying levels of granularity – from highly specific details to broad, abstract summaries.

The system provides two powerful retrieval modes:

- **Tree Traversal Retrieval**: Intelligently navigates the document hierarchy to find the most relevant context, descending to finer details only when necessary.
- **Collapsed Tree Retrieval**: Flattens the entire hierarchy into a single vector space for direct similarity search, offering a straightforward and often highly effective retrieval approach.

This project aims to demonstrate the power of hierarchical document representation for robust and efficient RAG applications, enhancing the accuracy and relevance of LLM-generated answers.

> _(Optional: Link to a demo video/GIF demonstrating the system in action - highly recommended for impact!)_

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

To install and run the RAPTOR RAG system locally, follow these steps:

```bash
# Clone the repository
git clone https://github.com/your-github-username/your-repo-name.git
cd your-repo-name

# Create a virtual environment and activate it (recommended)
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Create a `.env` file in the root directory of the project and add your OpenAI API key:

```env
OPENAI_API_KEY="your_openai_api_key_here"
```

> ⚠️ **Note:** Ensure you keep your API key secure and do not commit it to public repositories.

## 🚀 Usage

### 1. Start the FastAPI Application

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:  
📍 `http://127.0.0.1:8000`

You can access the Swagger UI at:  
📖 `http://127.0.0.1:8000/docs`

### 2. Ingest a Document

Use the `/ingest` endpoint to process a text file and build the RAPTOR hierarchy.

**Endpoint**: `POST /ingest`  
**Body**:

```json
{
  "file_path": "path/to/your/document.txt",
  "n_levels": 3
}
```

- `file_path`: The path to the text file you want to ingest (relative to the project root or an absolute path).
- `n_levels`: The number of hierarchical levels to build for the RAPTOR tree (e.g., 3 means chunks → Level 1 summaries → Level 2 summaries → Level 3 summaries).

**Example using curl**:

```bash
curl -X POST "http://127.0.0.1:8000/ingest" \
     -H "Content-Type: application/json" \
     -d '{"file_path": "data/sample_document.txt", "n_levels": 3}'
```

### 3. Ask a Question (Tree Traversal)

Query the document using the Tree Traversal method.

**Endpoint**: `POST /ask_tree_traversal`  
**Body**:

```json
{
  "query": "Your question here?"
}
```

**Example using curl**:

```bash
curl -X POST "http://127.0.0.1:8000/ask_tree_traversal" \
     -H "Content-Type: application/json" \
     -d '{"query": "What are the main inspirations behind the novel Dune, specifically related to Frank Herbert\'s personal experiences?"}'
```

### 4. Ask a Question (Collapsed Tree)

Query the document using the Collapsed Tree method.

**Endpoint**: `POST /ask_flattened`  
**Body**:

```json
{
  "query": "Your question here?"
}
```

**Example using curl**:

```bash
curl -X POST "http://127.0.0.1:8000/ask_flattened" \
     -H "Content-Type: application/json" \
     -d '{"query": "When was Dune first published and by what company, and how was its initial critical reception?"}'
```
