import sys
import os
import json
import numpy as np
import pandas as pd
import umap
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sklearn.mixture import GaussianMixture
from pydantic import BaseModel
import uvicorn
import asyncio # New import for async operations
import threading
import logging
from typing import List, Dict, Tuple, Optional, Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
import PyPDF2
import io
import hashlib # For embedding caching

# --- Pydantic Model for API Requests ---
class QuestionRequest(BaseModel):
    query: str

# --- Configuration and Initialization ---
# Ensure your OpenAI API key is set as an environment variable.
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable not set. Please set it to your OpenAI API key.")

# Initialize OpenAI Embedding and Chat models
openai_ef = OpenAIEmbeddings()
model = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")

print("OpenAI Embeddings initialized.")
print("OpenAI Chat model initialized.")

# --- ChromaDB Setup ---
CHROMA_DB_PATH = "chroma_db_raptor_strategies" # Path to store ChromaDB data
CHROMA_COLLECTION_NAME_FLATTENED = "raptor_flattened_collection"

# Initialize ChromaDB client and embedding function for ChromaDB
chroma_client = PersistentClient(path=CHROMA_DB_PATH)
chroma_openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"],
    model_name="text-embedding-ada-002"
)

# Get or create the flattened ChromaDB collection
try:
    chroma_collection_flattened = chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME_FLATTENED,
        embedding_function=chroma_openai_ef
    )
    print(f"Loaded existing Chroma flattened collection: {CHROMA_COLLECTION_NAME_FLATTENED}")
except Exception as e:
    print(f"Error getting/creating flattened collection: {e}. Attempting to create new.")
    chroma_collection_flattened = chroma_client.create_collection(
        name=CHROMA_COLLECTION_NAME_FLATTENED,
        embedding_function=chroma_openai_ef
    )
    print(f"Created new Chroma flattened collection: {CHROMA_COLLECTION_NAME_FLATTENED}")


# --- FastAPI Application Instance ---
app = FastAPI()

# --- Directories for file uploads and RAPTOR data ---
UPLOAD_FOLDER = "uploaded_documents"
RAPTOR_DATA_FOLDER = "raptor_data_strategies"
EMBEDDING_CACHE_DIR = "embedding_cache" # New directory for embedding cache

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RAPTOR_DATA_FOLDER, exist_ok=True)
os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True) # Ensure cache directory exists

# --- Global Constants for RAPTOR ---
RANDOM_SEED = 224
EMBEDDING_DIM = 10 # Dimensionality for UMAP reduction
GMM_THRESHOLD = 0.1 # Threshold for Gaussian Mixture Model clustering
N_LEVELS = 3 # Number of levels in the RAPTOR hierarchy (e.g., 3 means chunks -> L2 summaries -> L3 summaries)

# --- Global counter for unique IDs (for ingested documents) ---
global_id_counter = 0

# --- HELPER FUNCTIONS (Ordered for correct definition before use) ---

def global_cluster_embeddings(embeddings: np.ndarray, dim: int, n_neighbors: Optional[int] = None, metric: str = "cosine") -> np.ndarray:
    """Reduces dimensionality of embeddings globally using UMAP."""
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5) if len(embeddings) > 1 else 1
    if len(embeddings) <= dim + 1:
         # Handle cases where UMAP cannot be applied (not enough samples)
         if embeddings.shape[0] == 0:
             return np.array([])
         return embeddings[:, :min(dim, embeddings.shape[1])] if embeddings.shape[1] >= dim else np.pad(embeddings, ((0,0),(0,dim-embeddings.shape[1])), 'constant', constant_values=0)
    return umap.UMAP(n_neighbors=n_neighbors, n_components=dim, metric=metric, random_state=RANDOM_SEED).fit_transform(embeddings)

def local_cluster_embeddings(embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine") -> np.ndarray:
    """Reduces dimensionality of embeddings locally using UMAP."""
    if len(embeddings) <= dim + 1:
        if embeddings.shape[0] == 0:
            return np.array([])
        return embeddings[:, :min(dim, embeddings.shape[1])] if embeddings.shape[1] >= dim else np.pad(embeddings, ((0,0),(0,dim-embeddings.shape[1])), 'constant', constant_values=0)
    return umap.UMAP(n_neighbors=num_neighbors, n_components=dim, metric=metric, random_state=RANDOM_SEED).fit_transform(embeddings)

def get_optimal_clusters(embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED) -> int:
    """Determines optimal number of clusters using BIC for GMM."""
    num_samples = len(embeddings)
    max_clusters = min(max_clusters, num_samples)

    if num_samples < 2 or max_clusters < 1:
        return 1 # Not enough samples for clustering, default to 1 cluster
    
    n_clusters_range = np.arange(1, max_clusters + 1)
    bics = []
    
    for n in n_clusters_range:
        if n > num_samples: # Ensure n_components <= n_samples
            bics.append(np.inf) # Mark as infinity if too many components
            continue
        try:
            gm = GaussianMixture(n_components=n, random_state=random_state, n_init=10)
            gm.fit(embeddings)
            bics.append(gm.bic(embeddings))
        except ValueError as e:
            # print(f"Warning: BIC calculation failed for {n} clusters: {e}") # Debug print
            bics.append(np.inf) # Mark as infinity if calculation fails
    
    bics_np = np.array(bics)

    if not np.any(np.isfinite(bics_np)):
        # If all BIC calculations failed or are infinite, default to 1 cluster
        return 1
    
    # Find the index of the minimum finite BIC
    min_bic_idx_in_finite = np.argmin(bics_np[np.isfinite(bics_np)])
    
    # Map this index back to the original full bics_np array to get the corresponding 'n'
    # This requires finding the original index of the value found in finite_bics
    # np.where returns a tuple, so we need the first element and then the first index [0][0]
    
    # Get the value of the minimum finite BIC
    min_bic_value = bics_np[np.isfinite(bics_np)][min_bic_idx_in_finite]
    
    # Find all indices in the original bics_np that match this minimum value
    original_indices_of_min_bic = np.where(bics_np == min_bic_value)[0]
    
    # Pick the first one (or handle ties as desired, but first is common)
    optimal_n = n_clusters_range[original_indices_of_min_bic[0]]
    
    return optimal_n

def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = RANDOM_SEED):
    """Performs Gaussian Mixture Model clustering."""
    if len(embeddings) == 0:
        return [], 0
    if len(embeddings) == 1:
        return [np.array([0])], 1 # Single point is its own cluster

    n_clusters = get_optimal_clusters(embeddings)
    if n_clusters == 0: # Should not happen with the revised get_optimal_clusters, but as a safeguard
        return [], 0

    gm = GaussianMixture(n_components=n_clusters, random_state=random_state, n_init=10)
    try:
        gm.fit(embeddings)
        probs = gm.predict_proba(embeddings)
        labels = [np.where(prob > threshold)[0] for prob in probs]
    except Exception as e:
        print(f"Error in GMM_cluster: {e}. Falling back to single cluster or individual points.")
        # If GMM fails for any reason, treat each point as its own cluster (or assign all to one if desired)
        return [np.array([i]) for i in range(len(embeddings))], len(embeddings) if len(embeddings) > 0 else 1

    return labels, n_clusters

def perform_clustering(embeddings: np.ndarray, dim: int, threshold: float) -> List[np.ndarray]:
    """Applies global and local clustering to embeddings."""
    if len(embeddings) == 0:
        return []
    if len(embeddings) <= dim + 1:
        # If too few samples for UMAP, each point is its own cluster
        return [np.array([i]) for i in range(len(embeddings))]

    reduced_embeddings_global = global_cluster_embeddings(embeddings, dim)
    global_clusters, n_global_clusters = GMM_cluster(reduced_embeddings_global, threshold)

    if n_global_clusters == 0:
        return [np.array([i]) for i in range(len(embeddings))]

    all_local_clusters = [np.array([], dtype=int) for _ in range(len(embeddings))]
    total_clusters = 0

    for i in range(n_global_clusters):
        # Determine which original indices belong to this global cluster
        global_cluster_indices = [idx for idx, gc_labels in enumerate(global_clusters) if i in gc_labels]
        if not global_cluster_indices:
            continue

        global_cluster_embeddings_ = embeddings[global_cluster_indices]
        original_indices_in_global_cluster = np.array(global_cluster_indices)

        if len(global_cluster_embeddings_) == 0:
            continue

        if len(global_cluster_embeddings_) <= dim + 1:
            # Not enough samples for local UMAP/GMM, assign all to one local cluster
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            reduced_embeddings_local = local_cluster_embeddings(global_cluster_embeddings_, dim)
            try:
                local_clusters, n_local_clusters = GMM_cluster(reduced_embeddings_local, threshold)
            except ValueError as e:
                print(f"Warning: GMM_cluster failed for local cluster {i} with error: {e}. Assigning to single cluster.")
                local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
                n_local_clusters = 1

        for j in range(n_local_clusters):
            indices_in_global_cluster_for_local_cluster = [idx for idx, lc_labels in enumerate(local_clusters) if j in lc_labels]
            if not indices_in_global_cluster_for_local_cluster:
                continue

            original_indices_for_this_local_cluster = original_indices_in_global_cluster[indices_in_global_cluster_for_local_cluster]

            for original_idx in original_indices_for_this_local_cluster:
                all_local_clusters[original_idx] = np.append(all_local_clusters[original_idx], j + total_clusters)

        total_clusters += n_local_clusters

    # Ensure all original texts are assigned to at least one cluster
    for i, cluster_array in enumerate(all_local_clusters):
        if cluster_array.size == 0:
            all_local_clusters[i] = np.array([total_clusters])
            total_clusters += 1

    return all_local_clusters

# --- Embedding Caching Function ---
def get_text_hash(text: str) -> str:
    """Generates a SHA256 hash for a given text."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def embed_with_cache(texts: List[str]) -> np.ndarray:
    """
    Generates embeddings for a list of texts using the OpenAI embedding model,
    with a disk-based cache.
    """
    if 'openai_ef' not in globals() or openai_ef is None:
        raise RuntimeError("Embedding model 'openai_ef' not initialized.")
    if not texts:
        return np.array([])

    all_embeddings_ordered = [None] * len(texts)
    texts_to_embed_indices = []

    for i, text in enumerate(texts):
        text_hash = get_text_hash(text)
        cache_file = os.path.join(EMBEDDING_CACHE_DIR, f"{text_hash}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    all_embeddings_ordered[i] = np.array(json.load(f)['embedding'])
            except json.JSONDecodeError as e:
                print(f"Warning: Could not decode cached embedding for {text_hash}: {e}. Re-embedding.")
                texts_to_embed_indices.append(i)
        else:
            texts_to_embed_indices.append(i)

    if texts_to_embed_indices:
        batch_texts = [texts[idx] for idx in texts_to_embed_indices]
        try:
            # Langchain's embed_documents already handles batching internally
            new_embeddings_list = openai_ef.embed_documents(batch_texts)
        except Exception as e:
            raise RuntimeError(f"Error calling OpenAI embeddings API: {e}")

        for i, new_emb in zip(texts_to_embed_indices, new_embeddings_list):
            all_embeddings_ordered[i] = np.array(new_emb)
            text_hash = get_text_hash(texts[i])
            cache_file = os.path.join(EMBEDDING_CACHE_DIR, f"{text_hash}.json")
            try:
                with open(cache_file, 'w') as f:
                    json.dump({'text': texts[i], 'embedding': new_emb}, f)
            except Exception as e:
                print(f"Warning: Could not save embedding to cache for {text_hash}: {e}")
    
    # Filter out any None values if some embeddings failed or weren't processed
    final_embeddings = [emb for emb in all_embeddings_ordered if emb is not None]
    if not final_embeddings:
        return np.array([]) # Return empty array if no embeddings could be generated/retrieved

    return np.array(final_embeddings)

# Replace the original embed function with the cached one
def embed(texts: List[str]):
    return embed_with_cache(texts)

def fmt_txt(df: pd.DataFrame) -> str:
    """Formats text for summarization by joining unique texts."""
    unique_txt = df["text"].tolist()
    return "--- --- \n --- --- ".join(unique_txt)

# This function is called by embed_cluster_summarize_texts
def embed_cluster_texts(texts: List[str]) -> pd.DataFrame:
    """Embeds texts and performs clustering, returning a DataFrame with text, embeddings, and cluster labels."""
    text_embeddings_np = embed(texts)
    if len(texts) == 0 or text_embeddings_np.shape[0] == 0:
        return pd.DataFrame({"text": [], "embd": [], "cluster": []})

    cluster_labels = perform_clustering(text_embeddings_np, EMBEDDING_DIM, GMM_THRESHOLD)

    # Ensure cluster_labels matches texts length; fill with empty array if missing
    if len(cluster_labels) != len(texts):
        print(f"Warning: Mismatch in number of texts ({len(texts)}) and cluster labels ({len(cluster_labels)}). Adjusting.")
        if len(cluster_labels) < len(texts):
            cluster_labels.extend([np.array([], dtype=int) for _ in range(len(texts) - len(cluster_labels))])
        elif len(cluster_labels) > len(texts):
            cluster_labels = cluster_labels[:len(texts)]

    df = pd.DataFrame()
    df["text"] = texts
    df["embd"] = list(text_embeddings_np)
    df["cluster"] = cluster_labels
    return df

# This function is called by recursive_embed_cluster_summarize
async def embed_cluster_summarize_texts(texts: List[str], level: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Embeds, clusters, and then summarizes the texts for a given level."""
    df_clusters = embed_cluster_texts(texts) # This is synchronous as embed_cluster_texts is now synchronous (though embed is cached)

    if df_clusters.empty or all(len(c) == 0 for c in df_clusters["cluster"]):
        print(f"--No clusters generated at level {level}. Cannot summarize.--")
        return df_clusters, pd.DataFrame({"summaries": [], "level": [], "cluster": []})

    expanded_list = []
    for index, row in df_clusters.iterrows():
        if isinstance(row["cluster"], np.ndarray) and row["cluster"].size > 0:
            for cluster_id in row["cluster"]:
                if isinstance(cluster_id, (int, np.integer)):
                    expanded_list.append({"text": row["text"], "embd": row["embd"], "cluster": int(cluster_id)})
                else:
                    print(f"Warning: Skipping non-integer cluster ID: {cluster_id} at level {level}")
        elif isinstance(row["cluster"], (int, np.integer)):
            expanded_list.append({"text": row["text"], "embd": row["embd"], "cluster": int(row["cluster"])})
        else:
            print(f"Warning: Skipping row with unprocessable cluster type at level {level}: {type(row['cluster'])} value: {row['cluster']}")

    expanded_df = pd.DataFrame(expanded_list)

    if expanded_df.empty:
        print(f"--No expanded data for summarizing at level {level}--")
        return df_clusters, pd.DataFrame({"summaries": [], "level": [], "cluster": []})

    all_clusters = expanded_df["cluster"].unique()
    print(f"--Generated {len(all_clusters)} clusters at level {level}--")

    template = """Here is a sub-set of documentation. Give a detailed summary of the documentation provided. Documentation: {context}"""
    prompt = ChatPromptTemplate.from_template(template)

    if 'model' not in globals() or model is None:
        raise RuntimeError("Language model 'model' not initialized.")

    chain = prompt | model | StrOutputParser()
    
    # Prepare for async summarization
    tasks = []
    cluster_id_map = {} # Map task index to cluster_id to reconstruct later

    for idx, i in enumerate(all_clusters):
        df_cluster = expanded_df[expanded_df["cluster"] == i]
        if not df_cluster.empty:
            formatted_txt = fmt_txt(df_cluster)
            tasks.append(chain.ainvoke({"context": formatted_txt}))
            cluster_id_map[idx] = i

    # Await all summarization tasks concurrently
    summaries_raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    summaries = []
    processed_clusters = []

    for idx, result in enumerate(summaries_raw_results):
        cluster_id = cluster_id_map[idx]
        if isinstance(result, Exception):
            print(f"Warning: Failed to summarize cluster {cluster_id} at level {level} due to: {result}")
            # You might want to append a placeholder or a default empty string for failed summaries
            # For now, we skip them to avoid issues with empty summaries later
        else:
            summaries.append(result)
            processed_clusters.append(cluster_id)

    df_summary = pd.DataFrame({"summaries": summaries, "level": [level] * len(summaries), "cluster": list(processed_clusters)})
    return df_clusters, df_summary

# This is the main recursive function for building the RAPTOR hierarchy
async def recursive_embed_cluster_summarize(texts: List[str], level: int = 1, n_levels: int = N_LEVELS) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Recursively embeds, clusters, and summarizes texts to build the RAPTOR hierarchy."""
    results = {}
    if not texts:
        print(f"--No texts to process at level {level}--")
        return results

    # Await the call to the async function
    df_clusters, df_summary = await embed_cluster_summarize_texts(texts, level)
    results[level] = (df_clusters, df_summary)
    
    unique_clusters_count = df_summary["cluster"].nunique() if not df_summary.empty else 0

    if level < n_levels and unique_clusters_count > 1 and not df_summary.empty:
        new_texts = df_summary["summaries"].tolist()
        if new_texts:
            print(f"Proceeding to level {level + 1} with {len(new_texts)} summaries.")
            # Await the recursive call
            next_level_results = await recursive_embed_cluster_summarize(new_texts, level + 1, n_levels)
            results.update(next_level_results)
        else:
            print(f"--No summaries generated to proceed to level {level + 1}--")
    elif unique_clusters_count <= 1 and level < n_levels:
        print(f"--Only {unique_clusters_count} unique cluster(s) at level {level}. Stopping recursion.--")
    elif df_summary.empty and level < n_levels:
        print(f"--No summaries generated at level {level}. Stopping recursion.--")
    elif level >= n_levels:
        print(f"--Reached maximum recursion depth ({n_levels}). Stopping recursion.--")

    return results

# --- FastAPI ENDPOINTS ---

@app.post("/clear_db")
async def clear_chroma_db():
    """Clears and recreates the ChromaDB collections."""
    try:
        # Delete and recreate flattened collection
        chroma_client.delete_collection(name=CHROMA_COLLECTION_NAME_FLATTENED)
        global chroma_collection_flattened
        chroma_collection_flattened = chroma_client.create_collection(
            name=CHROMA_COLLECTION_NAME_FLATTENED,
            embedding_function=chroma_openai_ef
        )

        print("ChromaDB collections cleared and recreated.")
        return {"message": "ChromaDB collections cleared successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing ChromaDB: {str(e)}")

@app.post("/ingest")
async def ingest_documents(files: List[UploadFile] = []):
    """
    Ingests documents, builds the RAPTOR hierarchy, and indexes it in ChromaDB.
    Supports PDF and plain text files.
    """
    if 'openai_ef' not in globals() or openai_ef is None:
        raise HTTPException(status_code=500, detail="Embedding model not initialized.")

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    all_extracted_texts_from_files = [] 
    for file in files:
        try:
            contents = await file.read()
            filename = file.filename
            text = ""
            if filename.lower().endswith(".pdf"):
                try:
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(contents))
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        text += page.extract_text()
                    print(f"Successfully processed PDF: {filename}")
                except Exception as e:
                    print(f"Error processing PDF {filename}: {e}")
                    raise HTTPException(status_code=500, detail=f"Error reading PDF {filename}: {str(e)}")
            else:
                try:
                    text = contents.decode("utf-8")
                    print(f"Successfully processed text file: {filename}")
                except UnicodeDecodeError:
                    text = contents.decode("latin-1") # Fallback for common encoding issues
                    print(f"Successfully processed text file (latin-1): {filename}")
                except Exception as e:
                    print(f"Error reading file {filename}: {e}")
                    raise HTTPException(status_code=500, detail=f"Error reading non-PDF file {filename}: {str(e)}")
            
            if text:
                all_extracted_texts_from_files.append(text)
        except Exception as e:
            print(f"Error reading file {file.filename}: {e}")
            raise HTTPException(status_code=500, detail=f"Error reading file {file.filename}: {str(e)}")

    if not all_extracted_texts_from_files:
        raise HTTPException(status_code=400, detail="No readable text content found in uploaded files.")

    # --- NEW: Text Splitting for RAPTOR Base Chunks ---
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50) # Standard chunk sizes
    all_chunks = []
    for doc_text in all_extracted_texts_from_files:
        chunks = text_splitter.split_text(doc_text)
        all_chunks.extend(chunks)
    
    if not all_chunks:
        raise HTTPException(status_code=400, detail="No chunks generated from documents after splitting. Ensure documents are not empty.")

    print(f"DEBUG: Initial document(s) split into {len(all_chunks)} chunks for Level 1 clustering.")

    # --- RAPTOR Processing: Now feed the chunks to the recursive function ---
    print("Starting RAPTOR hierarchy generation...")
    # Await the top-level recursive call
    raptor_results = await recursive_embed_cluster_summarize(all_chunks, n_levels=N_LEVELS)
    print("RAPTOR hierarchy generation complete.")

    # Clear existing data before fresh ingest to prevent duplicates/old data
    await clear_chroma_db() # Ensure this is also awaited as it's an async FastAPI endpoint

    all_embeddings_flattened = []
    all_documents_flattened = []
    all_metadatas_flattened = []
    global global_id_counter

    print("Populating ChromaDB with flattened RAPTOR hierarchy...")
    for level, (df_clusters, df_summary) in raptor_results.items():
        # Store original chunks from the lowest level (level 1)
        if level == 1 and not df_clusters.empty:
            for idx, row in df_clusters.iterrows():
                # Ensure embedding is a list for ChromaDB
                embedding_to_add = row["embd"].tolist() if isinstance(row["embd"], np.ndarray) else row["embd"]
                all_embeddings_flattened.append(embedding_to_add)
                all_documents_flattened.append(row["text"])
                all_metadatas_flattened.append({
                    "type": "chunk",
                    "level": level,
                    "original_doc_idx": idx # This links back to original chunk index, useful for debugging
                })
        
        # Store summaries from all levels (including if level 1 somehow produced summaries for ingestion logic)
        # Note: Level 1 typically produces chunks. Summaries start from level 2 and above summarizing level 1 chunks.
        if not df_summary.empty:
            for idx, row in df_summary.iterrows():
                # Embed the summary text itself
                summary_embedding = embed([row["summaries"]])[0].tolist()
                all_embeddings_flattened.append(summary_embedding)
                all_documents_flattened.append(row["summaries"])
                all_metadatas_flattened.append({
                    "type": "summary",
                    "level": row["level"],
                    "cluster_id": row["cluster"] # This links summaries to their cluster
                })

    if all_documents_flattened:
        ids = [f"doc_flat_{global_id_counter + j}" for j in range(len(all_documents_flattened))]
        try:
            chroma_collection_flattened.add(
                embeddings=all_embeddings_flattened,
                documents=all_documents_flattened,
                metadatas=all_metadatas_flattened,
                ids=ids
            )
            print(f"Added {len(all_documents_flattened)} items (chunks and summaries) to flattened ChromaDB collection.")
            global_id_counter += len(all_documents_flattened)
        except Exception as e:
            print(f"Error adding to flattened ChromaDB: {e}")
            raise HTTPException(status_code=500, detail=f"Error adding documents to flattened ChromaDB: {str(e)}")
    else:
        print("No documents or summaries generated to add to flattened ChromaDB.")

    # Save RAPTOR results to a JSON file (optional, for inspection)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"raptor_results_{timestamp}.json"
    results_path = os.path.join(RAPTOR_DATA_FOLDER, results_filename)
    serialized_results = {}
    for level, (df_clusters, df_summary) in raptor_results.items():
        try:
            df_clusters_json_compatible = df_clusters.copy()
            df_clusters_json_compatible['embd'] = df_clusters_json_compatible['embd'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
            df_clusters_json_compatible['cluster'] = df_clusters_json_compatible['cluster'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
            serialized_clusters = df_clusters_json_compatible.replace({np.nan: None}).to_dict(orient="records")
            
            df_summary_json_compatible = df_summary.copy()
            serialized_summaries = df_summary_json_compatible.replace({np.nan: None}).to_dict(orient="records")
            serialized_results[level] = {"clusters": serialized_clusters, "summaries": serialized_summaries}
        except Exception as e:
            print(f"Error serializing level {level} results: {e}")

    try:
        with open(results_path, "w") as f:
            json.dump(serialized_results, f, indent=4)
        print(f"RAPTOR hierarchy saved to {results_path}")
    except Exception as e:
        print(f"Error saving results to file: {e}")

    return {"message": "Documents ingested, RAPTOR hierarchy built, and indexed in ChromaDB successfully for flattened retrieval.", "results_file": results_path, "total_indexed_items": len(all_documents_flattened)}




@app.post("/ask_flattened")
async def ask_question_flattened(request: QuestionRequest):
    """
    Answers a question by querying the flattened ChromaDB collection (all levels)
    and generating an answer based on the most relevant retrieved items.
    """
    if 'openai_ef' not in globals() or openai_ef is None or 'model' not in globals() or model is None:
        raise HTTPException(status_code=500, detail="Embedding or Language model not initialized.")

    query = request.query
    try:
        # Use embed_with_cache for query embedding as well
        query_embedding = embed_with_cache([query])[0] # embed_with_cache returns array of embeddings
        
        results = chroma_collection_flattened.query(
            query_embeddings=[query_embedding.tolist()], # ChromaDB expects list
            n_results=10, # Retrieve 10 most relevant items across all levels
            include=['documents', 'metadatas', 'distances']
        )
        
        retrieved_items = []
        if results and results.get('documents') and results.get('metadatas') and results.get('distances'):
            docs_list = results['documents'][0]
            metas_list = results['metadatas'][0]
            dists_list = results['distances'][0]

            for i in range(len(docs_list)):
                doc = docs_list[i]
                meta = metas_list[i]
                dist = dists_list[i]
                retrieved_items.append({"document": doc, "metadata": meta, "distance": dist})

        context_docs = [item["document"] for item in retrieved_items]
        
    except Exception as e:
        print(f"Error querying flattened ChromaDB: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving relevant documents from flattened index: {str(e)}")

    if not context_docs:
        return {"answer": "No relevant information found in the documents."}

    context = "\n\n---\n\n".join(context_docs)
    prompt_template = """You are a helpful assistant. Answer the question truthfully based on the provided context. If the answer is not in the context, say "I cannot answer based on the provided information."

    Context:
    {context}

    Question: {question}

    Answer: """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    answer_chain = prompt | model | StrOutputParser()

    try:
        # Use ainvoke for asynchronous LLM call
        answer = await answer_chain.ainvoke({"question": query, "context": context})
        return {"answer": answer, "retrieved_items_count": len(retrieved_items)}
    except Exception as e:
        print(f"Error generating answer with LLM: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")


@app.post("/ask_tree_traversal")
async def ask_question_tree_traversal(request: QuestionRequest):
    """
    Answers a question by traversing the RAPTOR hierarchy from the highest level down.
    Attempts to answer with a higher-level summary first before falling back to lower levels.
    """
    if 'openai_ef' not in globals() or openai_ef is None or 'model' not in globals() or model is None:
        raise HTTPException(status_code=500, detail="Embedding or Language model not initialized.")

    query = request.query
    final_context_docs = []
    retrieved_items_meta = []
    
    # Define the base prompt for answering
    answer_prompt_template = """You are a helpful assistant. Answer the question truthfully based on the provided context. If the answer is not in the context, say "I cannot answer based on the provided information."

    Context:
    {context}

    Question: {question}

    Answer: """
    answer_chain = ChatPromptTemplate.from_template(answer_prompt_template) | model | StrOutputParser()

    # Define a prompt for determining if a summary is sufficient
    sufficiency_prompt_template = """Given the question: "{question}" and the following summary/document: "{document}".
    Does this summary/document contain enough direct information to answer the question without needing more detail from underlying documents?
    Respond with 'YES' if it directly answers the question, 'NO' if it requires more detailed information from lower levels, or 'UNCERTAIN' if you are unsure.
    Response:"""
    sufficiency_chain = ChatPromptTemplate.from_template(sufficiency_prompt_template) | model | StrOutputParser()

    try:
        # Use embed_with_cache for query embedding
        query_embedding = embed_with_cache([query])[0] # embed_with_cache returns array of embeddings

        # Iterate from highest level down to 1
        for level in range(N_LEVELS, 0, -1):  
            print(f"Querying level {level}...")
            
            type_filter = "summary" if level > 1 else "chunk"
            
            num_results = 5 if level > 1 else 3 # More results for summaries, fewer for direct chunks initially
            
            where_clause = {"$and": [{"level": level}, {"type": type_filter}]}
            
            level_results = chroma_collection_flattened.query(
                query_embeddings=[query_embedding.tolist()], # ChromaDB expects list
                n_results=num_results,  
                where=where_clause,
                include=['documents', 'metadatas', 'distances']
            )
            
            level_docs = level_results.get('documents', [[]])[0]
            level_metadatas = level_results.get('metadatas', [[]])[0]
            level_distances = level_results.get('distances', [[]])[0]

            if not level_docs:
                print(f"No relevant documents found at level {level}.")
                continue

            print(f"Found {len(level_docs)} documents at level {level}.")

            # Try to answer directly with the current level's context
            current_level_context = "\n\n---\n\n".join(level_docs)
            
            # Asynchronously invoke the potential answer and sufficiency check
            potential_answer_task = answer_chain.ainvoke({"question": query, "context": current_level_context})
            sufficiency_check_task = sufficiency_chain.ainvoke({"question": query, "document": current_level_context}) # Check context, not just answer

            potential_answer, is_sufficient_response = await asyncio.gather(
                potential_answer_task, sufficiency_check_task, return_exceptions=True
            )

            if isinstance(potential_answer, Exception):
                print(f"Warning: Failed to generate potential answer at level {level}: {potential_answer}")
                potential_answer = "Error generating answer."
            
            if isinstance(is_sufficient_response, Exception):
                print(f"Warning: Failed sufficiency check at level {level}: {is_sufficient_response}")
                is_sufficient_response = "UNCERTAIN" # Default to uncertain if check fails

            print(f"Sufficiency check for Level {level} context: {is_sufficient_response}")

            if "YES" in is_sufficient_response.upper() and potential_answer != "I cannot answer based on the provided information.":
                final_context_docs = level_docs
                for i in range(len(level_docs)):
                    retrieved_items_meta.append({
                        "document": level_docs[i],
                        "metadata": level_metadatas[i],
                        "distance": level_distances[i]
                    })
                return {"answer": potential_answer, "retrieved_items_count": len(final_context_docs), "retrieved_from_level": level, "strategy": "Tree Traversal - Answered at higher level"}
            
            # If not sufficient at this level, we'll continue to the next lower level.
            # The current approach implicitly drills down by trying the next lower level.
            # A more sophisticated RAPTOR tree traversal would involve identifying the
            # specific children of the most relevant summary to retrieve. This would require
            # storing explicit parent-child links (e.g., child_ids) in ChromaDB metadata.
            # For this flattened structure, we proceed by trying the next level generally.

        # If after traversing all levels, no sufficient answer is found,
        # or if the loop completes without a 'YES' from the sufficiency check,
        # we provide the answer based on the best context found (which would be
        # the lowest level chunks if the highest levels weren't sufficient, or
        # an empty answer if nothing was found).
        if not final_context_docs and level_docs: # If no sufficient answer, but some documents were found at the last queried level
             final_context_docs = level_docs # Use the documents from the lowest level queried
             for i in range(len(level_docs)):
                retrieved_items_meta.append({
                    "document": level_docs[i],
                    "metadata": level_metadatas[i],
                    "distance": level_distances[i]
                })

        if final_context_docs:
            final_context = "\n\n---\n\n".join(final_context_docs)
            final_answer = await answer_chain.ainvoke({"question": query, "context": final_context})
            return {"answer": final_answer, "retrieved_items_count": len(final_context_docs), "strategy": "Tree Traversal - Consolidated lowest level"}
        else:
            return {"answer": "I cannot answer based on the provided information, as no relevant documents were found at any level.", "retrieved_items_count": 0, "strategy": "Tree Traversal - No documents found"}

    except Exception as e:
        print(f"Error during tree traversal: {e}")
        raise HTTPException(status_code=500, detail=f"Error during tree traversal: {str(e)}")