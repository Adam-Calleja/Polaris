"""
Retrieval layer of the RAG pipeline.

This package covers everything needed to turn raw sources into searchable
vectors and to fetch the most relevant chunks for a query. It includes
a document loader, text pre-processing/splitting utilities, embedding model
 wrappers, vector index backends, the high-level retriever interface, and 
 optional rerankers.

Submodules
----------
document_loader
    Loads documents from various sources.
preprocessors
    Source-specific cleaning and normalization routines applied prior to chunking.
text_splitter
    Functions/classes for chunking long documents into overlapping pieces.
embedder
    Embedding model wrappers and batching helpers.
index_store
    Vector store/index implementation
retriever
    High-level retrieval API orchestrating store lookups.
reranker
    (Optional) Re-ranking models to refine initial retrieval results.

Re-exports
----------
TextSplitter
    Primary interface for splitting documents into chunks.
Embedder
    Abstraction over embedding models.
IndexStore
    Vector index/store interface.
Retriever
    Main class/function to retrieve top-k relevant chunks.
Reranker
    Optional second-stage ranker.
"""