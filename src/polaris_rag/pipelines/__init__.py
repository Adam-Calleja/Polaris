"""polaris_rag.pipelines

Pipeline orchestration components for the Polaris RAG system.

This package contains high-level pipeline abstractions that coordinate
retrieval, prompt construction, and language model generation. Pipelines
are intentionally lightweight and stateless beyond their configured
components, making them safe to reuse across requests and execution
contexts.

Modules
-------
rag_pipeline
    End-to-end Retrieval-Augmented Generation (RAG) pipeline.
"""
