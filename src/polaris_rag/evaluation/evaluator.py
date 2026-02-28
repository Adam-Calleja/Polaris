"""
evaluator

Evaluation orchestration for ragas metrics.

This module provides classes to run a suite of Metric instances
over model outputs, collect per‐example scores, and generate summary reports.

Classes
-------
Evaluator
    Runs a suite of ragas metrics on RAG examples and aggregates results. 
"""
from ragas import EvaluationDataset, evaluate, RunConfig
from llama_index.llms.huggingface import HuggingFaceLLM
from os import environ
import torch
import pandas as pd
from typing import Optional
from ragas.metrics import Metric as RagasMetric
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from polaris_rag.evaluation.metrics import Metric
from polaris_rag.generation.llm_interface import BaseLLM
from polaris_rag.retrieval.embedder import BaseEmbedder, OpenAILikeEmbedder

class Evaluator():
    """
    Runs a suite of ragas metrics on RAG examples and aggregates results. 

    Methods
    -------
    register_metric
        Add or replace a metric by name. 
    list_metrics
        List all registered metric names. 
    evaluate
        Evaluate a set of samples across all registered metrics.
    """

    def __init__(
            self,
            *,
            metrics: list[Metric],
            llm: BaseLLM,
            embedder: BaseEmbedder,
            token: Optional[str] = None,
        ):
        """
        Initialise the evaluator with a list of Metric instances. 

        Parameters
        ----------
        metrics : list[Metric]
            Instances of Metric subclasses to apply.
        llm : BaseLLM
            The large language model to use. 
        embedder : BaseEmbedder,
            The embedding model to use for the metrics. 
        token : str
            HuggingFace API token to authenticate and run the LLM.
        """

        self.metrics = {m.name(): m.get_metric() for m in metrics}
        self.ragas_metrics = {m.get_metric().name: m.get_metric() for m in metrics}
        self.registered_metrics = {}

        if token:
            environ["HF_TOKEN"] = token

        self.llm = llm or HuggingFaceLLM(
            model_name="meta-llama/Llama-2-7b-chat-hf",
            tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
            device_map="auto",
            model_kwargs={
                "torch_dtype": torch.float16,
            },
            max_new_tokens=512,
            generate_kwargs={"temperature": 0.2, "top_p": 0.9},
        )

        self.embedder = embedder or OpenAILikeEmbedder(
            model_name="Qwen/Qwen3-Embedding-8B",
            api_base="http://localhost:8081/v1",
            callback_manager=None,
        )

    def register_metric(
            self,
            metric_name: str
        ) -> None:
        """
        Add or replace a metric selection by name.

        Parameters
        ----------
        metric_name : str
            The name of a metric available in `self.metrics` to activate for evaluation.
        """
        if metric_name not in self.metrics and metric_name not in self.ragas_metrics:
            raise KeyError(f"Metric '{metric_name}' is not available. The available metrics are: {sorted(self.metrics.keys())} or {sorted(self.ragas_metrics.keys())}")
        if metric_name in self.metrics:
            self.registered_metrics[metric_name] = self.metrics[metric_name]
        else:
            self.registered_metrics[metric_name] = self.ragas_metrics[metric_name]

    def clear_registered_metrics(self) -> None:
        """Clear the active/registered metric selection (revert to all metrics)."""
        self.registered_metrics.clear()

    def list_metric_names(self) -> list[str]:
        """
        List active metric names if any have been registered; otherwise list all available metrics.

        Returns
        -------
        list[str]
            Sorted list of metric names.
        """
        if self.registered_metrics:
            return sorted(self.registered_metrics.keys())
        return sorted(self.metrics.keys())
    
    def list_metrics(self) -> list[RagasMetric]:
        """
        Return the ragas metric instances that will be used for evaluation.
        If any metrics have been registered via `register_metric`, only those are returned;
        otherwise, all available metrics are returned.

        Returns
        -------
        list[RagasMetric]
            The active ragas metric instances.
        """
        source = self.registered_metrics if self.registered_metrics else self.metrics
        return [source[name] for name in sorted(source.keys())]
    
    def evaluate(
            self,
            dataset: EvaluationDataset,
            run_config: RunConfig = RunConfig(timeout=120,max_retries=3,max_wait=60,max_workers=1),
        ) -> pd.DataFrame:
        """
        Run the full evaluation loop on a dataset and return the results as a DataFrame.

        Parameters
        ----------
        dataset : EvaluationDataset
            Dataset wrapper providing inputs, references, and any metadata
            needed for evaluation.
        run_config : RunConfig
            Configuration parameters for the evaluation run, e.g.
            timeout, max_retries, max_wait, etc.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame where each row corresponds to one example from
            `dataset` and each column is either a metric score or other
            evaluation metadata (e.g. example ID, LLM latency, etc.).

        Raises
        ------
        Exception
            Propagates any errors from the underlying evaluation call
            (because `raise_exceptions=True`).
        """

        evaluator_llm = LangchainLLMWrapper(self.llm)
        embedder = LangchainEmbeddingsWrapper(self.embedder)
        callbacks = []

        score = evaluate(
            dataset=dataset,
            metrics=self.list_metrics(),
            llm=evaluator_llm,
            embeddings=embedder,
            raise_exceptions=False,
            callbacks=callbacks,
            run_config=run_config,
            show_progress=True,
        )
        
        return score.to_pandas()
