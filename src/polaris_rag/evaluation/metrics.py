"""
metrics

This module provides functionality for computing evaluation 
metrics on the retrieval-augmented-generation (RAG) system.
It defines a common Metric interface and concrete implementations
for a suite of RAGAS metrics including:

    - Metric
    - Context Precision
    - Context Recall
    - Context Entities Recall
    - Faithfulness
    - Response Relevancy
    - Factual Correctness
    - Semantic Similarity
    - Summarisation Score
    - Noise Sensitivity

Classes
-------
Metric
    Abstract Base Class for all ragas metrics. 
ContextPrecision
    Measures the proportion of relevant chunks in the retrieved 
    contexts. 
ContextRecall
    Measures the proportion of relevant chunks that were retrieved.
ContextEntitiesRecall
    A measure of the context recall based on the numbers of entities
    present in both the context and reference.
Faithfulness
    A metric that measures how factually consistent a response is 
    with the retrieved context.
ResponseRelevancy
    A metric that measures how relevant a response is to the user input.
FactualCorrectness
    A metric that compares and evaluates the factual accuracy of the 
    generated response with the reference.
SemanticSimilarity
    A metric that evaluates the semantic resemblance between the generated 
    answer and the ground truth.
SummarizationScore
    A metric that measures how well the summary captures the important 
    information from the retrieved contexts.
NoiseSensitivity
    A measure of how often a system makes errors by providing incorrect 
    responses when using either relevant or irrelevant retrieved documents.
"""
from abc import ABC, abstractmethod
from ragas.metrics import LLMContextPrecisionWithoutReference as RagasContextPrecision
from ragas.metrics import LLMContextRecall as RagasContextRecall
from ragas.metrics import ContextEntityRecall as RagasContextEntityRecall
from ragas.metrics import Faithfulness as RagasFaithfulness
from ragas.metrics import ResponseRelevancy as RagasResponseRelevancy
from ragas.metrics import FactualCorrectness as RagasFactualCorrectness
from ragas.metrics import SemanticSimilarity as RagasSemanticSimilarity
from ragas.metrics import SummarizationScore as RagasSummarizationScore
from ragas.metrics import NoiseSensitivity as RagasNoiseSensitivity

TEXT_ONLY_RULES = (
    "\n\nOutput format constraints (strict):\n"
    "- Place the ENTIRE answer strictly between the markers <ANS> and </ANS>.\n"
    "- Do NOT put ANY text before <ANS> or after </ANS>.\n"
    "- Return plain text only. Never respond in JSON, XML, YAML, lists with bullets, or Markdown code fences.\n"
    "- Do not include labels/keys like reason, verdict, entities, score.\n"
    "- Do not use brackets/braces/quotes around the answer.\n"
    "- Do not add commentary, apologies, or meta-instructions.\n"
    "- No leading or trailing whitespace inside the markers.\n"
    "- If the task requires a binary decision, output EXACTLY one character: 1 (yes/true) or 0 (no/false).\n"
    "- If the task requires listing items/entities, output ONE item per line with no bullets or numbering.\n"
    "- If an explanation is explicitly requested, put it on NEW lines after the first required line, as plain sentences.\n"
    "\nExamples (do not include the word 'Example' in outputs):\n"
    "Binary decision only:\n<ANS>\n1\n</ANS>\n"
    "\nBinary decision + explanation (only if requested):\n<ANS>\n0\nThe context lacks a time limit directive.\n</ANS>\n"
    "\nList of entities/items (one per line):\n<ANS>\nslurm\nMYPROJECT-CPU\n00:10:00\n</ANS>\n"
)

class Metric(ABC):
    """
    Abstract Base Class for all ragas metrics. 

    All metrics must inherit from this and implement the following methods:
        - name()
        - compute()

    Methods
    -------
    name
        Returns the metrics name. 
    """

    @abstractmethod
    def get_metric(self):
        """
        Return the ragas Metric instance.
        """

    @abstractmethod
    def name(self) -> str:
        """
        Returns the unique name of the metric

        Returns
        -------
        str
            The metric's name
        """

def _append_text_only_rules(metric, include_keys=None, exclude_keys=None):
    """
    Prepend TEXT_ONLY_RULES to the `.instruction` of selected prompts on a RAGAS metric.

    Parameters
    ----------
    metric : object
        RAGAS metric instance exposing get_prompts()/set_prompts().
    include_keys : Optional[Iterable[str]]
        If provided, only prompts whose keys are in this set will be modified.
    exclude_keys : Optional[Iterable[str]]
        If provided, prompts whose keys are in this set will be skipped.
    """
    prompts = metric.get_prompts()
    for key, p in prompts.items():
        if include_keys is not None and key not in include_keys:
            continue
        if exclude_keys is not None and key in exclude_keys:
            continue
        instruction = getattr(p, "instruction", None)
        if instruction is not None and TEXT_ONLY_RULES not in instruction:
            p.instruction = TEXT_ONLY_RULES + instruction
    metric.set_prompts(**prompts)

class ContextPrecision(Metric):
    """
    Measures the proportion of relevant chunks in the retrieved contexts.

    I.e., How many of the chunks in the retrieved contexts are relevant. 

    Methods
    -------
    name
        Returns "context_precision"
    """
    def get_metric(self):
        """Return the ragas ContextPrecision metric instance."""
        return RagasContextPrecision()

    def name(self) -> str:
        """
        Returns "context_precision"

        Returns
        -------
        str
            "context_precision"
        """

        return "context_precision"

class ContextRecall(Metric):
    """
    Measures the proportion of relevant chunks that were retrieved.

    I.e., how many of the relevant chunks in the vector database were
          actually retrieved.

    Methods
    -------
    name
        Returns "context_recall"
    """
    def get_metric(self):
        """Return the ragas ContextRecall metric instance."""
        return RagasContextRecall()

    def name(self) -> str:
        """
        Returns "context_recall"

        Returns
        -------
        str
            "context_recall"
        """

        return "context_recall"
    
class ContextEntityRecall(Metric):
    """
    A measure of the context recall based on the numbers of entities
    present in both the context and reference.

    Methods
    -------
    name
        Returns "context_entity_recall"
    """
    def get_metric(self):
        """Return the ragas ContextEntityRecall metric instance."""
        return RagasContextEntityRecall()

    def name(self) -> str:
        """
        Returns "context_entity_recall"

        Returns
        -------
        str
            "context_entity_recall"
        """

        return "context_entity_recall"

class Faithfulness(Metric):
    """
    A metric that measures how factually consistent a response is 
    with the retrieved context.

    Methods
    -------
    name
        Returns "faithfulness"
    """
    def get_metric(self):
        """Return a configured ragas Faithfulness metric instance."""
        metric= RagasFaithfulness()
        _append_text_only_rules(metric, include_keys={"statement_generator_prompt", "n_l_i_statement_prompt"})
        return metric

    def name(self) -> str:
        """
        Returns "faithfulness"

        Returns
        -------
        str
            "faithfulness"
        """

        return "faithfulness"
    
class ResponseRelevancy(Metric):
    """
    A metric that measures how relevant a response is to the user input.

    Methods
    -------
    name
        Returns "response_relevancy"
    """
    def get_metric(self):
        """Return the ragas ResponseRelevancy metric instance."""
        return RagasResponseRelevancy()

    def name(self) -> str:
        """
        Returns "response_relevancy"

        Returns
        -------
        str
            "response_relevancy"
        """

        return "response_relevancy"
    
class FactualCorrectness(Metric):
    """
    A metric that compares and evaluates the factual accuracy of the 
    generated response with the reference.

    Methods
    -------
    name
        Returns "factual_correctness"
    """
    def get_metric(self):
        """Return a configured ragas FactualCorrectness metric instance."""
        metric= RagasFactualCorrectness(coverage="high", atomicity="high")
        _append_text_only_rules(metric, include_keys={"claim_decomposition_prompt", "n_l_i_statement_prompt"})
        return metric

    def name(self) -> str:
        """
        Returns "factual_correctness"

        Returns
        -------
        str
            "factual_correctness"
        """

        return "factual_correctness"
    
class SemanticSimilarity(Metric):
    """
    A metric that evaluates the semantic resemblance between the generated 
    answer and the ground truth.

    Methods
    -------
    name
        Returns "semantic_similarity"
    """
    def get_metric(self):
        """Return the ragas SemanticSimilarity metric instance."""
        return RagasSemanticSimilarity()

    def name(self) -> str:
        """
        Returns "semantic_similarity"

        Returns
        -------
        str
            "semantic_similarity"
        """

        return "semantic_similarity"

class SummarizationScore(Metric):
    """
    A metric that measures how well the summary captures the important 
    information from the retrieved contexts.

    Methods
    -------
    name
        Returns "summarization_score"
    """
    def get_metric(self):
        """Return the ragas SummarizationScore metric instance."""
        return RagasSummarizationScore()

    def name(self) -> str:
        """
        Returns "summarization_score"

        Returns
        -------
        str
            "summarization_score"
        """

        return "summarization_score" 
    
class NoiseSensitivity(Metric):
    """
    A measure of how often a system makes errors by providing incorrect 
    responses when using either relevant or irrelevant retrieved documents.

    Methods
    -------
    name
        Returns "noise_sensitivity"
    """
    def get_metric(self):
        """Return a configured ragas NoiseSensitivity metric instance."""
        metric= RagasNoiseSensitivity()
        _append_text_only_rules(metric, include_keys={"statement_generator_prompt", "n_l_i_statement_prompt"})
        return metric

    def name(self) -> str:
        """
        Returns "noise_sensitivity"

        Returns
        -------
        str
            "noise_sensitivity"
        """

        return "noise_sensitivity"