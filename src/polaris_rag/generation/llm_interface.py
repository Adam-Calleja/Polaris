"""polaris_rag.generation.llm_interface

from __future__ import annotations

Unified interface and factory for large language model (LLM) backends.

This module defines a small, provider-agnostic abstraction for text
generation and concrete implementations backed by LangChain-compatible LLM
wrappers. A factory function is provided to instantiate the appropriate LLM
implementation from a configuration mapping.

Classes
-------
BaseLLM
    Abstract interface specifying the API used by the Polaris RAG pipeline.
OpenAILikeLLM
    Text generation using an OpenAI-compatible HTTP API via LangChain.
OpenAIChatLikeLLM
    Chat completions using an OpenAI-compatible HTTP API via LangChain.
HuggingFaceTGI
    Text generation using Hugging Face Text Generation Inference (TGI) via LangChain.

Functions
---------
create_llm
    Construct an LLM implementation from a configuration mapping.
"""

from abc import ABC, abstractmethod
from typing import Any
from typing import Mapping, Optional
import warnings
import yaml
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import OpenAI, ChatOpenAI
from langchain_core.language_models.llms import LLM as LangChainBaseLLM
from langchain_community.llms import HuggingFaceTextGenInference


def _is_gemini_openai_compat(api_base: str | None) -> bool:
    """Return ``True`` when the API base points to Gemini's OpenAI-compatible endpoint."""
    if not api_base:
        return False
    base = api_base.lower()
    return "generativelanguage.googleapis.com" in base and "/openai" in base


def _sanitize_openai_kwargs(
    api_base: str | None,
    kwargs: dict[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    """Drop provider-incompatible OpenAI kwargs for known OpenAI-compatible backends."""
    sanitized = dict(kwargs)

    if _is_gemini_openai_compat(api_base):
        unsupported_keys = {"frequency_penalty", "presence_penalty"}
        removed = sorted(k for k in unsupported_keys if k in sanitized)
        for key in removed:
            sanitized.pop(key, None)
        if removed:
            warnings.warn(
                "Dropping unsupported Gemini OpenAI-compatible params "
                f"during {context}: {', '.join(removed)}",
                UserWarning,
            )

    return sanitized

class _SimpleGeneration:
    """Minimal generation object compatible with LangChain result structures.

    This class exists to satisfy downstream access patterns that expect a
    ``.text`` attribute on generation items.

    Attributes
    ----------
    text : str
        Generated text.
    generation_info : Any or None
        Optional provider-specific metadata. Always ``None`` for this shim.
    """
    def __init__(
            self, 
            text: str
        ):
        """Initialise a _SimpleGeneration instance.

        Parameters
        ----------
        text : str
            Generated text.
        """
        self.text = text
        self.generation_info = None

class _SimpleResult:
    """Minimal result object compatible with LangChain result structures.

    This class provides a ``generations`` attribute shaped like LangChain's
    ``LLMResult.generations`` (a list of lists), enabling reuse of existing
    code that expects that structure.

    Attributes
    ----------
    generations : list[list[_SimpleGeneration]]
        Nested list of generation objects, one list per prompt.
    """
    def __init__(
            self, 
            generations: list[list[_SimpleGeneration]]
        ):
        """Initialise a _SimpleResult instance.

        Parameters
        ----------
        generations : list[list[_SimpleGeneration]]
            Nested list of generations, one list per prompt.
        """
        self.generations = generations

    def flatten(self) -> list["_SimpleResult"]:
        """Return this result wrapped in a list for compatibility.

        Returns
        -------
        list[_SimpleResult]
            A one-element list containing this instance (i.e., ``[self]``).

        Notes
        -----
        This method does not modify ``self.generations``; it only wraps the
        current instance in a list.
        """
        return [self]


class BaseLLM(ABC):
    """Abstract interface for LLM text generation.

    Concrete implementations wrap provider-specific clients and expose a
    small, consistent API used by the Polaris RAG pipeline.
    """
    @classmethod
    @abstractmethod
    def from_config(
            cls, 
            config_path: str, 
            callback_manager: BaseCallbackHandler = None
        ):
        """Create an LLM instance from a YAML configuration file.

        Parameters
        ----------
        config_path : str
            Path to the YAML configuration file.
        callback_manager : BaseCallbackHandler, optional
            Optional callback handler for logging/telemetry/streaming.

        Returns
        -------
        BaseLLM
            An initialised LLM implementation.

        Raises
        ------
        FileNotFoundError
            If ``config_path`` does not exist.
        ValueError
            If the configuration is invalid for the concrete implementation.
        """
        pass

    @classmethod
    @abstractmethod
    def from_config_dict(
            cls,
            config: dict,
            callback_manager: BaseCallbackHandler = None
        ):
        """Create an LLM instance from a configuration mapping.

        Parameters
        ----------
        config : dict
            Configuration parameters for the concrete implementation.
        callback_manager : BaseCallbackHandler, optional
            Optional callback handler for logging/telemetry/streaming.

        Returns
        -------
        BaseLLM
            An initialised LLM implementation.

        Raises
        ------
        ValueError
            If required configuration keys are missing or invalid.
        """
        pass

    @abstractmethod
    def get_llm(self) -> LangChainBaseLLM:
        """Return the underlying LangChain LLM object.

        Returns
        -------
        LangChainBaseLLM
            The wrapped LangChain-compatible LLM instance.
        """
        pass

    @abstractmethod
    def generate(
            self, 
            prompt: str, 
            **kwargs
        ) -> Any:
        """Generate text for a single prompt.

        Parameters
        ----------
        prompt : str
            Prompt text to send to the model.
        **kwargs
            Additional keyword arguments forwarded to the underlying model.

        Returns
        -------
        Any
            Provider-specific generation result (often a ``str``).
        """
        pass

class OpenAILikeLLM(BaseLLM):
    """LLM interface using an OpenAI-compatible API.

    This implementation wraps :class:`langchain_openai.OpenAI` and exposes a
    small synchronous/async surface used throughout the codebase.
    """

    def __init__(
        self,
        model_name: str,
        api_base: str,
        api_key: str = "fake",
        callback_manager: BaseCallbackHandler = None,
        **model_kwargs: Any,
    ):
        """Initialise an OpenAI-compatible LLM wrapper.

        Parameters
        ----------
        model_name : str
            Model identifier (e.g., ``"gpt-4"`` or a local model name).
        api_base : str
            Base URL for the OpenAI-compatible API endpoint.
        api_key : str, optional
            API key value. Defaults to ``"fake"`` for local deployments that do
            not require authentication.
        callback_manager : BaseCallbackHandler, optional
            Optional callback handler for logging/telemetry/streaming.
        **model_kwargs : Any
            Additional keyword arguments forwarded to the underlying LangChain
            OpenAI wrapper (e.g., ``temperature``, ``top_p``).

        Notes
        -----
        Stop sequences may be supplied at call time via ``stop`` or ``stop_list``.
        If none are provided, a conservative default of ``["User:"]`` is used.
        """
        self.api_base = api_base
        model_kwargs = _sanitize_openai_kwargs(api_base, model_kwargs, context="model init")
        self.model_kwargs = dict(model_kwargs)
        self.default_stop_list = model_kwargs.pop("stop_list", None)
        self.stop_list = self.default_stop_list

        top_p = model_kwargs.pop("top_p", None)
        if top_p is not None:
            try:
                top_p_val = float(top_p)
            except (TypeError, ValueError):
                top_p_val = None
            else:
                if not (0.0 < top_p_val < 1.0):
                    top_p_val = None
            top_p = top_p_val

        self.llm = OpenAI(
            model_name=model_name,
            openai_api_base=api_base,
            openai_api_key=api_key,
            top_p=top_p or 1,
            **model_kwargs,
        )

    def get_stop_list(self) -> list[str] | None:
        """Return the configured default stop sequences.

        Returns
        -------
        list[str] or None
            Default stop sequences, or ``None`` if not configured.
        """
        return self.default_stop_list

    @classmethod
    def from_config(
        cls, 
        config_path: str, 
        callback_manager: BaseCallbackHandler = None
    ):
        """Create an :class:`~polaris_rag.generation.llm_interface.OpenAILikeLLM` from YAML.

        Parameters
        ----------
        config_path : str
            Path to the YAML configuration file.
        callback_manager : BaseCallbackHandler, optional
            Optional callback handler for logging/telemetry/streaming.

        Returns
        -------
        OpenAILikeLLM
            An initialised instance.

        Notes
        -----
        The YAML is expected to contain ``model_name`` and ``api_base`` keys, plus
        optional ``api_key`` and ``model_kwargs``.
        """
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        return cls.from_config_dict(cfg, callback_manager)

    @classmethod
    def from_config_dict(
            cls,
            config: dict,
            callback_manager: BaseCallbackHandler = None
        ) -> "OpenAILikeLLM":
        """Create an :class:`~polaris_rag.generation.llm_interface.OpenAILikeLLM` from a mapping.

        Parameters
        ----------
        config : dict
            Configuration mapping.
        callback_manager : BaseCallbackHandler, optional
            Optional callback handler for logging/telemetry/streaming.

        Returns
        -------
        OpenAILikeLLM
            An initialised instance.

        Raises
        ------
        ValueError
            If required keys (e.g., ``model_name`` or ``api_base``) are missing.
        """
        model_name = config.get('model_name')
        api_base = config.get('api_base')
        model_kwargs = config.get('model_kwargs', {})
        api_key = config.get('api_key', None)
        return cls(
            model_name=model_name,
            api_base=api_base,
            api_key=api_key,
            callback_manager=callback_manager,
            **model_kwargs,
        )

    def get_llm(self) -> LangChainBaseLLM:
        """Return the underlying LangChain LLM object.

        Returns
        -------
        LangChainBaseLLM
            The wrapped :class:`langchain_openai.OpenAI` instance.
        """
        return self.llm

    def generate(
            self, 
            prompt: str, 
            **kwargs
        ) -> Any:
        """Generate text for a single prompt.

        Parameters
        ----------
        prompt : str
            Prompt text to send to the model.
        **kwargs
            Additional generation parameters forwarded to the underlying LLM.

        Returns
        -------
        str
            Generated text for the prompt.

        Notes
        -----
        Stop sequences are resolved in the following order: explicit ``stop``,
        per-call ``stop_list``, instance default stop list, then a fallback of
        ``["User:"]``.
        """
        if not isinstance(prompt, str):
            prompt = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)

        run_kwargs = _sanitize_openai_kwargs(self.api_base, kwargs, context="generation")

        explicit_stop = run_kwargs.pop("stop", None)
        alt_stop_list = run_kwargs.pop("stop_list", None)
        final_stop = explicit_stop or alt_stop_list or self.default_stop_list or ["User:"]

        response = self.llm.generate([prompt], stop=final_stop, **run_kwargs)
        return response.generations[0][0].text

    def generate_prompt(
            self, 
            prompt: str, 
            **kwargs
        ) -> Any:
        """Generate text for a single prompt.

        This is an alias for :meth:`~polaris_rag.generation.llm_interface.OpenAILikeLLM.generate`
        kept for compatibility with callers expecting a ``generate_prompt`` API.

        Parameters
        ----------
        prompt : str
            Prompt text to send to the model.
        **kwargs
            Additional generation parameters forwarded to :meth:`generate`.

        Returns
        -------
        str
            Generated text for the prompt.
        """
        return self.generate(prompt, **kwargs)

    async def acomplete(
            self, 
            prompt: str, 
            **kwargs
        ) -> Any:
        """Asynchronously generate text for a single prompt.

        Parameters
        ----------
        prompt : str
            Prompt text to send to the model.
        **kwargs
            Additional generation parameters forwarded to the underlying LLM.

        Returns
        -------
        Any
            Provider-specific generation result (often a ``str``).

        Notes
        -----
        If the underlying client does not expose an async API, synchronous
        generation is executed in a thread pool via ``run_in_executor``.
        """

        run_kwargs = dict(self.run_config) if hasattr(self, 'run_config') else {}

        if not isinstance(prompt, str):
            prompt = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)

        run_kwargs.update(kwargs)
        run_kwargs = _sanitize_openai_kwargs(self.api_base, run_kwargs, context="generation")

        if hasattr(self.llm, 'acomplete'):
            return await self.llm.acomplete(prompt, **run_kwargs)
        else:
            import asyncio
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.llm.generate([prompt], **run_kwargs).generations[0][0].text
            )

    async def agenerate_prompt(
            self, 
            prompts: list[str], 
            **kwargs
        ) -> list[Any]:
        """Asynchronously generate text for a batch of prompts.

        Parameters
        ----------
        prompts : list[str]
            Prompt texts to send to the model.
        **kwargs
            Additional generation parameters forwarded to the underlying LLM.

        Returns
        -------
        _SimpleResult
            LangChain-compatible result object containing a generation per prompt.

        Notes
        -----
        Callback-related keyword arguments are stripped since callbacks are handled
        at initialisation.
        """
        kwargs.pop("callbacks", None)
        kwargs.pop("callback_manager", None)

        gen_results: list[list[_SimpleGeneration]] = []

        for p in prompts:
            text = await self.acomplete(p, **kwargs)
            gen_results.append([_SimpleGeneration(text)])
            
        return _SimpleResult(gen_results)

class OpenAIChatLikeLLM(BaseLLM):
    """LLM interface using an OpenAI-compatible Chat Completions API via LangChain.

    This implementation wraps :class:`langchain_openai.ChatOpenAI`.
    """

    def __init__(
        self,
        model_name: str,
        api_base: str,
        api_key: str = "fake",
        callback_manager: BaseCallbackHandler = None,
        **model_kwargs: Any,
    ):
        """Initialise an OpenAI-compatible chat LLM wrapper.

        Parameters
        ----------
        model_name : str
            Model identifier (e.g., ``"llama-3.3-70b-versatile"``).
        api_base : str
            Base URL for the OpenAI-compatible API endpoint.
        api_key : str, optional
            API key value. Defaults to ``"fake"`` for local deployments that do
            not require authentication.
        callback_manager : BaseCallbackHandler, optional
            Optional callback handler for logging/telemetry/streaming.
        **model_kwargs : Any
            Additional keyword arguments forwarded to the underlying LangChain
            ChatOpenAI wrapper (e.g., ``temperature``, ``top_p``).
        """
        import inspect

        self.api_base = api_base
        model_kwargs = _sanitize_openai_kwargs(api_base, model_kwargs, context="model init")
        self.model_kwargs = dict(model_kwargs)
        self.default_stop_list = model_kwargs.pop("stop_list", None)
        self.stop_list = self.default_stop_list

        top_p = model_kwargs.pop("top_p", None)
        if top_p is not None:
            try:
                top_p_val = float(top_p)
            except (TypeError, ValueError):
                top_p_val = None
            else:
                if not (0.0 < top_p_val < 1.0):
                    top_p_val = None
            top_p = top_p_val

        sig = inspect.signature(ChatOpenAI)
        init_kwargs: dict[str, Any] = dict(model_kwargs)

        if "model" in sig.parameters:
            init_kwargs["model"] = model_name
        else:
            init_kwargs["model_name"] = model_name

        if "openai_api_base" in sig.parameters:
            init_kwargs["openai_api_base"] = api_base
        elif "base_url" in sig.parameters:
            init_kwargs["base_url"] = api_base
        else:
            init_kwargs["api_base"] = api_base

        if api_key is not None:
            if "openai_api_key" in sig.parameters:
                init_kwargs["openai_api_key"] = api_key
            else:
                init_kwargs["api_key"] = api_key

        if top_p is not None:
            init_kwargs["top_p"] = top_p

        self.llm = ChatOpenAI(**init_kwargs)

    def get_stop_list(self) -> list[str] | None:
        """Return the configured default stop sequences."""
        return self.default_stop_list

    @classmethod
    def from_config(
        cls,
        config_path: str,
        callback_manager: BaseCallbackHandler = None,
    ):
        """Create an OpenAI-compatible chat LLM from YAML."""
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        return cls.from_config_dict(cfg, callback_manager)

    @classmethod
    def from_config_dict(
        cls,
        config: dict,
        callback_manager: BaseCallbackHandler = None,
    ) -> "OpenAIChatLikeLLM":
        """Create an OpenAI-compatible chat LLM from a mapping."""
        model_name = config.get('model_name')
        api_base = config.get('api_base')
        model_kwargs = config.get('model_kwargs', {})
        api_key = config.get('api_key', None)
        return cls(
            model_name=model_name,
            api_base=api_base,
            api_key=api_key,
            callback_manager=callback_manager,
            **model_kwargs,
        )

    def get_llm(self) -> Any:
        """Return the underlying LangChain chat model object."""
        return self.llm

    def generate(self, prompt: str, **kwargs) -> Any:
        """Generate text for a single prompt."""
        if not isinstance(prompt, str):
            prompt = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)

        run_kwargs = _sanitize_openai_kwargs(self.api_base, kwargs, context="generation")

        explicit_stop = run_kwargs.pop("stop", None)
        alt_stop_list = run_kwargs.pop("stop_list", None)
        final_stop = explicit_stop or alt_stop_list or self.default_stop_list or ["User:"]

        response = self.llm.invoke(prompt, stop=final_stop, **run_kwargs)
        return response.content if hasattr(response, "content") else response

    def generate_prompt(self, prompt: str, **kwargs) -> Any:
        """Alias for ``generate`` kept for compatibility."""
        return self.generate(prompt, **kwargs)

    async def acomplete(self, prompt: str, **kwargs) -> Any:
        """Asynchronously generate text for a single prompt."""
        run_kwargs = dict(self.run_config) if hasattr(self, 'run_config') else {}

        if not isinstance(prompt, str):
            prompt = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)

        explicit_stop = kwargs.pop("stop", None)
        alt_stop_list = kwargs.pop("stop_list", None)
        final_stop = explicit_stop or alt_stop_list or self.default_stop_list or ["User:"]

        run_kwargs.update(kwargs)
        run_kwargs = _sanitize_openai_kwargs(self.api_base, run_kwargs, context="generation")

        if hasattr(self.llm, 'ainvoke'):
            response = await self.llm.ainvoke(prompt, stop=final_stop, **run_kwargs)
            return response.content if hasattr(response, "content") else response
        else:
            import asyncio
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.generate(prompt, stop=final_stop, **run_kwargs)
            )

    async def agenerate_prompt(self, prompts: list[str], **kwargs) -> list[Any]:
        """Asynchronously generate text for a batch of prompts."""
        kwargs.pop("callbacks", None)
        kwargs.pop("callback_manager", None)

        gen_results: list[list[_SimpleGeneration]] = []

        for p in prompts:
            text = await self.acomplete(p, **kwargs)
            gen_results.append([_SimpleGeneration(text)])

        return _SimpleResult(gen_results)

class HuggingFaceTGI(BaseLLM):
    """
    An LLM interface using Hugging Face Text Generation Inference (TGI).
    """

    def __init__(
        self,
        inference_server_url: str,
        callback_manager: BaseCallbackHandler = None,
        stop_sequences: list[str] | None = None,
        temperature: float | None = None,
        repetition_penalty: float | None = None,
        max_new_tokens: int | None = None,
        **model_kwargs: Any,
    ):
        """
        Initialize the Hugging Face TGI LLM interface.

        Parameters
        ----------
        inference_server_url : str
            Base URL of the TGI server, for example ``"http://localhost:8080"``. Must not include a path.
        callback_manager : BaseCallbackHandler, optional
            Callback manager for logging/telemetry. If provided, it is passed to the underlying LangChain LLM.
        stop_sequences : list of str or None, optional
            List of strings that, if generated, will cause decoding to stop. If ``None``, no stop sequences are enforced unless provided at call time.
        temperature : float or None, optional
            Sampling temperature. Higher values (e.g., 1.0) produce more random outputs; lower values (e.g., 0.3) are more deterministic. If ``None``, the backend default is used.
        repetition_penalty : float or None, optional
            Penalty applied to previously generated tokens. Values > 1 discourage repetition; 1.0 disables the penalty. If ``None``, the backend default is used.
        max_new_tokens : int or None, optional
            Maximum number of tokens to generate beyond the prompt. If ``None``, the backend default is used.
        **model_kwargs
            Additional generation parameters forwarded to the TGI backend (e.g., ``top_p``, ``top_k``, ``do_sample``, ``typical_p``).

        Notes
        -----
        This class wraps ``langchain_community.llms.HuggingFaceTextGenInference`` and forwards supported parameters to the TGI server.
        """
        self.llm = HuggingFaceTextGenInference(
            inference_server_url=inference_server_url,
            callbacks=[callback_manager] if callback_manager else None,
            stop_sequences=stop_sequences,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            **model_kwargs
        )

        self._default_stop_sequences = stop_sequences

    def get_stop_list(self) -> list[str] | None:
        """Return the configured default stop sequences.

        Returns
        -------
        list[str] or None
            Default stop sequences, or ``None`` if not configured.
        """
        return self._default_stop_sequences

    @classmethod
    def from_config(cls, config_path: str, callback_manager: BaseCallbackHandler = None):
        """
        Create a ``HuggingFaceTGI`` instance from a YAML configuration file.

        Parameters
        ----------
        config_path : str
            Path to the YAML configuration file.
        callback_manager : BaseCallbackHandler, optional
            Optional callback manager to attach to the LLM.

        Returns
        -------
        HuggingFaceTGI
            An initialized instance configured from the provided file.

        Raises
        ------
        FileNotFoundError
            If ``config_path`` does not exist.
        ValueError
            If required fields are missing in the configuration.
        """
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        return cls.from_config_dict(cfg, callback_manager)

    @classmethod
    def from_config_dict(cls, config: dict, callback_manager: BaseCallbackHandler = None) -> "HuggingFaceTGI":
        """
        Create a ``HuggingFaceTGI`` instance from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration mapping. Expected keys:

            - ``inference_server_url`` (str): Base URL of the TGI server.
            - ``model_kwargs`` (dict, optional): Additional generation parameters.
            - ``stop_sequences`` (list[str], optional)
            - ``temperature`` (float, optional)
            - ``repetition_penalty`` (float, optional)
            - ``max_new_tokens`` (int, optional)
        callback_manager : BaseCallbackHandler, optional
            Optional callback manager to attach to the LLM.

        Returns
        -------
        HuggingFaceTGI
            An initialized instance configured from the provided dictionary.

        Raises
        ------
        ValueError
            If ``inference_server_url`` is missing or empty.
        """
        inference_server_url = config.get('inference_server_url')
        model_kwargs = dict(config.get('model_kwargs', {}))
        stop_sequences = config.get('stop_sequences', None)
        temperature = config.get('temperature', None)
        repetition_penalty = config.get('repetition_penalty', None)
        max_new_tokens = config.get('max_new_tokens', None)

        return cls(
            inference_server_url=inference_server_url,
            callback_manager=callback_manager,
            stop_sequences=stop_sequences,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            **model_kwargs
        )

    def get_llm(self) -> LangChainBaseLLM:
        """
        Return the underlying LangChain LLM object.

        Returns
        -------
        langchain.llms.base.LLM
            The wrapped ``HuggingFaceTextGenInference`` instance.
        """
        return self.llm

    def generate(self, prompt: str, **kwargs) -> Any:
        """
        Generate text for a single prompt synchronously.

        Parameters
        ----------
        prompt : str
            The input prompt string.
        **kwargs
            Additional generation parameters that override instance defaults for this call.

        Returns
        -------
        str
            The generated text.

        Notes
        -----
        ``kwargs`` are passed through to the underlying LLM. Any incompatible keys will raise at the client level.
        """
        if not isinstance(prompt, str):
            prompt = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
        response = self.llm.generate([prompt], **kwargs)
        return response.generations[0][0].text

    def generate_prompt(self, prompt: str, **kwargs) -> Any:
        """
        Alias for ``generate`` kept for compatibility with callers expecting a ``generate_prompt`` API.

        Parameters
        ----------
        prompt : str
            The input prompt string.
        **kwargs
            Additional generation parameters forwarded to :meth:`generate`.

        Returns
        -------
        str
            The generated text.
        """
        return self.generate(prompt, **kwargs)

    async def acomplete(self, prompt: str, **kwargs) -> Any:
        """
        Asynchronously generate text for a single prompt.

        Parameters
        ----------
        prompt : str
            The input prompt string.
        **kwargs
            Additional generation parameters that override instance defaults for this call.

        Returns
        -------
        str
            The generated text.

        Notes
        -----
        This implementation delegates synchronous generation to a background executor to provide an async interface.
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate(prompt, **kwargs)
        )

    async def agenerate_prompt(self, prompts: list[str], **kwargs) -> list[Any]:
        """
        Asynchronously generate text for a batch of prompts.

        Parameters
        ----------
        prompts : list of str
            A list of prompt strings to generate for.
        **kwargs
            Additional generation parameters applied to each prompt.

        Returns
        -------
        _SimpleResult
            A lightweight LangChain-compatible result object containing generations for each prompt.

        Notes
        -----
        Callback-related kwargs are stripped since callbacks are handled at initialization.
        """
        kwargs.pop("callbacks", None)
        kwargs.pop("callback_manager", None)

        gen_results: list[list[_SimpleGeneration]] = []

        for p in prompts:
            text = await self.acomplete(p, **kwargs)
            gen_results.append([_SimpleGeneration(text)])

        return _SimpleResult(gen_results)


# ----------------- Factory helpers -----------------

def _get_llm_kind(cfg: Mapping[str, Any]) -> str:
    """Extract the LLM kind/type/provider discriminator from a config mapping.

    Parameters
    ----------
    cfg : Mapping[str, Any]
        Configuration mapping.

    Returns
    -------
    str
        The first non-empty discriminator value found, or an empty string if none
        is present.
    """
    for key in ("kind", "type", "provider", "backend", "impl"):
        val = cfg.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _normalize_llm_kind(kind: str) -> str:
    """Normalise an LLM kind/type string to a stable registry key.

    Parameters
    ----------
    kind : str
        Provider/type discriminator value.

    Returns
    -------
    str
        Normalised registry key (e.g., ``"OpenAILike"`` -> ``"openai_like"``).

    Notes
    -----
    The normalisation process:
    - converts CamelCase to snake_case
    - replaces whitespace and hyphens with underscores
    - collapses repeated underscores
    - applies a small set of provider-specific aliases
    """
    k = kind.strip()
    if not k:
        return ""

    out: list[str] = []
    prev = ""
    for ch in k:
        if prev and prev.islower() and ch.isupper():
            out.append("_")
        out.append(ch)
        prev = ch

    k2 = "".join(out)
    k2 = k2.replace("-", "_").replace(" ", "_")

    while "__" in k2:
        k2 = k2.replace("__", "_")

    k2 = k2.lower()

    k2 = k2.replace("openailike", "openai_like")
    k2 = k2.replace("open_ailike", "openai_like")
    k2 = k2.replace("open_ai_like", "openai_like")
    k2 = k2.replace("chatopenai", "openai_chat")
    k2 = k2.replace("chat_openai", "openai_chat")
    k2 = k2.replace("openai_chatlike", "openai_chat")
    k2 = k2.replace("open_ai_chatlike", "openai_chat")
    k2 = k2.replace("openai_chat_like", "openai_chat")
    k2 = k2.replace("open_aichat_like", "openai_chat")
    k2 = k2.replace("open_ai_chat_like", "openai_chat")

    k2 = k2.replace("huggingfacetgi", "huggingface_tgi")
    k2 = k2.replace("huggingface_tgi", "huggingface_tgi")

    return k2


def create_llm(config: dict, callback_manager: Optional[BaseCallbackHandler] = None) -> BaseLLM:
    """Create an LLM implementation from a configuration mapping.

    This is the preferred entry point for wiring LLMs (used by the application
    container). The concrete implementation is selected by a discriminator field
    in the configuration (one of: ``kind``, ``type``, ``provider``, ``backend``,
    or ``impl``).

    Parameters
    ----------
    config : dict
        Configuration mapping used to construct the LLM.
    callback_manager : BaseCallbackHandler, optional
        Optional callback handler for logging/telemetry/streaming.

    Returns
    -------
    BaseLLM
        An initialised LLM implementation.

    Raises
    ------
    TypeError
        If ``config`` is not a mapping.
    ValueError
        If the discriminator field is missing, or if the discriminator selects
        an unsupported implementation.

    Notes
    -----
    The returned object is a concrete :class:`BaseLLM` implementation. Call
    :meth:`BaseLLM.get_llm` to access the underlying LangChain LLM object.
    """

    if not isinstance(config, Mapping):
        raise TypeError(f"create_llm expected a mapping/dict, got {type(config)}")

    kind_raw = _get_llm_kind(config)
    kind = _normalize_llm_kind(kind_raw)

    if not kind:
        raise ValueError(
            "LLM config is missing a discriminator field (type/kind/provider/etc.). "
            "Add e.g. type: OpenAILike or type: HuggingFaceTGI."
        )

    registry: dict[str, type[BaseLLM]] = {
        "openai_like": OpenAILikeLLM,
        "openai": OpenAILikeLLM,
        "openailike": OpenAILikeLLM,
        "openai_chat": OpenAIChatLikeLLM,
        "openai_chatlike": OpenAIChatLikeLLM,
        "chat_openai": OpenAIChatLikeLLM,
        "chatopenai": OpenAIChatLikeLLM,
        "huggingface_tgi": HuggingFaceTGI,
        "huggingfacetgi": HuggingFaceTGI,
        "tgi": HuggingFaceTGI,
    }

    cls = registry.get(kind)
    if cls is None:
        raise ValueError(
            f"Unknown LLM kind '{kind_raw}' (normalized to '{kind}'). Supported kinds: {sorted(registry.keys())}."
        )

    return cls.from_config_dict(dict(config), callback_manager=callback_manager)


__all__ = [
    "BaseLLM",
    "OpenAILikeLLM",
    "OpenAIChatLikeLLM",
    "HuggingFaceTGI",
    "create_llm",
]
