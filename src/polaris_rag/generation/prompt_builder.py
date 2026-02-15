"""polaris_rag.generation.prompt_builder

Prompt template definitions and rendering utilities.

This module provides lightweight abstractions for defining, registering,
and rendering named prompt templates used by LLM interfaces. Templates
support optional system messages, few-shot examples, and user instructions,
and are rendered using Jinja2.

Classes
-------
PromptTemplate
    Represents a single named prompt template.
PromptBuilder
    Registry and factory for prompt templates.
"""
from typing import Optional, List, Dict, Any, Iterable, Union
from pathlib import Path
import json
from jinja2 import Template
import warnings
from importlib import resources

class PromptTemplate:
    """Represents a single named prompt template.

    A prompt template is composed of an optional system message, zero or more
    few-shot examples, and a user instruction block. The full prompt is rendered
    by concatenating these parts and applying Jinja2 templating.

    Parameters
    ----------
    name : str
        Name of the template.
    system : str or None, optional
        System-level instructions for the template.
    few_shot : list[dict[str, str]] or None, optional
        Few-shot examples for the template. Each entry is expected to contain
        a ``"content"`` key.
    user : str, optional
        User instruction part of the template.
    """

    def __init__(self, 
                 name: str, 
                 system: Optional[str] = None,
                 few_shot: Optional[List[Dict[str, str]]] = None,
                 user: Optional[str] = ''
        ):
        """Initialise a PromptTemplate.

        Parameters
        ----------
        name : str
            Name of the template.
        system : str or None, optional
            System-level instructions.
        few_shot : list[dict[str, str]] or None, optional
            Few-shot examples.
        user : str, optional
            User instruction text.
        """
        self.name = name
        self.system = system
        self.few_shot = few_shot or []
        self.user = user

    def render(self, **kwargs) -> str:
        """Render the full prompt by filling in placeholders.

        Parameters
        ----------
        **kwargs : Any
            Variables to substitute into the template parts.

        Returns
        -------
        str
            The rendered prompt string.

        Notes
        -----
        The system message, few-shot examples (in order), and user instruction
        are concatenated with newline separators before Jinja2 rendering.
        """
        parts = []
        if self.system:
            parts.append(self.system)
        for example in self.few_shot:
            parts.append(example.get('content', ''))
        if self.user:
            parts.append(self.user)
        template_str = "\n".join(parts)
        return Template(template_str).render(**kwargs)


class PromptBuilder:
    """Registry and factory for prompt templates.

    This class manages a collection of named :class:`PromptTemplate` instances
    and provides methods to register templates from dictionaries or files and
    to render prompts by name.
    """

    def __init__(self):
        """Initialise an empty PromptBuilder."""
        self.templates: Dict[str, PromptTemplate] = {}

    def register_from_dict(self, data: Dict[str, Any]):
        """Register a new template from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            Mapping containing the template definition. Expected keys are
            ``"name"``, ``"system"``, ``"few_shot"``, and ``"user"``.

        Raises
        ------
        KeyError
            If ``"name"`` is missing from ``data``.
        TypeError
            If fields are of invalid types.
        ValueError
            If ``"name"`` is empty.
        """
        if "name" not in data:
            raise KeyError("Template definition missing required key: 'name'")
        name = data["name"]
        if not isinstance(name, str):
            raise TypeError(f"Template 'name' must be a str, got {type(name)!r}")
        if not name.strip():
            raise ValueError("Template 'name' must be a non-empty string")

        system = data.get("system")
        few_shot = data.get("few_shot")
        if few_shot is not None and not isinstance(few_shot, list):
            raise TypeError(f"Template 'few_shot' must be a list or None, got {type(few_shot)!r}")
        user = data.get("user", "")
        if user is None:
            user = ""

        template = PromptTemplate(name=name, system=system, few_shot=few_shot, user=user)
        if name in self.templates:
            warnings.warn(f"Overwriting existing prompt template: {name}")
        self.templates[name] = template

    def register_from_file(self, path: Union[Path, str], base_dir: Optional[Path] = None) -> List[str]:
        """Load and register templates from a JSON file.

        Parameters
        ----------
        path : Path | str
            Path to a JSON file containing one or more template definitions.
        base_dir : Path | None, optional
            If provided and ``path`` is relative, resolve it relative to this directory.

        Returns
        -------
        list[str]
            Names of templates registered from this file.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file extension is not supported.
        """
        p = Path(path)
        if not p.is_absolute() and base_dir is not None:
            p = base_dir / p
        p = p.resolve()
        if not p.exists():
            raise FileNotFoundError(f"Prompt file not found: {p}")
        if p.suffix.lower() not in [".json"]:
            raise ValueError(f"Unsupported file type: {p.suffix}")

        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)

        registered: List[str] = []
        if isinstance(data, dict):
            self.register_from_dict(data)
            registered.append(data["name"])
        elif isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    raise TypeError(f"Template list items must be dicts, got {type(item)!r}")
                self.register_from_dict(item)
                registered.append(item["name"])
        else:
            raise TypeError(f"Prompt file must contain an object or list of objects, got {type(data)!r}")

        return registered

    def register_from_package(self, package: str, resource_path: str) -> List[str]:
        """Load and register templates from a JSON file bundled as a package resource.

        Parameters
        ----------
        package : str
            Dotted package name that contains the resource.
        resource_path : str
            Resource path within the package (e.g., ``"default.json"`` or ``"prompts/default.json"``).

        Returns
        -------
        list[str]
            Names of templates registered fromut.

        Raises
        ------
        FileNotFoundError
            If the resource does not exist.
        ValueError
            If the resource extension is not supported.
        """
        if not resource_path.lower().endswith(".json"):
            raise ValueError(f"Unsupported resource type: {resource_path}")

        try:
            res = resources.files(package).joinpath(resource_path)
        except Exception as e:
            raise FileNotFoundError(f"Could not locate resource '{resource_path}' in package '{package}'") from e

        if not res.is_file():
            raise FileNotFoundError(f"Prompt resource not found: pkg:{package}:{resource_path}")

        data = json.loads(res.read_text(encoding="utf-8"))

        registered: List[str] = []
        if isinstance(data, dict):
            self.register_from_dict(data)
            registered.append(data["name"])
        elif isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    raise TypeError(f"Template list items must be dicts, got {type(item)!r}")
                self.register_from_dict(item)
                registered.append(item["name"])
        else:
            raise TypeError(f"Prompt resource must contain an object or list of objects, got {type(data)!r}")

        return registered

    def register_from_source(self, source: str, base_dir: Optional[Path] = None) -> List[str]:
        """Register templates from a source spec.

        Supported formats
        -----------------
        - ``pkg:<package>:<resource_path>``
        - ``file:<path>``
        - ``<path>`` (plain filesystem path)

        Parameters
        ----------
        source : str
            Source spec.
        base_dir : Path | None, optional
            Base directory for resolving relative filesystem paths.

        Returns
        -------
        list[str]
            Names of templates registered.
        """
        if not isinstance(source, str):
            raise TypeError(f"source must be a str, got {type(source)!r}")

        if source.startswith("pkg:"):
            rest = source[len("pkg:"):]
            if ":" not in rest:
                raise ValueError("pkg: sources must be of the form 'pkg:<package>:<resource_path>'")
            package, resource_path = rest.split(":", 1)
            return self.register_from_package(package.strip(), resource_path.strip())

        if source.startswith("file:"):
            path_str = source[len("file:"):].strip()
            return self.register_from_file(Path(path_str), base_dir=base_dir)

        return self.register_from_file(Path(source), base_dir=base_dir)

    def list_prompts(self) -> List[str]:
        """Return a sorted list of registered prompt template names."""
        return sorted(self.templates.keys())

    def has_prompt(self, name: str) -> bool:
        """Return True if a prompt template with this name is registered."""
        return name in self.templates

    def get_template(self, name: str) -> PromptTemplate:
        """Get a registered PromptTemplate by name.

        Raises
        ------
        KeyError
            If no template is registered under ``name``.
        """
        if name not in self.templates:
            available = ", ".join(self.list_prompts())
            raise KeyError(f"No template registered under name: {name}. Available: [{available}]")
        return self.templates[name]

    def build(self, name: str, **kwargs) -> str:
        """Build and render a prompt by template name.

        Parameters
        ----------
        name : str
            Name of the registered template.
        **kwargs : Any
            Variables to fill into the template.

        Returns
        -------
        str
            The rendered prompt string.

        Raises
        ------
        KeyError
            If no template is registered under ``name``.
        """
        template = self.get_template(name)
        return template.render(**kwargs)
