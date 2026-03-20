"""Deterministic local-official authority registry builder."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import urlsplit

from polaris_rag.common import MarkdownDocument

SOURCE_SCOPE_LOCAL_OFFICIAL = "local_official"
REGISTRY_ENTITY_TYPES: tuple[str, ...] = (
    "system",
    "partition",
    "service",
    "software",
    "module",
    "toolchain",
)
STATUS_VALUES: tuple[str, ...] = (
    "current",
    "maintenance",
    "legacy",
    "eol",
    "unknown",
)
REVIEW_STATE_AUTO_VERIFIED = "auto_verified"
REVIEW_STATE_NEEDS_REVIEW = "needs_review"
EXTRACTION_VERSION = "authority_registry_v1"

_TITLE_SUFFIX_PATTERN = re.compile(
    r"\s+[—-]\s*CSD3(?:\s+\d+(?:\.\d+)*)?\s+documentation\s*$",
    flags=re.IGNORECASE,
)
_HEADING_PERMALINK_PATTERN = re.compile(r"\s*\[¶\]\([^)]*\)\s*$")
_VERSION_PATTERN = re.compile(
    r"\b\d+\.\d+(?:\.\d+){0,2}(?:[-+_][A-Za-z0-9][A-Za-z0-9.+_-]*)?\b"
)
_SBATCH_PARTITION_PATTERN = re.compile(
    r"^\s*#SBATCH\s+(?:--partition(?:=|\s+)|-p\s+)([A-Za-z0-9][A-Za-z0-9_-]*)\b",
    flags=re.IGNORECASE,
)
_SCHEDULER_PARTITION_PATTERN = re.compile(
    r"\b(?:sbatch|srun|salloc)\b[^\n#]*(?:--partition(?:=|\s+)|-p\s+)([A-Za-z0-9][A-Za-z0-9_-]*)\b",
    flags=re.IGNORECASE,
)
_LONG_PARTITION_PATTERN = re.compile(r"\b([A-Za-z0-9][A-Za-z0-9_-]*-long)\b", flags=re.IGNORECASE)
_MODULE_LOAD_PATTERN = re.compile(
    r"(?:^|&&|;)\s*(?:\$+\s*)?module\s+load\s+([^\n#;]+)",
    flags=re.IGNORECASE | re.MULTILINE,
)
_INLINE_CODE_PATTERN = re.compile(r"`([^`]+)`")
_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SLUG_SPLIT_PATTERN = re.compile(r"[-_]+")
_MULTISPACE_PATTERN = re.compile(r"\s+")
_PUNCT_NORMALISE_PATTERN = re.compile(r"[^a-z0-9]+")

_STATUS_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("eol", re.compile(r"\b(?:eol|end[- ]of[- ]life|end of life)\b", flags=re.IGNORECASE)),
    ("legacy", re.compile(r"\b(?:legacy|retired|deprecated)\b", flags=re.IGNORECASE)),
    ("maintenance", re.compile(r"\bmaintenance\b", flags=re.IGNORECASE)),
    ("current", re.compile(r"\b(?:current|recommended|supported)\b", flags=re.IGNORECASE)),
)

_SERVICE_HINTS = (
    "interface",
    "service",
    "portal",
    "gateway",
    "login-web",
    "login web",
)
_SYSTEM_HINTS = (
    "nodes",
    "node",
    "platform",
    "cluster",
    "system",
)
_SOFTWARE_PATH_HINT = "/software-packages/"
_SOFTWARE_TOOLS_PATH_HINT = "/software-tools/"

_PAGE_CLASS_ENTITY = "entity_page"
_PAGE_CLASS_PROCEDURAL = "procedural_page"
_PAGE_CLASS_NOTICE = "notice_page"
_PAGE_CLASS_EXCLUDE = "exclude_page"

_PRIMARY_ENTITY_OVERRIDES: dict[str, tuple[str, str]] = {
    "a100": ("system", "Ampere GPU Nodes"),
    "castep": ("software", "CASTEP"),
    "cclake": ("system", "Cascade Lake Nodes"),
    "gaussian": ("software", "Gaussian"),
    "icelake": ("system", "Ice Lake Nodes"),
    "login-web": ("service", "Login-Web Interface"),
    "mfa": ("service", "MultiFactor Authentication (MFA)"),
    "pvc": ("system", "Dawn - Intel GPU (PVC) Nodes"),
    "python": ("software", "Python"),
    "sapphire-hbm": ("system", "Sapphire Rapid Nodes with High Bandwidth Memory"),
    "sapphire": ("system", "Sapphire Rapid Nodes"),
    "spack": ("software", "Spack"),
    "turbovnc": ("software", "TurboVNC"),
    "wbic": ("system", "WBIC-HPHI Platform"),
}
_PROCEDURAL_PAGE_SLUGS = {
    "batch",
    "connecting",
    "getting_support",
    "hostkeys",
    "index",
    "interactive",
    "io_management",
    "long",
    "modules",
    "performance-tips",
    "policies",
    "quickstart",
    "reading",
    "transfer",
}
_NOTICE_PAGE_SLUGS = {
    "data_centre_upgrade",
    "rsecon25",
    "sbs",
}
_EXCLUDED_PAGE_SUFFIXES = {
    "/_sources/index.rst.txt",
}
_MODULE_TOKEN_DENYLIST = {
    "#",
    "all",
    "basic",
    "cpu",
    "environment",
    "for",
    "load",
    "loads",
    "module",
    "partitions",
    "recommended",
    "required",
    "the",
}
_PARTITION_TOKEN_DENYLIST = {
    "partition",
    "partitions",
    "queue",
}
_PRIMARY_ENTITY_STATUS_OVERRIDES = {
    "a100": "current",
    "cclake": "current",
    "icelake": "current",
    "login-web": "current",
    "mfa": "current",
    "pvc": "current",
    "sapphire": "current",
    "sapphire-hbm": "current",
    "turbovnc": "current",
    "wbic": "eol",
}
_LEGACY_MODULE_NAMES = {
    "default-impi",
    "default-wilkes",
    "rhel7/default-peta4",
}
_KNOWN_BARE_MODULE_NAMES = {
    "alphafold",
    "castep",
    "cuda",
    "cudnn",
    "gromacs",
    "intelpython-conda",
    "jupyter",
    "jupyterlab",
    "lammps",
    "matlab",
    "openfoam",
    "openmm",
    "pgi",
    "python",
    "pytorch",
    "r",
    "spack",
    "tensorflow",
    "turbovnc",
    "use.own",
}
_SOFTWARE_VERSION_PREFIXES: dict[str, tuple[str, ...]] = {
    "alphafold": ("alphafold",),
    "castep": ("castep",),
    "gaussian": ("gaussian", "g16", "g09"),
    "gromacs": ("gromacs",),
    "jupyter": ("jupyter", "jupyterlab"),
    "lammps": ("lammps",),
    "matlab": ("matlab",),
    "openfoam": ("openfoam",),
    "openmm": ("openmm",),
    "python": ("python",),
    "pytorch": ("pytorch",),
    "r": ("r",),
    "spack": ("spack",),
    "tensorflow": ("tensorflow",),
    "turbovnc": ("turbovnc",),
}
_SOFTWARE_CURRENT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bavailable on csd3\b", flags=re.IGNORECASE),
    re.compile(r"\bcurrently installed\b", flags=re.IGNORECASE),
    re.compile(r"\binstalled on csd3\b", flags=re.IGNORECASE),
    re.compile(r"\bcentral installations?\b", flags=re.IGNORECASE),
    re.compile(r"\bsupported on (?:each|all)\b", flags=re.IGNORECASE),
    re.compile(r"\bversions? .* are available\b", flags=re.IGNORECASE),
    re.compile(r"\bmodule load\b", flags=re.IGNORECASE),
)
_PARTITION_ALIASES_BY_DOC_SLUG: dict[str, dict[str, tuple[str, ...]]] = {
    "pvc": {
        "pvc9": ("pvc",),
    },
}

_TOOLCHAIN_HINTS = (
    "foss",
    "gompi",
    "intel",
    "cuda",
    "default",
    "oneapi",
    "gcccore",
    "nvhpc",
    "amp",
    "openmpi",
)
_MODULE_COMMAND_PATTERN = re.compile(r"\bmodule\s+load\s+", flags=re.IGNORECASE)
_COMMAND_SNIPPET_PATTERN = re.compile(r"\b(?:module|source|conda|sbatch|srun|salloc)\b|#SBATCH", flags=re.IGNORECASE)


@dataclass(frozen=True)
class ReviewQueueRow:
    """Manual-audit row for ambiguous registry candidates."""

    reason: str
    entity_type: str
    canonical_name: str
    source_scope: str
    candidate_count: int
    status_values: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass(frozen=True)
class RegistryEntity:
    """One normalized authority-registry entity."""

    entity_id: str
    entity_type: str
    canonical_name: str
    aliases: list[str]
    source_scope: str
    status: str
    known_versions: list[str]
    doc_id: str
    doc_title: str
    heading_path: list[str]
    evidence_spans: list[dict[str, object]]
    extraction_method: str
    review_state: str


@dataclass(frozen=True)
class RegistryArtifact:
    """Serialized output for a deterministic authority build."""

    build: dict[str, object]
    source_urls: list[str]
    entities: list[RegistryEntity]
    summary: dict[str, object]


@dataclass(frozen=True)
class _Section:
    heading_path: tuple[str, ...]
    text: str


@dataclass(frozen=True)
class _Candidate:
    entity_type: str
    canonical_name: str
    aliases: tuple[str, ...]
    source_scope: str
    status: str
    known_versions: tuple[str, ...]
    doc_id: str
    doc_title: str
    heading_path: tuple[str, ...]
    evidence_text: str
    extraction_method: str


def _clean_title(title: str) -> str:
    text = str(title or "").strip()
    text = _TITLE_SUFFIX_PATTERN.sub("", text)
    return text.strip()


def _clean_heading_text(text: str) -> str:
    value = _HEADING_PERMALINK_PATTERN.sub("", str(text or "").strip())
    return _normalize_spaces(value)


def _normalize_spaces(text: str) -> str:
    return _MULTISPACE_PATTERN.sub(" ", str(text or "").strip())


def _normalize_alias(text: str) -> str:
    return _normalize_spaces(text)


def _normalize_key(text: str) -> str:
    lowered = str(text or "").strip().lower()
    return _PUNCT_NORMALISE_PATTERN.sub(" ", lowered).strip()


def _sorted_unique(values: Iterable[str]) -> list[str]:
    return sorted({str(value).strip() for value in values if str(value or "").strip()})


def _canonical_url(document: MarkdownDocument) -> str:
    metadata = dict(document.metadata or {})
    source = str(metadata.get("source") or document.id or "").strip()
    return source or str(document.id)


def _url_slug(source_url: str) -> str:
    path = urlsplit(source_url).path.strip("/")
    if not path:
        return ""
    return path.rsplit("/", 1)[-1]


def _strip_slug_suffix(slug: str) -> str:
    text = str(slug or "").strip()
    for suffix in (".html", ".rst.txt"):
        if text.lower().endswith(suffix):
            return text[: -len(suffix)]
    return text


def _slug_aliases(source_url: str) -> list[str]:
    slug = _strip_slug_suffix(_url_slug(source_url))
    if not slug:
        return []
    parts = [part for part in _SLUG_SPLIT_PATTERN.split(slug) if part]
    candidates = {slug, slug.replace("-", " ")}
    if len(parts) > 1:
        candidates.add(" ".join(parts))
    return _sorted_unique(candidates)


def _extract_sections(markdown_text: str) -> list[_Section]:
    lines = str(markdown_text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    sections: list[_Section] = []
    stack: list[str] = []
    current_path: tuple[str, ...] = tuple()
    current_lines: list[str] = []
    in_fence = False

    def _flush() -> None:
        if not current_lines:
            return
        text = "\n".join(current_lines).strip()
        if text:
            sections.append(_Section(heading_path=current_path, text=text))

    for raw_line in lines:
        stripped_line = raw_line.strip()
        if stripped_line.startswith("```"):
            current_lines.append(raw_line)
            in_fence = not in_fence
            continue

        if in_fence:
            current_lines.append(raw_line)
            continue

        match = _HEADING_PATTERN.match(raw_line)
        if match:
            _flush()
            level = len(match.group(1))
            title = _clean_heading_text(match.group(2))
            if level <= 0 or not title:
                current_lines.clear()
                continue
            while len(stack) >= level:
                stack.pop()
            stack.append(title)
            current_path = tuple(stack)
            current_lines = []
            continue
        current_lines.append(raw_line)

    _flush()
    if not sections:
        return [_Section(heading_path=tuple(), text=_normalize_spaces(markdown_text))]
    return sections


def _extract_versions(text: str) -> list[str]:
    return _sorted_unique(_VERSION_PATTERN.findall(str(text or "")))


def _extract_status(text: str) -> str:
    body = str(text or "")
    for status, pattern in _STATUS_PATTERNS:
        if pattern.search(body):
            return status
    return "unknown"


def _iter_command_snippets(text: str) -> list[str]:
    body = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    snippets: list[str] = []
    current_fence: list[str] = []
    current_indented: list[str] = []
    in_fence = False

    def _flush_indented() -> None:
        nonlocal current_indented
        snippet = "\n".join(current_indented).strip()
        if snippet and _COMMAND_SNIPPET_PATTERN.search(snippet):
            snippets.append(snippet)
        current_indented = []

    for line in body.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            if current_indented:
                _flush_indented()
            if in_fence:
                snippet = "\n".join(current_fence).strip()
                if snippet:
                    snippets.append(snippet)
                current_fence = []
                in_fence = False
            else:
                in_fence = True
                current_fence = []
            continue

        if in_fence:
            current_fence.append(line)
            continue

        if line.startswith("    ") or line.startswith("\t"):
            if line.startswith("    "):
                current_indented.append(line[4:])
            else:
                current_indented.append(line.lstrip("\t"))
            continue

        if current_indented:
            _flush_indented()

        if stripped.startswith(("$ ", "module ", "source ", "conda ", "sbatch ", "srun ", "salloc ", "#SBATCH")):
            snippets.append(stripped)

    if current_indented:
        _flush_indented()
    snippets.extend(_INLINE_CODE_PATTERN.findall(body))
    return [snippet for snippet in snippets if str(snippet or "").strip()]


def _extract_partition_names(text: str) -> list[str]:
    matches: list[str] = []
    body = str(text or "")
    for snippet in _iter_command_snippets(text):
        for line in str(snippet).splitlines():
            sbatch_match = _SBATCH_PARTITION_PATTERN.search(line)
            if sbatch_match:
                matches.append(sbatch_match.group(1))
            for scheduler_match in _SCHEDULER_PARTITION_PATTERN.finditer(line):
                matches.append(scheduler_match.group(1))
    for line in body.splitlines():
        line_text = str(line or "").strip()
        lowered = line_text.lower()
        if "partition" not in lowered and "-long" not in lowered:
            continue
        for match in _LONG_PARTITION_PATTERN.finditer(line_text):
            matches.append(match.group(1))
    filtered = [
        match
        for match in matches
        if _normalize_key(match) and _normalize_key(match) not in _PARTITION_TOKEN_DENYLIST
    ]
    return _sorted_unique(filtered)


def _split_module_tokens(raw_value: str) -> list[str]:
    if not raw_value:
        return []
    cutoff = raw_value
    for separator in ("&&", ";"):
        cutoff = cutoff.split(separator, 1)[0]
    cutoff = cutoff.split("#", 1)[0]
    tokens = []
    for token in re.split(r"[\s,]+", cutoff.strip()):
        cleaned = token.strip().strip("`'\"()[]{}")
        if not cleaned:
            continue
        if cleaned.startswith("-"):
            continue
        cleaned_lower = cleaned.lower()
        if cleaned_lower in _MODULE_TOKEN_DENYLIST:
            continue
        if cleaned.startswith("<") and cleaned.endswith(">"):
            continue
        if not _looks_like_module_token(cleaned):
            continue
        tokens.append(cleaned)
    return tokens


def _looks_like_module_token(token: str) -> bool:
    cleaned = str(token or "").strip()
    if not cleaned:
        return False
    if any(char in cleaned for char in "()"):
        return False
    normalized = cleaned.lower()
    if normalized in _MODULE_TOKEN_DENYLIST:
        return False
    if "/" in cleaned or "." in cleaned:
        return True
    if any(char.isdigit() for char in cleaned):
        return True
    if "-" in cleaned:
        return normalized in _KNOWN_BARE_MODULE_NAMES or not cleaned.isalpha()
    return normalized in _KNOWN_BARE_MODULE_NAMES


def _extract_module_names(text: str) -> list[str]:
    matches: list[str] = []
    for snippet in _iter_command_snippets(text):
        for line in str(snippet).splitlines():
            remaining = line
            while True:
                match = _MODULE_COMMAND_PATTERN.search(remaining)
                if not match:
                    break
                tail = remaining[match.end() :]
                next_match = _MODULE_COMMAND_PATTERN.search(tail)
                raw_value = tail[: next_match.start()] if next_match else tail
                matches.extend(_split_module_tokens(raw_value))
                if not next_match:
                    break
                remaining = tail[next_match.start() :]
    return _sorted_unique(matches)


def _infer_toolchain_name(module_name: str) -> str | None:
    token = str(module_name or "").strip()
    if not token:
        return None

    token_lower = token.lower()
    if token_lower.startswith("rhel") and "/" in token:
        return token

    if token_lower.startswith(("default-", "cuda", "cudnn", "intelpython-conda")):
        return token

    if any(token_lower == hint or token_lower.startswith(f"{hint}/") for hint in _TOOLCHAIN_HINTS):
        return token

    return None


def _primary_heading_path(sections: Sequence[_Section]) -> tuple[str, ...]:
    for section in sections:
        if section.heading_path:
            return section.heading_path
    return tuple()


def _primary_heading_title(sections: Sequence[_Section]) -> str:
    path = _primary_heading_path(sections)
    if not path:
        return ""
    return _clean_heading_text(path[0])


def _document_slug(document: MarkdownDocument) -> str:
    return _strip_slug_suffix(_url_slug(_canonical_url(document))).lower()


def _classify_document_page(document: MarkdownDocument, sections: Sequence[_Section]) -> str:
    source_url = _canonical_url(document)
    path = urlsplit(source_url).path.lower()
    slug = _document_slug(document)
    display_title = _normalize_spaces(
        _primary_heading_title(sections) or _clean_title(str((document.metadata or {}).get("title") or ""))
    ).lower()

    if (
        "/_sources/" in path
        or "/_images/" in path
        or "/images/" in path
        or path.startswith("/storage/")
        or path.endswith((".png", ".jpg", ".jpeg", ".gif", ".svg", ".pdf", ".css", ".js", ".xml"))
    ):
        return _PAGE_CLASS_EXCLUDE
    if any(path.endswith(suffix) for suffix in _EXCLUDED_PAGE_SUFFIXES):
        return _PAGE_CLASS_EXCLUDE
    if slug in _NOTICE_PAGE_SLUGS:
        return _PAGE_CLASS_NOTICE
    if slug in _PROCEDURAL_PAGE_SLUGS:
        return _PAGE_CLASS_PROCEDURAL
    if _SOFTWARE_PATH_HINT in path or _SOFTWARE_TOOLS_PATH_HINT in path:
        return _PAGE_CLASS_ENTITY
    if slug in _PRIMARY_ENTITY_OVERRIDES:
        return _PAGE_CLASS_ENTITY
    if any(hint in display_title for hint in _SERVICE_HINTS) or any(hint in display_title for hint in _SYSTEM_HINTS):
        return _PAGE_CLASS_ENTITY
    return _PAGE_CLASS_PROCEDURAL


def _resolve_document_title(document: MarkdownDocument, sections: Sequence[_Section]) -> str:
    heading_title = _primary_heading_title(sections)
    if heading_title:
        return heading_title

    metadata_title = _clean_title(str((document.metadata or {}).get("title") or ""))
    if metadata_title:
        return metadata_title

    slug = _strip_slug_suffix(_url_slug(_canonical_url(document)))
    return _normalize_spaces(slug.replace("-", " "))


def _extract_entity_versions(title: str, sections: Sequence[_Section]) -> list[str]:
    candidate_texts: list[str] = [title]
    for section in sections:
        if section.heading_path:
            candidate_texts.extend(section.heading_path)
        for line in section.text.splitlines():
            line_text = str(line or "").strip()
            if not line_text:
                continue
            if re.search(r"\bversions?\b", line_text, flags=re.IGNORECASE):
                candidate_texts.append(line_text)
        candidate_texts.extend(_extract_module_names(section.text))
    return _sorted_unique(_extract_versions("\n".join(candidate_texts)))


def _looks_like_software_version(value: str) -> bool:
    token = str(value or "").strip()
    if not token:
        return False
    if re.fullmatch(r"r\d{4}[a-z]?", token, flags=re.IGNORECASE):
        return True
    if re.fullmatch(r"\d+(?:\.\d+){0,3}(?:[-_][A-Za-z0-9][A-Za-z0-9._-]*)?", token):
        return True
    return False


def _extract_versions_from_module_references(text: str, prefixes: Sequence[str]) -> list[str]:
    versions: list[str] = []
    body = str(text or "")
    for prefix in prefixes:
        pattern = re.compile(rf"(?<![A-Za-z0-9_-]){re.escape(prefix)}/([A-Za-z0-9][A-Za-z0-9._-]*)", flags=re.IGNORECASE)
        for match in pattern.finditer(body):
            candidate = _clean_version_candidate(match.group(1))
            if _looks_like_software_version(candidate):
                versions.append(candidate)
    return _sorted_unique(versions)


def _clean_version_candidate(value: str) -> str:
    token = str(value or "").strip().strip("`'\".,;:!?)]}")
    if not token:
        return ""

    direct_match = re.fullmatch(
        r"(?:v)?(r\d{4}[a-z]?|\d+(?:\.\d+){0,3})(?:[-_+].*)?",
        token,
        flags=re.IGNORECASE,
    )
    if direct_match:
        return direct_match.group(1)

    prefixed_match = re.fullmatch(
        r"[A-Za-z][A-Za-z0-9+._-]*?[-_/](r\d{4}[a-z]?|\d+(?:\.\d+){0,3})(?:[-_+].*)?",
        token,
        flags=re.IGNORECASE,
    )
    if prefixed_match:
        return prefixed_match.group(1)

    return token


def _extract_module_like_versions(module_name: str) -> list[str]:
    values: list[str] = []
    for segment in str(module_name or "").split("/"):
        cleaned = _clean_version_candidate(segment)
        if _looks_like_software_version(cleaned):
            values.append(cleaned)
    return _sorted_unique(values)


def _extract_software_versions(slug: str, title: str, sections: Sequence[_Section]) -> list[str]:
    prefixes = _SOFTWARE_VERSION_PREFIXES.get(slug, (slug,))
    candidate_versions: list[str] = []
    for section in sections:
        if section.heading_path:
            candidate_versions.extend(_extract_versions_from_module_references("\n".join(section.heading_path), prefixes))
        candidate_versions.extend(_extract_versions_from_module_references(section.text, prefixes))
        for line in section.text.splitlines():
            line_text = str(line or "").strip()
            if not line_text:
                continue
            lowered = line_text.lower()
            if any(prefix.lower() in lowered for prefix in prefixes) and re.search(r"\b(?:version|versions?)\b", lowered):
                if "http://" in lowered or "https://" in lowered:
                    continue
                candidate_versions.extend(
                    _clean_version_candidate(value)
                    for value in _extract_versions(line_text)
                    if _looks_like_software_version(_clean_version_candidate(value))
                )
    candidate_versions.extend(
        _clean_version_candidate(value)
        for value in _extract_versions(title)
        if _looks_like_software_version(_clean_version_candidate(value))
    )
    return _sorted_unique(candidate_versions)


def _status_for_primary_entity(slug: str, title: str, sections: Sequence[_Section]) -> str:
    override = _PRIMARY_ENTITY_STATUS_OVERRIDES.get(slug)
    if override:
        return override

    focus_parts: list[str] = [title]
    if sections:
        focus_parts.append(sections[0].text[:600])
    return _extract_status("\n".join(part for part in focus_parts if str(part).strip()))


def _status_for_primary_software(slug: str, title: str, sections: Sequence[_Section]) -> str:
    override = _PRIMARY_ENTITY_STATUS_OVERRIDES.get(slug)
    if override:
        return override

    focus_text = "\n".join(part for part in [title, *(section.text[:800] for section in sections[:2])] if str(part).strip())
    explicit = _extract_status(focus_text)
    if explicit in {"eol", "legacy", "maintenance"}:
        return explicit
    if any(pattern.search(focus_text) for pattern in _SOFTWARE_CURRENT_PATTERNS):
        return "current"
    if _extract_software_versions(slug, title, sections):
        return "current"
    return "current"


def _status_for_partition(partition_name: str, section_text: str) -> str:
    status = _extract_status(section_text)
    if status != "unknown":
        return status
    return "current"


def _status_for_module_like(module_name: str, section_text: str) -> str:
    normalized = _normalize_key(module_name)
    if normalized in {_normalize_key(value) for value in _LEGACY_MODULE_NAMES}:
        return "legacy"
    return _extract_status(section_text)


def _partition_aliases(partition_name: str, doc_slug: str) -> list[str]:
    aliases = [partition_name.lower()]
    for alias in _PARTITION_ALIASES_BY_DOC_SLUG.get(doc_slug, {}).get(partition_name, ()):
        aliases.append(alias)
    return _sorted_unique(aliases)


def _primary_evidence_text(title: str, sections: Sequence[_Section]) -> str:
    for section in sections:
        if section.text.strip():
            return section.text.strip()[:400]
    return title


def _infer_primary_entity(document: MarkdownDocument, sections: Sequence[_Section], source_scope: str) -> _Candidate | None:
    source_url = _canonical_url(document)
    slug = _document_slug(document)
    title = _resolve_document_title(document, sections)
    if not title:
        return None

    override = _PRIMARY_ENTITY_OVERRIDES.get(slug)
    entity_type: str | None = None
    if override is not None:
        entity_type, title = override
    else:
        path = urlsplit(source_url).path.lower()
        title_lower = title.lower()
        if _SOFTWARE_PATH_HINT in path or _SOFTWARE_TOOLS_PATH_HINT in path:
            entity_type = "software"
            title = re.sub(r"^\s*using\s+", "", title, flags=re.IGNORECASE)
        elif any(hint in title_lower for hint in _SERVICE_HINTS):
            entity_type = "service"
        elif any(hint in title_lower for hint in _SYSTEM_HINTS):
            entity_type = "system"

    if entity_type is None:
        return None

    heading_path = _primary_heading_path(sections)
    if entity_type == "software":
        versions = _extract_software_versions(slug, title, sections)
        status = _status_for_primary_software(slug, title, sections)
    else:
        versions = _extract_entity_versions(title, sections)
        status = _status_for_primary_entity(slug, title, sections)
    aliases = _sorted_unique(
        [
            title,
            _clean_title(str((document.metadata or {}).get("title") or "")),
            *_slug_aliases(source_url),
        ]
    )
    evidence_text = _primary_evidence_text(title, sections)
    return _Candidate(
        entity_type=entity_type,
        canonical_name=title,
        aliases=tuple(aliases),
        source_scope=source_scope,
        status=status,
        known_versions=tuple(versions),
        doc_id=document.id,
        doc_title=title,
        heading_path=heading_path,
        evidence_text=evidence_text,
        extraction_method="title_path_heuristic",
    )


def _build_candidate(
    *,
    entity_type: str,
    canonical_name: str,
    aliases: Iterable[str],
    source_scope: str,
    status: str,
    known_versions: Iterable[str],
    doc_id: str,
    doc_title: str,
    heading_path: Sequence[str],
    evidence_text: str,
    extraction_method: str,
) -> _Candidate | None:
    canonical = _normalize_spaces(canonical_name)
    if not canonical:
        return None
    aliases_sorted = _sorted_unique([canonical, *aliases])
    versions_sorted = _sorted_unique(known_versions)
    return _Candidate(
        entity_type=entity_type,
        canonical_name=canonical,
        aliases=tuple(aliases_sorted),
        source_scope=source_scope,
        status=status if status in STATUS_VALUES else "unknown",
        known_versions=tuple(versions_sorted),
        doc_id=str(doc_id),
        doc_title=_normalize_spaces(doc_title),
        heading_path=tuple(str(item) for item in heading_path if str(item).strip()),
        evidence_text=str(evidence_text or "").strip(),
        extraction_method=extraction_method,
    )


def extract_registry_candidates(
    markdown_documents: Iterable[MarkdownDocument],
    *,
    source_scope: str = SOURCE_SCOPE_LOCAL_OFFICIAL,
) -> tuple[list[_Candidate], list[ReviewQueueRow]]:
    """Extract raw registry candidates from markdown-normalized documents."""

    candidates: list[_Candidate] = []
    review_rows: list[ReviewQueueRow] = []

    for document in sorted(markdown_documents, key=lambda item: str(item.id)):
        sections = _extract_sections(document.text)
        page_class = _classify_document_page(document, sections)
        title = _resolve_document_title(document, sections)
        doc_slug = _document_slug(document)
        primary = _infer_primary_entity(document, sections, source_scope) if page_class == _PAGE_CLASS_ENTITY else None
        if primary is not None:
            candidates.append(primary)
        elif page_class == _PAGE_CLASS_ENTITY:
            review_rows.append(
                ReviewQueueRow(
                    reason="missing_canonical_name",
                    entity_type="unknown",
                    canonical_name="",
                    source_scope=source_scope,
                    candidate_count=1,
                    doc_ids=[str(document.id)],
                    notes="Document title and slug could not be normalized into a canonical page entity.",
                )
            )

        for section in sections:
            doc_title = title or _strip_slug_suffix(_url_slug(str(document.id)))

            for partition_name in _extract_partition_names(section.text):
                candidate = _build_candidate(
                    entity_type="partition",
                    canonical_name=partition_name,
                    aliases=_partition_aliases(partition_name, doc_slug),
                    source_scope=source_scope,
                    status=_status_for_partition(partition_name, section.text),
                    known_versions=(),
                    doc_id=str(document.id),
                    doc_title=doc_title,
                    heading_path=section.heading_path,
                    evidence_text=section.text[:400],
                    extraction_method="partition_parse",
                )
                if candidate is not None:
                    candidates.append(candidate)

            for module_name in _extract_module_names(section.text):
                module_candidate = _build_candidate(
                    entity_type="module",
                    canonical_name=module_name,
                    aliases=[module_name.lower()],
                    source_scope=source_scope,
                    status=_status_for_module_like(module_name, section.text),
                    known_versions=_extract_module_like_versions(module_name),
                    doc_id=str(document.id),
                    doc_title=doc_title,
                    heading_path=section.heading_path,
                    evidence_text=section.text[:400],
                    extraction_method="module_load_parse",
                )
                if module_candidate is not None:
                    candidates.append(module_candidate)

                toolchain_name = _infer_toolchain_name(module_name)
                if toolchain_name:
                    toolchain_candidate = _build_candidate(
                        entity_type="toolchain",
                        canonical_name=toolchain_name,
                        aliases=[toolchain_name.lower()],
                        source_scope=source_scope,
                        status=_status_for_module_like(toolchain_name, section.text),
                        known_versions=_extract_module_like_versions(toolchain_name),
                        doc_id=str(document.id),
                        doc_title=doc_title,
                        heading_path=section.heading_path,
                        evidence_text=section.text[:400],
                        extraction_method="toolchain_parse",
                    )
                    if toolchain_candidate is not None:
                        candidates.append(toolchain_candidate)

    return candidates, review_rows


def _make_entity_id(source_scope: str, entity_type: str, canonical_name: str) -> str:
    payload = f"{source_scope}|{entity_type}|{_normalize_key(canonical_name)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _merge_candidates(
    candidates: Iterable[_Candidate],
    *,
    source_scope: str,
) -> tuple[list[RegistryEntity], list[ReviewQueueRow]]:
    grouped: dict[tuple[str, str], list[_Candidate]] = defaultdict(list)
    for candidate in candidates:
        key = (candidate.entity_type, _normalize_key(candidate.canonical_name))
        grouped[key].append(candidate)

    entities: list[RegistryEntity] = []
    review_rows: list[ReviewQueueRow] = []

    for (entity_type, _), group in sorted(grouped.items(), key=lambda item: item[0]):
        group_sorted = sorted(
            group,
            key=lambda item: (
                item.canonical_name.lower(),
                item.doc_id,
                item.heading_path,
                item.extraction_method,
            ),
        )
        primary = group_sorted[0]
        statuses = sorted({item.status for item in group_sorted if item.status != "unknown"})
        aliases = _sorted_unique(alias for item in group_sorted for alias in item.aliases)
        versions = _sorted_unique(version for item in group_sorted for version in item.known_versions)
        evidence_spans = [
            {
                "doc_id": item.doc_id,
                "doc_title": item.doc_title,
                "heading_path": list(item.heading_path),
                "evidence_text": item.evidence_text,
                "extraction_method": item.extraction_method,
            }
            for item in group_sorted
        ]

        status = statuses[0] if len(statuses) == 1 else "unknown"
        review_state = REVIEW_STATE_AUTO_VERIFIED
        if len(statuses) > 1:
            review_state = REVIEW_STATE_NEEDS_REVIEW
            review_rows.append(
                ReviewQueueRow(
                    reason="conflicting_status",
                    entity_type=entity_type,
                    canonical_name=primary.canonical_name,
                    source_scope=source_scope,
                    candidate_count=len(group_sorted),
                    status_values=statuses,
                    aliases=aliases,
                    doc_ids=_sorted_unique(item.doc_id for item in group_sorted),
                    notes="Multiple explicit lifecycle statuses were extracted for the same canonical entity.",
                )
            )

        entities.append(
            RegistryEntity(
                entity_id=_make_entity_id(source_scope, entity_type, primary.canonical_name),
                entity_type=entity_type,
                canonical_name=primary.canonical_name,
                aliases=aliases,
                source_scope=source_scope,
                status=status,
                known_versions=versions,
                doc_id=primary.doc_id,
                doc_title=primary.doc_title,
                heading_path=list(primary.heading_path),
                evidence_spans=evidence_spans,
                extraction_method=primary.extraction_method if len({item.extraction_method for item in group_sorted}) == 1 else "merged",
                review_state=review_state,
            )
        )

    alias_index: dict[tuple[str, str], list[RegistryEntity]] = defaultdict(list)
    for entity in entities:
        for alias in entity.aliases:
            alias_index[(entity.entity_type, _normalize_key(alias))].append(entity)

    entities_by_id = {entity.entity_id: entity for entity in entities}
    needs_review_ids: set[str] = set()
    for (entity_type, alias_key), alias_entities in sorted(alias_index.items(), key=lambda item: item[0]):
        distinct_canonicals = {_normalize_key(entity.canonical_name) for entity in alias_entities}
        if len(distinct_canonicals) <= 1 or not alias_key:
            continue
        for entity in alias_entities:
            needs_review_ids.add(entity.entity_id)
        review_rows.append(
            ReviewQueueRow(
                reason="alias_ambiguity",
                entity_type=entity_type,
                canonical_name=" / ".join(sorted(entity.canonical_name for entity in alias_entities)),
                source_scope=source_scope,
                candidate_count=len(alias_entities),
                aliases=_sorted_unique(alias for entity in alias_entities for alias in entity.aliases),
                doc_ids=_sorted_unique(entity.doc_id for entity in alias_entities),
                notes=f"Alias key '{alias_key}' maps to multiple canonical entities of the same type.",
            )
        )

    final_entities: list[RegistryEntity] = []
    for entity in sorted(entities, key=lambda item: (item.entity_type, item.canonical_name.lower(), item.entity_id)):
        if entity.entity_id in needs_review_ids and entity.review_state != REVIEW_STATE_NEEDS_REVIEW:
            final_entities.append(
                RegistryEntity(
                    entity_id=entity.entity_id,
                    entity_type=entity.entity_type,
                    canonical_name=entity.canonical_name,
                    aliases=list(entity.aliases),
                    source_scope=entity.source_scope,
                    status=entity.status,
                    known_versions=list(entity.known_versions),
                    doc_id=entity.doc_id,
                    doc_title=entity.doc_title,
                    heading_path=list(entity.heading_path),
                    evidence_spans=list(entity.evidence_spans),
                    extraction_method=entity.extraction_method,
                    review_state=REVIEW_STATE_NEEDS_REVIEW,
                )
            )
            continue
        final_entities.append(entity)

    review_rows_sorted = sorted(
        review_rows,
        key=lambda row: (
            row.reason,
            row.entity_type,
            row.canonical_name.lower(),
            tuple(row.doc_ids),
        ),
    )
    return final_entities, review_rows_sorted


def _build_summary(entities: Sequence[RegistryEntity], review_rows: Sequence[ReviewQueueRow]) -> dict[str, object]:
    type_counts = Counter(entity.entity_type for entity in entities)
    status_counts = Counter(entity.status for entity in entities)
    review_counts = Counter(row.reason for row in review_rows)
    return {
        "entity_count": len(entities),
        "review_count": len(review_rows),
        "counts_by_entity_type": dict(sorted(type_counts.items())),
        "counts_by_status": dict(sorted(status_counts.items())),
        "counts_by_review_reason": dict(sorted(review_counts.items())),
    }


def build_registry_artifact(
    markdown_documents: Iterable[MarkdownDocument],
    *,
    homepage: str,
    source_urls: Sequence[str],
    source_scope: str = SOURCE_SCOPE_LOCAL_OFFICIAL,
    extraction_version: str = EXTRACTION_VERSION,
) -> tuple[RegistryArtifact, list[ReviewQueueRow]]:
    """Build a deterministic registry artifact from local official markdown docs."""

    if source_scope != SOURCE_SCOPE_LOCAL_OFFICIAL:
        raise ValueError(
            f"Unsupported source_scope {source_scope!r}. Stage 1 only supports {SOURCE_SCOPE_LOCAL_OFFICIAL!r}."
        )

    candidates, extraction_review_rows = extract_registry_candidates(
        markdown_documents,
        source_scope=source_scope,
    )
    entities, merge_review_rows = _merge_candidates(candidates, source_scope=source_scope)
    review_rows = sorted(
        [*extraction_review_rows, *merge_review_rows],
        key=lambda row: (
            row.reason,
            row.entity_type,
            row.canonical_name.lower(),
            tuple(row.doc_ids),
        ),
    )
    artifact = RegistryArtifact(
        build={
            "homepage": str(homepage),
            "source_scope": source_scope,
            "extraction_version": str(extraction_version),
        },
        source_urls=sorted({str(url).strip() for url in source_urls if str(url).strip()}),
        entities=entities,
        summary=_build_summary(entities, review_rows),
    )
    return artifact, review_rows


def persist_registry_artifact(artifact: RegistryArtifact, path: str | Path) -> Path:
    """Persist a registry artifact as stable JSON."""

    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "build": artifact.build,
        "source_urls": list(artifact.source_urls),
        "entities": [asdict(entity) for entity in artifact.entities],
        "summary": artifact.summary,
    }
    resolved.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return resolved


def persist_review_rows(rows: Sequence[ReviewQueueRow], path: str | Path) -> Path:
    """Persist manual-audit rows as CSV."""

    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "reason",
        "entity_type",
        "canonical_name",
        "source_scope",
        "candidate_count",
        "status_values",
        "aliases",
        "doc_ids",
        "notes",
    ]
    with resolved.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "reason": row.reason,
                    "entity_type": row.entity_type,
                    "canonical_name": row.canonical_name,
                    "source_scope": row.source_scope,
                    "candidate_count": row.candidate_count,
                    "status_values": ";".join(row.status_values),
                    "aliases": ";".join(row.aliases),
                    "doc_ids": ";".join(row.doc_ids),
                    "notes": row.notes,
                }
            )
    return resolved


__all__ = [
    "EXTRACTION_VERSION",
    "REGISTRY_ENTITY_TYPES",
    "REVIEW_STATE_AUTO_VERIFIED",
    "REVIEW_STATE_NEEDS_REVIEW",
    "SOURCE_SCOPE_LOCAL_OFFICIAL",
    "STATUS_VALUES",
    "RegistryArtifact",
    "RegistryEntity",
    "ReviewQueueRow",
    "build_registry_artifact",
    "extract_registry_candidates",
    "persist_registry_artifact",
    "persist_review_rows",
]
