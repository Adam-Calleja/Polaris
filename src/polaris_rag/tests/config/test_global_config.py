from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[2] / "config" / "global_config.py"
_SPEC = spec_from_file_location("test_global_config_module", _MODULE_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
GlobalConfig = _MODULE.GlobalConfig


def test_vector_stores_uses_explicit_multi_store_mapping() -> None:
    cfg = GlobalConfig(
        {
            "generator_llm": {},
            "evaluator_llm": {},
            "embedder": {},
            "vector_stores": {
                "docs": {"collection_name": "support_docs"},
                "tickets": {"collection_name": "support_tickets"},
            },
        }
    )

    assert cfg.vector_stores == {
        "docs": {"collection_name": "support_docs"},
        "tickets": {"collection_name": "support_tickets"},
    }
    assert cfg.vector_store == {"collection_name": "support_docs"}


def test_vector_stores_falls_back_to_legacy_single_store() -> None:
    cfg = GlobalConfig(
        {
            "generator_llm": {},
            "evaluator_llm": {},
            "embedder": {},
            "vector_store": {"collection_name": "legacy"},
        }
    )

    assert cfg.vector_stores == {
        "default": {"collection_name": "legacy"},
    }
    assert cfg.vector_store == {"collection_name": "legacy"}


def test_vector_store_prefers_default_when_docs_missing() -> None:
    cfg = GlobalConfig(
        {
            "generator_llm": {},
            "evaluator_llm": {},
            "embedder": {},
            "vector_stores": {
                "default": {"collection_name": "default_collection"},
                "tickets": {"collection_name": "support_tickets"},
            },
        }
    )

    assert cfg.vector_store == {"collection_name": "default_collection"}
