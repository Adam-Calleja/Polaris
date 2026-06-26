from __future__ import annotations

import json
from pathlib import Path

from polaris_rag.retrieval.query_constraints import AuthorityQueryConstraintParser, serialize_query_constraints


def _entity(
    *,
    entity_id: str,
    entity_type: str,
    canonical_name: str,
    aliases: list[str],
    known_versions: list[str] | None = None,
    status: str = "current",
    doc_id: str = "https://docs.example.org/hpc/example.html",
) -> dict[str, object]:
    return {
        "entity_id": entity_id,
        "entity_type": entity_type,
        "canonical_name": canonical_name,
        "aliases": aliases,
        "source_scope": "local_official",
        "status": status,
        "known_versions": list(known_versions or []),
        "doc_id": doc_id,
        "doc_title": canonical_name,
        "heading_path": [canonical_name],
        "evidence_spans": [{"text": canonical_name}],
        "extraction_method": "unit_test",
        "review_state": "auto_verified",
    }


def _write_registry(tmp_path: Path, entities: list[dict[str, object]]) -> Path:
    path = tmp_path / "registry.local_official.v1.json"
    path.write_text(
        json.dumps(
            {
                "build": {
                    "extraction_version": "authority_registry_v1",
                    "source_scope": "local_official",
                },
                "source_urls": sorted({str(entity["doc_id"]) for entity in entities}),
                "entities": entities,
                "summary": {"entity_count": len(entities)},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def test_query_parser_extracts_unique_software_alias(tmp_path: Path) -> None:
    registry_path = _write_registry(
        tmp_path,
        entities=[
            _entity(
                entity_id="software-lammps",
                entity_type="software",
                canonical_name="LAMMPS",
                aliases=["LAMMPS", "lammps"],
                known_versions=["2024.4"],
            ),
        ],
    )
    parser = AuthorityQueryConstraintParser.from_registry_artifact(registry_path)

    constraints = parser.parse("How do I compile LAMMPS on CSD3?")

    assert constraints.software_names == ["LAMMPS"]
    assert constraints.partition_names == []
    assert constraints.system_names == []
    assert constraints.scope_family_names == []
    assert constraints.query_type == "general_how_to"


def test_query_parser_leaves_ambiguous_partition_system_alias_neutral_without_context(tmp_path: Path) -> None:
    registry_path = _write_registry(
        tmp_path,
        entities=[
            _entity(
                entity_id="partition-cclake",
                entity_type="partition",
                canonical_name="cclake",
                aliases=["cclake"],
            ),
            _entity(
                entity_id="system-cclake",
                entity_type="system",
                canonical_name="Cascade Lake Nodes",
                aliases=["cclake", "cascade lake nodes"],
            ),
        ],
    )
    parser = AuthorityQueryConstraintParser.from_registry_artifact(registry_path)

    constraints = parser.parse("Can I use cclake?")

    assert constraints.partition_names == []
    assert constraints.system_names == []
    assert constraints.scope_family_names == ["cclake"]
    assert constraints.scope_required is True
    assert constraints.query_type == "local_operational"


def test_query_parser_resolves_partition_alias_from_scheduler_syntax(tmp_path: Path) -> None:
    registry_path = _write_registry(
        tmp_path,
        entities=[
            _entity(
                entity_id="partition-cclake",
                entity_type="partition",
                canonical_name="cclake",
                aliases=["cclake"],
            ),
            _entity(
                entity_id="system-cclake",
                entity_type="system",
                canonical_name="Cascade Lake Nodes",
                aliases=["cclake", "cascade lake nodes"],
            ),
        ],
    )
    parser = AuthorityQueryConstraintParser.from_registry_artifact(registry_path)

    constraints = parser.parse("sbatch -p cclake run_job.sh")

    assert constraints.partition_names == ["cclake"]
    assert constraints.system_names == []
    assert constraints.scope_family_names == ["cclake"]
    assert constraints.scope_required is True
    assert constraints.query_type == "local_operational"


def test_query_parser_resolves_family_alias_from_run_on_phrase(tmp_path: Path) -> None:
    registry_path = _write_registry(
        tmp_path,
        entities=[
            _entity(
                entity_id="software-gromacs",
                entity_type="software",
                canonical_name="GROMACS",
                aliases=["GROMACS", "gromacs"],
                known_versions=["2024.4"],
            ),
            _entity(
                entity_id="partition-cclake",
                entity_type="partition",
                canonical_name="cclake",
                aliases=["cclake"],
            ),
            _entity(
                entity_id="system-cclake",
                entity_type="system",
                canonical_name="Cascade Lake Nodes",
                aliases=["cclake", "cascade lake nodes"],
            ),
        ],
    )
    parser = AuthorityQueryConstraintParser.from_registry_artifact(registry_path)

    constraints = parser.parse("How do I run GROMACS 2024.4 on cclake?")

    assert constraints.software_names == ["GROMACS"]
    assert constraints.software_versions == ["2024.4"]
    assert constraints.partition_names == []
    assert constraints.system_names == []
    assert constraints.scope_family_names == ["cclake"]
    assert constraints.scope_required is True
    assert constraints.version_sensitive_guess is True
    assert constraints.query_type == "software_version"


def test_query_parser_extracts_software_versions_and_marks_version_sensitive(tmp_path: Path) -> None:
    registry_path = _write_registry(
        tmp_path,
        entities=[
            _entity(
                entity_id="software-lammps",
                entity_type="software",
                canonical_name="LAMMPS",
                aliases=["LAMMPS", "lammps"],
                known_versions=["2024.4", "2025.1"],
            ),
        ],
    )
    parser = AuthorityQueryConstraintParser.from_registry_artifact(registry_path)

    constraints = parser.parse("How do I compile LAMMPS 2024.4?")

    assert constraints.software_names == ["LAMMPS"]
    assert constraints.software_versions == ["2024.4"]
    assert constraints.scope_family_names == []
    assert constraints.version_sensitive_guess is True
    assert constraints.query_type == "software_version"


def test_query_parser_keeps_generic_on_phrase_neutral_for_ambiguous_scope(tmp_path: Path) -> None:
    registry_path = _write_registry(
        tmp_path,
        entities=[
            _entity(
                entity_id="partition-cclake",
                entity_type="partition",
                canonical_name="cclake",
                aliases=["cclake"],
            ),
            _entity(
                entity_id="system-cclake",
                entity_type="system",
                canonical_name="Cascade Lake Nodes",
                aliases=["cclake", "cascade lake nodes"],
            ),
        ],
    )
    parser = AuthorityQueryConstraintParser.from_registry_artifact(registry_path)

    constraints = parser.parse("What documentation exists on cclake?")

    assert constraints.partition_names == []
    assert constraints.system_names == []
    assert constraints.scope_family_names == []
    assert constraints.scope_required is None


def test_query_parser_marks_explicit_version_mentions_sensitive_without_registry_version_match(tmp_path: Path) -> None:
    registry_path = _write_registry(
        tmp_path,
        entities=[
            _entity(
                entity_id="software-lammps",
                entity_type="software",
                canonical_name="LAMMPS",
                aliases=["LAMMPS", "lammps"],
                known_versions=[],
            ),
        ],
    )
    parser = AuthorityQueryConstraintParser.from_registry_artifact(registry_path)

    constraints = parser.parse("How do I compile LAMMPS 2024.4?")

    assert constraints.software_names == ["LAMMPS"]
    assert constraints.software_versions == []
    assert constraints.scope_family_names == []
    assert constraints.version_sensitive_guess is True
    assert constraints.query_type == "software_version"


def test_query_parser_extracts_module_and_toolchain_constraints(tmp_path: Path) -> None:
    registry_path = _write_registry(
        tmp_path,
        entities=[
            _entity(
                entity_id="software-gromacs",
                entity_type="software",
                canonical_name="GROMACS",
                aliases=["GROMACS", "gromacs"],
                known_versions=["2024.4"],
            ),
            _entity(
                entity_id="module-gromacs",
                entity_type="module",
                canonical_name="gromacs/2024.4",
                aliases=["gromacs/2024.4"],
                known_versions=["2024.4"],
            ),
            _entity(
                entity_id="toolchain-cuda",
                entity_type="toolchain",
                canonical_name="cuda/12.1",
                aliases=["cuda/12.1"],
                known_versions=["12.1"],
            ),
            _entity(
                entity_id="module-rhel8-cclake-base",
                entity_type="module",
                canonical_name="rhel8/cclake/base",
                aliases=["rhel8/cclake/base"],
                doc_id="https://docs.example.org/hpc/cclake.html",
            ),
        ],
    )
    parser = AuthorityQueryConstraintParser.from_registry_artifact(registry_path)

    constraints = parser.parse("module load rhel8/cclake/base gromacs/2024.4 cuda/12.1")

    assert constraints.software_names == ["GROMACS"]
    assert constraints.scope_family_names == ["cclake"]
    assert constraints.module_names == ["gromacs/2024.4", "rhel8/cclake/base"]
    assert constraints.toolchain_names == ["cuda/12.1"]
    assert constraints.software_versions == ["2024.4"]
    assert constraints.toolchain_versions == ["12.1"]


def test_query_parser_returns_neutral_constraints_for_generic_query(tmp_path: Path) -> None:
    registry_path = _write_registry(
        tmp_path,
        entities=[
            _entity(
                entity_id="software-lammps",
                entity_type="software",
                canonical_name="LAMMPS",
                aliases=["LAMMPS", "lammps"],
            ),
        ],
    )
    parser = AuthorityQueryConstraintParser.from_registry_artifact(registry_path)

    constraints = parser.parse("Can you help me?")

    assert serialize_query_constraints(constraints) == {
        "query_type": None,
        "system_names": [],
        "partition_names": [],
        "service_names": [],
        "scope_family_names": [],
        "software_names": [],
        "software_versions": [],
        "module_names": [],
        "toolchain_names": [],
        "toolchain_versions": [],
        "scope_required": None,
        "version_sensitive_guess": None,
    }


def test_query_parser_does_not_infer_system_scope_for_generic_software_query(tmp_path: Path) -> None:
    registry_path = _write_registry(
        tmp_path,
        entities=[
            _entity(
                entity_id="software-gromacs",
                entity_type="software",
                canonical_name="GROMACS",
                aliases=["GROMACS", "gromacs"],
            ),
            _entity(
                entity_id="partition-cclake",
                entity_type="partition",
                canonical_name="cclake",
                aliases=["cclake"],
            ),
            _entity(
                entity_id="system-cclake",
                entity_type="system",
                canonical_name="Cascade Lake Nodes",
                aliases=["cclake", "cascade lake nodes"],
            ),
        ],
    )
    parser = AuthorityQueryConstraintParser.from_registry_artifact(registry_path)

    constraints = parser.parse("How do I install GROMACS?")

    assert constraints.software_names == ["GROMACS"]
    assert constraints.partition_names == []
    assert constraints.system_names == []
    assert constraints.scope_family_names == []
    assert constraints.scope_required is None


def test_query_parser_resolves_system_alias_and_family_from_nodes_phrase(tmp_path: Path) -> None:
    registry_path = _write_registry(
        tmp_path,
        entities=[
            _entity(
                entity_id="partition-cclake",
                entity_type="partition",
                canonical_name="cclake",
                aliases=["cclake"],
            ),
            _entity(
                entity_id="system-cclake",
                entity_type="system",
                canonical_name="Cascade Lake Nodes",
                aliases=["cclake", "cascade lake nodes"],
            ),
        ],
    )
    parser = AuthorityQueryConstraintParser.from_registry_artifact(registry_path)

    constraints = parser.parse("Which cclake nodes should I use?")

    assert constraints.partition_names == []
    assert constraints.system_names == ["Cascade Lake Nodes"]
    assert constraints.scope_family_names == ["cclake"]
    assert constraints.scope_required is True
    assert constraints.query_type == "local_operational"


def test_query_parser_is_deterministic(tmp_path: Path) -> None:
    registry_path = _write_registry(
        tmp_path,
        entities=[
            _entity(
                entity_id="software-lammps",
                entity_type="software",
                canonical_name="LAMMPS",
                aliases=["LAMMPS", "lammps"],
                known_versions=["2024.4"],
            ),
        ],
    )
    parser = AuthorityQueryConstraintParser.from_registry_artifact(registry_path)
    query = "How do I compile LAMMPS 2024.4?"

    first = json.dumps(serialize_query_constraints(parser.parse(query)), sort_keys=True)
    second = json.dumps(serialize_query_constraints(parser.parse(query)), sort_keys=True)

    assert first == second
