from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from polaris_rag.authority import (
    REVIEW_STATE_NEEDS_REVIEW,
    SOURCE_SCOPE_LOCAL_OFFICIAL,
    build_registry_artifact,
    extract_registry_candidates,
)
from polaris_rag.common import MarkdownDocument


def _doc(*, doc_id: str, source: str, title: str, text: str) -> MarkdownDocument:
    return MarkdownDocument(
        id=doc_id,
        document_type="html",
        text=text,
        metadata={"source": source, "title": title},
    )


def test_build_registry_artifact_is_deterministic() -> None:
    documents = [
        _doc(
            doc_id="https://docs.example.org/hpc/software-packages/tensorflow",
            source="https://docs.example.org/hpc/software-packages/tensorflow",
            title="TensorFlow - CSD3 1.0 documentation",
            text="# TensorFlow\n\nSupported version 2.16.1.\n\n```bash\nmodule load TensorFlow/2.16.1-foss-2023a-CUDA-12.1.1\n```",
        )
    ]

    first_artifact, first_review = build_registry_artifact(
        documents,
        homepage="https://docs.example.org/hpc/index.html",
        source_urls=["https://docs.example.org/hpc/software-packages/tensorflow"],
    )
    second_artifact, second_review = build_registry_artifact(
        documents,
        homepage="https://docs.example.org/hpc/index.html",
        source_urls=["https://docs.example.org/hpc/software-packages/tensorflow"],
    )

    assert first_artifact == second_artifact
    assert first_review == second_review


def test_extract_registry_candidates_uses_title_and_path_heuristics() -> None:
    documents = [
        _doc(
            doc_id="https://docs.example.org/hpc/software-packages/tensorflow",
            source="https://docs.example.org/hpc/software-packages/tensorflow",
            title="TensorFlow - CSD3 1.0 documentation",
            text="# TensorFlow\n\nVersion 2.16.1 is supported.",
        ),
        _doc(
            doc_id="https://docs.example.org/hpc/user-guide/login-web",
            source="https://docs.example.org/hpc/user-guide/login-web",
            title="Login-Web Interface - CSD3 1.0 documentation",
            text="# Login-Web Interface\n\nThis service is available to users.",
        ),
        _doc(
            doc_id="https://docs.example.org/hpc/user-guide/a100",
            source="https://docs.example.org/hpc/user-guide/a100",
            title="Ampere GPU Nodes - CSD3 1.0 documentation",
            text="# Ampere GPU Nodes\n\nThese nodes provide accelerator access.",
        ),
    ]

    candidates, review_rows = extract_registry_candidates(
        documents,
        source_scope=SOURCE_SCOPE_LOCAL_OFFICIAL,
    )

    assert review_rows == []
    by_type = {(candidate.entity_type, candidate.canonical_name): candidate for candidate in candidates}
    assert ("software", "TensorFlow") in by_type
    assert ("service", "Login-Web Interface") in by_type
    assert ("system", "Ampere GPU Nodes") in by_type


def test_extract_registry_candidates_parses_module_and_toolchain_entities() -> None:
    document = _doc(
        doc_id="https://docs.example.org/hpc/software-packages/tensorflow",
        source="https://docs.example.org/hpc/software-packages/tensorflow",
        title="TensorFlow - CSD3 1.0 documentation",
        text=(
            "# TensorFlow\n\n"
            "```bash\n"
            "module load TensorFlow/2.16.1-foss-2023a-CUDA-12.1.1\n"
            "module load rhel8/default-amp\n"
            "```"
        ),
    )

    candidates, _ = extract_registry_candidates([document], source_scope=SOURCE_SCOPE_LOCAL_OFFICIAL)
    names = {(candidate.entity_type, candidate.canonical_name) for candidate in candidates}

    assert ("module", "TensorFlow/2.16.1-foss-2023a-CUDA-12.1.1") in names
    assert ("toolchain", "rhel8/default-amp") in names
    assert ("toolchain", "foss-2023a-CUDA-12.1.1") not in names


def test_build_registry_artifact_extracts_versions_and_lifecycle_status() -> None:
    document = _doc(
        doc_id="https://docs.example.org/hpc/user-guide/wbic",
        source="https://docs.example.org/hpc/user-guide/wbic",
        title="WBIC Platform End-of-Life - CSD3 1.0 documentation",
        text="# WBIC Platform\n\nThe platform is end of life. Version 1.2 remains documented for legacy migrations.",
    )

    artifact, review_rows = build_registry_artifact(
        [document],
        homepage="https://docs.example.org/hpc/index.html",
        source_urls=[document.metadata["source"]],
    )

    assert review_rows == []
    entity = next(item for item in artifact.entities if item.entity_type == "system")
    assert entity.canonical_name == "WBIC-HPHI Platform"
    assert entity.status == "eol"
    assert "1.2" in entity.known_versions


def test_build_registry_artifact_emits_review_rows_for_conflicting_statuses() -> None:
    documents = [
        _doc(
            doc_id="https://docs.example.org/hpc/user-guide/login-web",
            source="https://docs.example.org/hpc/user-guide/login-web",
            title="Login-Web Interface - CSD3 1.0 documentation",
            text="# Login-Web Interface\n\nThis service is in maintenance.",
        ),
        _doc(
            doc_id="https://docs.example.org/hpc/user-guide/login-web-alt",
            source="https://docs.example.org/hpc/user-guide/login-web-alt",
            title="Login Web Interface - CSD3 1.0 documentation",
            text="# Login Web Interface\n\nThis service is end of life.",
        ),
    ]

    artifact, review_rows = build_registry_artifact(
        documents,
        homepage="https://docs.example.org/hpc/index.html",
        source_urls=[doc.metadata["source"] for doc in documents],
    )

    assert any(row.reason == "conflicting_status" for row in review_rows)
    entity = next(item for item in artifact.entities if item.entity_type == "service")
    assert entity.review_state == REVIEW_STATE_NEEDS_REVIEW
    assert entity.status == "unknown"


def test_build_registry_artifact_emits_review_rows_for_alias_ambiguity() -> None:
    documents = [
        _doc(
            doc_id="https://docs.example.org/hpc/user-guide/login-web",
            source="https://docs.example.org/hpc/user-guide/login-web",
            title="Login-Web Interface - CSD3 1.0 documentation",
            text="# Login-Web Interface\n\nCurrent service documentation.",
        ),
        _doc(
            doc_id="https://docs.example.org/hpc/user-guide/login_web",
            source="https://docs.example.org/hpc/user-guide/login_web",
            title="Login Web Portal - CSD3 1.0 documentation",
            text="# Login Web Portal\n\nCurrent service documentation.",
        ),
    ]

    artifact, review_rows = build_registry_artifact(
        documents,
        homepage="https://docs.example.org/hpc/index.html",
        source_urls=[doc.metadata["source"] for doc in documents],
    )

    assert any(row.reason == "alias_ambiguity" for row in review_rows)
    assert any(entity.review_state == REVIEW_STATE_NEEDS_REVIEW for entity in artifact.entities)


def test_registry_entities_always_include_provenance() -> None:
    document = _doc(
        doc_id="https://docs.example.org/hpc/software-packages/tensorflow",
        source="https://docs.example.org/hpc/software-packages/tensorflow",
        title="TensorFlow - CSD3 1.0 documentation",
        text="# TensorFlow\n\nSupported version 2.16.1.\n\n```bash\nmodule load TensorFlow/2.16.1-foss-2023a-CUDA-12.1.1\n```",
    )

    artifact, _ = build_registry_artifact(
        [document],
        homepage="https://docs.example.org/hpc/index.html",
        source_urls=[document.metadata["source"]],
    )

    assert artifact.entities
    for entity in artifact.entities:
        assert entity.doc_id
        assert entity.doc_title
        assert entity.evidence_spans
        for span in entity.evidence_spans:
            assert span["doc_id"]
            assert span["doc_title"]
            assert "heading_path" in span
            assert span["evidence_text"]


def test_procedural_pages_do_not_emit_missing_canonical_review_rows() -> None:
    document = _doc(
        doc_id="https://docs.example.org/hpc/user-guide/quickstart.html",
        source="https://docs.example.org/hpc/user-guide/quickstart.html",
        title="",
        text="# Quick Start\n\nUse `module load <module>` to load the module you need.",
    )

    _, review_rows = extract_registry_candidates([document], source_scope=SOURCE_SCOPE_LOCAL_OFFICIAL)

    assert review_rows == []


def test_source_pages_are_excluded_from_primary_entity_review() -> None:
    document = _doc(
        doc_id="https://docs.example.org/hpc/_sources/index.rst.txt",
        source="https://docs.example.org/hpc/_sources/index.rst.txt",
        title="",
        text="Welcome to the docs source.",
    )

    candidates, review_rows = extract_registry_candidates([document], source_scope=SOURCE_SCOPE_LOCAL_OFFICIAL)

    assert candidates == []
    assert review_rows == []


def test_primary_entities_use_clean_names_from_overrides_and_headings() -> None:
    documents = [
        _doc(
            doc_id="https://docs.example.org/hpc/user-guide/login-web.html",
            source="https://docs.example.org/hpc/user-guide/login-web.html",
            title="",
            text="# Login-Web Interface[¶](#login-web-interface)\n\nThis service is available.",
        ),
        _doc(
            doc_id="https://docs.example.org/hpc/software-packages/openmm.html",
            source="https://docs.example.org/hpc/software-packages/openmm.html",
            title="",
            text="# OpenMM[¶](#openmm)\n\nOpenMM is available.",
        ),
    ]

    candidates, review_rows = extract_registry_candidates(documents, source_scope=SOURCE_SCOPE_LOCAL_OFFICIAL)

    assert review_rows == []
    names = {(candidate.entity_type, candidate.canonical_name) for candidate in candidates}
    assert ("service", "Login-Web Interface") in names
    assert ("software", "OpenMM") in names


def test_partition_extraction_only_uses_scheduler_contexts() -> None:
    document = _doc(
        doc_id="https://docs.example.org/hpc/user-guide/icelake.html",
        source="https://docs.example.org/hpc/user-guide/icelake.html",
        title="Ice Lake Nodes - CSD3 1.0 documentation",
        text=(
            "# Ice Lake Nodes\n\n"
            "In this case you should be able to simply specify the icelake partition to the -p sbatch directive.\n\n"
            "```bash\n"
            "#SBATCH -p icelake\n"
            "sbatch -p cclake job.sh\n"
            "./run_alphafold.sh -p monomer_ptm\n"
            "```\n"
        ),
    )

    candidates, _ = extract_registry_candidates([document], source_scope=SOURCE_SCOPE_LOCAL_OFFICIAL)
    partitions = {candidate.canonical_name for candidate in candidates if candidate.entity_type == "partition"}

    assert "icelake" in partitions
    assert "cclake" in partitions
    assert "sbatch" not in partitions
    assert "monomer_ptm" not in partitions


def test_module_extraction_rejects_placeholders_and_prose_false_positives() -> None:
    document = _doc(
        doc_id="https://docs.example.org/hpc/user-guide/quickstart.html",
        source="https://docs.example.org/hpc/user-guide/quickstart.html",
        title="Quick Start - CSD3 1.0 documentation",
        text=(
            "# Quick Start\n\n"
            "Use `module load <module>` as a placeholder example.\n\n"
            "```bash\n"
            "module purge && module load rhel8/default-amp\n"
            "```\n"
        ),
    )

    candidates, _ = extract_registry_candidates([document], source_scope=SOURCE_SCOPE_LOCAL_OFFICIAL)
    modules = {candidate.canonical_name for candidate in candidates if candidate.entity_type == "module"}

    assert "rhel8/default-amp" in modules
    assert "<module>" not in modules
    assert "load" not in modules


def test_toolchain_extraction_is_conservative() -> None:
    document = _doc(
        doc_id="https://docs.example.org/hpc/software-packages/lammps.html",
        source="https://docs.example.org/hpc/software-packages/lammps.html",
        title="LAMMPS - CSD3 1.0 documentation",
        text=(
            "# LAMMPS\n\n"
            "```bash\n"
            "module load intel-oneapi-mkl/2024.1.0/intel/vnktbkgm\n"
            "module load rhel8/default-ccl\n"
            "```\n"
        ),
    )

    candidates, _ = extract_registry_candidates([document], source_scope=SOURCE_SCOPE_LOCAL_OFFICIAL)
    toolchains = {candidate.canonical_name for candidate in candidates if candidate.entity_type == "toolchain"}

    assert "rhel8/default-ccl" in toolchains
    assert "oneapi-mkl/2024.1.0/intel/vnktbkgm" not in toolchains


def test_partition_placeholders_are_rejected() -> None:
    document = _doc(
        doc_id="https://docs.example.org/hpc/user-guide/quickstart.html",
        source="https://docs.example.org/hpc/user-guide/quickstart.html",
        title="Quick Start - CSD3 1.0 documentation",
        text=(
            "# Quick Start\n\n"
            "```bash\n"
            "#SBATCH -p partition\n"
            "```\n"
        ),
    )

    candidates, _ = extract_registry_candidates([document], source_scope=SOURCE_SCOPE_LOCAL_OFFICIAL)
    partitions = {candidate.canonical_name for candidate in candidates if candidate.entity_type == "partition"}

    assert "partition" not in partitions


def test_long_partitions_are_recovered_from_partition_prose() -> None:
    document = _doc(
        doc_id="https://docs.example.org/hpc/user-guide/long.html",
        source="https://docs.example.org/hpc/user-guide/long.html",
        title="Long jobs - CSD3 1.0 documentation",
        text=(
            "# Long jobs\n\n"
            "Use of QOSL is tied to the -long partitions. The currently available long partitions are "
            "`ampere-long`, `cclake-long`, and `icelake-long`.\n"
        ),
    )

    candidates, _ = extract_registry_candidates([document], source_scope=SOURCE_SCOPE_LOCAL_OFFICIAL)
    partitions = {
        (candidate.canonical_name, candidate.status)
        for candidate in candidates
        if candidate.entity_type == "partition"
    }

    assert ("ampere-long", "current") in partitions
    assert ("cclake-long", "current") in partitions
    assert ("icelake-long", "current") in partitions


def test_pvc_partition_includes_pvc_alias() -> None:
    document = _doc(
        doc_id="https://docs.example.org/hpc/user-guide/pvc.html",
        source="https://docs.example.org/hpc/user-guide/pvc.html",
        title="Dawn - Intel GPU (PVC) Nodes - CSD3 1.0 documentation",
        text=(
            "# Dawn - Intel GPU (PVC) Nodes\n\n"
            "These nodes entered Early Access service in January 2024.\n\n"
            "```bash\n"
            "#SBATCH -p pvc9\n"
            "```\n"
        ),
    )

    artifact, _ = build_registry_artifact(
        [document],
        homepage="https://docs.example.org/hpc/index.html",
        source_urls=[document.metadata["source"]],
    )

    pvc_partition = next(
        entity for entity in artifact.entities if entity.entity_type == "partition" and entity.canonical_name == "pvc9"
    )
    assert "pvc" in pvc_partition.aliases
    assert pvc_partition.status == "current"


def test_primary_status_overrides_prevent_page_level_status_leakage() -> None:
    documents = [
        _doc(
            doc_id="https://docs.example.org/hpc/user-guide/mfa.html",
            source="https://docs.example.org/hpc/user-guide/mfa.html",
            title="MultiFactor Authentication (MFA) - CSD3 1.0 documentation",
            text="# MultiFactor Authentication (MFA)\n\nMandatory MFA was introduced. You can read the archived full maintenance here.",
        ),
        _doc(
            doc_id="https://docs.example.org/hpc/user-guide/pvc.html",
            source="https://docs.example.org/hpc/user-guide/pvc.html",
            title="Dawn - Intel GPU (PVC) Nodes - CSD3 1.0 documentation",
            text="# Dawn - Intel GPU (PVC) Nodes\n\nThese new nodes entered Early Access service in January 2024.\n\nOld compilers are deprecated.",
        ),
        _doc(
            doc_id="https://docs.example.org/hpc/user-guide/turbovnc.html",
            source="https://docs.example.org/hpc/user-guide/turbovnc.html",
            title="Connecting to CSD3 via TurboVNC (3D Visualisation) - CSD3 1.0 documentation",
            text="# Connecting to CSD3 via TurboVNC (3D Visualisation)\n\nThe old nodes retired in 2023, but TurboVNC remains the access route.",
        ),
        _doc(
            doc_id="https://docs.example.org/hpc/user-guide/cclake.html",
            source="https://docs.example.org/hpc/user-guide/cclake.html",
            title="Cascade Lake Nodes - CSD3 1.0 documentation",
            text="# Cascade Lake Nodes\n\nThe Cascade Lake upgrade entered general service in November 2020.",
        ),
        _doc(
            doc_id="https://docs.example.org/hpc/user-guide/sapphire-hbm.html",
            source="https://docs.example.org/hpc/user-guide/sapphire-hbm.html",
            title="Sapphire Rapid Nodes with High Bandwidth Memory - CSD3 1.0 documentation",
            text="# Sapphire Rapid Nodes with High Bandwidth Memory\n\nThese nodes entered general service in October 2024.",
        ),
    ]

    artifact, _ = build_registry_artifact(
        documents,
        homepage="https://docs.example.org/hpc/index.html",
        source_urls=[document.metadata["source"] for document in documents],
    )

    by_name = {entity.canonical_name: entity for entity in artifact.entities}
    assert by_name["MultiFactor Authentication (MFA)"].status == "current"
    assert by_name["Dawn - Intel GPU (PVC) Nodes"].status == "current"
    assert by_name["TurboVNC"].status == "current"
    assert by_name["Cascade Lake Nodes"].status == "current"
    assert by_name["Sapphire Rapid Nodes with High Bandwidth Memory"].status == "current"


def test_legacy_module_names_are_not_marked_current() -> None:
    document = _doc(
        doc_id="https://docs.example.org/hpc/user-guide/quickstart.html",
        source="https://docs.example.org/hpc/user-guide/quickstart.html",
        title="Quick Start - CSD3 1.0 documentation",
        text=(
            "# Quick Start\n\n"
            "Please delete old module lines such as:\n\n"
            "```bash\n"
            "module load default-impi\n"
            "module load default-wilkes\n"
            "module load rhel7/default-peta4\n"
            "```\n"
        ),
    )

    artifact, _ = build_registry_artifact(
        [document],
        homepage="https://docs.example.org/hpc/index.html",
        source_urls=[document.metadata["source"]],
    )

    by_key = {(entity.entity_type, entity.canonical_name): entity for entity in artifact.entities}
    assert by_key[("module", "default-impi")].status == "legacy"
    assert by_key[("module", "default-wilkes")].status == "legacy"
    assert by_key[("module", "rhel7/default-peta4")].status == "legacy"
    assert by_key[("toolchain", "default-impi")].status == "legacy"
    assert by_key[("toolchain", "default-wilkes")].status == "legacy"
    assert by_key[("toolchain", "rhel7/default-peta4")].status == "legacy"


def test_module_extraction_handles_chained_module_commands() -> None:
    document = _doc(
        doc_id="https://docs.example.org/hpc/software-packages/matlab.html",
        source="https://docs.example.org/hpc/software-packages/matlab.html",
        title="MATLAB - CSD3 1.0 documentation",
        text=(
            "# MATLAB\n\n"
            "```bash\n"
            "module purge module load rhel8/global matlab/r2024a\n"
            "module purge && module load rhel8/default-ccl && module load openfoam/2312/gcc/intel-oneapi-mpi/2ioxyvgw\n"
            "```\n"
        ),
    )

    candidates, _ = extract_registry_candidates([document], source_scope=SOURCE_SCOPE_LOCAL_OFFICIAL)
    names = {(candidate.entity_type, candidate.canonical_name) for candidate in candidates}

    assert ("module", "matlab/r2024a") in names
    assert ("module", "openfoam/2312/gcc/intel-oneapi-mpi/2ioxyvgw") in names


def test_module_extraction_handles_indented_code_blocks() -> None:
    document = _doc(
        doc_id="https://docs.example.org/hpc/software-packages/openfoam.html",
        source="https://docs.example.org/hpc/software-packages/openfoam.html",
        title="OpenFOAM - CSD3 1.0 documentation",
        text=(
            "# OpenFOAM\n\n"
            "    #!/bin/bash\n"
            "    #SBATCH -p cclake\n"
            "    module purge\n"
            "    module load rhel8/cclake/base\n"
            "    module load openfoam/2312/gcc/intel-oneapi-mpi/2ioxyvgw\n"
        ),
    )

    candidates, _ = extract_registry_candidates([document], source_scope=SOURCE_SCOPE_LOCAL_OFFICIAL)
    names = {(candidate.entity_type, candidate.canonical_name) for candidate in candidates}

    assert ("module", "rhel8/cclake/base") in names
    assert ("module", "openfoam/2312/gcc/intel-oneapi-mpi/2ioxyvgw") in names


def test_module_extraction_handles_shell_comments_inside_fenced_code_blocks() -> None:
    document = _doc(
        doc_id="https://docs.example.org/hpc/software-packages/matlab.html",
        source="https://docs.example.org/hpc/software-packages/matlab.html",
        title="MATLAB - CSD3 1.0 documentation",
        text=(
            "# MATLAB\n\n"
            "## Running MATLAB on CSD3\n\n"
            "```\n"
            "#!/bin/bash\n"
            "#SBATCH -A MYACCOUNT-CHANGEME\n"
            "#SBATCH -p icelake # Can change this to a different partition\n"
            "#SBATCH -N 1\n"
            "#SBATCH -n 1 # MATLAB r2024b minimum requirements are 2 CPUs and 8 GB RAM.\n"
            "#SBATCH -t 00:10:00\n"
            "module purge\n"
            "module load rhel8/global matlab/r2024a\n"
            "# Using -nojvm to turn off MATLAB JVM and its long, associated overhead.\n"
            "# Use with care, as JVM is used in several toolboxes like the parallel toolbox\n"
            "matlab -nodisplay -nojvm -r \"example('output_file'); quit\"\n"
            "```\n"
        ),
    )

    candidates, _ = extract_registry_candidates([document], source_scope=SOURCE_SCOPE_LOCAL_OFFICIAL)
    names = {(candidate.entity_type, candidate.canonical_name) for candidate in candidates}

    assert ("module", "rhel8/global") in names
    assert ("module", "matlab/r2024a") in names


def test_module_extraction_ignores_inline_explanations_after_module_names() -> None:
    document = _doc(
        doc_id="https://docs.example.org/hpc/software-packages/pytorch.html",
        source="https://docs.example.org/hpc/software-packages/pytorch.html",
        title="PyTorch - CSD3 1.0 documentation",
        text=(
            "# PyTorch\n\n"
            "```bash\n"
            "module load rhel8/default-ccl # recommended for all RHEL8 CPU partitions (cclake, icelake, sapphire)\n"
            "```\n"
        ),
    )

    candidates, _ = extract_registry_candidates([document], source_scope=SOURCE_SCOPE_LOCAL_OFFICIAL)
    modules = {candidate.canonical_name for candidate in candidates if candidate.entity_type == "module"}

    assert "rhel8/default-ccl" in modules
    assert "#" not in modules
    assert "recommended" not in modules
    assert "all" not in modules
    assert "CPU" not in modules
    assert "sapphire)" not in modules


def test_software_entities_default_to_current_and_use_targeted_versions() -> None:
    documents = [
        _doc(
            doc_id="https://docs.example.org/hpc/software-packages/gromacs.html",
            source="https://docs.example.org/hpc/software-packages/gromacs.html",
            title="GROMACS - CSD3 1.0 documentation",
            text=(
                "# GROMACS\n\n"
                "GROMACS is supported on each of the hardware partitions on CSD3.\n\n"
                "```bash\n"
                "module load gromacs/2024.4/openmpi-4.1.1/gcc-9.4.0-hzwzjqx\n"
                "```\n"
            ),
        ),
        _doc(
            doc_id="https://docs.example.org/hpc/software-tools/python.html",
            source="https://docs.example.org/hpc/software-tools/python.html",
            title="Using Python - CSD3 1.0 documentation",
            text=(
                "# Using Python\n\n"
                "CSD3 provides central installations of both Python 2 and Python 3.\n"
                "Recent versions of the Python interpreter can be accessed as python/2.7, python/3.5, "
                "python/3.6, python/3.7 or python/3.8. The default is currently python/3.6.\n"
            ),
        ),
        _doc(
            doc_id="https://docs.example.org/hpc/software-packages/tensorflow.html",
            source="https://docs.example.org/hpc/software-packages/tensorflow.html",
            title="Tensorflow - CSD3 1.0 documentation",
            text=(
                "# Tensorflow\n\n"
                "A python package for tensorflow is available on CSD3.\n\n"
                "```bash\n"
                "module load python/3.8.11/gcc-9.4.0-yb6rzr6\n"
                "module load cuda/12.1 cudnn/8.9_cuda-12.1\n"
                "```\n"
            ),
        ),
        _doc(
            doc_id="https://docs.example.org/hpc/software-packages/pytorch.html",
            source="https://docs.example.org/hpc/software-packages/pytorch.html",
            title="PyTorch - CSD3 1.0 documentation",
            text=(
                "# PyTorch\n\n"
                "The python installation will depend on the target partition.\n\n"
                "```bash\n"
                "module load rhel8/default-ccl\n"
                "module load python/3.11.9/gcc/nptrdpll\n"
                "```\n"
                "# install pytorch into your virtual env "
                "(instructions from Intel: https://pytorch-extension.intel.com/installation?platform=cpu&version=v2.7.0%2Bcpu&os=linux%2Fwsl2&package=pip)\n"
            ),
        ),
        _doc(
            doc_id="https://docs.example.org/hpc/software-packages/openfoam.html",
            source="https://docs.example.org/hpc/software-packages/openfoam.html",
            title="OpenFOAM - CSD3 1.0 documentation",
            text=(
                "# OpenFOAM\n\n"
                "Version 2312 is currently installed.\n\n"
                "```bash\n"
                "module load openfoam/2312/gcc/intel-oneapi-mpi/2ioxyvgw\n"
                "```\n"
            ),
        ),
        _doc(
            doc_id="https://docs.example.org/hpc/software-packages/matlab.html",
            source="https://docs.example.org/hpc/software-packages/matlab.html",
            title="MATLAB - CSD3 1.0 documentation",
            text=(
                "# MATLAB\n\n"
                "The matlab module loads the latest version installed on CSD3.\n\n"
                "```bash\n"
                "module purge module load rhel8/global matlab/r2024a\n"
                "```\n"
            ),
        ),
    ]

    artifact, _ = build_registry_artifact(
        documents,
        homepage="https://docs.example.org/hpc/index.html",
        source_urls=[document.metadata["source"] for document in documents],
    )

    by_key = {(entity.entity_type, entity.canonical_name): entity for entity in artifact.entities}

    assert by_key[("software", "GROMACS")].status == "current"
    assert by_key[("software", "GROMACS")].known_versions == ["2024.4"]

    assert by_key[("software", "Python")].status == "current"
    assert by_key[("software", "Python")].known_versions == ["2.7", "3.5", "3.6", "3.7", "3.8"]

    assert by_key[("software", "Tensorflow")].status == "current"
    assert by_key[("software", "Tensorflow")].known_versions == []

    assert by_key[("software", "PyTorch")].status == "current"
    assert by_key[("software", "PyTorch")].known_versions == []

    assert by_key[("software", "OpenFOAM")].status == "current"
    assert by_key[("software", "OpenFOAM")].known_versions == ["2312"]

    assert by_key[("software", "MATLAB")].status == "current"
    assert by_key[("software", "MATLAB")].known_versions == ["r2024a"]


def test_module_and_toolchain_versions_come_from_tokens_not_whole_sections() -> None:
    document = _doc(
        doc_id="https://docs.example.org/hpc/software-packages/mixed.html",
        source="https://docs.example.org/hpc/software-packages/mixed.html",
        title="Mixed Modules - CSD3 1.0 documentation",
        text=(
            "# Mixed Modules\n\n"
            "```bash\n"
            "#SBATCH -p icelake\n"
            "module purge\n"
            "module load rhel8/global matlab/r2024a\n"
            "module load python/3.8.11/gcc-9.4.0-yb6rzr6\n"
            "module load gromacs/2021.3/openmpi-4.1.1/gcc-9.4.0-hzwzjqx\n"
            "```\n"
        ),
    )

    artifact, _ = build_registry_artifact(
        [document],
        homepage="https://docs.example.org/hpc/index.html",
        source_urls=[document.metadata["source"]],
    )

    by_key = {(entity.entity_type, entity.canonical_name): entity for entity in artifact.entities}

    assert by_key[("module", "matlab/r2024a")].known_versions == ["r2024a"]
    assert by_key[("toolchain", "rhel8/global")].known_versions == []
    assert by_key[("module", "python/3.8.11/gcc-9.4.0-yb6rzr6")].known_versions == ["3.8.11", "9.4.0"]
    assert by_key[("module", "gromacs/2021.3/openmpi-4.1.1/gcc-9.4.0-hzwzjqx")].known_versions == [
        "2021.3",
        "4.1.1",
        "9.4.0",
    ]


def test_prefixed_module_versions_keep_first_semantic_version() -> None:
    document = _doc(
        doc_id="https://docs.example.org/hpc/software-tools/spack.html",
        source="https://docs.example.org/hpc/software-tools/spack.html",
        title="Using Spack - CSD3 1.0 documentation",
        text=(
            "# Using Spack\n\n"
            "```bash\n"
            "module load boost-1.64.0-gcc-6.2.0-pftxg46\n"
            "```\n"
        ),
    )

    artifact, _ = build_registry_artifact(
        [document],
        homepage="https://docs.example.org/hpc/index.html",
        source_urls=[document.metadata["source"]],
    )

    by_key = {(entity.entity_type, entity.canonical_name): entity for entity in artifact.entities}

    assert by_key[("module", "boost-1.64.0-gcc-6.2.0-pftxg46")].known_versions == ["1.64.0"]


def test_primary_overrides_cover_additional_software_pages() -> None:
    documents = [
        _doc(
            doc_id="https://docs.example.org/hpc/software-packages/castep.html",
            source="https://docs.example.org/hpc/software-packages/castep.html",
            title="",
            text="# CASTEP\n\nCASTEP is available on CSD3.",
        ),
        _doc(
            doc_id="https://docs.example.org/hpc/software-packages/gaussian.html",
            source="https://docs.example.org/hpc/software-packages/gaussian.html",
            title="",
            text="# Gaussian\n\nGaussian is available on CSD3.",
        ),
    ]

    candidates, review_rows = extract_registry_candidates(documents, source_scope=SOURCE_SCOPE_LOCAL_OFFICIAL)

    assert review_rows == []
    names = {(candidate.entity_type, candidate.canonical_name) for candidate in candidates}
    assert ("software", "CASTEP") in names
    assert ("software", "Gaussian") in names
