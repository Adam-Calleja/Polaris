from __future__ import annotations

from polaris_rag.app.demo_anonymizer import anonymize_query_payload


class _FakeDetectorLLM:
    def generate(self, prompt: str, **kwargs):  # noqa: ANN001
        return """
        {
          "names": [["Jane Smith"]],
          "email_addresses": ["jane.smith@example.com"],
          "phone_numbers": [],
          "usernames": ["abc123"],
          "project_codes": [],
          "account_numbers": [],
          "institutions": ["University of Cambridge"],
          "locations": [],
          "other_identifiable_info": [],
          "special_category_info": []
        }
        """


def test_anonymize_query_payload_redacts_answer_context_and_doc_ids() -> None:
    payload = anonymize_query_payload(
        answer=(
            "Resolved [1] for Jane Smith (jane.smith@example.com). "
            "Please ask abc123 to retry.\n\n"
            "REFERENCE KEY\n"
            "[1] : HPC-12345"
        ),
        context=[
            {
                "rank": 1,
                "doc_id": "HPC-12345",
                "text": (
                    "Jane Smith from University of Cambridge reported this via "
                    "jane.smith@example.com. Username abc123."
                ),
                "source": "tickets",
            }
        ],
        llm=_FakeDetectorLLM(),
        timeout_seconds=2.0,
    )

    assert payload.answer == (
        "Resolved [1] for PERSON_001 (EMAIL_001). "
        "Please ask USER_001 to retry.\n\n"
        "REFERENCE KEY\n"
        "[1] : TICKET_001"
    )
    assert payload.context == [
        {
            "rank": 1,
            "doc_id": "TICKET_001",
            "text": "PERSON_001 from ORG_001 reported this via EMAIL_001. Username USER_001.",
            "score": None,
            "source": "tickets",
        }
    ]
    assert payload.aliases["HPC-12345"] == "TICKET_001"
    assert payload.aliases["Jane Smith"] == "PERSON_001"


def test_anonymize_query_payload_leaves_documentation_context_unchanged() -> None:
    payload = anonymize_query_payload(
        answer=(
            "Resolved [1] for Jane Smith while the docs at [2] still recommend `module load gromacs`.\n\n"
            "REFERENCE KEY\n"
            "[1] : HPC-12345\n"
            "[2] : docs-gromacs-install"
        ),
        context=[
            {
                "rank": 1,
                "doc_id": "HPC-12345",
                "text": "Jane Smith reported this via jane.smith@example.com. Username abc123.",
                "source": "tickets",
            },
            {
                "rank": 2,
                "doc_id": "docs-gromacs-install",
                "text": "Load the module with `module load gromacs` and verify `gmx --version`.",
                "source": "docs",
            },
        ],
        llm=_FakeDetectorLLM(),
        timeout_seconds=2.0,
    )

    assert payload.answer == (
        "Resolved [1] for PERSON_001 while the docs at [2] still recommend `module load gromacs`.\n\n"
        "REFERENCE KEY\n"
        "[1] : TICKET_001\n"
        "[2] : docs-gromacs-install"
    )
    assert payload.context == [
        {
            "rank": 1,
            "doc_id": "TICKET_001",
            "text": "PERSON_001 reported this via EMAIL_001. Username USER_001.",
            "score": None,
            "source": "tickets",
        },
        {
            "rank": 2,
            "doc_id": "docs-gromacs-install",
            "text": "Load the module with `module load gromacs` and verify `gmx --version`.",
            "score": None,
            "source": "docs",
        },
    ]
