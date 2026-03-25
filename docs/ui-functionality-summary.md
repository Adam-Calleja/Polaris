# Polaris UI Functionality Summary

This summary reflects the current Streamlit UI implementation in `src/polaris_rag/streamlit` as of March 25, 2026.

## Overview

Polaris provides a Streamlit frontend for interacting with the RAG backend, running curated evaluation demos, and inspecting system status. The UI is organised into three workspaces:

- `Assistant`
- `Evaluation`
- `System`

The app launches from `src/polaris_rag/streamlit/polaris_interface.py` and is designed as a thin frontend over the backend API rather than a self-contained application.

## Global UI Structure

### Navigation

- The app opens in the `Assistant` workspace by default.
- A collapsed drawer can be opened from the main canvas via a menu button.
- The drawer provides navigation between `Assistant`, `Evaluation`, and `System`.

### Backend Controls

The drawer also exposes runtime configuration for the frontend:

- API base URL
- Query endpoint path
- HTTP timeout
- Debug mode toggle
- Manual query constraints

The timeout is adjusted in 5-second steps and is clamped between 1 and 600 seconds.

### Manual Query Constraints

The UI can optionally attach manual constraints to assistant queries and demo runs. The currently supported manual fields are:

- `query_type`
- `scope_family_names`
- `service_names`
- `software_names`
- `software_versions`

These constraints are only sent when at least one field is populated.

### Session State

The frontend keeps most UI state in Streamlit session state, including:

- current workspace
- backend settings
- assistant messages
- whether conversation history is expanded
- evaluation results
- feedback submission fingerprints

This state is session-local except for evaluation feedback, which is persisted to a JSONL log file.

## Assistant Workspace

The Assistant workspace is the main interactive query UI.

### Landing State

Before the first assistant response, the user sees:

- two quick-prompt buttons
- a single-line prompt composer for free-text questions

The current quick prompts shown are:

- "We need to renew RDS and transfer ownership. How should this be handled?"
- "Can you confirm the exact new path for my project data?"

### Active Conversation State

After a response is received, the layout changes to an active workspace:

- the latest user query is shown prominently
- the latest Polaris answer is shown in the main left column
- a diagnostics panel is shown in the right column
- a follow-up input box is shown below the main content

Older exchanges can be revealed through a `Conversation History` toggle.

### Answer Rendering

Assistant answers are rendered as structured cards. If the backend returns headings in the expected template format, the UI splits the answer into sections such as:

- Classification
- Quick Assessment
- Action
- Action Steps (Helpdesk)
- Questions to Ask
- Example Customer Reply
- Safety / Policy Notes
- Reference Key

If no recognised headings are present, the answer is rendered as a single response block.

### Evidence and Diagnostics

For successful responses, the Assistant workspace shows:

- answer status badges (`Grounded`, `Limited Evidence`, or `No Evidence`)
- answer-status detail text
- evidence chunk count
- retrieval latency
- generation latency
- interpreted query constraints returned by the backend

Expandable panels are available for:

- retrieved evidence chunks
- debug metadata, when debug mode is enabled and the backend returns evaluation metadata

Each evidence item is shown with:

- rank
- document ID
- source label
- score
- raw chunk text

### Error Handling

The Assistant workspace handles several failure modes and renders them as UI cards:

- timeout
- network error
- backend/API error

The UI provides short recovery guidance, such as checking backend reachability, shortening the query, disabling debug mode, or increasing the timeout.

### Important Current Behaviour

The interface stores conversation history locally, but each backend request only sends the current prompt plus optional manual constraints. The previous messages are not submitted as conversational context to the API.

In practice, this means the UI behaves like a sequence of independent queries with visible history, not a true backend-backed multi-turn chat session.

## Evaluation Workspace

The Evaluation workspace is built for demos, screenshots, and lightweight usability/evaluation evidence collection.

### Scenario Catalog

The UI currently includes six built-in scenarios:

- Grounded Support Answer
- Ambiguous User Query
- Unsupported / Missing Evidence
- Potentially Conflicting Guidance
- Freshness / Version Sensitivity
- Timeout / Recovery State

Each scenario includes:

- title
- description
- focus note
- fixed query text
- a `Run` button

### Scenario Execution

When a scenario is run, the frontend sends the scenario query to the backend. A scenario may also:

- include predefined query constraints
- request evaluation metadata
- override the server timeout

Results are stored in session state per scenario and shown below the scenario card.

### Scenario Result Display

For each executed scenario, the UI displays:

- the rendered answer or error card
- the same diagnostics panel used in the Assistant workspace
- evidence/debug expanders where applicable

### Evaluation Feedback Capture

After a scenario is run, the UI provides a feedback form with:

- helpfulness (`yes`, `partly`, `no`)
- groundedness (`yes`, `partly`, `no`)
- citation quality (`strong`, `adequate`, `weak`)
- failure type
- free-text notes

Submitted feedback is appended to a persistent JSONL log.

### Feedback Summary

At the top of the Evaluation workspace, the UI shows a persistent summary of logged feedback:

- total saved feedback count
- number of `Helpful = yes`
- number of `Grounded = yes`
- counts by scenario
- counts by failure type

The UI also prevents duplicate feedback submission for the exact same response within the same session by fingerprinting the query, answer, evidence doc IDs, and scenario ID.

## System Workspace

The System workspace is an inspection and readiness page rather than a control panel for backend internals.

### Architecture View

The page includes a high-level pipeline summary with five stages:

- Corpus
- Retrieval
- Reranking
- Prompting
- Answer

### Corpus Legend

The UI explains the main evidence/source categories used by the product:

- Docs
- Tickets
- Multi-source

### Live Backend Checks

The frontend probes backend readiness endpoints and shows the result for:

- `/health`
- `/ready`

For each check, the UI shows:

- pass/fail state
- probed URL
- optional payload viewer

There is also a `Refresh Status` button to rerun the checks.

### Frontend Runtime Summary

The System page displays the frontend runtime configuration currently in use:

- API base URL
- query endpoint
- HTTP timeout
- debug mode state
- feedback log path

## API Integration

The Streamlit frontend communicates with the backend through a POST request to the configured query endpoint, which defaults to:

- base URL: `http://rag-api:8000`
- endpoint: `/v1/query`

The query payload can include:

- `query`
- `query_constraints`
- `include_evaluation_metadata`

The frontend also supports a per-request server timeout header for evaluation scenarios.

The response is normalised into frontend data structures for:

- answer text
- retrieved context items
- interpreted query constraints
- evaluation metadata
- answer status
- timings

## Current Limitations and Notable Constraints

- The Assistant workspace does not send prior chat history to the backend.
- Only the latest exchange is foregrounded; earlier turns are hidden behind a conversation-history toggle.
- Manual constraint entry supports only a subset of the possible constraint fields the backend can return in diagnostics.
- The UI is primarily an inspection/demo frontend; it does not include authentication, user roles, corpus management, or admin mutation flows.
- Evaluation feedback persistence is file-based JSONL logging rather than a database-backed workflow.

## Main Source Files

The summary above is based primarily on these files:

- `src/polaris_rag/streamlit/polaris_interface.py`
- `src/polaris_rag/streamlit/views/assistant.py`
- `src/polaris_rag/streamlit/views/evaluation.py`
- `src/polaris_rag/streamlit/views/system.py`
- `src/polaris_rag/streamlit/api_client.py`
- `src/polaris_rag/streamlit/demo_catalog.py`
- `src/polaris_rag/streamlit/feedback.py`
