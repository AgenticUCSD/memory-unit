# Agentic RAG Memory Unit

A memory unit that provides **agentic RAG (Retrieval-Augmented Generation)** capabilities for workflow automation systems.

## Overview

This memory unit combines **two data sources** into a **unified searchable index**:

### 1. User Provided Context
- **Contents**: Diverse, unstructured documents (PDFs, notes, guidelines)
- **Processing**: Chunked and added to unified index
- **Use Case**: Relevant documents retrieved for any task/workflow query

### 2. Machine Generated Context
- **Contents**: Text files with natural language summaries of trends/preferences
- **Processing**: Chunked into unified index + metadata extracted for enrichment
- **Categories**:
  - `user_preferences` - Communication style, meeting preferences, etc.
  - `task_patterns` - Common task types, automation candidates
  - `workflow_trends` - Successful patterns, optimization opportunities

### Key Design
- **Unified Index**: All content (docs + preferences) searchable together
- **Holistic Retrieval**: Agent finds best sources regardless of source folder
- **Metadata Enrichment**: Preference categories enhance component-specific context
- **Agentic RAG**: Intelligent routing with combined document + preference results

## Architecture

### Package Structure
```
memory_unit/
├── __init__.py              # Package exports
├── core.py                  # Main MemoryUnit class
├── models/
│   ├── documents.py         # ContextDocument, FolderSummary
│   └── query.py             # ContextQueryResult, DriveFolderConfig
├── storage/
│   ├── vector_store.py      # ChromaDB wrapper
│   └── bm25_search.py       # Keyword search
├── processing/
│   ├── document_processor.py # Text chunking
│   └── preference_analyzer.py # Preference analysis
├── drive/
│   └── client.py            # Google Drive API
└── agents/
    └── tools.py             # LangChain tools
```

### System Architecture
```
System Architecture
========================================

[Google Drive]
    ├── Root Folder
    │   ├── User Provided Context (subfolder 1) - Diverse documents
    │   │   ├── project_guidelines.pdf
    │   │   ├── meeting_notes.md
    │   │   └── contact_list.csv
    │   │
    │   └── Machine Generated Context (subfolder 2) - Preference files
    │       ├── user_preferences.txt
    │       ├── task_patterns.txt
    │       └── workflow_trends.txt
    └──

[Extension Component]
    │
    │── GET ephemeral OAuth token (chrome.identity.getAuthToken)
    │── POST /hydrate with {folder_id, auth_token}
    │    └── Memory Unit hydrates from Drive
    │
    └── Extension can now query: POST /query?q=...

[Context Injection]
    ├── /context/extension - Unified context
    ├── /context/task-identifier - Context + task patterns
    └── /context/workflow-builder - Context + workflow trends
```

## Installation

```bash
# Create conda environment (if needed)
conda create -n "agents_ucsd" python==3.11

conda activate agents_ucsd
pip install -r requirements.txt

# Run the API
python api.py
```

## API Endpoints

### Hydration

The extension calls this with the ephemeral token to populate the memory unit:

```http
POST /hydrate
Content-Type: application/json
Authorization: Bearer ya29.a0...
X-User-Id: optional-user-id
X-Thread-Id: optional-thread-id

{
  "root_folder_id": "1ABC123..."
}
```

**Note**: The `auth_token` comes from the `Authorization` header, not the JSON body.

### Query

```http
POST /query
Content-Type: application/json

{
  "query": "What are common task patterns for project management?"
}
```

Response:
```json
{
  "answer": "Based on your documents...",
  "sources": [{"filename": "workflows.pdf"}],
  "context_for_extension": "Brief summary...",
  "context_for_task_identifier": "Task patterns found...",
  "context_for_workflow_builder": "Workflow examples...",
  "user_preferences": ["Prefers async updates..."],
  "task_patterns": ["Weekly reports every Friday..."],
  "workflow_trends": ["Successful workflows include..."]
}
```

### Context Injection (System Diagram Integration)

```http
POST /context/extension
POST /context/task-identifier
POST /context/workflow-builder
```

### Direct Preference Access

```http
GET /preferences
GET /preferences?category=user_preferences
```

Returns raw machine-generated preferences for direct consumption.


### Component-Specific Context

```python
# Get context for specific components
extension_context = memory.get_context_for_extension("user question")
task_context = memory.get_context_for_task_identifier("create a report")
workflow_context = memory.get_context_for_workflow_builder("automate email")

# Direct preference access
prefs = memory.get_direct_preferences("user_preferences")
```

## Supported Document Types

- `.txt` - Plain text
- `.md` - Markdown
- `.pdf` - PDF documents
- `.csv` - CSV files
- `.docx` - Word documents
- Google Docs (exported as text)
- Google Sheets (exported as CSV)

## Internal Architecture

```
[Memory Unit - Unified Index]

    Everything goes into the same index:
    ┌─────────────────────────────────────────────────────┐
    │  Unified Vector Store (Chroma) + Keyword Search    │
    │                                                     │
    │  • User doc chunks + Preference file chunks        │
    │  • All searchable via semantic + keyword           │
    │  • Metadata tracks source folder for context       │
    └─────────────────────────────────────────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────────────────────┐
    │  PreferenceAnalyzer (metadata enrichment)          │
    │                                                     │
    │  • Categorizes preference files by type            │
    │  • Provides quick category lookup for enriching    │
    │    component-specific context                      │
    └─────────────────────────────────────────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────────────────────┐
    │  QueryAgent (LangChain) - Intelligent routing      │
    │                                                     │
    │  • Hybrid search finds best sources anywhere       │
    │  • Preference metadata enriches context            │
    └─────────────────────────────────────────────────────┘
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/unit/test_processing.py -v

# Run with coverage
python -m pytest tests/ --cov=memory_unit
```

