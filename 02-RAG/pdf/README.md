# 02-RAG

This folder contains the implementation and resources for a Retrieval-Augmented Generation (RAG) pipeline using PDF documents. The main components and their purposes are described below:

## Structure

- **pdf/**
  - **docker-compose.yaml**: Docker configuration for running services related to the RAG pipeline.
  - **index.py**: Main entry point for orchestrating the RAG workflow.
  - **injection_phase.py**: Handles the injection of PDF data into the retrieval system.
  - **retrieval_phase.py**: Implements the retrieval logic to fetch relevant information from indexed PDFs.
  - **summarization_phase.py**: Summarizes retrieved content for downstream tasks or user consumption.
  - **node-dev.pdf**: Example PDF document used for testing or demonstration.
  - \***\*pycache**/\*\*: Python bytecode cache directory (auto-generated).

## Usage

1. **Indexing PDFs**: Use `injection_phase.py` to process and inject PDF data into the retrieval system.
2. **Retrieval**: Run `retrieval_phase.py` to query and fetch relevant information from the indexed PDFs.
3. **Summarization**: Use `summarization_phase.py` to summarize the retrieved content.
4. **Orchestration**: The `index.py` script can be used to coordinate the above phases.
5. **Docker**: Use `docker-compose.yaml` to set up and run the pipeline in a containerized environment.

## Requirements

- Python 3.12+
- (Optional) Docker for containerized execution
- Additional dependencies may be listed in the main project `pyproject.toml` or `requirements.txt`

## Environment Variables

Create a `.env` file in the project root and add your Google API key:

```env
GOOGLE_API_KEY=
```

## Notes

- The folder is designed for modular experimentation with RAG workflows using PDF documents.
- Example PDF (`node-dev.pdf`) is provided for testing.
- For more details, refer to the code in each phase script.
