# Geni-Ai

A project for learning and experimenting with Generative AI concepts.

## Project Structure

```
main.py                  # Main entry point for the application
pyproject.toml           # Python project configuration
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- `uv` package manager (recommended over pip for faster dependency management)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/ShreyasPrabhu26/Geni-Ai.git
   cd Geni-Ai
   ```

2. **Install uv (if not already installed)**

   ```bash
   # Using pip
   pip install uv

   # Or using curl (for Unix-based systems)
   curl -sSf https://install.python-uv.org | python3
   ```

3. **Install project dependencies**

   ```bash
   # Install dependencies defined in pyproject.toml
   uv pip install -e .

   # Or add specific packages
   uv add openai
   ```

## Running the Project

### Main Application

```bash
uv run main.py
```

### Tokenization Examples

```bash
# Run the tokenization script using uv
uv run 01-tokenization/tokenization.py

# Run the embeddings script using uv
uv run 01-tokenization/embeddings.py
```

## Development

To add new dependencies:

```bash
# Add a package to your project
uv add package_name

# Add multiple packages
uv add package1 package2
```
