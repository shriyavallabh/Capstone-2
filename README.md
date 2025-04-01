# TalkToCode

## Overview
TalkToCode is a Graph RAG (Retrieval Augmented Generation) implementation for code analysis. It allows users to interact with and analyze codebases using natural language, leveraging graph-based representations of code structures.

## Features
- **Code Indexing**: Parses and creates graph representations of code
- **Graph-based Retrieval**: Uses graph algorithms to find relevant code snippets
- **Interactive UI**: Streamlit-based interface for querying and visualizing code
- **Natural Language Processing**: Query code using plain English

## Project Structure
- `indexing/`: Code parsing and graph construction
- `retrieval/`: Graph traversal and retrieval algorithms
- `ui/`: Streamlit user interface components
- `utils/`: Helper functions and utilities

## Installation
```bash
pip install -r requirements.txt
```

## API Key Setup
TalkToCode requires an OpenAI API key for generating embeddings and answering queries. You can set up your API key in one of the following ways:

1. **Create a .env file in the project root**:
   ```
   cp .env.example .env
   ```
   Then edit `.env` and replace `your_openai_api_key_here` with your actual API key.

2. **Enter the API key directly in the application**:
   The application will prompt you for your API key if one isn't found.

3. **Set an environment variable**:
   ```bash
   export OPENAI_API_KEY=your_openai_api_key_here
   ```

You can obtain an API key by signing up at [OpenAI's platform](https://platform.openai.com/api-keys).

## Getting Started
1. Set up your OpenAI API key as described above
2. Run the Streamlit interface:
   ```bash
   streamlit run app.py
   ```
3. Upload your codebase through the UI
4. Explore the graph visualization and start querying your code

## Keyboard Shortcuts
- **Ctrl+/** or **Cmd+/**: Focus on chat input
- **Ctrl+Shift+F** or **Cmd+Shift+F**: Toggle sidebar
- **Ctrl+G** or **Cmd+G**: Switch to graph visualization tab
- **Ctrl+R** or **Cmd+R**: Switch to search results tab
- **F11**: Toggle fullscreen mode

## License
MIT 