# AI-Powered Research Assistant

## Overview

The AI-Powered Research Assistant is a Python application designed to process and analyze research papers. It extracts text from PDF documents, identifies key terms, and searches for related research papers using academic databases like ArXiv and CrossRef. The application features an interactive chatbot that can summarize documents, extract key findings, and handle custom queries.

## Architecture
![Flowchart](https://github.com/user-attachments/assets/85903cf3-b512-44b3-9404-16bc1a033d0f)
The AI-Powered Research Assistant is structured as follows:

1. **PDF Text Extraction**:
   - Utilizes the `fitz` library (PyMuPDF) to extract text from PDF documents.
   - The `extract_text_from_pdf` function reads the PDF file page by page and collects the text.

2. **Text Analysis**:
   - **TF-IDF Vectorization**:
     - Uses `TfidfVectorizer` from `scikit-learn` to extract significant terms from the text.
     - The `get_top_terms` function identifies the top terms based on their TF-IDF scores.

3. **Language Learning Model (LLM)**:
   - Integrated with Anthropic's Claude AI model via `langchain-anthropic`.
   - The `process_text_with_llm` function sends prompts to the LLM to generate summaries, extract key findings, and respond to custom queries.

4. **Academic Database Semantic Search**:
   - **ArXiv**:
     - Uses the ArXiv API to search for related research papers.
     - The `search_arxiv` function sends a query(extracted top terms) to the ArXiv API and parses the XML response to retrieve paper details.
   - **CrossRef**:
     - Uses the CrossRef API to search for related research papers.
     - The `search_crossref` function sends a query(extracted top terms) to the CrossRef API and processes the JSON response to extract paper details.

5. **Chatbot Interface**:
   - Implements an interactive menu system for user interactions.
   - The `chatbot` function handles user choices and interacts with the LLM based on the selected option.
   - **State Management**:
     - Uses `langgraph` to manage the state of the chatbot and handle transitions between different tasks.

## Features


- **PDF Text Extraction**: Extracts text from PDF documents using PyMuPDF.
- **Top Terms Extraction**: Identifies the most significant terms in the text using TF-IDF.
- **LLM Processing**: Uses Anthropic's Claude AI model for summarization, key point extraction, and answering custom queries.
- **Academic Database Search**: Searches ArXiv and CrossRef for research papers related to the extracted terms.
- **Chatbot Interface**: Offers an interactive menu for various tasks related to document processing and research queries.
## Installation
Install the required dependencies:

```python 
pip install fitz requests numpy scikit-learn langchain-anthropic langgraph icecream
```

## Setup
- **Set up your Anthropic API key**:

    To use the Claude AI model, you need to provide your Anthropic API key. When you run the application, you will be prompted to enter your API key.
  
## Necessary Imports

The following Python libraries and modules are required for the AI-Powered Research Assistant project:

```python
import os
import fitz  # PyMuPDF for PDF extraction
import requests
import numpy as np
import getpass
from xml.etree import ElementTree as ET
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from icecream import ic
from typing import Annotated, List
from typing_extensions import TypedDict
```



## Usage
**Run the application**:

Start the application by running the following command:

```python
python research_assistant_cli.py
```


**Interact with the application**:

**Enter PDF Path**: You will be prompted to enter the path to the PDF file you want to process.

**Choose an Option**: 
After processing the PDF, you can select from the menu options to summarize the document, extract key findings, enter a custom query, or find related research papers.

Menu Options:
- Summarize the document.
- Extract key findings.
- Enter a custom query.
- Find related research papers from ArXiv and CrossRef databases.
  
## Information Provided in Relevant Research Papers

When searching for relevant research papers using the AI-Powered Research Assistant, the following information is typically provided for each paper:

### For ArXiv Papers

- **Title**: The title of the research paper.
- **Authors**: A list of authors who contributed to the research paper.
- **Published Date**: The date when the research paper was published.
- **URL**: A link to the full research paper on ArXiv.

### For CrossRef Papers

- **Title**: The title of the research paper.
- **Authors**: A list of authors who contributed to the research paper, including their given and family names.
- **Published Date**: The date when the research paper was published.
- **Publisher**: The publisher of the research paper.
- **DOI**: The Digital Object Identifier of the paper, which provides a persistent link to its location on the internet.
- **URL**: A direct link to the research paper or its metadata.

## Custom Query Mode

If you select option 3, you can enter a custom query to ask questions about the research paper until you want to quit.

## Example
Here's an example of how the application might be used:
![image](https://github.com/user-attachments/assets/f4588785-5ef4-4651-bcf2-101fbba9f747)
