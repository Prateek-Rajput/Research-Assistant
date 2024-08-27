import os
import fitz  # PyMuPDF for PDF extraction
import requests
import numpy as np
import getpass
from flask import Flask, render_template, request, redirect, url_for, flash
from xml.etree import ElementTree as ET
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_anthropic import ChatAnthropic

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Function to set an environment variable for the API key
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

# Set the environment variable for the Anthropic API key
_set_env("ANTHROPIC_API_KEY")

# Initialize the LLM (Language Learning Model) from Anthropic
llm = ChatAnthropic(model="claude-3-haiku-20240307")

# Extract text from a PDF file
def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    pdf_document = fitz.open(pdf_path)
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    pdf_document.close()
    return text

# Extract the top terms from a given text using TF-IDF
def get_top_terms(text: str, num_terms: int = 5) -> list:
    vectorizer = TfidfVectorizer(stop_words='english', max_features=num_terms)
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]
    top_terms = feature_array[tfidf_sorting][:num_terms]
    return top_terms

# Process the text with the LLM and return the result
def process_text_with_llm(prompt: str, text=None) -> str:
    full_prompt = f"{prompt}\n\n{text}"
    try:
        response = llm.invoke([{"role": "user", "content": full_prompt}])
        return response.content
    except Exception as e:
        return f"Error during LLM processing: {e}"

# Search for research papers in ArXiv using a query
def search_arxiv(query: str) -> str:
    base_url = "http://export.arxiv.org/api/query"
    params = {
        'search_query': query,
        'start': 0,
        'max_results': 5
    }
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    data = response.text

    # Parse the XML response
    root = ET.fromstring(data)
    entries = root.findall('{http://www.w3.org/2005/Atom}entry')

    output = ""
    for entry in entries:
        title = entry.find('{http://www.w3.org/2005/Atom}title').text
        authors = [author.find('{http://www.w3.org/2005/Atom}name').text for author in entry.findall('{http://www.w3.org/2005/Atom}author')]
        published = entry.find('{http://www.w3.org/2005/Atom}published').text
        arxiv_url = entry.find('{http://www.w3.org/2005/Atom}id').text

        output += (
            f"<h3>Title: {title}</h3>"
            f"<p>Authors: {', '.join(authors)}</p>"
            f"<p>Published: {published}</p>"
            f"<p><a href='{arxiv_url}' target='_blank'>URL</a></p>"
        )
    
    return output if output else "<p>No results found.</p>"

# Search for research papers in CrossRef using a query
def search_crossref(query: str) -> str:
    base_url = "https://api.crossref.org/works"
    params = {'query': query, 'rows': 5}
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    data = response.json()
    items = data.get('message', {}).get('items', [])
    result_str = ""
    for item in items:
        title = item.get('title', ['N/A'])[0]
        authors = ', '.join([f"{author.get('given', '')} {author.get('family', '')}" for author in item.get('author', [])])
        published_date = item.get('published-print', {}).get('date-parts', [['N/A']])[0]
        publisher = item.get('publisher', 'N/A')
        doi = item.get('DOI', 'N/A')
        url = item.get('URL', 'N/A')
        result_str += (
            f"<h3>Title: {title}</h3>"
            f"<p>Authors: {authors}</p>"
            f"<p>Published Date: {'-'.join(map(str, published_date))}</p>"
            f"<p>Publisher: {publisher}</p>"
            f"<p>DOI: {doi}</p>"
            f"<p><a href='{url}' target='_blank'>URL</a></p>"
        )
    return result_str if result_str else "<p>No results found.</p>"

# Search academic databases (ArXiv and CrossRef) for papers related to the query
def search_academic_databases(query, source):
    if source == "arxiv":
        return search_arxiv(query)
    elif source == "crossref":
        return search_crossref(query)
    else:
        return "<p>Invalid source specified.</p>"

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        if "pdf" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["pdf"]
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        
        # Save the uploaded file
        pdf_path = os.path.join("uploads", file.filename)
        file.save(pdf_path)
        
        # Process the PDF based on user's choice
        choice = request.form.get("choice")
        text = extract_text_from_pdf(pdf_path)
        
        if choice == "1":
            prompt = "Summarize the research paper provided above."
            result = process_text_with_llm(prompt, text)
        elif choice == "2":
            prompt = "Extract the key points from the above provided research paper."
            result = process_text_with_llm(prompt, text)
        elif choice == "3":
            user_query = request.form.get("custom_query")
            prompt = f"Research paper provided below. User query: {user_query}"
            result = process_text_with_llm(prompt, text)
        elif choice == "4":
            top_terms = get_top_terms(text)
            top_query = " ".join(top_terms)
            arxiv_results = search_academic_databases(top_query, source="arxiv")
            crossref_results = search_academic_databases(top_query, source="crossref")
            result = {
                'arxiv': arxiv_results,
                'crossref': crossref_results
            }
        else:
            result = "Invalid option selected."
        
    return render_template("index.html", result=result)

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
