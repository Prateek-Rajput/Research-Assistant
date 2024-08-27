from flask import Flask, request, render_template, jsonify
import os
import tempfile
import fitz  # PyMuPDF for PDF extraction
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_anthropic import ChatAnthropic
from xml.etree import ElementTree as ET

app = Flask(__name__)

# Initialize the LLM (Language Learning Model) from Anthropic
llm = ChatAnthropic(model="claude-3-haiku-20240307")

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with fitz.open(pdf_path) as pdf_document:
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
    return text

def get_top_terms(text: str, num_terms: int = 5) -> list[str]:
    vectorizer = TfidfVectorizer(stop_words='english', max_features=num_terms)
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]
    top_terms = feature_array[tfidf_sorting][:num_terms]
    return top_terms

def process_text_with_llm(prompt: str, text: str) -> str:
    full_prompt = f"{prompt}\n\n{text}"
    try:
        response = llm.invoke([{"role": "user", "content": full_prompt}])
        return response.content
    except Exception as e:
        return f"Error during LLM processing: {e}"

def search_arxiv(query: str) -> str:
    base_url = "http://export.arxiv.org/api/query"
    params = {
        'search_query': query,
        'start': 0,
        'max_results': 5
    }
    response = requests.get(base_url, params=params)
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
            f"Title: {title}\n"
            f"Authors: {', '.join(authors)}\n"
            f"Published: {published}\n"
            f"URL: {arxiv_url}\n"
            f"------------------------\n"
        )
    
    return output if output else "No results found."

def search_crossref(query: str) -> str:
    base_url = "https://api.crossref.org/works"
    params = {'query': query, 'rows': 5}
    response = requests.get(base_url, params=params)
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
            f"Title: {title}\n"
            f"Authors: {authors}\n"
            f"Published Date: {'-'.join(map(str, published_date))}\n"
            f"Publisher: {publisher}\n"
            f"DOI: {doi}\n"
            f"URL: {url}\n"
            f"------------------------\n"
        )
    return result_str if result_str else "No results found."

def search_academic_databases(query: str) -> str:
    results = ""
    results += "ArXiv Results:\n"
    results += search_arxiv(query) + "\n"
    results += "CrossRef Results:\n"
    results += search_crossref(query)
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    api_key = request.form['api_key']
    option = request.form['option']
    custom_query = request.form.get('custom_query')
    
    if 'pdf_file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['pdf_file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the PDF to a temporary file and extract its text
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file_path = temp_file.name
        file.save(temp_file_path)
        
        try:
            text = extract_text_from_pdf(temp_file_path)
        except Exception as e:
            return jsonify({"error": f"Error extracting text: {e}"}), 500
        finally:
            os.remove(temp_file_path)  # Clean up the temporary file
        
        if option == '1':
            prompt = "Act as a research assistant with 20 years of experience. Summarize the research paper provided above in 1000 words."
            result = process_text_with_llm(prompt, text)
        elif option == '2':
            prompt = "Please extract the key points from the above provided research paper."
            result = process_text_with_llm(prompt, text)
        elif option == '3':
            if custom_query:
                prompt = (
                    "Act as a research assistant with 20 years of experience. A research paper is provided below. "
                    "Read it and the user will ask questions regarding the research paper. Respond accordingly.\n\n"
                    f"User query: {custom_query}"
                )
                result = process_text_with_llm(prompt, text)
            else:
                result = "Custom query was not provided."
        else:
            result = "Invalid option selected."

        # Extract top terms and search for related papers
        top_terms = get_top_terms(text, num_terms=5)
        top_query = " ".join(top_terms)
        relevant_papers = search_academic_databases(top_query)
        
        return jsonify({
            "result": result,
            "relevant_papers": relevant_papers
        })

if __name__ == '__main__':
    app.run(debug=True)
