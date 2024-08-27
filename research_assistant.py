import os
import fitz  # PyMuPDF for PDF extraction
import requests
import numpy as np
import getpass
from xml.etree import ElementTree as ET
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from icecream import ic
from typing import Annotated, List
from typing_extensions import TypedDict

# Function to set an environment variable for the API key
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

# Set the environment variable for the Anthropic API key
_set_env("ANTHROPIC_API_KEY")

class State(TypedDict):
    messages: Annotated[List[dict], add_messages]

# Initialize the LLM (Language Learning Model) from Anthropic
llm = ChatAnthropic(model="claude-3-haiku-20240307")

def get_pdf_path_from_user() -> str:
    """Prompt the user to enter the path to the PDF file."""
    return input("Please enter the path to the PDF file: ")

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
def get_top_terms(text: str, num_terms: int = 5) -> List[str]:
    vectorizer = TfidfVectorizer(stop_words='english', max_features=num_terms)
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]
    top_terms = feature_array[tfidf_sorting][:num_terms]
    print(top_terms)
    return top_terms

# Process the text with the LLM and return the result
def process_text_with_llm(prompt: str, text = None) -> str:
    full_prompt = f"{prompt}\n\n{text}"
    try:
        response = llm.invoke([{"role": "user", "content": full_prompt}])
        return response.content
    except Exception as e:
        return f"Error during LLM processing: {e}"

# Search for research papers in ArXiv using a query
def search_arxiv(query: str):
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

# Search for research papers in CrossRef using a query
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

# Search academic databases (ArXiv and CrossRef) for papers related to the query
def search_academic_databases(query: str) -> str:
    results = ""
    results += "ArXiv Results:\n"
    results += search_arxiv(query) + "\n"
    results += "CrossRef Results:\n"
    results += search_crossref(query)
    return results

# Process the PDF and search databases based on top terms extracted from the PDF
def process_pdf_and_search_databases(pdf_path: str) -> str:
    text = extract_text_from_pdf(pdf_path)
    top_terms = get_top_terms(text, num_terms=5)
    top_query = " ".join(top_terms)
    return search_academic_databases(top_query)

# Show menu options to the user and get their input
def show_menu_and_get_input() -> str:
    menu = (
        "What would you like to do?\n"
        "1. Summarize the document\n"
        "2. Extract key findings\n"
        "3. Enter a custom query\n"
        "4. Find related research papers from ArXiv and CrossRef databases\n"
        "Please enter the number corresponding to your choice."
    )
    print(menu)
    return input("Your choice: ")

# Define the chatbot logic based on user choice
def chatbot(state: State):
    last_message = state["messages"][-1]
    user_message = last_message.content
    if user_message.isdigit() and int(user_message) in [1, 2, 3, 4]:
        option = int(user_message)
        
        if option in [1, 2, 3]:
            if option == 1:
                prompt = "Act as a research assistant with 20 years of experience. Summarize the research paper provided above in 1000 words."
            elif option == 2:
                prompt = "Please extract the key points from the above provided research paper."
            elif option == 3:
                while True:
                    user_custom_query = input("Please enter your custom query (or type 'quit', 'q', 'exit' to stop): ")
                    if user_custom_query.lower() in ["quit", "q", "exit"]:
                        return {"messages": [{"role": "system", "content": "Exiting custom query mode. You can choose another option."}]}
                    
                    prompt = (
                        "Act as a research assistant with 20 years of experience. A research paper is provided below. "
                        "Read it and the user will ask questions regarding the research paper. Respond accordingly.\n\n"
                        f"User query: {user_custom_query}"
                    )
                    result = process_text_with_llm(prompt, text)
                    ic(result)
                    print("Assistant:", result)
        elif option == 4:
            result = process_pdf_and_search_databases(pdf_path)
            return {"messages": [{"role": "system", "content": result}]}
                   
        result = process_text_with_llm(prompt, text)
        print("Assistant:", result) 
    
    else:
        return {"messages": [{"role": "system", "content": "Invalid choice. Please enter a number between 1 and 4."}]}

# Initialize the state graph for the chatbot logic
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

# Get the PDF path from the user and process the PDF
pdf_path = get_pdf_path_from_user()
try:
    text = extract_text_from_pdf(pdf_path)
except Exception as e:
    print(f"Error extracting text: {e}")

prompt = "Below provided text is a research paper. User wants you to read and perform tasks according to his query. Say 'I have received the research paper and I am ready to help'"
result = process_text_with_llm(prompt, text)
print("Assistant:", result)

# Main loop to handle user input and chatbot interactions
while True:
    user_input = show_menu_and_get_input()

    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    
    try:
        for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
            for value in event.values():
                print("Assistant:", value["messages"][-1]['content'])
    except Exception as e:
        print(f"Error during processing: {e}")
    
    # Ask if the user wants to perform another task
    print("\nWould you like to perform another task? (Type 'quit', 'exit', 'q' to exit)")
