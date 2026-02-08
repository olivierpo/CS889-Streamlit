"""Literature Review Assistant
features: local search, query expansion.
"""

import streamlit as st
import google.generativeai as genai
import json
import os
from dotenv import load_dotenv

# Load environment variables (optional)
load_dotenv()

st.set_page_config(page_title="Literature Review Assistant", page_icon="📚", layout="wide")

_CSS = """
.ai-response { background-color: #f0f7ff; border-radius: 8px; padding: 12px; }
"""
st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


def init_session_state():
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
    if 'last_query' not in st.session_state:
        st.session_state.last_query = None
    if 'last_expanded_query' not in st.session_state:
        st.session_state.last_expanded_query = None
    if 'gemini_api_key' not in st.session_state:
        st.session_state.gemini_api_key = st.secrets.get("GEMINI_API_KEY", "")


init_session_state()


class LocalBibliography:
    BIB_FILE = "example-bib.json"

    @staticmethod
    @st.cache_data
    def load_bibliography():
        try:
            base = os.path.dirname(__file__) if '__file__' in globals() else '.'
            path = os.path.join(base, LocalBibliography.BIB_FILE)
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('references', [])
        except Exception:
            return []

    @staticmethod
    def _normalize_article(article: dict) -> dict:
        return {
            'paperId': str(article.get('id', '')),
            'title': article.get('title', 'Untitled'),
            'abstract': article.get('abstract', ''),
            'authors': [{'name': a} for a in article.get('authors', [])],
            'year': article.get('year'),
            'venue': article.get('journal', ''),
            'keywords': article.get('keywords', []),
            'doi': article.get('doi', ''),
        }

    @staticmethod
    def search_papers(query: str, limit: int = 20, year_start: int = None, year_end: int = None):
        articles = LocalBibliography.load_bibliography()
        if not articles or not query:
            return []
        
        # Tokenize query and remove punctuation
        import string
        tokens = query.lower().translate(str.maketrans('', '', string.punctuation)).split()
        
        results = []
        for a in articles:
            score = 0
            title = a.get('title', '').lower()
            abstract = a.get('abstract', '').lower()
            authors = ' '.join(a.get('authors', [])).lower()
            
            # Apply year filter
            year = a.get('year')
            if year_start and year and year < year_start:
                continue
            if year_end and year and year > year_end:
                continue
            
            # Score each token
            for token in tokens:
                if token:
                    if token in title:
                        score += 10
                    if token in abstract:
                        score += 5
                    if token in authors:
                        score += 3
            
            if score > 0:
                results.append((score, a))
        
        results.sort(key=lambda x: x[0], reverse=True)
        return [LocalBibliography._normalize_article(a) for _, a in results[:limit]]

    @staticmethod
    def get_all_papers():
        return [LocalBibliography._normalize_article(a) for a in LocalBibliography.load_bibliography()]


class GeminiAI:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
    
    def expand_query(self, query: str, query_num: int = 10) -> str:
        """Expand a search query with related keywords and synonyms."""
        prompt = f"""Given the search query: "{query}"

Generate an expanded search query that includes:
1. Synonyms and related terms
2. Broader concepts
3. Specific techniques or methods related to the topic

Return only the {query_num} most relevant terms as a single line of comma-separated terms, no explanations.

Example: if input is "machine learning", output might be: "machine learning, deep learning, artificial intelligence, neural networks, supervised learning, classification"

Expanded query:"""
        try:
            resp = self.model.generate_content(prompt)
            text = getattr(resp, 'text', str(resp)).strip()
            if len(text.split(",")) > query_num:
                for term in text.split(",")[:query_num]:
                    text_verify = term.strip()
                return text_verify
            return text
        except Exception as e:
            return str(e)


def display_article_card(article: dict, index: int):
    pid = article.get('paperId', '')
    title = article.get('title', 'No title')
    authors = ', '.join([a.get('name', '') for a in article.get('authors', [])])
    year = article.get('year', '')
    abstract = article.get('abstract', '')

    st.markdown(f"### {title}")
    st.markdown(f"**Authors:** {authors}  |  **Year:** {year}")
    if abstract:
        with st.expander("Abstract"):
            st.write(abstract)
    st.divider()


def search_page():
    st.header("Search Literature")
    all_papers = LocalBibliography.get_all_papers()
    st.info(f"Local bibliography: {len(all_papers)} articles")

    # API key input outside form (persistent across reruns)
    if not st.session_state.gemini_api_key:
        st.session_state.gemini_api_key = st.text_input(
            'Gemini API Key (optional, for AI query expansion)',
            value=st.session_state.gemini_api_key,
            type='password',
            help='Your key is stored only in session memory'
        )

    with st.form('search'):
        query = st.text_input('Search query')
        num = st.slider('Max results', 5, 100, 20)
        query_num = st.slider('Number of terms for AI expansion', 1, 20, 5)
        year_start = st.number_input('From year', value=0)
        year_end = st.number_input('To year', value=9999)
        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button('Search')
        with col2:
            use_ai = st.form_submit_button('Search with AI')

    if use_ai and query:
        if st.session_state.gemini_api_key:
            ai = GeminiAI(st.session_state.gemini_api_key)
            with st.spinner('Expanding query with AI...'):
                expanded = ai.expand_query(query, query_num=query_num)
                st.session_state.last_query = query
                st.session_state.last_expanded_query = expanded
                results = LocalBibliography.search_papers(query=expanded, limit=num, year_start=year_start or None, year_end=year_end or None)
                st.session_state.search_results = results
        else:
            st.error('Please provide a Gemini API key to use AI query expansion')
            return
    elif submitted and query:
        st.session_state.last_query = query
        st.session_state.last_expanded_query = None
        results = LocalBibliography.search_papers(query=query, limit=num, year_start=year_start or None, year_end=year_end or None)
        st.session_state.search_results = results

    if st.session_state.search_results:
        if st.session_state.last_expanded_query:
            st.info(f"📊 **Original:** {st.session_state.last_query}\n\n**Expanded:** {st.session_state.last_expanded_query}")
        st.subheader(f"Found {len(st.session_state.search_results)} papers")
        for i, a in enumerate(st.session_state.search_results):
            display_article_card(a, i)
    elif st.session_state.last_query:
        st.warning("❌ No results found for your search. Try different keywords or use AI query expansion.")


def main():
    with st.sidebar:
        st.title('📚 Literature Review Assistant')
        page = st.radio('Navigation', ['Search'])

    if page == 'Search':
        search_page()


if __name__ == '__main__':
    main()
