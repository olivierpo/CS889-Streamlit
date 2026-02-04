"""
Literature Review Assistant
A Streamlit application for searching, selecting, and analyzing research articles
using the Semantic Scholar API and Google Gemini AI.
"""

import streamlit as st
import pandas as pd
import google.generativeai as genai
from datetime import datetime
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Literature Review Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .article-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
    }
    .selected-card {
        border-left: 4px solid #2196F3;
        background-color: #e3f2fd;
    }
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .ai-response {
        background-color: #f0f7ff;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #bbdefb;
    }
    .stButton>button {
        border-radius: 20px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    if 'selected_articles' not in st.session_state:
        st.session_state.selected_articles = {}
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    if 'ai_chat_history' not in st.session_state:
        st.session_state.ai_chat_history = []


init_session_state()


# Local Bibliography Search
class LocalBibliography:
    """Search and manage articles from a local JSON bibliography file."""
    
    BIB_FILE = "example-bib.json"
    
    @staticmethod
    @st.cache_data
    def load_bibliography() -> list:
        """Load the bibliography from the JSON file."""
        try:
            bib_path = os.path.join(os.path.dirname(__file__), LocalBibliography.BIB_FILE)
            with open(bib_path, 'r') as f:
                data = json.load(f)
                return data.get("references", [])
        except FileNotFoundError:
            st.error(f"Bibliography file '{LocalBibliography.BIB_FILE}' not found.")
            return []
        except json.JSONDecodeError:
            st.error(f"Error parsing '{LocalBibliography.BIB_FILE}'. Please check the JSON format.")
            return []
    
    @staticmethod
    def _normalize_article(article: dict) -> dict:
        """Convert local bibliography format to a common format for display."""
        return {
            'paperId': str(article.get('id', '')),
            'title': article.get('title', 'Untitled'),
            'abstract': article.get('abstract', ''),
            'authors': [{'name': name} for name in article.get('authors', [])],
            'year': article.get('year'),
            'venue': article.get('journal', ''),
            'citationCount': None,  # Not available in local data
            'url': f"https://doi.org/{article.get('doi', '')}" if article.get('doi') else None,
            'doi': article.get('doi', ''),
            'volume': article.get('volume'),
            'issue': article.get('issue'),
            'pages': article.get('pages'),
            'keywords': article.get('keywords', []),
            'openAccessPdf': None
        }
    
    @staticmethod
    def search_papers(query: str, limit: int = 20, year_start: int = None, 
                      year_end: int = None, keywords_filter: list = None) -> list:
        """Search for papers in the local bibliography."""
        articles = LocalBibliography.load_bibliography()
        
        if not articles:
            return []
        
        query_lower = query.lower()
        query_terms = query_lower.split()
        
        results = []
        for article in articles:
            # Calculate relevance score
            score = 0
            
            # Search in title (highest weight)
            title = article.get('title', '').lower()
            for term in query_terms:
                if term in title:
                    score += 10
            
            # Search in abstract
            abstract = article.get('abstract', '').lower()
            for term in query_terms:
                if term in abstract:
                    score += 5
            
            # Search in keywords
            keywords = [k.lower() for k in article.get('keywords', [])]
            for term in query_terms:
                for keyword in keywords:
                    if term in keyword:
                        score += 3
            
            # Search in authors
            authors = ' '.join(article.get('authors', [])).lower()
            for term in query_terms:
                if term in authors:
                    score += 2
            
            # Apply year filter
            year = article.get('year')
            if year_start and year and year < year_start:
                continue
            if year_end and year and year > year_end:
                continue
            
            # Apply keywords filter
            if keywords_filter:
                article_keywords = [k.lower() for k in article.get('keywords', [])]
                if not any(kf.lower() in ' '.join(article_keywords) for kf in keywords_filter):
                    continue
            
            if score > 0:
                results.append((score, article))
        
        # Sort by relevance score (descending)
        results.sort(key=lambda x: x[0], reverse=True)
        
        # Normalize and return top results
        return [LocalBibliography._normalize_article(article) for score, article in results[:limit]]
    
    @staticmethod
    def get_all_keywords() -> list:
        """Get all unique keywords from the bibliography."""
        articles = LocalBibliography.load_bibliography()
        keywords = set()
        for article in articles:
            for keyword in article.get('keywords', []):
                keywords.add(keyword)
        return sorted(list(keywords))
    
    @staticmethod
    def get_year_range() -> tuple:
        """Get the min and max years in the bibliography."""
        articles = LocalBibliography.load_bibliography()
        years = [a.get('year') for a in articles if a.get('year')]
        if years:
            return min(years), max(years)
        return 2000, 2024
    
    @staticmethod
    def get_all_papers() -> list:
        """Get all papers from the bibliography."""
        articles = LocalBibliography.load_bibliography()
        return [LocalBibliography._normalize_article(a) for a in articles]


# Gemini AI functions
class GeminiAI:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
    
    def summarize_article(self, title: str, abstract: str) -> str:
        """Generate a concise summary of an article."""
        prompt = f"""Summarize the following research article in 2-3 sentences, focusing on the main contribution and findings:

Title: {title}

Abstract: {abstract}

Provide a clear, concise summary suitable for a literature review."""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def compare_articles(self, articles: list) -> str:
        """Compare multiple articles and identify themes."""
        articles_text = "\n\n".join([
            f"Title: {a.get('title', 'N/A')}\nAbstract: {a.get('abstract', 'N/A')}"
            for a in articles
        ])
        
        prompt = f"""Analyze the following research articles and provide:
1. Common themes and topics
2. Key differences in approaches or findings
3. Research gaps that could be addressed
4. How these papers relate to each other

Articles:
{articles_text}

Provide a structured analysis suitable for a literature review."""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error comparing articles: {str(e)}"
    
    def generate_research_questions(self, articles: list) -> str:
        """Generate potential research questions based on selected articles."""
        articles_text = "\n\n".join([
            f"Title: {a.get('title', 'N/A')}\nAbstract: {a.get('abstract', 'N/A')}"
            for a in articles
        ])
        
        prompt = f"""Based on the following research articles, suggest 5 potential research questions that could extend or build upon this work:

Articles:
{articles_text}

Format each question with a brief explanation of why it's relevant and how it connects to the existing literature."""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating questions: {str(e)}"
    
    def write_literature_section(self, articles: list, topic: str) -> str:
        """Help write a literature review section."""
        articles_text = "\n\n".join([
            f"Title: {a.get('title', 'N/A')}\nAuthors: {', '.join([author.get('name', '') for author in a.get('authors', [])])}\nYear: {a.get('year', 'N/A')}\nAbstract: {a.get('abstract', 'N/A')}"
            for a in articles
        ])
        
        prompt = f"""Write a literature review section about "{topic}" using the following articles as sources. 
Include proper in-text citations (Author, Year) and organize the content thematically.

Articles:
{articles_text}

Write 2-3 paragraphs that synthesize the key findings and show how the papers relate to each other."""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error writing section: {str(e)}"
    
    def suggest_search_terms(self, topic: str) -> str:
        """Suggest related search terms for a topic."""
        prompt = f"""For the research topic "{topic}", suggest:
1. 5 related search queries to find more papers
2. 3 broader terms to explore the general field
3. 3 narrower terms to focus on specific aspects
4. Key authors or seminal papers to look for

Format the suggestions clearly with brief explanations."""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error suggesting terms: {str(e)}"
    
    def chat(self, message: str, context: str = "") -> str:
        """General chat about literature review."""
        prompt = f"""You are a helpful research assistant specializing in literature reviews. 
{f'Context about selected papers: {context}' if context else ''}

User question: {message}

Provide a helpful, informative response."""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error in chat: {str(e)}"


def display_article_card(article: dict, index: int, show_select: bool = True):
    """Display an article as a card."""
    paper_id = article.get('paperId', '')
    title = article.get('title', 'No title')
    abstract = article.get('abstract', 'No abstract available')
    authors = article.get('authors', [])
    year = article.get('year', 'N/A')
    venue = article.get('venue', 'N/A')
    url = article.get('url', '')
    doi = article.get('doi', '')
    keywords = article.get('keywords', [])
    volume = article.get('volume', '')
    issue = article.get('issue', '')
    pages = article.get('pages', '')
    
    is_selected = paper_id in st.session_state.selected_articles
    
    with st.container():
        col1, col2 = st.columns([0.9, 0.1])
        
        with col1:
            # Title with selection indicator
            if is_selected:
                st.markdown(f"### ✓ {title}")
            else:
                st.markdown(f"### {title}")
            
            # Authors
            author_names = [a.get('name', '') for a in authors[:5]]
            if len(authors) > 5:
                author_names.append(f"et al. ({len(authors)} total)")
            st.markdown(f"**Authors:** {', '.join(author_names)}")
            
            # Metadata row
            meta_col1, meta_col2 = st.columns(2)
            with meta_col1:
                st.markdown(f"**Year:** {year}")
            with meta_col2:
                # Build venue string with volume/issue/pages
                venue_str = venue if venue else ''
                if volume:
                    venue_str += f", Vol. {volume}"
                if issue:
                    venue_str += f"({issue})"
                if pages:
                    venue_str += f", pp. {pages}"
                st.markdown(f"**Journal:** {venue_str if venue_str else 'N/A'}")
            
            # Keywords
            if keywords:
                st.markdown(f"**Keywords:** {', '.join(keywords)}")
            
            # Abstract
            if abstract:
                with st.expander("View Abstract"):
                    st.write(abstract)
            
            # DOI Link
            if doi:
                st.markdown(f"[DOI: {doi}](https://doi.org/{doi})")
        
        with col2:
            if show_select:
                if is_selected:
                    if st.button("✓", key=f"remove_{paper_id}_{index}", help="Remove from selection"):
                        del st.session_state.selected_articles[paper_id]
                        st.rerun()
                else:
                    if st.button("+", key=f"add_{paper_id}_{index}", help="Add to selection"):
                        st.session_state.selected_articles[paper_id] = article
                        st.rerun()
        
        st.divider()


def search_page():
    """Main search page."""
    st.header("Search Literature")
    
    # Get available keywords and year range from bibliography
    available_keywords = LocalBibliography.get_all_keywords()
    min_year, max_year = LocalBibliography.get_year_range()
    
    # Show bibliography info
    all_papers = LocalBibliography.get_all_papers()
    st.info(f"Searching {len(all_papers)} articles from local bibliography ({min_year}-{max_year})")
    
    # Search form
    with st.form("search_form"):
        query = st.text_input("Search Query", placeholder="Enter your search terms...")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            num_results = st.slider("Number of results", 5, min(50, len(all_papers)), min(20, len(all_papers)))
        with col2:
            year_start = st.number_input("From Year", min_value=min_year, max_value=max_year, value=min_year)
        with col3:
            year_end = st.number_input("To Year", min_value=min_year, max_value=max_year, value=max_year)
        
        # Keywords filter
        selected_keywords = st.multiselect("Filter by Keywords (optional)", available_keywords)
        
        col_search, col_browse = st.columns(2)
        with col_search:
            search_submitted = st.form_submit_button("Search", type="primary")
        with col_browse:
            browse_all = st.form_submit_button("Browse All Articles")
    
    if search_submitted and query:
        results = LocalBibliography.search_papers(
            query=query,
            limit=num_results,
            year_start=year_start,
            year_end=year_end,
            keywords_filter=selected_keywords if selected_keywords else None
        )
        st.session_state.search_results = results
        st.session_state.search_history.append({
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "results_count": len(results)
        })
    
    if browse_all:
        # Show all articles with optional filters
        results = LocalBibliography.get_all_papers()
        # Apply year filter
        results = [r for r in results if (not year_start or (r.get('year') and r.get('year') >= year_start))]
        results = [r for r in results if (not year_end or (r.get('year') and r.get('year') <= year_end))]
        # Apply keyword filter
        if selected_keywords:
            results = [r for r in results if any(k.lower() in ' '.join([kw.lower() for kw in r.get('keywords', [])]) for k in selected_keywords)]
        st.session_state.search_results = results[:num_results]
        st.session_state.search_history.append({
            "query": "[Browse All]",
            "timestamp": datetime.now().isoformat(),
            "results_count": len(st.session_state.search_results)
        })
    
    # Display results
    if st.session_state.search_results:
        st.subheader(f"Found {len(st.session_state.search_results)} papers")
        
        # Sort options
        sort_option = st.selectbox(
            "Sort by",
            ["Relevance", "Year (Newest)", "Year (Oldest)"]
        )
        
        results = st.session_state.search_results.copy()
        
        if sort_option == "Year (Newest)":
            results.sort(key=lambda x: x.get('year', 0) or 0, reverse=True)
        elif sort_option == "Year (Oldest)":
            results.sort(key=lambda x: x.get('year', 9999) or 9999)
        
        # Display articles
        for i, article in enumerate(results):
            display_article_card(article, i)


def selected_articles_page():
    """Page showing selected articles."""
    st.header("Selected Articles")
    
    if not st.session_state.selected_articles:
        st.info("No articles selected yet. Use the Search page to find and select articles.")
        return
    
    # Summary stats
    col1, col2, col3 = st.columns(3)
    articles = list(st.session_state.selected_articles.values())
    
    with col1:
        st.metric("Total Articles", len(articles))
    with col2:
        total_citations = sum(a.get('citationCount', 0) or 0 for a in articles)
        st.metric("Total Citations", total_citations)
    with col3:
        years = [a.get('year') for a in articles if a.get('year')]
        if years:
            st.metric("Year Range", f"{min(years)} - {max(years)}")
    
    st.divider()
    
    # Actions
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        if st.button("Export as CSV", type="secondary"):
            df = pd.DataFrame([
                {
                    'Title': a.get('title', ''),
                    'Authors': ', '.join([auth.get('name', '') for auth in a.get('authors', [])]),
                    'Year': a.get('year', ''),
                    'Venue': a.get('venue', ''),
                    'Citations': a.get('citationCount', 0),
                    'URL': a.get('url', ''),
                    'Abstract': a.get('abstract', '')
                }
                for a in articles
            ])
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="literature_review_articles.csv",
                mime="text/csv"
            )
    
    with action_col2:
        if st.button("Export as BibTeX", type="secondary"):
            bibtex_entries = []
            for a in articles:
                authors_str = " and ".join([auth.get('name', '') for auth in a.get('authors', [])])
                paper_id = a.get('paperId', 'unknown')
                entry = f"""@article{{{paper_id},
  title = {{{a.get('title', '')}}},
  author = {{{authors_str}}},
  year = {{{a.get('year', '')}}},
  journal = {{{a.get('venue', '')}}}
}}"""
                bibtex_entries.append(entry)
            bibtex_content = "\n\n".join(bibtex_entries)
            st.download_button(
                label="Download BibTeX",
                data=bibtex_content,
                file_name="literature_review.bib",
                mime="text/plain"
            )
    
    with action_col3:
        if st.button("Clear All", type="secondary"):
            st.session_state.selected_articles = {}
            st.rerun()
    
    st.divider()
    
    # Display selected articles
    for i, (paper_id, article) in enumerate(st.session_state.selected_articles.items()):
        display_article_card(article, i)


def ai_assistant_page():
    """AI Assistant page for analysis and writing help."""
    st.header("AI Research Assistant")
    
    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        st.warning("Please set your GEMINI_API_KEY in a .env file to use AI features.")
        api_key = st.text_input("Or enter your Gemini API key:", type="password")
        if not api_key:
            return
    
    ai = GeminiAI(api_key)
    articles = list(st.session_state.selected_articles.values())
    
    # Tabs for different AI features
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Summarize", "Compare & Analyze", "Research Questions", "Write Section", "Chat"
    ])
    
    with tab1:
        st.subheader("Summarize Articles")
        if not articles:
            st.info("Select articles first to summarize them.")
        else:
            article_titles = {a.get('paperId'): a.get('title', 'Untitled') for a in articles}
            selected_paper = st.selectbox(
                "Select an article to summarize",
                options=list(article_titles.keys()),
                format_func=lambda x: article_titles[x]
            )
            
            if st.button("Generate Summary", key="summarize_btn"):
                article = st.session_state.selected_articles[selected_paper]
                with st.spinner("Generating summary..."):
                    summary = ai.summarize_article(
                        article.get('title', ''),
                        article.get('abstract', '')
                    )
                    st.markdown("### Summary")
                    st.markdown(f'<div class="ai-response">{summary}</div>', unsafe_allow_html=True)
    
    with tab2:
        st.subheader("Compare & Analyze Articles")
        if len(articles) < 2:
            st.info("Select at least 2 articles to compare them.")
        else:
            st.write(f"Comparing {len(articles)} selected articles...")
            if st.button("Analyze Articles", key="compare_btn"):
                with st.spinner("Analyzing articles..."):
                    analysis = ai.compare_articles(articles)
                    st.markdown("### Analysis")
                    st.markdown(analysis)
    
    with tab3:
        st.subheader("Generate Research Questions")
        if not articles:
            st.info("Select articles first to generate research questions.")
        else:
            if st.button("Generate Questions", key="questions_btn"):
                with st.spinner("Generating research questions..."):
                    questions = ai.generate_research_questions(articles)
                    st.markdown("### Potential Research Questions")
                    st.markdown(questions)
    
    with tab4:
        st.subheader("Write Literature Review Section")
        if not articles:
            st.info("Select articles first to write a literature review section.")
        else:
            topic = st.text_input("What topic should this section cover?")
            if st.button("Write Section", key="write_btn") and topic:
                with st.spinner("Writing literature review section..."):
                    section = ai.write_literature_section(articles, topic)
                    st.markdown("### Draft Literature Review Section")
                    st.markdown(section)
                    
                    # Copy button
                    st.download_button(
                        label="Download as Text",
                        data=section,
                        file_name="literature_review_section.txt",
                        mime="text/plain"
                    )
    
    with tab5:
        st.subheader("Research Chat")
        st.write("Ask questions about your literature review or get research advice.")
        
        # Chat history
        for msg in st.session_state.ai_chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        
        # Chat input
        user_input = st.chat_input("Ask a question...")
        if user_input:
            st.session_state.ai_chat_history.append({"role": "user", "content": user_input})
            
            # Build context from selected articles
            context = ""
            if articles:
                context = f"User has {len(articles)} selected articles: " + ", ".join(
                    [a.get('title', '')[:50] for a in articles[:5]]
                )
            
            with st.spinner("Thinking..."):
                response = ai.chat(user_input, context)
                st.session_state.ai_chat_history.append({"role": "assistant", "content": response})
                st.rerun()


def search_suggestions_page():
    """Page for AI-powered search suggestions."""
    st.header("Search Suggestions")
    
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        api_key = st.text_input("Enter your Gemini API key:", type="password")
        if not api_key:
            st.warning("Please enter a Gemini API key to get search suggestions.")
            return
    
    ai = GeminiAI(api_key)
    
    topic = st.text_input("Enter your research topic")
    
    if st.button("Get Suggestions") and topic:
        with st.spinner("Generating suggestions..."):
            suggestions = ai.suggest_search_terms(topic)
            st.markdown("### Search Suggestions")
            st.markdown(suggestions)


def main():
    """Main application."""
    # Sidebar
    with st.sidebar:
        st.title("📚 Literature Review Assistant")
        st.divider()
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["Search", "Selected Articles", "AI Assistant", "Search Suggestions"]
        )
        
        st.divider()
        
        # Quick stats
        st.markdown("### Quick Stats")
        all_papers = LocalBibliography.get_all_papers()
        st.markdown(f"**Bibliography Size:** {len(all_papers)} articles")
        st.markdown(f"**Selected Articles:** {len(st.session_state.selected_articles)}")
        st.markdown(f"**Searches Made:** {len(st.session_state.search_history)}")
        
        st.divider()
        
        # Recent searches
        if st.session_state.search_history:
            st.markdown("### Recent Searches")
            for search in st.session_state.search_history[-5:]:
                st.markdown(f"- {search['query']} ({search['results_count']} results)")
        
        st.divider()
        
        # Settings
        with st.expander("Settings & Info"):
            has_gemini_key = bool(os.getenv("GEMINI_API_KEY"))
            
            st.markdown("**Gemini AI Status:**")
            if has_gemini_key:
                st.success("Configured - AI features available")
            else:
                st.warning("Not configured - Add GEMINI_API_KEY to .env")
            
            st.markdown("---")
            st.markdown("**Data Source:** Local bibliography")
            st.markdown(f"**File:** `{LocalBibliography.BIB_FILE}`")
    
    # Main content
    if page == "Search":
        search_page()
    elif page == "Selected Articles":
        selected_articles_page()
    elif page == "AI Assistant":
        ai_assistant_page()
    elif page == "Search Suggestions":
        search_suggestions_page()


if __name__ == "__main__":
    main()
