# Literature Review Assistant

A Streamlit application for searching, selecting, and analyzing research articles for literature reviews. Uses the Semantic Scholar API for paper discovery and Google Gemini AI for intelligent analysis features.

## Features

### Search & Discovery
- **Semantic Scholar Integration**: Search millions of academic papers
- **Advanced Filters**: Filter by year range and fields of study
- **Sort Options**: Sort by relevance, citations, or year
- **Article Selection**: Save interesting papers for later review

### AI-Powered Analysis (Gemini)
- **Article Summarization**: Get concise summaries of papers
- **Comparative Analysis**: Compare multiple papers to find themes and gaps
- **Research Question Generation**: Generate potential research questions
- **Literature Section Writing**: Draft literature review sections with proper citations
- **Research Chat**: Ask questions and get advice about your literature review

### Explore Connections
- **Citation Tracking**: Find papers that cite your selected articles
- **Reference Exploration**: Discover the papers your selections are built upon
- **Network Discovery**: Build a comprehensive understanding of a research area

### Export Options
- **CSV Export**: Export selected articles as a spreadsheet
- **BibTeX Export**: Generate citation file for reference managers

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API Key**:
   Create a `.env` file in the project root:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
   
   Or enter it directly in the app when prompted.

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## Usage

### Basic Workflow

1. **Search**: Start on the Search page, enter your research topic, and optionally set filters
2. **Select**: Click the "+" button on articles you want to include in your review
3. **Review**: Go to "Selected Articles" to see all your saved papers
4. **Analyze**: Use the "AI Assistant" to summarize, compare, and write about your selections
5. **Explore**: Use the "Explore" page to find related papers through citations
6. **Export**: Download your selection as CSV or BibTeX

### Tips

- Use specific search terms for better results
- Select papers from different years to show the evolution of a topic
- Use the AI comparison feature to identify themes across papers
- Export to BibTeX for easy import into Zotero, Mendeley, or EndNote

## API Information

### Semantic Scholar API
- Free to use, no API key required
- Rate limited to 100 requests per 5 minutes
- [API Documentation](https://api.semanticscholar.org/)

### Google Gemini API
- Requires API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
- Free tier available with generous limits

## Project Structure

```
activity3/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env.example        # Example environment file
└── README.md           # This file
```

## Technologies Used

- **Streamlit**: Web application framework
- **Semantic Scholar API**: Academic paper search
- **Google Gemini**: AI-powered analysis
- **Pandas**: Data manipulation and export

## License

MIT License - Feel free to use and modify for your research needs.
