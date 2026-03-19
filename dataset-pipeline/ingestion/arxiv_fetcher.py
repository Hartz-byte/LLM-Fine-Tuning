import arxiv

def fetch_papers(query="LLM", max_results=50):
    search = arxiv.Search(query=query, max_results=max_results)
    
    papers = []
    for result in search.results():
        papers.append({
            "title": result.title,
            "abstract": result.summary,
            "authors": [a.name for a in result.authors]
        })
    
    return papers
