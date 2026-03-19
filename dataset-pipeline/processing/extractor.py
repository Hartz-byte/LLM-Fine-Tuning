def extract_basic_info(paper):
    return {
        "title": paper["title"],
        "abstract": paper["abstract"],
        "length": len(paper["abstract"])
    }
