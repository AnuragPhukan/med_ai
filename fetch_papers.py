import requests
import time

def fetch_papers(keyword):
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={keyword}&limit=10&fields=title,abstract,year,authors.name,authors.authorId,authors.affiliations,url,publicationTypes,venue,citationCount"
    retries = 5
    for i in range(retries):
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            print("Full API Response:", data)  # Print full API response for inspection
            return data.get('data', [])
        elif response.status_code == 429:
            wait = 2 ** i
            print(f"Rate limit exceeded. Retrying in {wait} seconds...")
            time.sleep(wait)
        else:
            response.raise_for_status()
    return []
