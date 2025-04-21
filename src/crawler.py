import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque

def crawl_and_extract(start_url: str, max_depth: int = 1):
    visited = set()
    queue = deque([(start_url, 0)])
    pages = []

    while queue:
        url, depth = queue.popleft()
        if url in visited or depth > max_depth:
            continue
        try:
            response = requests.get(url, timeout=10)
            if 'text/html' not in response.headers.get('Content-Type', ''):
                continue
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.title.string if soup.title else ''
            text = ' '.join(p.get_text() for p in soup.find_all('p'))
            pages.append({
                'url': url,
                'title': title,
                'content': text.strip()
            })
            visited.add(url)

            # Extract and queue internal links
            base_url = urlparse(start_url).netloc
            for link in soup.find_all('a', href=True):
                next_url = urljoin(url, link['href'])
                if urlparse(next_url).netloc == base_url:
                    queue.append((next_url, depth + 1))

        except requests.RequestException:
            continue

    return pages
