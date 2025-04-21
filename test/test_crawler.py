
import pytest
from crawler import crawl_and_extract

# Test a known simple website (could be replaced by a mock server or static page in production tests)
def test_crawl_and_extract():
    url = "https://httpbin.org/html"  # Static HTML page for testing
    pages = crawl_and_extract(url, max_depth=1)

    assert isinstance(pages, list)
    assert len(pages) > 0

    for page in pages:
        assert 'url' in page
        assert 'title' in page
        assert 'content' in page
        assert page['url'].startswith("http")
        assert isinstance(page['content'], str)
