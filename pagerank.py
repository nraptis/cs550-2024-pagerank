import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    
    all_pages = set(corpus.keys())
    num_pages = len(all_pages)

    probabilities = {_page: 0.0 for _page in all_pages}

    page_links = corpus[page]
    if page_links and isinstance(page_links, set) and len(page_links) > 0:
        num_links = len(page_links)
        for _page in all_pages:
            probabilities[_page] += (1 / num_pages) * (1.0 - damping_factor)
        for link in page_links:
            probabilities[link] += (1 / num_links) * damping_factor
    else:
        # An equal chance to travel to any page, no links.
        for _page in all_pages:
            probabilities[_page] += (1 / num_pages)

    return probabilities

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    all_pages = list(corpus.keys())
    num_pages = len(all_pages)
    visits = {page: 0 for page in all_pages}
    page = random.choice(all_pages)
    for _ in range(n):
        visits[page] += 1
        _transition_model = transition_model(corpus, page, damping_factor)
        weights = [_transition_model[page] for page in all_pages]
        page = random.choices(all_pages, weights)[0]
    page_ranks = {page: visits[page] / n for page in all_pages}
    return page_ranks

def difference(ranks1, ranks2):
    greatest_difference = 0.0
    for page in ranks1.keys():
        _difference = ranks1[page] - ranks2[page]
        greatest_difference = max(greatest_difference, _difference)
    return greatest_difference

def iterate_pagerank_iteration(corpus, damping_factor, previous_ranks):

    all_pages = list(corpus.keys())
    num_pages = len(all_pages)    

    updated_ranks = {page: (1.0 - damping_factor) / num_pages for page in all_pages}

    for page in all_pages:
        page_links = corpus[page]
        if page_links and isinstance(page_links, set) and len(page_links) > 0:
            num_links = len(page_links)
            for link in page_links:
                updated_ranks[link] += damping_factor * (previous_ranks[page] / num_links)
        else:
            for _page in all_pages:
                updated_ranks[link] += damping_factor * (previous_ranks[_page] / num_pages)

    return updated_ranks

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    all_pages = list(corpus.keys())
    num_pages = len(all_pages)    
    page_ranks = {page: 1.0 / num_pages for page in all_pages}

    while True:
        updated_ranks = iterate_pagerank_iteration(corpus, damping_factor, page_ranks)
        if difference(updated_ranks, page_ranks) < 0.001:
            break
        page_ranks = updated_ranks

    return page_ranks


if __name__ == "__main__":
    main()
