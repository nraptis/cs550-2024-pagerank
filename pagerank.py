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
    print(corpus)
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
    all_pages = list(corpus.keys())
    num_pages = len(all_pages)

    result = {}

    links = corpus[page]

    if links:
        #navigate to any random page, weight = (1 - d)
        for _page in all_pages:
            result[_page] = (1.0 - damping_factor) / num_pages

        #navigate to a link, weight = d
        num_links = len(links)
        for link in links:
            result[link] += damping_factor / num_links
    else:
        #navigate to any random page, weight = 1
        for _page in all_pages:
            result[_page] = 1.0 / num_pages

    return result


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

    #how many times we visit each page
    visits = {page: 0 for page in all_pages}

    #start on a random page
    page = random.choice(all_pages)

    for _ in range(n):

        #visit the page
        visits[page] += 1

        #our model
        _transition_model = transition_model(corpus, page, damping_factor)

        #just the weights from each page
        weights = [_transition_model[page] for page in all_pages]

        #pick a new random page
        page = random.choices(all_pages, weights)[0]

    return {page: visits[page] / n for page in all_pages}

def greatest_difference(ranks_a, ranks_b):
    all_pages = list(ranks_a.keys())
    result = 0.0
    for page in all_pages:
        result = max(result, abs(ranks_a[page] - ranks_b[page]))
    return result


def iterate_pagerank_iteration(corpus, damping_factor, previous_ranks):

    all_pages = list(corpus.keys())
    num_pages = len(all_pages)

    updated_ranks = {page: (1.0 - damping_factor) / num_pages for page in all_pages}

    for page in all_pages:
        links = corpus[page]
        if links:
            num_links = len(links)

            #contribution from each link, per the formula.
            contribution = damping_factor * (previous_ranks[page] / num_links)
            for link in links:
                updated_ranks[link] += contribution

        else:
            #contribution from each page, per the formula.
            contribution = damping_factor * (previous_ranks[page] / num_pages)
            for _page in all_pages:
                updated_ranks[_page] += contribution

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
    num_pages = len(all_pages) # N

    page_ranks = {page: 1.0 / num_pages for page in all_pages}

    while True:
        updated_ranks = iterate_pagerank_iteration(corpus, damping_factor, page_ranks)
        difference = greatest_difference(page_ranks, updated_ranks)
        page_ranks = updated_ranks
        if difference < 0.001:
            break

    return page_ranks


if __name__ == "__main__":
    main()
