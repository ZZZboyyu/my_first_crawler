"""Simple Hacker News crawler.

Day 2 goal:
- Fetch the Hacker News front page.
- Parse story titles and URLs.
- Save the result to hacker_news.csv.
"""

from __future__ import annotations

import csv
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import requests
from bs4 import BeautifulSoup


TARGET_URL = "https://news.ycombinator.com/"
OUTPUT_CSV = "hacker_news.csv"


def fetch_html(url: str) -> str:
    """Fetch HTML from the target URL."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        return response.text
    except requests.RequestException as exc:
        print(f"requests failed: {exc}")
        print("Falling back to urllib...")

    request = Request(url, headers=headers)
    with urlopen(request, timeout=15) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def parse_stories(html: str) -> list[dict[str, str]]:
    """Parse story title and URL from Hacker News HTML."""
    soup = BeautifulSoup(html, "html.parser")
    story_rows = soup.find_all("tr", class_="athing")

    stories: list[dict[str, str]] = []
    for rank, row in enumerate(story_rows, start=1):
        titleline = row.find("span", class_="titleline")
        if titleline is None:
            continue

        link = titleline.find("a")
        if link is None:
            continue

        title = link.get_text(strip=True)
        href = link.get("href", "")
        url = urljoin(TARGET_URL, href)

        stories.append(
            {
                "rank": str(rank),
                "title": title,
                "url": url,
            }
        )

    return stories


def save_csv(stories: list[dict[str, str]], output_path: Path) -> None:
    """Save stories to a CSV file that opens cleanly in Excel."""
    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["rank", "title", "url"])
        writer.writeheader()
        writer.writerows(stories)


def main() -> None:
    print(f"Fetching {TARGET_URL}")
    html = fetch_html(TARGET_URL)

    stories = parse_stories(html)
    if not stories:
        raise RuntimeError("No Hacker News stories were found. The page structure may have changed.")

    output_path = Path(__file__).resolve().parent / OUTPUT_CSV
    save_csv(stories, output_path)

    print(f"Saved {len(stories)} stories to {output_path}")
    print("Top 5:")
    for story in stories[:5]:
        print(f"  {story['rank']}. {story['title']}")


if __name__ == "__main__":
    main()
