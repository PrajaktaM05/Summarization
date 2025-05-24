import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppresses TensorFlow logs
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Now import libraries
import tensorflow as tf
from transformers import pipeline





# Initialize summarization pipeline
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Your NewsAPI key here
NEWSAPI_KEY = "ab8ed41cbb7242debbdf07bd94608bc3"

def fetch_news(company):
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={company}&"
        f"language=en&"
        f"sortBy=relevance&"
        f"apiKey={NEWSAPI_KEY}"
    )
    response = requests.get(url)
    if response.status_code != 200:
        print("Error fetching news:", response.status_code)
        return []
    data = response.json()
    if data.get("status") != "ok" or data.get("totalResults", 0) == 0:
        return []
    return data.get("articles", [])

def summarize_text(text):
    input_len = len(text.split())
    max_len = min(50, input_len)
    min_len = max(10, max_len // 2)
    summary = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
    return summary[0]['summary_text']

def main():
    company = input("Enter company name to search news: ").strip()
    print(f"\nFetching news for: {company}...\n")
    articles = fetch_news(company)
    if not articles:
        print("No news found or unable to fetch.")
        return
    for idx, article in enumerate(articles[:10], 1):  # Only take top 10 articles
        title = article.get("title", "No title")
        description = article.get("description")
        print(f"{idx}. {title}\n")
        if description:
            summary = summarize_text(description)
            print(f"Summary: {summary}\n")
        else:
            print("Summary: No description available.\n")

if __name__ == "__main__":
    main()