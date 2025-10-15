import streamlit as st
from newspaper import Article
from serpapi import GoogleSearch
from transformers import pipeline
import pandas as pd
import torch

# ----------------- CONFIG -----------------
SERPAPI_KEY = "347946e7d3d427a6ce6f4970e34a0a4652360fab131c675c01e17c511c936a6a"  # Replace with your SerpAPI key
st.set_page_config(page_title="News Research Assistant", layout="wide")

# ----------------- LOAD SUMMARIZER -----------------
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# ----------------- SCRAPE FUNCTION -----------------
def fetch_article_data(url):
    """Extracts title, image, and text safely from an article URL"""
    try:
        article = Article(url)
        article.download()
        article.parse()
        data = {
            "title": article.title,
            "image": article.top_image if article.top_image else None,
            "text": article.text[:5000]  # limit length
        }
        return data
    except Exception as e:
        return {"title": "Error fetching article", "image": None, "text": str(e)}

# ----------------- SUMMARIZE FUNCTION -----------------
def summarize_text(text):
    try:
        if len(text.split()) < 50:
            return "Text too short to summarize."
        summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception:
        return "Summary unavailable."

# ----------------- SEARCH FUNCTION -----------------
def google_search(query):
    params = {
        "q": query,
        "api_key": SERPAPI_KEY,
        "engine": "google",
        "num": 10,
        "tbm": "nws"
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    return results.get("news_results", [])

# ----------------- UI -----------------
st.title("🧠 News Research Assistant")
st.write("Search your company’s articles across the web, get summaries, and export results.")

keyword = st.text_input("🔍 Enter keyword (e.g. 'Edelman AI Campaign')", "")
use_summary = st.checkbox("Generate AI summaries (slower but detailed)", value=True)

if st.button("Search Now") and keyword:
    with st.spinner("Searching Google News..."):
        news_results = google_search(keyword)

    if news_results:
        st.success(f"Found {len(news_results)} results.")
        data_list = []

        for result in news_results:
            url = result.get("link")
            source = result.get("source")
            st.markdown(f"### [{result.get('title')}]({url})")
            st.caption(f"📰 Source: {source}")

            article_data = fetch_article_data(url)

            if article_data["image"]:
                st.image(article_data["image"], width=400)
            else:
                st.image("https://via.placeholder.com/400x200?text=No+Image", width=400)

            summary = summarize_text(article_data["text"]) if use_summary else article_data["text"][:500]
            st.write(summary)

            data_list.append({
                "Title": article_data["title"],
                "Source": source,
                "URL": url,
                "Summary": summary,
                "Image": article_data["image"]
            })

        df = pd.DataFrame(data_list)
        st.dataframe(df)

        # Export option
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"{keyword.replace(' ', '_')}_news.csv",
            mime='text/csv'
        )
    else:
        st.warning("No results found. Try different keywords.")

