import streamlit as st
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from create_vectordb import create_recent_db
from utils.price import get_price
from utils.llms import generate_overall_summary, get_trading_recommendation, summarize, generate_smart_query
from sentence_transformers import SentenceTransformer
import faiss
from utils.finbert import get_sentiment

# Streamlit page config
st.set_page_config(page_title="FinSights", layout="wide")

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .big-font {
        font-size:24px !important;
        font-weight: bold;
        color: #2E8B57; /* Green color for headers */
        margin-top: 20px;
    }
    .medium-font {
        font-size:18px !important;
        font-weight: bold;
        color: #006400; /* Dark green color for sub-headers */
        margin-top: 20px;
    }
    .small-font {
        font-size:14px !important;
        color: #006400; /* Dark green color for small text */
    }
    .stButton>button {
        font-size:18px !important;
        padding: 0.8em 1.5em;
        background-color: #2E8B57; /* Green color for buttons */
        color: white;
        border: none;
        border-radius: 4px;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #228B22; /* Darker green on hover */
    }
    .stButton {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .stSidebar .stSelectbox, .stSidebar .stNumberInput, .stSidebar .stDateInput {
        margin-bottom: 20px;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #2E8B57; /* Green color for markdown headers */
    }
    .summary-container {
        background-color: #F0FFF0; /* Honeydew background for summaries */
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .recommendation-container {
        background-color: #F5FFFA; /* Mintcream background for recommendations */
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar with collapsible input section
with st.sidebar.expander("Input Parameters", expanded=True):
    st.header("Input Parameters")
    stock = st.text_input("Stock Name", "Nvidia")

    stock_info = pd.read_csv("data/stock_info.csv")
    symbol_value = stock_info[stock_info["Stock"] == stock]["Symbol"].values[0]
    symbol = st.text_input("Stock Symbol", symbol_value)

    company_info = stock_info[stock_info["Stock"] == stock]["company_info"].values[0]
    company = st.text_input("Company Info", company_info)

    sector_info = stock_info[stock_info["Stock"] == stock]["Sector_info"].values[0]
    sector = st.text_input("Sector Info", sector_info)

    date = st.date_input("Analysis Date")
    news_duration = st.number_input("News Duration (days)", value=3, min_value=1)
    price_duration = st.number_input("Price Duration (days)", value=7, min_value=1)

# Initialize session state for relevant_news_df and price
if 'relevant_news_df' not in st.session_state:
    st.session_state.relevant_news_df = pd.DataFrame()

if 'price' not in st.session_state:
    st.session_state.price = pd.DataFrame()

# Main content
st.title("FinSights: AI-Driven Financial Analysis & Trading Strategies")

# Smart Query Generation
with st.expander("Generate Smart Query", expanded=False):
    if st.button("Generate Smart Query"):
        with st.spinner("Generating Smart Query..."):
            response = generate_smart_query(stock, company_info, sector_info)
            pattern = r'\n\n"(.*?)"\n\n'
            match = re.search(pattern, response, re.DOTALL)
            query_text = match.group(1) if match else "Nvidia, GPU, semiconductor, gaming, AI, data center, autonomous vehicle, edge computing, IoT, machine learning, autonomous driving"
    else:
        # Default user input for query
        query_text = "Nvidia, GPU, semiconductor, gaming, AI, data center, autonomous vehicle, edge computing, IoT, machine learning, autonomous driving"
    query = st.text_input("Enter Smart Query", query_text)

# News and Price Fetching
st.markdown('<p class="medium-font">Fetch Latest Data</p>', unsafe_allow_html=True)

with st.expander("Get Latest News", expanded=True):
    k = st.number_input("Number of Top Responses", value=5, min_value=1)
    if st.button("Get Latest News"):
        with st.spinner("Fetching latest news..."):
            create_recent_db(stock, str(date), news_duration)
            # Load the FAISS index
            index = faiss.read_index("data/news_content_index.faiss")
            filtered_news_df = pd.read_csv("data/news_content.csv")

            # Load the embedding model
            model = SentenceTransformer('all-MiniLM-L6-v2')

            def search_faiss(query, k):
                query_vector = model.encode([query])[0].astype('float32')
                query_vector = np.expand_dims(query_vector, axis=0)
                distances, indices = index.search(query_vector, k)
                return distances, indices

            with st.spinner("Searching relevant news..."):
                distances, indices = search_faiss(query, k)
                st.session_state.relevant_news_df = filtered_news_df[filtered_news_df.index.isin(indices[0])]
            
            if not st.session_state.relevant_news_df.empty:
                st.markdown('<p class="medium-font">Latest Relevant News:</p>', unsafe_allow_html=True)
                st.write(st.session_state.relevant_news_df['topic'].values)
            else:
                st.warning("No relevant news found.")

with st.expander("Get Latest Price", expanded=True):
    if st.button("Get Latest Price"):
        with st.spinner("Fetching price data..."):
            st.session_state.price = get_price(symbol, str(date), price_duration)
            st.markdown('<p class="medium-font">Price Data:</p>', unsafe_allow_html=True)
            st.dataframe(st.session_state.price)

# Run Analysis Section
if st.button("Run Analysis"):
    if st.session_state.relevant_news_df.empty:
        st.warning("Please get the latest news before running the analysis.")
    else:
        summary_list = []
        with st.spinner("Generating summaries..."):
            for i in tqdm(range(len(st.session_state.relevant_news_df))):
                topic = st.session_state.relevant_news_df.topic.values[i]
                content = st.session_state.relevant_news_df.content.values[i]
                summary = summarize(stock, topic, content)
                summary_list.append(summary)
                with st.expander(f"News Article {i+1}: {topic}"):
                    st.write(summary)
            st.session_state.relevant_news_df['summary'] = summary_list

        with st.spinner("Generating overall summary..."):
            overall_summary = generate_overall_summary(stock, st.session_state.relevant_news_df)
        st.markdown('<div class="summary-container"><p class="big-font">Overall Summary:</p>', unsafe_allow_html=True)
        st.write(overall_summary)
        st.markdown('</div>', unsafe_allow_html=True)

        with st.spinner("Generating trading recommendation..."):
            trade_recommendation = get_trading_recommendation(sector_info, str(date), st.session_state.price.to_string(index=False), overall_summary)
        st.markdown('<div class="recommendation-container"><p class="big-font">Trading Recommendation:</p>', unsafe_allow_html=True)
        st.write(trade_recommendation)
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Enter your query and click 'Run Analysis' to start the analysis process.")


