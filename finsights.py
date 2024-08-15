import streamlit as st
import pandas as pd
import numpy as np
from tqdm import tqdm
from create_vectordb import create_recent_db
from utils.price import get_price
from utils.llms import generate_overall_summary, get_trading_recommendation, summarize
from sentence_transformers import SentenceTransformer
import faiss
from utils.finbert import get_sentiment
# import plotly.graph_objects as go

# Streamlit page config
st.set_page_config(page_title="FinSights", layout="wide")

# Custom CSS
st.markdown("""
<style>
.big-font {
    font-size:20px !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.header("Input Parameters")
stock = st.sidebar.text_input("Stock Name", "Nvidia")
symbol = st.sidebar.text_input("Stock Symbol", "NVDA")
date = st.sidebar.date_input("Analysis Date")
price_duration = st.sidebar.number_input("Price Duration (days)", value=7, min_value=1)

news_duration = 1
sector_info = '''Nvidia operates in the semiconductor industry, specializing in the design and manufacture of graphics processing units (GPUs) for gaming, professional visualization, data centers, AI, and automotive applications. 
Known for its GeForce brand, Nvidia leads the gaming market while also providing high-performance solutions for AI and machine learning in data centers. 
Additionally, Nvidia's DRIVE platform supports autonomous vehicle technology, and its Jetson platform powers edge computing and IoT devices. 
The company's innovations position it at the forefront of emerging trends in AI, data processing, and autonomous driving, driving significant growth and industry influence.
'''

# Main content
st.title("FinSights: AI-Powered Stock Analysis")

# def display_sentiment_box(sentiment, summary):
#     if sentiment == "Negative":
#         color = "red"
#     elif sentiment == "Positive":
#         color = "green"
#     elif sentiment == "Neutral":
#         color = "yellow"

#     st.markdown(f"""
#         <div style="border-radius:5px; padding: 10px; margin: 5px 0; background-color:{color};">
#             {summary}
#         </div>
#     """, unsafe_allow_html=True)

# User input for query
query = st.text_input("Enter your query about the stock", f"Which article would have the highest impact on {stock} stock price? Ticker: {symbol}")

if st.button("Run Analysis"):
    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        with st.spinner("Fetching latest news..."):
            create_recent_db(stock, str(date), news_duration)

        # Load the FAISS index
        index = faiss.read_index("data/news_content_index.faiss")
        filtered_news_df = pd.read_csv("data/news_content.csv")

        # Load the embedding model
        model = SentenceTransformer('all-MiniLM-L6-v2')

        def search_faiss(query, k=5):
            query_vector = model.encode([query])[0].astype('float32')
            query_vector = np.expand_dims(query_vector, axis=0)
            distances, indices = index.search(query_vector, k)
            return distances, indices

        with st.spinner("Searching relevant news..."):
            distances, indices = search_faiss(query)
            relevant_news_df = filtered_news_df[filtered_news_df.index.isin(indices[0])]
        st.markdown('<p class="big-font">Latest Relevant News:</p>', unsafe_allow_html=True)
        st.dataframe(relevant_news_df[['topic', 'content']], height=300)

        summary_list = []
        # sentiment_list = []
        with st.spinner("Generating summaries..."):
            for i in tqdm(range(len(relevant_news_df))):
                topic = relevant_news_df.topic.values[i]
                content = relevant_news_df.content.values[i]
                summary = summarize(stock, topic, content)
                summary_list.append(summary)
                # sentiment = get_sentiment(summary)
                # sentiment_list.append(sentiment)
                with st.expander(f"News Article {i+1}: {topic}"):
                    st.write(summary)
            relevant_news_df['summary'] = summary_list

    with col2:
        with st.spinner("Generating overall summary..."):
            overall_summary = generate_overall_summary(stock, relevant_news_df)
        st.markdown('<p class="big-font">Overall Summary:</p>', unsafe_allow_html=True)
        st.write(overall_summary)

        with st.spinner("Fetching price data..."):
            price = get_price(symbol, str(date), price_duration)
            table_string = price.to_string(index=False)
        st.markdown('<p class="big-font">Price Data:</p>', unsafe_allow_html=True)
        st.dataframe(price)

        # # Create a line chart for the stock price
        # fig = go.Figure(data=go.Scatter(x=price['Date'], y=price['Close'], mode='lines+markers'))
        # fig.update_layout(title=f'{stock} Stock Price', xaxis_title='Date', yaxis_title='Price')
        # st.plotly_chart(fig)

        with st.spinner("Generating trading recommendation..."):
            trade_recommendation = get_trading_recommendation(sector_info, str(date), table_string, overall_summary)
        st.markdown('<p class="big-font">Trading Recommendation:</p>', unsafe_allow_html=True)
        st.write(trade_recommendation)

else:
    st.info("Enter your query and click 'Run Analysis' to start the analysis process.")