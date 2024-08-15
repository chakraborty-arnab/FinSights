import pandas as pd
import numpy as np
from tqdm import tqdm
from create_vectordb import create_recent_db
from utils.price import get_price
from utils.llms import generate_impact_summary, generate_overall_summary, get_trading_recommendation, summarize
from sentence_transformers import SentenceTransformer
import faiss

stock = "Nvidia"
symbol = "NVDA"  
date = "2024-05-02"
news_duration = 1
price_duration = 7
sector_info='''Nvidia operates in the semiconductor industry, specializing in the design and manufacture of graphics processing units (GPUs) for gaming, professional visualization, data centers, AI, and automotive applications. 
Known for its GeForce brand, Nvidia leads the gaming market while also providing high-performance solutions for AI and machine learning in data centers. 
Additionally, Nvidia's DRIVE platform supports autonomous vehicle technology, and its Jetson platform powers edge computing and IoT devices. 
The company's innovations position it at the forefront of emerging trends in AI, data processing, and autonomous driving, driving significant growth and industry influence.
'''
# create_recent_db(stock, date, news_duration)

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

# Example usage
query = "Which article would have the highest impact on nvidia stock price? Ticker: NVDA"

distances, indices = search_faiss(query)

relevant_news_df = filtered_news_df[filtered_news_df.index.isin(indices[0])]

summary_list=[]
for i in tqdm(range(len(relevant_news_df))):
    topic = relevant_news_df.topic.values[i]
    content = relevant_news_df.content.values[i]
    summary = summarize(stock, topic, content)
    summary_list.append(summary)
relevant_news_df['summary'] = summary_list

overall_summary = generate_overall_summary(stock, relevant_news_df)

price = get_price(symbol, date, price_duration)
table_string = price.to_string(index=False)
trade_recommendation = get_trading_recommendation(sector_info, date, table_string, overall_summary)

print(trade_recommendation)
