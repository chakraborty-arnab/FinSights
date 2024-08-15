import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta
from utils.news import get_news, get_url_content

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def create_recent_db(symbol, date, news_duration):
    try:
        # Calculate the previous date
        current_date = datetime.strptime(date, "%Y-%m-%d")
        yesterday_date_str = (current_date - timedelta(days=1)).strftime("%Y-%m-%d")
        
        # Load the existing news content if available, else initialize an empty DataFrame
        try:
            filtered_news_df = pd.read_csv('data/news_content.csv')
            filtered_news_df['date'] = pd.to_datetime(filtered_news_df['updated_at'], format='%Y-%m-%dT%H:%M:%SZ', utc=True)
        except FileNotFoundError:
            filtered_news_df = pd.DataFrame()

        # Check if news data for the given date already exists
        if not filtered_news_df[filtered_news_df['date'].dt.strftime("%Y-%m-%d") == yesterday_date_str].empty:
            print("News data for the given date already exists. Skipping fetching new data.")
        else:
            # Fetch new news data
            news_df = get_news(symbol, date, news_duration)
            new_filtered_news_df = news_df[news_df['symbols'].apply(lambda x: symbol in x)]
            
            # Extract content from URLs
            new_filtered_news_df['topic'], new_filtered_news_df['content'] = zip(*[get_url_content(url) for url in new_filtered_news_df['url'].values])
            
            # Append the new data
            filtered_news_df = pd.concat([filtered_news_df, new_filtered_news_df], ignore_index=True)
            filtered_news_df.to_csv('data/news_content.csv', index=False)
            print("News data fetched and appended.")

        # Generate embeddings for the relevant content
        relevant_news = filtered_news_df[filtered_news_df['date'].dt.strftime("%Y-%m-%d") == yesterday_date_str]
        embeddings_np = model.encode(relevant_news.content.values).astype('float32')
        
        # Create and save FAISS index
        index = faiss.IndexFlatL2(embeddings_np.shape[1])
        index.add(embeddings_np)
        faiss.write_index(index, "data/news_content_index.faiss")
        print("Database and FAISS index created successfully.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    symbol = "NVDA"  
    date = "2024-05-02"
    news_duration = 1
    create_recent_db(symbol, date, news_duration)

