from dotenv import load_dotenv
import os
import requests
import json
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

load_dotenv()  

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

url = "https://data.alpaca.markets/v1beta1/news"

def get_news(symbol, date, news_duration):

    # Convert the string date to a datetime object
    end_date = datetime.strptime(date, "%Y-%m-%d")

    # Subtract news_duration days
    start_date = end_date - timedelta(days=news_duration)

    # Convert back to string format if needed
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": SECRET_KEY,
    }

    params = {
        "limit": 50,  
        "symbols": symbol, 
        "start": start_date_str,
        "end": end_date_str,
        "sort":"desc",
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        response_json = json.loads(response.text)

        data = []
        for news_item in response_json['news']:
            item_data = {
                'id': news_item['id'],
                'author': news_item['author'],
                'created_at': news_item['created_at'],
                'updated_at': news_item['updated_at'],
                'headline': news_item['headline'],
                'summary': news_item['summary'],
                'source': news_item['source'],
                'url': news_item['url'],
                'symbols': news_item['symbols'],  
            }
            data.append(item_data)

        df = pd.DataFrame(data)

        return df
    else:
        return None
    
def get_url_content(url):
    params = {
    "token":"BENZINGA_KEY"
    }
    response = requests.request("GET", url, params=params)
    if response.status_code == 200:

        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string
        paragraphs = soup.find_all('p')
        content=''
        for p in paragraphs:
            content += p.get_text()
            content += '\n '

        return title, content
    else:
        return None, None
    
if __name__ == "__main__":
    symbol = "NVDA"  
    date = "2024-05-02"
    news_duration = 1
    df = get_news(symbol, date, news_duration)
    if df is not None:
        print(df.head())
    else:
        print("Failed to fetch news")