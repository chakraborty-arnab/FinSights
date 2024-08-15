import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_price(symbol, date, duration):

    # Convert the string date to a datetime object
    end_date = datetime.strptime(date, "%Y-%m-%d")

    # Subtract news_duration days
    start_date = end_date - timedelta(days=duration)

    # Convert back to string format if needed
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    price = yf.download(symbol, start=start_date_str, end=end_date_str)

    return price