from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama

def generate_overall_summary(stock, news_df):
    llm = Ollama(model="llama3")

    overall_summary_template = """
    You are a financial analyst tasked with summarizing key information that could impact the stock price of {stock}. Below is a list of news topics and their summaries:

    {summaries}

    Based on these news items, please provide a concise summary of the major key points that might impact {stock}'s stock price. Focus on the following:

    1. Identify the most significant positive and negative factors.
    2. Highlight any recurring themes or trends across multiple news items.
    3. Mention any upcoming events or announcements that could influence the stock.
    4. Provide a brief overview of the general sentiment towards {stock} based on these news items.

    Your summary should be clear, concise, and directly related to potential stock price impacts. Aim for a length of 3-5 paragraphs.

    Summary:
    """

    # Prepare the summaries string
    summaries_text = ""
    for i in range(len(news_df)):
        summaries_text += f"Topic {i}: {news_df.topic.values[i]}\n"
        summaries_text += f"Summary: {news_df.summary.values[i]}\n\n"

    overall_summary_prompt = PromptTemplate(template=overall_summary_template, input_variables=["stock", "summaries"])
    
    # Create the chain
    chain = LLMChain(llm=llm, prompt=overall_summary_prompt)

    # Run the chain to get the overall summary
    overall_summary = chain.run(stock=stock, summaries=summaries_text)
    
    return overall_summary

def generate_impact_summary(stock, title, content):

    llm = Ollama(model="llama3")

    stock_price_impact_template = """
    You are an AI assistant specialized in investment advisory. 
    Provide a concise summary of the news article focusing only on the specified stock, interpret the expected impact of this news on the stock price as a score between -1 and 1 (where -1 indicates a negative impact and 1 indicates a positive impact):

    Stock: {stock}

    Title: {title}

    Content: {content}
    Summary:
    Impact Score:
    """

    stock_price_impact_prompt = PromptTemplate(template=stock_price_impact_template, input_variables=["stock", "title", "content"])

    # Create the chain
    chain = LLMChain(llm=llm, prompt=stock_price_impact_prompt)

    # Run the chain to get the impact score on stock price
    impact_score = chain.run(stock=stock, title=title, content=content)
    
    return impact_score

def summarize(stock, title, content):

    llm = Ollama(model="llama3")

    summary_template = """
    You are an AI assistant specialized in investment advisory. 
    Provide a concise summary of the news article focusing only on the specified stock:

    Stock: {stock}

    Title: {title}

    Content: {content}

    Summary:
    """

    summary_prompt = PromptTemplate(template=summary_template, input_variables=["stock", "title", "content"])

    # Create the chain
    chain = LLMChain(llm=llm, prompt=summary_prompt)

    # Run the chain to get the summary
    summary = chain.run(stock = stock, title=title, content=content)
    
    return summary

def get_trading_recommendation(sector_info, date, table_string, summary):

    llm = Ollama(model="llama3")

    trade_template = """
    As an expert trading agent with extensive experience in trading Nvidia stock, analyze the following information:

    Sector: {sector_info}
    Date: {date}
    Recent Stock Performance:
    {table_string}

    Recent News Summary:
    {summary}

    Based on the provided data, please offer an investment decision. Consider factors such as:
    1. Current market trends in the technology sector
    2. Nvidia's recent stock price momentum
    3. Potential impact of the news on stock performance

    Provide your analysis using the following format:

    Decision: (Buy/Sell/Hold)
    Reasoning: (Explain your decision in 3-5 concise points)
    Confidence Level: (Low/Medium/High)

    Answer:
    """

    trade_prompt = PromptTemplate(template=trade_template, input_variables=["sector_info", "date", "table_string", "summary"])

    chain = LLMChain(llm=llm, prompt=trade_prompt)

    trade = chain.run(sector_info=sector_info, date=date, table_string=table_string, summary=summary)

    return trade

def get_trading_recommendation_on_current_holding(holding, sector_info, date, table_string, summary):

    llm = Ollama(model="llama3")

    trade_template = """
    As an expert trading agent with extensive experience in trading Nvidia stock, analyze the following information:

    Sector: {sector_info}
    Date: {date}
    Recent Stock Performance:
    {table_string}

    Recent News Summary:
    {summary}

    Based on the provided data, please offer an investment decision. Consider factors such as:
    1. Current market trends in the technology sector
    2. Nvidia's recent stock price momentum
    3. Potential impact of the news on stock performance

    Provide your analysis using the following format:

    Decision: (Buy/Sell/Hold)
    Reasoning: (Explain your decision in 3-5 concise points)
    Confidence Level: (Low/Medium/High)

    Answer:
    """

    trade_prompt = PromptTemplate(template=trade_template, input_variables=["sector_info", "date", "table_string", "summary"])

    chain = LLMChain(llm=llm, prompt=trade_prompt)

    trade = chain.run(sector_info=sector_info, date=date, table_string=table_string, summary=summary)

    return trade

