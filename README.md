# FinSights
## AI-Driven Financial Analysis & Trading Strategies

![image](https://github.com/user-attachments/assets/06f67fb2-c428-4800-916c-8d351257f763)


### Introduction
FinSights aims to harness the power of generative AI and retrieval-augmented generation (RAG) to create memory retrieval modules that can autonomously assist in decision-making for investors. FinSights combines price action and news to analyze earnings reports and market sentiment from news articles. The project’s objective is to build an AI-powered agent that can enhance trading strategies by offering data-driven insights, thus helping investors achieve better risk-adjusted returns.

### To Run

1. **Download Ollama:**
   - Visit the [Ollama website](https://ollama.com) to download and install the tool.
     
2. **Pull the Llama 3.1 model using Ollama:**
   ```bash
    ollama pull llama3.1
   ```
3. **Clone the Repository:**
   ```bash
    git clone https://github.com/your-username/FinSights.git
    cd FinSights
   ```
4. **Install the required dependencies:**
   ```
    pip install -r requirements.txt
   ```
   * Ensure you have Python 3.8 or above installed.

5. **Set Up Environment Variables**
 
     Create .env File:
 
    * Copy the .env.template file and rename it to .env:
      
```
  cp .env.template .env
```

## Add your API keys:
1. [Alpaca API Key and Secret](https://alpaca.markets/)
2. [Benzinga API Key](https://www.benzinga.com/apis/)

# Fine-Tune
The fine-tuned folder contains models that have been fine-tuned, including a fine-tuned Llama 3.1 model that incorporates trading rationale. You can explore the folder for more details on how the models have been customized for FinSights.

## Running the Application

Start the Streamlit app:
```
streamlit run app.py
```
The application will be accessible in your web browser at `http://localhost:8501`.

# Results

| Type of Decision  | Buy | Sell |
|------------------ |-----|------|
| Correct Decisions | 34  | 5    |
| Wrong Decisions   | 26  | 0    |

## Youtube Video
Feel free to check our youtube Demo : https://youtu.be/NvtjVLcBGwA

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For any questions or feedback, feel free to reach out via [Aditi's LinkedIn](https://www.linkedin.com/in/yaditi/) [Arnab's LinkedIn](https://www.linkedin.com/in/arnab-chakraborty13/)
