# Fine-Tuning Guide

This guide outlines the steps for fine-tuning the LLaMA 3.1 8B model using the SFT Trainer and QLoRA techniques.

## Steps for Fine-Tuning

1. **Generate Historical News Data:**
   - Use the `generate_news` notebook to gather historical news data.
   - This notebook utilizes the Alpaca and Benzinga APIs to retrieve news data relevant to your analysis.

2. **Prepare the Dataset:**
   - Run the `generate_dataset` notebook to simulate trading scenarios based on the news data.
   - The notebook processes the data and filters out trade rationales where returns are in your favor.

3. **Perform QLoRA Fine-Tuning:**
   - Use the filtered dataset to fine-tune the LLaMA 3.1 8B model.
   - The fine-tuning process employs the QLoRA technique to adapt the model to trading rationales and improve its performance.


By following these steps, you'll effectively fine-tune the LLaMA 3.1 8B model to enhance its performance in trading-related tasks.
