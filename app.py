import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
import faiss
import numpy as np
import pickle

# Set page config
st.set_page_config(page_title="NutriNudge", page_icon="🍎", layout="wide")