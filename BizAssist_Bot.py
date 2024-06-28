# -*- coding: utf-8 -*-
"""
"""
import numpy as np
import streamlit as st
import pandas as pd
import re
from transformers import BertTokenizer, BertModel

from sklearn.metrics.pairwise import cosine_similarity
import torch

# Load the dataset
df = pd.read_csv('custom_qa_dataset.csv')

# Preprocess the data
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

df['questions'] = df['questions'].apply(clean_text)
df['answers'] = df['answers'].apply(clean_text)

# Load BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Tokenize and encode questions
encoded_questions = tokenizer(df['questions'].tolist(), padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    question_embeddings = model(**encoded_questions).last_hidden_state.mean(dim=1)

def get_most_similar_answer(question, question_embeddings, df):
    # Clean and encode the input question
    cleaned_question = clean_text(question)
    encoded_question = tokenizer(cleaned_question, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        question_embedding = model(**encoded_question).last_hidden_state.mean(dim=1)
    
    # Calculate cosine similarity
    similarities = cosine_similarity(question_embedding, question_embeddings).flatten()
    
    # Find the index of the most similar question
    most_similar_idx = similarities.argmax()
    
    # Return the corresponding answer
    return df['answers'].iloc[most_similar_idx]

def main():
    # Title of the web app
    st.title('Business Assistant Bot')

    # Getting the input data from the user
    question = st.text_input("Enter your question")

    # Code for prediction
    answer = ''

    # Creating a button for prediction
    if st.button('Get Answer'):
        answer = get_most_similar_answer(question, question_embeddings, df)
        
    st.success(answer)

if __name__ == '__main__':
    main()
