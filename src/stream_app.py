import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
from comparing_prompts import prepare_data, collabPrompt, genrePrompts, twoStepPrompt1, twoStepPrompt2, wikiPrompt
from groq import Client

# Load API key từ .env file
dotenv_path = 'D:/test/LLM-Recommender-System-with-RAG/key_api.env'
load_dotenv(dotenv_path)
api_key = os.getenv('GROQ_API_KEY')
client = Client(api_key=api_key)

def calculate_hit_rate(recommended_movies, watched_movies):
    hit_rate = len(set(recommended_movies).intersection(watched_movies)) / len(recommended_movies)
    return hit_rate

# Load dữ liệu
def load_data():
    movie100k = pd.read_csv('D:/test/LLM-Recommender-System-with-RAG/data/processed_movie100k.csv')
    movie_wiki = pd.read_csv('D:/test/LLM-Recommender-System-with-RAG/data/movie_wiki.csv')
    train_data = pd.read_csv('D:/test/LLM-Recommender-System-with-RAG/data/train.csv')
    test_data = pd.read_csv('D:/test/LLM-Recommender-System-with-RAG/data/test.csv')
    return movie100k, movie_wiki, train_data, test_data

# Tải dữ liệu
movie100k, movie_wiki, train_data, test_data = load_data()
user_prompt_data = prepare_data()

def generate_recommendations(prompt):
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=150,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

# Giao diện Streamlit
st.title('Movie Recommender System')

st.sidebar.header('User Input')
option = st.sidebar.selectbox('Recommendation Method', ('Most Popular', 'Collaborative Filtering', 'Candidate Genre', 'Two Step', 'Wiki Summary'))

user_id = st.sidebar.number_input('User ID', min_value=1, max_value=943, value=1)

if option == 'Most Popular':
    st.header('Most Popular Movies Recommendation')
    viewings = train_data.groupby('movie_title').count().sort_values('user_id', ascending=False)
    top_10_movies = viewings.head(10).index.tolist()
    st.subheader('Top 10 Most Popular Movies')
    st.table(pd.DataFrame({'Movies': top_10_movies}))
    
    user_movies = test_data[test_data['user_id'] == user_id]['movie_title'].tolist()
    hit_rate = calculate_hit_rate(top_10_movies, user_movies)
    st.subheader(f'User {user_id} watched movies')
    st.table(pd.DataFrame({'Movies': user_movies}))
    st.write(f'Hit rate for user {user_id}: {hit_rate:.2f}')

elif option == 'Collaborative Filtering':
    st.header('Collaborative Filtering Recommendation')
    if user_id not in user_prompt_data.index:
        st.write(f"User ID {user_id} not found in the dataset.")
    else:
        prompt = collabPrompt(user_id, user_prompt_data)
        recommendations = generate_recommendations(prompt)
        st.write(f'Recommendations for User {user_id}:')
        st.write(recommendations)

elif option == 'Candidate Genre':
    st.header('Candidate Genre Recommendation')
    if user_id not in user_prompt_data.index:
        st.write(f"User ID {user_id} not found in the dataset.")
    else:
        prompt = genrePrompts(user_id, user_prompt_data)
        recommendations = generate_recommendations(prompt)
        st.write(f'Recommendations for User {user_id}:')
        st.write(recommendations)

elif option == 'Two Step':
    st.header('Two Step Recommendation')
    if user_id not in user_prompt_data.index:
        st.write(f"User ID {user_id} not found in the dataset.")
    else:
        step1_prompt = twoStepPrompt1(user_id, user_prompt_data)
        step1_response = generate_recommendations(step1_prompt)
        step2_prompt = twoStepPrompt2(user_id, user_prompt_data, step1_response)
        recommendations = generate_recommendations(step2_prompt)
        st.write(f'Recommendations for User {user_id}:')
        st.write(recommendations)

elif option == 'Wiki Summary':
    st.header('Wiki Summary Recommendation')
    if user_id not in user_prompt_data.index:
        st.write(f"User ID {user_id} not found in the dataset.")
    else:
        prompt = wikiPrompt(user_id, user_prompt_data, movie_wiki)
        recommendations = generate_recommendations(prompt)
        st.write(f'Recommendations for User {user_id}:')
        st.write(recommendations)
