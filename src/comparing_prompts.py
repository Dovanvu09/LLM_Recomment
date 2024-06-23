import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load dữ liệu
train = pd.read_csv('D:\\test\\LLM-Recommender-System-with-RAG\\data\\train.csv')
train.head()

# Các bộ phim mà mỗi người dùng đã xem trong dữ liệu huấn luyện
user_movies = train.groupby('user_id', group_keys=False).apply(lambda x: x['movie_title'].tolist())

# Số lượng người dùng để chạy recommender (sử dụng mẫu nhỏ để kết quả nhanh hơn)
num_users = 20

# Uncomment below to run on all users
# n = user_movies.shape[0]

# Các bộ phim gần đây nhất mà mỗi người dùng đã xem
def get_user_recent_movies(dataframe, n=10):
    return dataframe.groupby('user_id', group_keys=False).apply(lambda x: x['movie_title'].tail(n).to_list())

user_recent_movies = get_user_recent_movies(train)

# Các bộ phim mà mỗi người dùng đã đánh giá cao nhất
def get_user_top_movies(dataframe, n=10):
    return dataframe.groupby('user_id', group_keys=False).apply(lambda x: x.sort_values('rating', ascending=False).head(n)['movie_title'].to_list())

user_top_movies = get_user_top_movies(train)

# Thể loại của mỗi bộ phim
movie_genres = train.groupby('movie_title', group_keys=False).apply(lambda x: x['genres'].unique().tolist()).apply(lambda x: x[0].split(", "))

user_top_movie_genres = user_top_movies.apply(lambda x: pd.Series(movie_genres.loc[x].sum()).unique().tolist())

# Đánh giá của mỗi người dùng cho mỗi bộ phim mà họ đã xem trong dữ liệu huấn luyện
user_ratings = train.drop_duplicates(['user_id', 'movie_title']).pivot(index='user_id', columns='movie_title', values='avg_rating').fillna(0)
sparse_user_ratings = csr_matrix(user_ratings)

# Tìm người dùng tương tự dựa trên cosine similarity
def get_similar_users(sparse_user_ratings, index, n=10):
    return pd.DataFrame(cosine_similarity(sparse_user_ratings) - np.identity(sparse_user_ratings.shape[0]), index=index, columns=index).apply(lambda x: list(x.sort_values(ascending=False).head(n).index), axis=1)

similar_users = get_similar_users(sparse_user_ratings, user_ratings.index)

# Xác định các bộ phim ứng viên mà mỗi người dùng tương tự đã đánh giá cao nhất
def get_candidate_movies(similar_users, dataframe, n=20):
    return similar_users.apply(lambda x: list(dataframe.loc[dataframe['user_id'].isin(x)].groupby('movie_title').sum().sort_values('rating', ascending=False).index[:n]))

candidate_movies = get_candidate_movies(similar_users, train)

# Dữ liệu cho lời nhắc
user_prompt_data = pd.concat((user_movies, user_top_movies, user_recent_movies, similar_users, candidate_movies, user_top_movie_genres), axis=1, keys=['user_movies', 'user_top_movies', 'user_recent_movies', 'similar_users', 'candidate_movies', 'user_top_movie_genres'])

new_line = '\n'


def collabPrompt(each, user_prompt_data, new_line = '\n'):
    return f"""I am user {each}.
            The most recent ten movies I have seen are:
            {", ".join(user_prompt_data.loc[each, 'user_recent_movies'])}.
            My top rated movies are:
            {", ".join(user_prompt_data.loc[each, 'user_top_movies'])}.
            The users who are most like me are {", ".join([str(each) for each in user_prompt_data.loc[each, 'similar_users']])}.
            The top movies for each of these users are:
            {new_line.join([f"{each}: {', '.join(user_prompt_data.loc[each, 'user_top_movies'])}" for each in user_prompt_data.loc[each, 'similar_users']])}.
            Please recommend ten movies for me to watch that I have not seen. Provide brackets around your recommendations so I can easily parse them.
            For example ([Midnight Cowboy (1969){new_line}Lost in Translation (2003){new_line}etc.])"""
collab_prompts = pd.Series(user_prompt_data.index, index = user_prompt_data.index).apply(lambda x: collabPrompt(x, user_prompt_data))
from dotenv import load_dotenv
from groq import Client
import os
dotenv_path = 'D:\\test\\LLM-Recommender-System-with-RAG\\key_api.env'  # Thay thế bằng đường dẫn thực tế
load_dotenv(dotenv_path)
api_key = os.getenv('GROQ_API_KEY')
client = Client(api_key=api_key)
collab_response = collab_prompts.iloc[:num_users].apply(lambda x: client.chat.completions.create(
    model="llama3-8b-8192",
        messages= [{ 'role':'user','content' : x}],
        # temperature=0,
        # max_tokens=512,
        # top_p=1,
        # frequency_penalty=0,
        # presence_penalty=0,
        ))
collab_recommendations = collab_response.apply(lambda x: x.choices[0].message.content.split("[")[1].split("]")[0].split("\n"))
collab_recommendations.head()
test = pd.read_csv('D:\\test\\LLM-Recommender-System-with-RAG\\data\\test.csv')
user_movies_test = test.groupby('user_id').apply(lambda x: x['movie_title'].tolist())
collab_hits = pd.concat((user_movies, user_movies_test, collab_recommendations), axis=1, keys = ['user_movies', 'user_movies_test', 'collab_recommendations']).dropna(subset = 'collab_recommendations').apply(lambda x: len(set(x.iloc[0]).union(x.iloc[1]).intersection(x.iloc[2])) if x.iloc[1] is not np.nan else len(set(x.iloc[0]).intersection(x.iloc[2])), axis = 1)
# Prompt Similar to Zero-Shot Paper (reduced to one prompt, added genr# In[37]:
def genrePrompts(each, user_prompt_data, new_line = '\n'):
    return f"""
Candidate Set (candidate movies): {user_prompt_data.loc[each, 'candidate_movies']}.
The movies I have rated highly (watched movies): {user_prompt_data.loc[each, 'user_top_movies']}.
Their genres are: {user_prompt_data.loc[each, 'user_top_movie_genres']}.
Can you recommend 10 movies from the Candidate Set similar to but not in the selected movies I've watched?.
Please use brackets around the movies you recommend and separate the titles by new lines so I can easily parse them.
(Format Example: Here are the 10 movies recommended for you: [Midnight Cowboy (1969){new_line}Lost in Translation (2003){new_line}etc.])
Answer: 
"""
genre_prompts = pd.Series(user_prompt_data.index, index = user_prompt_data.index).apply(lambda x: genrePrompts(x, user_prompt_data))
genre_prompts.head()
genre_response = genre_prompts.iloc[:num_users].apply(lambda x: client.chat.completions.create(
    model="llama3-8b-8192",
        messages= [{ 'role':'user','content' : x}],
        # temperature=0,
        # max_tokens=512,
        # top_p=1,
        # frequency_penalty=0,
        # presence_penalty=0,
        ))
genre_recommendations = genre_response.apply(lambda x: x.choices[0].message.content.split("[")[-1].split("]")[0].split("\n"))
genre_recommendations.head()
genre_hits = pd.concat((user_movies, user_movies_test, genre_recommendations), axis=1, keys = ['user_movies', 'user_movies_test', 'genre_recommendations']).dropna(subset = 'genre_recommendations').apply(lambda x: len(set(x.iloc[0]).union(x.iloc[1]).intersection(x.iloc[2])) if x.iloc[1] is not np.nan else len(set(x.iloc[0]).intersection(x.iloc[2])), axis = 1)
def twoStepPrompt1(each, user_prompt_data):
    return f"""
Candidate Set (candidate movies): {user_prompt_data.loc[each, 'candidate_movies']}.
The movies I have rated highly (watched movies): {user_prompt_data.loc[each, 'user_top_movies']}.
Their genres are: {user_prompt_data.loc[each, 'user_top_movie_genres']}.
Step 1: What features are most important to me when selecting movies (Summarize my preferences briefly)? 
Answer: 
"""

def twoStepPrompt2(each, user_prompt_data, response1):
    return f"""
Candidate Set (candidate movies): {user_prompt_data.loc[each, 'candidate_movies']}.
The movies I have rated highly (watched movies): {user_prompt_data.loc[each, 'user_top_movies']}.
Their genres are: {user_prompt_data.loc[each, 'user_top_movie_genres']}.
Step 1: What features are most important to me when selecting movies (Summarize my preferences briefly)? 
Answer: {response1.loc[each]}.
Step 2: Can you recommend 10 movies from the Candidate Set similar to but not in the selected movies I've watched?
Please use brackets around the movies you recommend and separate the titles by new lines so I can easily parse them.
(Format Example: Here are the 10 movies recommended for you: [Midnight Cowboy (1969){new_line}Lost in Translation (2003){new_line}etc.])
Answer: 
"""
prompt1 = pd.Series(user_prompt_data.index, index = user_prompt_data.index).apply(lambda x: twoStepPrompt1(x, user_prompt_data))
response1 = prompt1.iloc[:num_users].apply(lambda x: client.chat.completions.create(
    model="llama3-8b-8192",
        messages= [{ 'role':'user','content' : x}],
        # temperature=0,
        # max_tokens=512,
        # top_p=1,
        # frequency_penalty=0,
        # presence_penalty=0,
        ))
response1.head()
prompt2 = pd.Series(user_prompt_data.index, index = user_prompt_data.index).iloc[:num_users].apply(lambda x: twoStepPrompt2(x, user_prompt_data, response1))
response2 = prompt2.apply(lambda x: client.chat.completions.create(
    model="llama3-8b-8192",
        messages= [{ 'role':'user','content' : x}],
        # temperature=0,
        # max_tokens=512,
        # top_p=1,
        # frequency_penalty=0,
        # presence_penalty=0,
        ))
twoStep_recommendations = response2.apply(lambda x: x.choices[0].message.content.split("[")[-1].split("]")[0].split("\n"))
twoStep_recommendations.head()
twoStep_hits = pd.concat((user_movies, user_movies_test, twoStep_recommendations), axis=1, keys = ['user_movies', 'user_movies_test', 'twoStep_recommendations']).dropna(subset = 'twoStep_recommendations').apply(lambda x: len(set(x.iloc[0]).union(x.iloc[1]).intersection(x.iloc[2])) if x.iloc[1] is not np.nan else len(set(x.iloc[0]).intersection(x.iloc[2])), axis = 1)
twoStep_hits.head()
movie_wiki = pd.read_csv('D:\\test\\LLM-Recommender-System-with-RAG\\data\\movie_wiki.csv')
movie_wiki.head()

def wikiPrompt(each, user_prompt_data, movie_wiki, new_line = '\n'):
    return f"""
Candidate Set (candidate movies): {user_prompt_data.loc[each, 'candidate_movies']}.
The movies I have rated highly (watched movies): {user_prompt_data.loc[each, 'user_top_movies']}.
Summary of the movies I have watched: {new_line.join([f"{eachMovie}: {movie_wiki.loc[movie_wiki['movie_title'] == eachMovie, 'wiki_summary'].iloc[0]}" for eachMovie in user_prompt_data.loc[1, 'user_top_movies']])}
Can you recommend 10 movies from the Candidate Set similar to but not in the selected movies I've watched?.
Please use brackets around the movies you recommend and separate the titles by new lines so I can easily parse them.
(Format Example: Here are the 10 movies recommended for you: [Midnight Cowboy (1969){new_line}Lost in Translation (2003){new_line}etc.])
Answer: 
"""
wiki_prompts = pd.Series(user_prompt_data.index, index = user_prompt_data.index).apply(lambda x: wikiPrompt(x, user_prompt_data, movie_wiki))
wiki_prompts.head()
wiki_response = wiki_prompts.iloc[:num_users].apply(lambda x: client.chat.completions.create(
    model="llama3-8b-8192",
        messages= [{ 'role':'user','content' : x}],
        # temperature=0,
        # max_tokens=512,
        # top_p=1,
        # frequency_penalty=0,
        # presence_penalty=0,
        ))
wiki_recommendations = wiki_response.apply(lambda x: x.choices[0].message.content.split("[")[-1].split("]")[0].split("\n"))
wiki_recommendations.head()
wiki_hits = pd.concat((user_movies, user_movies_test, wiki_recommendations), axis=1, keys = ['user_movies', 'user_movies_test', 'wiki_recommendations']).dropna(subset = 'wiki_recommendations').apply(lambda x: len(set(x.iloc[0]).union(x.iloc[1]).intersection(x.iloc[2])) if x.iloc[1] is not np.nan else len(set(x.iloc[0]).intersection(x.iloc[2])), axis = 1)
viewings = train.groupby('movie_title').count().sort_values('user_id', ascending=False)
viewings.head()
top_10_movies = viewings.head(10).index.tolist()
top_10_movies
baseline_hits = pd.concat((user_movies, user_movies_test), axis=1).dropna().iloc[:num_users].apply(lambda x: len(set(x.iloc[0]).union(x.iloc[1]).intersection(top_10_movies)), axis = 1)
comparison = pd.concat((collab_hits, genre_hits, twoStep_hits, wiki_hits, baseline_hits), axis = 1, keys = ['collab_hits', 'genre_hits', 'twoStep_hits', 'wiki_hits', 'baseline_hits'])
comparison.describe().iloc[1:] / 10 * 100

