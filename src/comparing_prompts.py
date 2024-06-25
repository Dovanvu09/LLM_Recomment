import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def prepare_data():
    train = pd.read_csv('D:\\test\\LLM-Recommender-System-with-RAG\\data\\train.csv')

    # Các bộ phim mà mỗi người dùng đã xem trong dữ liệu huấn luyện
    user_movies = train.groupby('user_id', group_keys=False).apply(lambda x: x['movie_title'].tolist())
    user_recent_movies = train.groupby('user_id', group_keys=False).apply(lambda x: x['movie_title'].tail(10).to_list())
    user_top_movies = train.groupby('user_id', group_keys=False).apply(lambda x: x.sort_values('rating', ascending=False).head(10)['movie_title'].to_list())
    movie_genres = train.groupby('movie_title', group_keys=False).apply(lambda x: x['genres'].unique().tolist()).apply(lambda x: x[0].split(", "))
    user_top_movie_genres = user_top_movies.apply(lambda x: pd.Series(movie_genres.loc[x].sum()).unique().tolist())

    user_ratings = train.drop_duplicates(['user_id', 'movie_title']).pivot(index='user_id', columns='movie_title', values='avg_rating').fillna(0)
    sparse_user_ratings = csr_matrix(user_ratings)

    def get_similar_users(sparse_user_ratings, index, n=10):
        return pd.DataFrame(cosine_similarity(sparse_user_ratings) - np.identity(sparse_user_ratings.shape[0]), index=index, columns=index).apply(lambda x: list(x.sort_values(ascending=False).head(n).index), axis=1)

    similar_users = get_similar_users(sparse_user_ratings, user_ratings.index)
    candidate_movies = similar_users.apply(lambda x: list(train.loc[train['user_id'].isin(x)].groupby('movie_title').sum().sort_values('rating', ascending=False).index[:20]))

    # Kiểm tra dữ liệu thiếu và xác định người dùng cần loại bỏ
    users_to_remove = set()

    for series in [user_movies, user_recent_movies, user_top_movies, user_top_movie_genres, similar_users, candidate_movies]:
        missing_users = series[series.apply(lambda x: len(x) == 0)].index
        users_to_remove.update(missing_users)

    # Loại bỏ người dùng có dữ liệu thiếu
    user_movies = user_movies.drop(index=users_to_remove)
    user_recent_movies = user_recent_movies.drop(index=users_to_remove)
    user_top_movies = user_top_movies.drop(index=users_to_remove)
    similar_users = similar_users.drop(index=users_to_remove)
    candidate_movies = candidate_movies.drop(index=users_to_remove)
    user_top_movie_genres = user_top_movie_genres.drop(index=users_to_remove)

    # Kết hợp tất cả dữ liệu vào một DataFrame duy nhất
    user_prompt_data = pd.concat([user_movies, user_top_movies, user_recent_movies, similar_users, candidate_movies, user_top_movie_genres], axis=1, keys=['user_movies', 'user_top_movies', 'user_recent_movies', 'similar_users', 'candidate_movies', 'user_top_movie_genres'])

    return user_prompt_data



def get_similar_users(sparse_user_ratings, index, n=10):
    return pd.DataFrame(cosine_similarity(sparse_user_ratings) - np.identity(sparse_user_ratings.shape[0]), index=index, columns=index).apply(lambda x: list(x.sort_values(ascending=False).head(n).index), axis=1)

# You can run this to prepare data and save it to a CSV for reuse
def collabPrompt(each, user_prompt_data, new_line='\n'):
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

def genrePrompts(each, user_prompt_data, new_line='\n'):
    return f"""
Candidate Set (candidate movies): {user_prompt_data.loc[each, 'candidate_movies']}.
The movies I have rated highly (watched movies): {user_prompt_data.loc[each, 'user_top_movies']}.
Their genres are: {user_prompt_data.loc[each, 'user_top_movie_genres']}.
Can you recommend 10 movies from the Candidate Set similar to but not in the selected movies I've watched?.
Please use brackets around the movies you recommend and separate the titles by new lines so I can easily parse them.
(Format Example: Here are the 10 movies recommended for you: [Midnight Cowboy (1969){new_line}Lost in Translation (2003){new_line}etc.])
Answer: 
"""

def twoStepPrompt1(each, user_prompt_data):
    return f"""
Candidate Set (candidate movies): {user_prompt_data.loc[each, 'candidate_movies']}.
The movies I have rated highly (watched movies): {user_prompt_data.loc[each, 'user_top_movies']}.
Their genres are: {user_prompt_data.loc[each, 'user_top_movie_genres']}.
Step 1: What features are most important to me when selecting movies (Summarize my preferences briefly)? 
Answer: 
"""

def twoStepPrompt2(each, user_prompt_data, response1,new_line = '\n'):
    return f"""
Candidate Set (candidate movies): {user_prompt_data.loc[each, 'candidate_movies']}.
The movies I have rated highly (watched movies): {user_prompt_data.loc[each, 'user_top_movies']}.
Their genres are: {user_prompt_data.loc[each, 'user_top_movie_genres']}.
Step 1: What features are most important to me when selecting movies (Summarize my preferences briefly)? 
Answer: {response1}.
Step 2: Can you recommend 10 movies from the Candidate Set similar to but not in the selected movies I've watched?
Please use brackets around the movies you recommend and separate the titles by new lines so I can easily parse them.
(Format Example: Here are the 10 movies recommended for you: [Midnight Cowboy (1969){new_line}Lost in Translation (2003){new_line}etc.])
Answer: 
"""

def wikiPrompt(each, user_prompt_data, movie_wiki, new_line='\n'):
    return f"""
Candidate Set (candidate movies): {user_prompt_data.loc[each, 'candidate_movies']}.
The movies I have rated highly (watched movies): {user_prompt_data.loc[each, 'user_top_movies']}.
Summary of the movies I have watched: {new_line.join([f"{eachMovie}: {movie_wiki.loc[movie_wiki['movie_title'] == eachMovie, 'wiki_summary'].iloc[0]}" for eachMovie in user_prompt_data.loc[each, 'user_top_movies']])}
Can you recommend 10 movies from the Candidate Set similar to but not in the selected movies I've watched?.
Please use brackets around the movies you recommend and separate the titles by new lines so I can easily parse them.
(Format Example: Here are the 10 movies recommended for you: [Midnight Cowboy (1969){new_line}Lost in Translation (2003){new_line}etc.])
Answer: 
"""

