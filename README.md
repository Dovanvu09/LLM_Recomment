
# LLM Recommender System
This repository uses ChatGPT via the Groq API to create a movie recommender system and compares the recommender's performance across four prompt templates.

To run the code in this repository using Groq, first pip install -r requirements.txt, then get a Groq API Key and include it in a python file called `key.py`. The file only needs one line stating:
```python
GROQ_API_KEY = "YOUR_KEY_HERE"
This will allow all of the other files to import your API key.

To run all four prompts and compare performance, click "Run All" on the comparing_prompts.ipynb Jupyter Notebook.

The recommender's performance was compared using the MovieLens 100k dataset from the below paper. It can be found here: https://grouplens.org/datasets/movielens/

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets:
History and Context. ACM Transactions on Interactive Intelligent
Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages.
DOI=http://dx.doi.org/10.1145/2827872

The four prompt templates used are:

A collaborative filter type prompt:

I am user {User ID}.
The most recent ten movies I have seen are:
{List of ten recent movies}.
My top rated movies are:
{List of ten top rated movies}.
The users who are most like me are {10 user id's of similar users}.
The top movies for each of these users are:
{Similar User ID: List of ten top rated movies}.
Please recommend ten movies for me to watch that I have not seen. Provide brackets around your recommendations so I can easily parse them.
For example ([Midnight Cowboy (1969)\nLost in Translation (2003)\netc.])
A prompt that provides a candidate set and the genres of a user's top rated movies:

Candidate Set (candidate movies): {List of candidate movies}.
The movies I have rated highly (watched movies): {List of ten top rated movies}.
Their genres are: {List of genres from the ten top rated movies}.
Can you recommend 10 movies from the Candidate Set similar to but not in the selected movies I've watched?.
Please use brackets around the movies you recommend and separate the titles by new lines so I can easily parse them.
(Format Example: Here are the 10 movies recommended for you: [Midnight Cowboy (1969)\nLost in Translation (2003)\netc.])
Answer:
A two-step prompt, which is a slightly modified version of the prompt provided in the paper: Wang, L., & Lim, E.-P. (2023). Zero-Shot Next-Item Recommendation using Large Pretrained Language Models. arXiv [Cs.IR]. Retrieved from http://arxiv.org/abs/2304.03153
Step 1:


Candidate Set (candidate movies): {List of Candidate movies}.
The movies I have rated highly (watched movies): {List of ten top rated movies}.
Their genres are: {List of genres from the ten top rated movies}.
Step 1: What features are most important to me when selecting movies (Summarize my preferences briefly)? 
Answer:
Step 2:


Candidate Set (candidate movies): {List of Candidate movies}.
The movies I have rated highly (watched movies): {List of ten top rated movies}.
Their genres are: {List of genres from the ten top rated movies}.
Step 1: What features are most important to me when selecting movies (Summarize my preferences briefly)? 
Answer: {Response from Step 1}.
Step 2: Can you recommend 10 movies from the Candidate Set similar to but not in the selected movies I've watched?
Please use brackets around the movies you recommend and separate the titles by new lines so I can easily parse them.
(Format Example: Here are the 10 movies recommended for you: [Midnight Cowboy (1969)\nLost in Translation (2003)\netc.])
Answer:
A prompt that includes a 1 sentence summary of the Wikipedia page for the movie:

Candidate Set (candidate movies): {List of Candidate movies}.
The movies I have rated highly (watched movies): {List of ten top rated movies}.
Summary of the movies I have watched: {Each Movie: 1 sentence summary of the wikipedia page for the movie}
Can you recommend 10 movies from the Candidate Set similar to but not in the selected movies I've watched?.
Please use brackets around the movies you recommend and separate the titles by new lines so I can easily parse them.
(Format Example: Here are the 10 movies recommended for you: [Midnight Cowboy (1969)\nLost in Translation (2003)\netc.])
Answer:
Additionally, all four prompts were compared against a baseline model, which recommended the top ten most popular movies from the training set to every user.

The comparative performance for the prompts is shown in the below table:

Hit Rate	Collab Prompt	Genre Prompt	Two Step Prompt	Wiki Prompt	Baseline
mean	2.500000 %	82.000000 %	72.500000 %	76.000000 %	65.000000 %
std	5.501196 %	17.947291 %	31.601965 %	26.437613 %	20.900768 %
min	0.000000 %	30.000000 %	0.000000 %	0.000000 %	20.000000 %
max	20.000000 %	100.000000 %	100.000000 %	100.000000 %	100.00000 %
Note: candidate sets are selected from top rated movies by similar users; similar users are selected by cosine similarity of a vector of a user's ratings; hit rate is the proportion of recommended movies that the user watched.