"""This function takes a list or pandas series of movies and creates a prompt"""


def createPrompt(candidateMovies, watchedMovies):
    return f"Candidate Set (candidate movies): {', '.join(candidateMovies)}\n\nThe movies I have watched (watched movies): {', '.join(watchedMovies)}\n\nStep 1: What features are most important to me when selecting movies (Summarize my preferences briefly)?"
