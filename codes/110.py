import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("movies.csv")

df["genres"] = df["genres"].fillna("")

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(df["genres"])

similarity = cosine_similarity(tfidf_matrix)

def recommend(movie_name):

    index = df[
        df["title"].str.lower() == movie_name.lower()
    ].index[0]

    scores = list(enumerate(similarity[index]))

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    scores = scores[1:6]

    print("Recommended Movies:")

    for i, score in scores:
        print(df.iloc[i]["title"])

recommend("Toy Story")