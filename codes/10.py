import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("movies.csv")

df["genres"] = df["genres"].fillna("")

vectorizer = TfidfVectorizer()
genre_matrix = vectorizer.fit_transform(df["genres"])

similarity = cosine_similarity(genre_matrix)

movie_name = "Toy Story"

index = df[df["title"] == movie_name].index[0]

scores = list(enumerate(similarity[index]))

scores = sorted(scores, key=lambda x: x[1], reverse=True)

print("Movies recommended for", movie_name, ":")

for i in scores[1:6]:
    print(df.iloc[i[0]]["title"])