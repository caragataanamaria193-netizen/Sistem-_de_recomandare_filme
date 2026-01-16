import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os


file_name = "movie_dataset_final_ana.csv"

if not os.path.exists(file_name):

    file_name = "movie_dataset_final.csv"

try:
    df = pd.read_csv(file_name)
    print(f"✅ Succes: Am încărcat {file_name}")
except FileNotFoundError:
    print(f"❌ Eroare: Nu am găsit fișierul {file_name} în folderul proiectului!")
    exit()


genre_cols = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]


movies_unique = df.drop_duplicates(subset=['title']).copy()


def extract_genres(row):
    """Transformă valorile 0/1 în text (numele genului)"""
    active = []
    for gen in genre_cols:

        if gen in row and (str(row[gen]).startswith('1')):
            active.append(gen)
    return " ".join(active) if active else "unknown"


movies_unique["metadata_genres"] = movies_unique.apply(extract_genres, axis=1)

tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w[\w-]+\b")
tfidf_matrix = tfidf.fit_transform(movies_unique["metadata_genres"])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

indices = pd.Series(movies_unique.index, index=movies_unique["title"]).drop_duplicates()


def recomanda_content_based(titlu_film, top_n=10):
    if titlu_film not in indices:
        return [f"Filmul '{titlu_film}' nu a fost găsit."]

    idx = indices[titlu_film]
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

    pos = movies_unique.index.get_loc(idx)

    sim_scores = list(enumerate(cosine_sim[pos]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:top_n + 1]

    movie_indices = [i[0] for i in sim_scores]
    return movies_unique['title'].iloc[movie_indices].tolist()


if __name__ == "__main__":
    print("\n--- Sistem de Căutare Filme (Content-Based) ---")

    nume_cautat = input("Introdu numele filmului (ex: Toy Story (1995)): ")

    rezultate = recomanda_content_based(nume_cautat)

    if isinstance(rezultate, list) and len(rezultate) > 0:
        if "nu a fost găsit" in rezultate[0]:
            print(f"\n❌ {rezultate[0]}")
            print("Sfat: Verifică dacă ai scris anul corect între paranteze.")
        else:
            print(f"\n✅ Filme similare cu '{nume_cautat}':")
            for i, film in enumerate(rezultate, 1):
                print(f"{i}. {film}")
    else:
        print("\nNu s-au găsit recomandări.")