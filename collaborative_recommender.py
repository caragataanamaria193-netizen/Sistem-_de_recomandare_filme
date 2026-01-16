import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import os

# --- 1. Încărcarea datelor ---
# Folosim fișierul generat de Studentul 1 (care conține user_id și rating-uri)
file_name = "movie_dataset.csv"

# Verificăm dacă fișierul există (pentru robustețe, similar cu Student 2)
if not os.path.exists(file_name):
    # Fallback pe fișierul Anei dacă cel standard nu există
    file_name = "movie_dataset_final.csv"

try:
    df = pd.read_csv(file_name)
    print(f"✅ [Colaborativ] Succes: Am încărcat {file_name}")
except FileNotFoundError:
    print(f"❌ [Colaborativ] Eroare: Nu am găsit fișierul {file_name}!")
    exit()

# --- 2. Curățarea datelor (Data Cleaning) ---
# Pentru ca sistemul colaborativ să fie bun, eliminăm filmele care au prea puține voturi.
# Dacă un film are doar 1 vot de 5 stele, nu e relevant statistic.

# Calculăm numărul de rating-uri pentru fiecare film
movie_stats = df.groupby('title')['rating'].agg('count')

# Păstrăm doar filmele care au mai mult de 30 de voturi (prag ajustabil)
popular_movies_titles = movie_stats[movie_stats > 30].index
df_filtered = df[df['title'].isin(popular_movies_titles)].copy()

print(
    f"Statistici: Din {df['title'].nunique()} filme, am păstrat {df_filtered['title'].nunique()} pentru analiză (cele cu >30 voturi).")

# --- 3. Crearea Matricei User-Item (PIVOT TABLE) ---
# Transformăm tabelul lung într-o matrice unde:
# Rânduri = Filme (Titluri)
# Coloane = User ID
# Valori = Rating
movie_user_matrix = df_filtered.pivot_table(index='title', columns='user_id', values='rating').fillna(0)

# Transformăm în matrice rară (CSR) pentru eficiență (scikit-learn lucrează mai rapid așa)
movie_user_matrix_sparse = csr_matrix(movie_user_matrix.values)

# --- 4. Construirea Modelului (Nearest Neighbors) ---
# Folosim 'cosine similarity' pentru a calcula distanța dintre vectorii filmelor
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
model_knn.fit(movie_user_matrix_sparse)


# --- 5. Funcția de Recomandare Colaborativă ---
def recomanda_collaborative(titlu_film, top_n=10):
    """
    Recomandă filme bazat pe tiparul de votare al utilizatorilor.
    """
    # Verificăm dacă filmul există în matricea noastră filtrată
    if titlu_film not in movie_user_matrix.index:
        return [f"Filmul '{titlu_film}' nu are suficiente date pentru recomandări colaborative sau nu există."]

    # Obținem indexul filmului în matrice
    query_index = movie_user_matrix.index.get_loc(titlu_film)

    # Întrebăm modelul care sunt vecinii cei mai apropiați
    # reshape(1, -1) e necesar pentru că prezicem pentru un singur item
    distances, indices = model_knn.kneighbors(movie_user_matrix.iloc[query_index, :].values.reshape(1, -1),
                                              n_neighbors=top_n + 1)

    recomandari = []
    # Iterăm prin rezultate (primul rezultat e chiar filmul căutat, deci îl sărim)
    for i in range(1, len(distances.flatten())):
        idx = indices.flatten()[i]
        # Putem accesa și distanța (cât de similar e), dar aici returnăm doar titlul
        film_recomandat = movie_user_matrix.index[idx]
        recomandari.append(film_recomandat)

    return recomandari


# --- Zona de testare (Se rulează doar dacă pornești acest fișier direct) ---
if __name__ == "__main__":
    print("\n--- Sistem de Recomandare Colaborativ (User-Patterns) ---")

    test_movie = "Star Wars (1977)"  # Un exemplu clasic care sigur are voturi
    print(f"Căutăm recomandări pentru: {test_movie}")

    rezultate = recomanda_collaborative(test_movie)

    for i, film in enumerate(rezultate, 1):
        # Afișăm doar dacă nu e mesaj de eroare
        if "nu are suficiente date" in film:
            print(film)
        else:
            print(f"{i}. {film}")