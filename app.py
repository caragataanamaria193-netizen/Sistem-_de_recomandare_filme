import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# --- CONFIGURARE PAGINĂ ---
st.set_page_config(
    page_title="Sistem de Recomandare Filme",
    page_icon="🎬",
    layout="wide"
)

# Stil CSS pentru estetică modernă
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div.stButton > button:first-child {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        border: none;
        height: 3em;
        width: 100%;
        font-weight: bold;
    }
    .genre-tag {
        display: inline-block;
        background-color: #262730;
        color: #ff4b4b;
        border-radius: 5px;
        padding: 2px 8px;
        margin-right: 5px;
        font-size: 0.8em;
        border: 1px solid #ff4b4b;
        margin-bottom: 5px;
    }
    .stSelectbox label, .stSlider label {
        font-weight: bold;
        color: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)

GENRES = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama",
          "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

DATASET_CANDIDATES = ["movie_dataset_final.csv", "movie_dataset_final (1).csv"]
MATRIX_CANDIDATES = ["user_movie_matrix.csv", "user_movie_matrix (2).csv"]


def first_existing(paths):
    for p in paths:
        if os.path.exists(p): return p
    return paths[0]


@st.cache_data(show_spinner=False)
def load_dataset(path):
    df = pd.read_csv(path)
    df["title"] = df["title"].astype(str)
    for g in GENRES:
        if g not in df.columns: df[g] = 0
    return df


@st.cache_data(show_spinner=False)
def load_matrix(path):
    um = pd.read_csv(path)
    return um.set_index("user_id")


def get_movie_genres(df, title):
    movie_row = df[df["title"] == title].iloc[0]
    return [g for g in GENRES if movie_row[g] == 1]


def display_genre_tags(genres):
    html = "".join([f'<span class="genre-tag">{g}</span>' for g in genres])
    st.markdown(html, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def build_models(df, um):
    # Model bazat pe conținut
    movies = df.drop_duplicates("title")[["title"] + GENRES].copy()
    G = movies[GENRES].fillna(0).values
    sim_content = cosine_similarity(G)
    titles_content = movies["title"].tolist()
    t2i_content = {t: i for i, t in enumerate(titles_content)}

    # Model colaborativ
    m = um.fillna(0).astype(float)
    titles_collab = m.columns.tolist()
    item_user = csr_matrix(m.T.values)
    knn = NearestNeighbors(metric="cosine", algorithm="brute").fit(item_user)
    t2i_collab = {t: i for i, t in enumerate(titles_collab)}

    return (titles_content, sim_content, t2i_content), (knn, item_user, titles_collab, t2i_collab)


def get_recs(title, n, method, model_cb, model_cf):
    titles_cb, sim_cb, t2i_cb = model_cb
    knn, item_user, titles_cf, t2i_cf = model_cf
    cb_list, cf_list = [], []

    if title in t2i_cb:
        idx = np.argsort(sim_cb[t2i_cb[title]])[::-1]
        cb_list = [titles_cb[j] for j in idx if titles_cb[j] != title]

    if title in t2i_cf:
        _, idx = knn.kneighbors(item_user[t2i_cf[title]], n_neighbors=n + 1)
        cf_list = [titles_cf[j] for j in idx.flatten() if titles_cf[j] != title]

    if "conținut" in method: return cb_list[:n]
    if "utilizatori" in method: return cf_list[:n]

    out, seen, i = [], {title}, 0
    while len(out) < n and (i < len(cb_list) or i < len(cf_list)):
        if i < len(cb_list) and cb_list[i] not in seen:
            out.append(cb_list[i]);
            seen.add(cb_list[i])
        if len(out) < n and i < len(cf_list) and cf_list[i] not in seen:
            out.append(cf_list[i]);
            seen.add(cf_list[i])
        i += 1
    return out


def hitrate_at_k(df, rec_fn, k=10, thr=4, max_users=100):
    rel = df[df["rating"] >= thr].groupby("user_id")["title"].apply(lambda s: list(pd.unique(s.astype(str))))
    users = [u for u, lst in rel.items() if len(lst) >= 2]
    if not users: return 0.0

    rng = np.random.default_rng(42)
    users = rng.choice(users, size=min(len(users), max_users), replace=False)
    hits = 0
    for u in users:
        liked = rel[u].copy()
        rng.shuffle(liked)
        seed, targets = liked[0], set(liked[1:])
        recs = set(rec_fn(seed, k))
        if recs & targets: hits += 1
    return hits / len(users)


# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configurare")
    with st.expander("📁 Gestionare Fișiere"):
        ds_path = st.text_input("Sursă Dataset", first_existing(DATASET_CANDIDATES))
        mx_path = st.text_input("Sursă Matrice", first_existing(MATRIX_CANDIDATES))

    st.subheader("🤖 Algoritm")
    selected_method = st.selectbox("Metodă utilizată:",
                                   ["Hibrid (Recomandat)",
                                    "Bazat pe conținut (Genuri)",
                                    "Bazat pe utilizatori similari"])
    n_recs = st.slider("Număr de recomandări afișate", 5, 20, 10)

    st.subheader("📊 Parametri Sistem")
    min_rating = st.select_slider("Prag de apreciere (Rating minim)", options=[1, 2, 3, 4, 5], value=4)
    # REDENUMIT DIN VALOAREA K
    k_eval = st.slider("Profunzimea analizei pentru test", 5, 25, 10)

# --- PANOU PRINCIPAL ---
st.title("🎬 Sistem de Recomandare Filme")

try:
    df = load_dataset(ds_path)
    um = load_matrix(mx_path)
    model_cb, model_cf = build_models(df, um)
    common_titles = sorted(list(set(df["title"].unique()) & set(um.columns)))

    c1, c2 = st.columns([3, 1])
    with c1:
        chosen_movie = st.selectbox("Alege un film care ți-a plăcut:", common_titles)
        display_genre_tags(get_movie_genres(df, chosen_movie))

    with c2:
        st.write("##")
        run_btn = st.button("Generează Recomandări")

    if run_btn:
        st.markdown("---")
        results = get_recs(chosen_movie, n_recs, selected_method, model_cb, model_cf)
        if results:
            st.subheader(f"✨ Recomandări relevante (Rating ≥ {min_rating})")
            for i, movie in enumerate(results, 1):
                st.markdown(f"*{i}. {movie}*")
                display_genre_tags(get_movie_genres(df, movie))
                st.write("")
        else:
            st.warning("Sistemul nu a găsit corelații suficiente.")

    st.markdown("---")
    with st.expander("📈 Evaluarea Sistemului (Acuratețe)"):
        st.write(f"Calculăm rata de succes a sistemului pentru filmele cu nota *{min_rating}+*.")
        if st.button("Calculează Rata de Succes"):
            with st.spinner("Se analizează baza de date..."):
                hr_val = hitrate_at_k(
                    df,
                    lambda t, k: get_recs(t, k, selected_method, model_cb, model_cf),
                    k=k_eval,
                    thr=min_rating
                )
                st.metric(label=f"Acuratețe la {k_eval} sugestii", value=f"{hr_val:.2%}")
                st.caption(f"Procentul în care sistemul a prezis corect un film pe care utilizatorul l-ar aprecia.")

except Exception as e:
    st.error(f"Eroare sistem: {e}")