import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
import json
import seaborn as sns

# Configuration de la page
st.set_page_config(page_title="Movie Rating Predictor", layout="wide")

# Titre de l'application
st.title("Movie Rating Predictor")
st.subheader("Prédiction de la note d'un film")

# Configuration de l'URL de l'API
API_URL = "https://backend-latest-prjc.onrender.com"  # Update this to your backend URL

# Liste des genres disponibles (triés par fréquence)
GENRES = [
    'drama', 'comedy', 'action', 'adventure', 'animation',
    'crime', 'documentary', 'thriller', 'mystery', 'romance',
    'fantasy', 'horror', 'family', 'reality tv', 'history',
    'biography', 'sci fi', 'short', 'music', 'sport',
    'game show', 'talk show', 'unspecified', 'musical',
    'war', 'western', 'news', 'film noir'
]

# Types de contenu
CONTENT_TYPES = ['Movie', 'TV Special', 'TV Movie', 'Video', 'TV Short', 'Video Game']

# Formulaire principal
with st.form("movie_form", clear_on_submit=False):
    col1, col2 = st.columns(2)
    
    with col1:
        movies = st.text_input("Titre du film", "The Matrix")
        
        year_col, type_col = st.columns(2)
        with year_col:
            year = st.text_input("Année (YYYY ou YYYY-YYYY)", "1999", 
                               help="Entrez une année simple (ex: 2021) ou une période (ex: 2021-2023)")
        with type_col:
            content_type = st.selectbox("Type de contenu", CONTENT_TYPES)
            
        # Combinaison année et type
        year_type = f"{year} {content_type}" if content_type else year
        st.info(f"Format final: {year_type}")
        
        runtime = st.number_input("Durée (minutes)", min_value=0, max_value=500, value=120)
        genre = st.multiselect(
            "Genre", 
            GENRES,
            ['drama']
        )
        
    with col2:
        one_line = st.text_area("Résumé du film", "A computer programmer discovers a mysterious world...")
        stars = st.text_area("Directeur et Acteurs", "Director: Lana Wachowski | Stars: Keanu Reeves, Laurence Fishburne")
        votes = st.number_input("Nombre de votes", min_value=0, value=1000)
        gross = st.number_input("Revenus (en millions $)", min_value=0.0, value=100.0)
        
    # Création du dictionnaire de données
    data = {
        "movies": movies,
        "one_line": one_line,
        "year": year_type,
        "runtime": runtime,
        "genre": ",".join(genre),
        "stars": stars,
        "votes": votes,
        "gross": f"{gross}M"
    }
    
    # Bouton de soumission
    submit = st.form_submit_button("Prédire la note")
    
    if submit:
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=data
            )
            if response.status_code == 200:
                result = response.json()
                
                st.subheader("Résultat de la prédiction")
                
                rating_mapping = {
                    0: "⭐ (0-4) Faible",
                    1: "⭐⭐ (4-6) Moyen",
                    2: "⭐⭐⭐ (6-8) Bon",
                    3: "⭐⭐⭐⭐ (8-10) Excellent"
                }
                
                st.success(f"Note prédite : {rating_mapping[result['predictions'][0]]}")
            else:
                st.error(f"Erreur API: {response.text}")
            
        except Exception as e:
            st.error(f"Une erreur s'est produite : {str(e)}")

# Section upload CSV
st.markdown("---")
st.subheader("Upload CSV pour prédictions en lot")

with st.expander("Format du fichier CSV requis"):
    st.write("""
    Votre fichier CSV doit contenir les colonnes suivantes:
    - movies: Titre du film
    - one_line: Résumé du film
    - year: Format 'YYYY' ou 'YYYY-YYYY' avec type de contenu
    - runtime: Durée en minutes
    - genre: Genres séparés par des virgules
    - stars: Format 'Director: Nom | Stars: Noms'
    - votes: Nombre de votes
    - gross: Format nombre ou 'XM' pour millions
    """)

    # Exemple de données
    example_data = {
        'movies': ['The Matrix', 'Inception'],
        'one_line': ['A computer hacker learns...', 'A thief who enters...'],
        'year': ['1999 Movie', '2010-2012 TV Series'],
        'runtime': [136, 148],
        'genre': ['action,sci fi', 'action,adventure'],
        'stars': ['Director: Wachowski | Stars: Keanu Reeves', 'Director: Nolan | Stars: DiCaprio'],
        'votes': [1500000, 2000000],
        'gross': ['463.5M', '836.8M']
    }
    st.dataframe(pd.DataFrame(example_data))

uploaded_file = st.file_uploader("Choisissez votre fichier CSV", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Aperçu des données chargées :")
        st.dataframe(df.head())
        
        if st.button("Lancer les prédictions", key="predict_batch"):
            with st.spinner('Prédiction en cours...'):
                try:
                    files = {"file": uploaded_file.getvalue()}
                    response = requests.post(
                        f"{API_URL}/predict/csv",
                        files=files
                    )
                    
                    if response.status_code == 200:
                        results = response.json()
                        
                        # Création DataFrame avec résultats
                        results_df = df.copy()
                        results_df['Predicted_Rating'] = results['predictions']
                        results_df['Rating_Category'] = results_df['Predicted_Rating'].map({
                            0: "Faible (0-4)",
                            1: "Moyen (4-6)",
                            2: "Bon (6-8)",
                            3: "Excellent (8-10)"
                        })
                        
                        # Onglets pour les résultats
                        tab1, tab2, tab3 = st.tabs(["Résultats", "Visualisations", "Export"])
                        
                        with tab1:
                            st.dataframe(results_df[['movies', 'year', 'Rating_Category']])
                        
                        with tab2:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig1, ax1 = plt.subplots(figsize=(10, 6))
                                results_df['Rating_Category'].value_counts().plot(kind='bar')
                                plt.title('Distribution des notes prédites')
                                plt.xticks(rotation=45)
                                st.pyplot(fig1)
                            
                            with col2:
                                results_df['decade'] = pd.to_numeric(
                                    results_df['year'].str.extract('(\d{4})')[0]
                                ).apply(lambda x: f"{str(x)[:3]}0s")
                                
                                fig2, ax2 = plt.subplots(figsize=(10, 6))
                                results_df.groupby('decade')['Predicted_Rating'].mean().plot(kind='line', marker='o')
                                plt.title('Évolution des notes par décennie')
                                plt.grid(True)
                                st.pyplot(fig2)
                        
                        with tab3:
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="Télécharger les résultats (CSV)",
                                data=csv,
                                file_name='predictions_films.csv',
                                mime='text/csv'
                            )
                    else:
                        st.error(f"Erreur API: {response.text}")
                        
                except Exception as e:
                    st.error(f"Erreur lors des prédictions : {str(e)}")
                    
    except Exception as e:
        st.error(f"Erreur lors du traitement du fichier : {str(e)}")

# Section À propos
with st.expander("À propos"):
    st.write("""
    Cette application permet de prédire la note d'un film en fonction de ses caractéristiques.
    Les prédictions sont basées sur un modèle entraîné sur un large ensemble de données de films.
    Les notes sont classées en 4 catégories :
    - 0-4 : Faible
    - 4-6 : Moyen
    - 6-8 : Bon
    - 8-10 : Excellent
    """)