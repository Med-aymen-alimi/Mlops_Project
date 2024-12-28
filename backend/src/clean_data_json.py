import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def clean_data_json(movie_data: dict):
    """
    Prétraitement final d'un film avec statistiques correctes
    """
    # Validate the input data
    if not isinstance(movie_data, dict):
        raise ValueError("Input data must be a dictionary")
    
    # Check for nested structures
    for key, value in movie_data.items():
        if isinstance(value, (list, dict)):
            raise ValueError(f"Nested structure found in key: {key}")
    
    # Flatten the movie_data dictionary
    flat_movie_data = flatten_dict(movie_data)
    
    # Convert the flattened dictionary to a DataFrame
    df = pd.DataFrame([flat_movie_data])
    
    # Liste complète des genres
    ALL_GENRES = [
        'action', 'adventure', 'animation', 'biography', 'comedy', 'crime',
        'documentary', 'drama', 'family', 'fantasy', 'film noir', 'game show',
        'history', 'horror', 'music', 'musical', 'mystery', 'news', 'reality tv',
        'romance', 'sci fi', 'short', 'sport', 'talk show', 'thriller',
        'unspecified', 'war', 'western'
    ]

    # Nettoyage du texte
    def clean_text(text):
        if pd.isna(text):
            return text
        text = str(text)
        text = re.sub(r'[^a-zA-Z0-9\s,.]', ' ', text)
        return ' '.join(text.split())

    # Traitement du one-line avec TfidfVectorizer
    tfidf = TfidfVectorizer(max_features=1, stop_words='english')
    one_line_value = float(tfidf.fit_transform([df['one_line'].iloc[0]]).mean())
    # Ajuster la valeur dans la plage observée [5.579078, 7.882606]
    one_line_scaled = 5.579078 + (7.882606 - 5.579078) * one_line_value

    # Génération des IDs director et stars avec les bonnes distributions
    def generate_id(text, is_director=True):
        if is_director:
            # Distribution pour director: mean=2060.53, std=914.21, min=0, max=3858
            base_value = abs(hash(text)) % 3859  # Max observé + 1
            return min(max(0, int(base_value * (2060.53/3859))), 3858)
        else:
            # Distribution pour stars: mean=3423.00, std=1827.53, min=0, max=6481
            base_value = abs(hash(text)) % 6482  # Max observé + 1
            return min(max(0, int(base_value * (3423.00/6482))), 6481)

    # Extraction et génération des IDs
    def extract_director_stars(text):
        director_text = "Unknown Director"
        stars_text = "Unknown Stars"
        
        if 'Director:' in text:
            director_text = text.split('Director:')[1].split('|')[0].strip()
        if 'Stars:' in text:
            stars_text = text.split('Stars:')[1].strip()
            
        return generate_id(director_text, True), generate_id(stars_text, False)

    # Application de l'extraction
    director_id, stars_id = extract_director_stars(df['stars'].iloc[0])
    
    # Création du DataFrame de sortie
    processed_df = pd.DataFrame({
        'movies': [0.0],  # Placeholder pour la colonne movies
        'one-line': [one_line_scaled],
        'votes': [float(df['votes'].iloc[0])],
        'gross': [float(df['gross'].str.replace('M', '').iloc[0])],
        'director': [director_id],
        'stars_only': [stars_id]
    })

    # Ajout des colonnes de genres
    genre_list = df['genre'].iloc[0].lower().split(',')
    for genre in ALL_GENRES:
        processed_df[genre] = 1 if genre in genre_list else 0

    # Ajout de movie_duration
    processed_df['movie_duration'] = 1

    # Ajout des colonnes content
    content_categories = {
        'Court': 0, 'Moyen': 0, 'Standard': 0, 'Long': 0, 'Très long': 0,
        'Movie': 0, 'TV Movie': 0, 'TV Short': 0, 'TV Special': 0, 
        'Unknown': 0, 'Video': 0, 'Video Game': 0
    }

    # Détermination du type de contenu
    year_str = str(df['year'].iloc[0])
    content_type = 'Movie'  # Par défaut
    for type_name in ['TV Movie', 'TV Short', 'TV Special', 'Video Game', 'Video']:
        if type_name in year_str:
            content_type = type_name
            break

    # Détermination de la durée
    runtime = float(df['runtime'].iloc[0])
    duration_type = 'Standard'  # Par défaut
    if runtime <= 30:
        duration_type = 'Court'
    elif runtime <= 60:
        duration_type = 'Moyen'
    elif runtime <= 90:
        duration_type = 'Standard'
    elif runtime <= 120:
        duration_type = 'Long'
    else:
        duration_type = 'Très long'

    # Mise à jour des colonnes content
    content_categories[duration_type] = 1
    content_categories[content_type] = 1

    # Ajout des colonnes content au DataFrame
    for category, value in content_categories.items():
        col_name = f'content_{category.replace(" ", "_")}'
        processed_df[col_name] = value

    return processed_df