#import the needed librairies
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import re
from sklearn.preprocessing import StandardScaler, LabelEncoder, QuantileTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder, QuantileTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge


"""
This fonction serves to clean the data for trainining and it will split it into train and test needed for further steps
"""
# Function to convert rating to classes
# preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler, LabelEncoder, QuantileTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

def clean_data(df):
    """
    Effectue un prétraitement complet des données de films.
    
    Args:
        df: DataFrame pandas contenant les données brutes des films
        
    Returns:
        DataFrame pandas prétraité
    """
    # Fonctions utilitaires internes
    
    def clean_text(text):
        print("DataFrame Columns:", df.columns)
        print(df.head())

        if pd.isna(text):
            return text
        text = str(text)
        text = re.sub(r'[^a-zA-Z0-9\s,.]', ' ', text)
        text = ' '.join(text.split())
        return text

    def clean_column_names(df):
        df = df.copy()
        df.columns = df.columns.str.strip().str.lower()
        return df

    def standardize_movie_genres(df):
        df['genre'] = df['genre'].fillna('unspecified')
        
        def normalize_genre(genre_string):
            if pd.isna(genre_string):
                return "unspecified"
            genre_string = str(genre_string).lower().strip()
            genres = [g.strip() for g in genre_string.split(',')]
            genres = sorted(set(genres))
            return ','.join(genres) if genres else "unspecified"
        
        df['genre'] = df['genre'].apply(normalize_genre)
        return df

    def clean_gross(df):
        df['gross'] = df['gross'].str.replace('M', '').astype(float)
        return df

    def impute_gross(df):
        df['gross'] = df['gross'].fillna(0)
        return df

    def fill_rating_by_genre(df):
        df_copy = df.copy()
        genre_median = df_copy.groupby('genre')['rating'].transform('median')
        df_copy['rating'] = df_copy['rating'].fillna(genre_median)
        global_median = df_copy['rating'].median()
        df_copy['rating'] = df_copy['rating'].fillna(global_median)
        return df_copy

    def clean_and_extract_years(year_str):
        if pd.isna(year_str) or not isinstance(year_str, str):
            return None, None
        
        year_str = year_str.strip('() ')
        
        try:
            if re.match(r'^[IVX]+\s*\(', year_str):
                year_str = re.sub(r'^[IVX]+\s*\((.+?)\)$', r'\1', year_str)
            
            base_year = re.search(r'(\d{4})', year_str)
            if not base_year:
                return None, None
            
            start_year = int(base_year.group(1))
            
            if year_str.endswith('– )') or year_str.endswith('-)'):
                return start_year, 2024
            
            if '–' in year_str or '-' in year_str:
                end_match = re.search(r'[–-]\s*(\d{4})', year_str)
                if end_match:
                    return start_year, int(end_match.group(1))
                return start_year, 2024
            
            if ' ' in year_str:
                years = year_str.split()
                if len(years) == 2 and years[0].isdigit() and years[1].isdigit():
                    return int(years[0]), int(years[1])
            
            return start_year, start_year
        
        except Exception as e:
            print(f"Erreur avec '{year_str}': {str(e)}")
            return None, None

    def get_content_type(year_str):
        if not isinstance(year_str, str):
            return 'Unknown'
        
        if 'TV Special' in year_str:
            return 'TV Special'
        elif 'TV Movie' in year_str:
            return 'TV Movie'
        elif 'TV Short' in year_str:
            return 'TV Short'
        elif 'Video Game' in year_str:
            return 'Video Game'
        elif 'Video' in year_str:
            return 'Video'
        elif '–' in year_str or '-' in year_str:
            return 'TV Series'
        else:
            return 'Movie'

    def process_df_years(df, year_column='year'):
        years_df = pd.DataFrame(df[year_column].apply(clean_and_extract_years).tolist(),
                              columns=['year_start', 'year_end'])
        df['year_start'] = years_df['year_start'].astype('Int64')
        df['year_end'] = years_df['year_end'].astype('Int64')
        df['content_type'] = df[year_column].apply(get_content_type)
        return df

    def fill_year_from_context(df):
        for idx in df[df['year_start'].isna()].index:
            text = str(df.loc[idx, 'one-line']) + str(df.loc[idx, 'stars']) + str(df.loc[idx, 'movies'])
            years = re.findall(r'\b(19|20)\d{2}\b', text)
            
            if years:
                found_year = int(years[0])
                df.loc[idx, 'year_start'] = found_year
        
        df['year_end'] = df['year_end'].fillna(df['year_start'])
        return df

    def impute_runtime(df):
        df['runtime_imputed'] = df.groupby('genre')['runtime'].transform(lambda x: x.fillna(x.median()))
        df['runtime_imputed'] = df['runtime_imputed'].fillna(df['runtime'].median())
        df['runtime'] = df['runtime_imputed']
        df.drop(columns=['runtime_imputed'], inplace=True)
        return df

    def clean_votes(df):
        df['votes'] = df['votes'].str.replace(',', '').astype(float)
        return df

    def impute_votes(df):
        features = ['year_start', 'runtime', 'rating']
        train_data = df[df['votes'].notna()]
        missing_data = df[df['votes'].isna()]
        model = RandomForestRegressor()
        model.fit(train_data[features], train_data['votes'])
        df.loc[df['votes'].isna(), 'votes'] = model.predict(missing_data[features])
        return df

    def extract_director_stars(row):
        text = str(row)
        director = 'Not Specified'
        stars = 'Not Specified'
        
        if 'Director' in text or 'DIRECTOR' in text or 'director' in text:
            for dir_keyword in ['Director', 'DIRECTOR', 'director']:
                if dir_keyword in text:
                    director_part = text.split(dir_keyword)[-1].split('Stars')[0]
                    director = director_part.strip()
                    break
        
        if 'Stars' in text or 'STARS' in text or 'stars' in text:
            for star_keyword in ['Stars', 'STARS', 'stars']:
                if star_keyword in text:
                    stars = text.split(star_keyword)[-1].strip()
                    break
        
        return pd.Series({'director': director, 'stars_only': stars})

    def scale_numeric_features(df):
        scaler = StandardScaler()
        numeric_cols = ['votes', 'gross']
        df_scaled = df.copy()
        df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        return df_scaled

    def codage_genre(df):
        genres = df['genre'].str.get_dummies(sep=',')
        df = pd.concat([df, genres], axis=1)
        df.drop(['genre'], axis=1, inplace=True)
        return df

    def calculate_movie_duration(df):
        df['movie_duration'] = df['year_end'] - df['year_start'] + 1
        df.drop(['year_start', 'year_end'], axis=1, inplace=True)
        return df

    def encode_content_duration(df):
        return pd.get_dummies(df, columns=['duration_category'], prefix='content')

    def encode_content_type(df):
        return pd.get_dummies(df, columns=['content_type'], prefix='content')

    def encode_high_cardinality(df):
        le = LabelEncoder()
        df['director'] = le.fit_transform(df['director'])
        df['stars_only'] = le.fit_transform(df['stars_only'])
        return df

    def text_to_value_supervised(df):
        tfidf = TfidfVectorizer(max_features=200, ngram_range=(1, 2), stop_words='english')
        X = tfidf.fit_transform(df['one-line'].fillna(''))
        model = Ridge(alpha=1.0)
        model.fit(X, df['rating'])
        df['one-line'] = model.predict(X)
        return df

    def encode_movie_titles_improved(df):
        def extract_advanced_features(title):
            title = str(title).lower()
            features = {
                'length': len(title),
                'word_count': len(title.split()),
                'has_number': any(c.isdigit() for c in title),
                'has_colon': ':' in title,
                'has_parentheses': '(' in title or ')' in title,
                'capital_ratio': sum(1 for c in title if c.isupper()) / len(title) if title else 0,
                'digit_ratio': sum(1 for c in title if c.isdigit()) / len(title) if title else 0,
                'processed_text': ' '.join(title.replace(':', ' ').replace('-', ' ').split())
            }
            
            genre_keywords = {
                'action': ['action', 'war', 'battle'],
                'drama': ['drama', 'life', 'story'],
                'comedy': ['comedy', 'funny', 'laugh'],
                'horror': ['horror', 'scary', 'night'],
                'romance': ['love', 'romance', 'heart']
            }
            
            for genre, keywords in genre_keywords.items():
                features[f'is_{genre}'] = any(keyword in title for keyword in keywords)
            
            return features

        processed_data = df['movies'].apply(extract_advanced_features)
        processed_titles = [d['processed_text'] for d in processed_data]

        tfidf = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 5),
            max_features=200,
            min_df=2,
            max_df=0.95
        )
        text_features = tfidf.fit_transform(processed_titles).toarray()

        numeric_features = []
        boolean_features = []
        for d in processed_data:
            numeric_feat = [
                d['length'],
                d['word_count'],
                d['capital_ratio'],
                d['digit_ratio']
            ]
            bool_feat = [
                int(d['has_number']),
                int(d['has_colon']),
                int(d['has_parentheses']),
                int(d['is_action']),
                int(d['is_drama']),
                int(d['is_comedy']),
                int(d['is_horror']),
                int(d['is_romance'])
            ]
            numeric_features.append(numeric_feat)
            boolean_features.append(bool_feat)

        numeric_features = np.array(numeric_features)
        boolean_features = np.array(boolean_features)

        qt = QuantileTransformer(output_distribution='normal')
        numeric_features_scaled = qt.fit_transform(numeric_features)

        X = np.hstack([text_features, numeric_features_scaled, boolean_features])

        model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42
        )

        model.fit(X, df['rating'])
        predictions = model.predict(X)

        qt_final = QuantileTransformer(output_distribution='uniform')
        predictions_calibrated = qt_final.fit_transform(predictions.reshape(-1, 1))

        min_rating = df['rating'].min()
        max_rating = df['rating'].max()
        df['movies'] = min_rating + predictions_calibrated * (max_rating - min_rating)

        return df

    # Application séquentielle de toutes les transformations
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(clean_text)
    
    df = clean_column_names(df)
    df = standardize_movie_genres(df)
    df = clean_gross(df)
    df = impute_gross(df)
    df = fill_rating_by_genre(df)
    df = process_df_years(df, year_column='year')
    df = fill_year_from_context(df)
    
    median_value = df['year_start'].median()
    df['year_start'] = df['year_start'].fillna(median_value)
    df['year_end'] = df['year_end'].fillna(df['year_start'])
    df = df.drop(['year'], axis=1)
    
    df = impute_runtime(df)
    df = clean_votes(df)
    df = impute_votes(df)
    
    df[['director', 'stars_only']] = df['stars'].apply(extract_director_stars)
    df['director'] = df['director'].str.replace('...', '').str.strip()
    df['stars_only'] = df['stars_only'].str.replace('...', '').str.strip()
    df = df.drop(['stars'], axis=1)
    
    df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')
    df['duration_category'] = pd.cut(
        df['runtime'],
        bins=[0, 30, 60, 90, 120, float('inf')],
        labels=['Court', 'Moyen', 'Standard', 'Long', 'Très long']
    )
    df = df.drop(['runtime'], axis=1)
    df = df.dropna(subset=['rating'])
    df = scale_numeric_features(df)
    df = codage_genre(df)
    df = calculate_movie_duration(df)
    df = encode_content_duration(df)
    df = encode_content_type(df)
    df = encode_high_cardinality(df)
    df = text_to_value_supervised(df)
    df = encode_movie_titles_improved(df)
    
    return df