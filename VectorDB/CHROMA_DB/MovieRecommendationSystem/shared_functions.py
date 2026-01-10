# ------------------------------- Movies Recommendation ----------------------
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import ast
import numpy as np
from typing import List, Dict, Any, Optional
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs
import warnings
warnings.filterwarnings("ignore")


# Initialize ChromaDB client
client = chromadb.Client()

def parse_list_column(col_value, key='name'):
    if pd.isna(col_value) or col_value == '[]' or not isinstance(col_value, str):
        return []
    try:
        # Safe eval with literal_eval
        items = ast.literal_eval(col_value)
        if isinstance(items, list):
            return [item[key] for item in items if isinstance(item, dict) and key in item]
    except (ValueError, SyntaxError):
        # Some rows have broken JSON-like strings – skip gracefully
        return []
    return []

def load_movie_data(df: pd.DataFrame) -> List[Dict]:
    """Convert TMDB DataFrame to list of dictionaries with cleaned fields"""
    movies = []
    
    for _, row in df.iterrows():
        movie = {}
        
        movie['movie_id'] = str(row['id'])  # TMDB ID as string
        movie['title'] = row['title'] if pd.notna(row['title']) else "Untitled"
        movie['original_title'] = row['original_title'] if pd.notna(row['original_title']) else movie['title']
        movie['overview'] = row['overview'] if pd.notna(row['overview']) else ""
        movie['tagline'] = row['tagline'] if pd.notna(row['tagline']) else ""
        movie['vote_average'] = float(row['vote_average'])
        movie['vote_count'] = int(row['vote_count'])
        movie['release_date'] = row['release_date'] if pd.notna(row['release_date']) else ""
        movie['runtime'] = int(row['runtime']) if pd.notna(row['runtime']) else 0
        movie['original_language'] = row['original_language']
        movie['popularity'] = float(row['popularity'])
        movie['adult'] = row['adult']
        
        movie['genres'] = parse_list_column(row['genres'])
        movie['production_companies'] = parse_list_column(row['production_companies'])
        movie['production_countries'] = parse_list_column(row['production_countries'], key='iso_3166_1')  # e.g., 'US', 'IN'
        movie['country_names'] = parse_list_column(row['production_countries'], key='name')  # e.g., 'United States', 'India'
        movie['spoken_languages'] = parse_list_column(row['spoken_languages'], key='english_name')
        movie['keywords'] = parse_list_column(row['keywords'])
        
        movies.append(movie)
    
    print(f"Successfully processed {len(movies)} movies from TMDB dataset")
    return movies

def create_similarity_search_collection(collection_name: str = "movies_collection", 
                                       collection_metadata: dict = None):
    """Create ChromaDB collection with sentence transformer embeddings"""
    try:
        client.delete_collection(collection_name)  # Start fresh
    except:
        pass
    
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    return client.create_collection(
        name=collection_name,
        metadata=collection_metadata or {"hnsw:space": "cosine"},
        embedding_function=sentence_transformer_ef
    )

def populate_similarity_collection(collection, movie_items: List[Dict], batch_size: int = 5000):
    """Populate collection with movie data in safe batches"""
    total = len(movie_items)
    print(f"Adding {total:,} movies in batches of {batch_size}...")
    
    used_ids = set()
    
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = movie_items[start:end]
        
        documents = []
        metadatas = []
        ids = []
        
        for movie in batch:
            # --- Build rich embedding text ---
            text_parts = []
            text_parts.append(f"Title: {movie['title']}")
            if movie['original_title'] != movie['title']:
                text_parts.append(f"Original Title: {movie['original_title']}")
            if movie['overview']:
                text_parts.append(f"Plot: {movie['overview']}")
            if movie['tagline']:
                text_parts.append(f"Tagline: {movie['tagline']}")
            if movie['genres']:
                text_parts.append(f"Genres: {', '.join(movie['genres'])}")
            if movie['keywords']:
                text_parts.append(f"Keywords: {', '.join(movie['keywords'][:20])}")
            if movie['country_names']:
                text_parts.append(f"Countries: {', '.join(movie['country_names'])}")
            if movie['spoken_languages']:
                text_parts.append(f"Languages: {', '.join(movie['spoken_languages'])}")
            if movie['production_companies']:
                text_parts.append(f"Studios: {', '.join(movie['production_companies'][:5])}")
            text_parts.append(f"Release Year: {movie['release_date'][:4] if movie['release_date'] else 'Unknown'}")
            text_parts.append(f"Rating: {movie['vote_average']:.1f} ({movie['vote_count']} votes)")
            
            text = ". ".join(text_parts)
            
            # --- Unique ID ---
            base_id = movie['movie_id']
            unique_id = base_id
            counter = 1
            while unique_id in used_ids:
                unique_id = f"{base_id}_{counter}"
                counter += 1
            used_ids.add(unique_id)
            
            documents.append(text)
            ids.append(unique_id)
            
            # --- Safe metadata (all scalar types) ---
            metadatas.append({
                "movie_id": movie['movie_id'],
                "title": movie['title'],
                "original_title": movie['original_title'],
                "overview": movie['overview'][:1000] if movie['overview'] else "",
                "genres": ", ".join(movie['genres']) if movie['genres'] else "Unknown",
                "production_countries": ", ".join(movie['production_countries']) if movie['production_countries'] else "Unknown",
                "country_names": ", ".join(movie['country_names']) if movie['country_names'] else "Unknown",
                "original_language": movie['original_language'],
                "spoken_languages": ", ".join(movie['spoken_languages']) if movie['spoken_languages'] else "Unknown",
                "keywords": ", ".join(movie['keywords'][:20]) if movie['keywords'] else "None",
                "vote_average": float(movie['vote_average']),
                "vote_count": int(movie['vote_count']),
                "release_year": movie['release_date'][:4] if movie['release_date'] else "Unknown",
                "runtime": int(movie['runtime']),
                "popularity": float(movie['popularity']),
                "adult": bool(movie['adult'])
            })
        
        # Add this batch
        try:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"   ✓ Added batch {start//batch_size + 1}: {len(batch)} movies (total: {end}/{total})")
        except Exception as e:
            print(f"   ❌ Error adding batch {start//batch_size + 1}: {e}")
            raise
    
    print(f"Successfully added all {total:,} movies to the database!")

def perform_similarity_search(collection, query: str, n_results: int = 10) -> List[Dict]:
    """Basic similarity search based on query text"""
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    if not results['ids'][0]:
        return []
    
    formatted_results = []
    for i in range(len(results['ids'][0])):
        meta = results['metadatas'][0][i]
        similarity_score = 1 - results['distances'][0][i]
        
        formatted_results.append({
            'movie_id': meta['movie_id'],
            'title': meta['title'],
            'overview': meta['overview'],
            'genres': meta['genres'],
            'vote_average': meta['vote_average'],
            'release_year': meta['release_year'],
            'original_language': meta['original_language'],
            'country_names': meta['country_names'],
            'similarity_score': round(similarity_score, 4),
            'distance': results['distances'][0][i]
        })
    
    return formatted_results

def perform_filtered_similarity_search(collection, query: str,
                                      genre: str = None,
                                      country: str = None,
                                      language: str = None,
                                      min_rating: float = None,
                                      year_range: tuple = None,
                                      n_results: int = 10) -> List[Dict]:
    """Filtered search with robust string matching (case-insensitive partial)"""
    
    # Get more candidates to allow room for filtering
    broad_results = collection.query(
        query_texts=[query],
        n_results=n_results * 5,  # Increased buffer
        include=["metadatas", "distances"]
    )
    
    if not broad_results['ids'][0]:
        return []
    
    filtered_results = []
    candidates = broad_results['metadatas'][0]
    distances = broad_results['distances'][0]
    
    # Normalize inputs for matching
    genre_lower = genre.lower().strip() if genre else None
    country_lower = country.lower().strip() if country else None
    
    for i, meta in enumerate(candidates):
        genres_str = meta['genres'].lower()
        countries_str = meta['production_countries'].lower()
        country_names_str = meta['country_names'].lower()
        
        # Genre filter: partial match (e.g., "sci-fi" in "action, sci-fi, thriller")
        if genre_lower and genre_lower not in genres_str:
            continue
        
        # Country filter: match either code or name
        if country_lower:
            if country_lower not in countries_str and country_lower not in country_names_str:
                continue
        
        # Language filter
        if language and meta['original_language'] != language:
            continue
        
        # Rating filter
        if min_rating and meta['vote_average'] < min_rating:
            continue
        
        # Year filter
        if year_range:
            year = meta['release_year']
            if not year or year == "Unknown":
                continue
            try:
                year_int = int(year)
                start, end = year_range
                if start and year_int < int(start):
                    continue
                if end and year_int > int(end):
                    continue
            except:
                continue
        
        similarity_score = 1 - distances[i]
        
        filtered_results.append({
            'movie_id': meta['movie_id'],
            'title': meta['title'],
            'overview': meta['overview'],
            'genres': meta['genres'],
            'vote_average': meta['vote_average'],
            'release_year': meta['release_year'],
            'original_language': meta['original_language'],
            'country_names': meta['country_names'],
            'similarity_score': round(similarity_score, 4)
        })
        
        if len(filtered_results) >= n_results:
            break
    if len(filtered_results) == 0:
        return get_top_n_broad_results(broad_results, n_results)
    return filtered_results

def get_top_n_broad_results(broad_results, n: int = 10) -> List[Dict]:
    """Get top N results from broad_results, sorted by decreasing similarity (increasing distances)."""
    if not broad_results['ids'][0]:
        return []
    
    # Pair indices with distances, sort by increasing distance (best similarity first)
    indexed_distances = list(enumerate(broad_results['distances'][0]))
    indexed_distances.sort(key=lambda x: x[1])
    
    # Take min(n, available) top indices
    top_indices = [idx for idx, _ in indexed_distances[:n]]
    
    # Build formatted results matching your filtered output format
    metadatas = broad_results['metadatas'][0]
    top_results = []
    for i in top_indices:
        meta = metadatas[i]
        dist = broad_results['distances'][0][i]
        similarity_score = 1 - dist
        top_results.append({
            'movie_id': meta['movie_id'],
            'title': meta['title'],
            'overview': meta['overview'],
            'genres': meta['genres'],
            'vote_average': meta['vote_average'],
            'release_year': meta['release_year'],
            'original_language': meta['original_language'],
            'country_names': meta['country_names'],
            'similarity_score': round(similarity_score, 4)
        })
    
    return top_results
