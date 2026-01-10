from shared_functions import *  # Your updated movie functions
import pandas as pd
import chromadb
import os

# --------------------------- PERSISTENT CHROMA SETUP ---------------------------
DB_PATH = "./chroma_movie_db"                    # Folder where DB will be saved
COLLECTION_NAME = "advanced_movie_search"

def get_or_create_collection():
    """Create or load persistent Chroma collection with embedding function"""
    # Use PersistentClient to save/load from disk
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # Define embedding function (must be the same every time!)
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    try:
        # Try to get existing collection
        collection = client.get_collection(name=COLLECTION_NAME)
        print(f"Loaded existing collection with {collection.count()} movies")
        return collection
    except:
        # Collection doesn't exist → create new one
        print("No existing collection found. Building new database...")
        collection = client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        return collection

def build_database_if_needed(collection):
    """Only populate if collection is empty"""
    if collection.count() > 0:
        print("Database already populated. Ready to search!")
        return
    
    print("Processing movies and generating embeddings (this may take a few minutes)...")
    
    # Load data
    movie_data = pd.read_csv('./Data/TMDB_movie_dataset_v11.csv')
    movie_items = load_movie_data(movie_data.head(20000))  # Increase as needed: 10k-50k works well
    
    # Smart filtering
    filtered_movies = []
    for m in movie_items:
        if not m['title'] or m['title'] == "Untitled":
            continue
        has_content = bool(m['overview']) or bool(m['genres']) or bool(m['keywords'])
        if has_content:
            filtered_movies.append(m)
    
    movie_items = filtered_movies
    
    if len(movie_items) == 0:
        print("No valid movies to add!")
        return
    
    print(f"Adding {len(movie_items):,} movies to persistent database...")
    populate_similarity_collection(collection, movie_items)
    
    print(f"Database built and saved to '{DB_PATH}'!")
    print(f"Total movies in DB: {collection.count()}")
    print("Next runs will start instantly!\n")

# --------------------------- MAIN ---------------------------
def main():
    """Main function for advanced movie search demonstrations"""
    try:
        print("Advanced Movie Recommendation System")
        print("=" * 60)
        
        # Get or create persistent collection
        collection = get_or_create_collection()
        
        # Build database only if empty
        build_database_if_needed(collection)
        
        print("Search engine ready!\n")
        
        # Start interactive interface
        interactive_advanced_movie_search(collection)
        
    except Exception as error:
        print(f"Error initializing system: {error}")
        import traceback
        traceback.print_exc()

# --------------------------- REST OF YOUR CODE (UNCHANGED) ---------------------------
def interactive_advanced_movie_search(collection):
    """Highly interactive advanced movie search menu"""
    print("="*60)
    print("ADVANCED MOVIE SEARCH WITH FILTERS")
    print("="*60)
    print("Choose a search mode:")
    print("  1. Basic Similarity Search")
    print("  2. Genre-Filtered Search")
    print("  3. Country/Region Search")
    print("  4. Language Search")
    print("  5. High-Rated Movies Only")
    print("  6. Year Range Search")
    print("  7. Combined Advanced Filters")
    print("  8. Live Recommendations (Free Chat)")
    print("  9. Demo Mode (See Examples)")
    print(" 10. Help & Tips")
    print(" 11. Exit")
    print("-" * 60)
    
    while True:
        try:
            choice = input("\nSelect option (1-11): ").strip()
            
            if choice == '1':
                perform_basic_movie_search(collection)
            elif choice == '2':
                perform_genre_filtered_search(collection)
            elif choice == '3':
                perform_country_filtered_search(collection)
            elif choice == '4':
                perform_language_filtered_search(collection)
            elif choice == '5':
                perform_high_rated_search(collection)
            elif choice == '6':
                perform_year_filtered_search(collection)
            elif choice == '7':
                perform_combined_advanced_search(collection)
            elif choice == '8':
                perform_live_chat_mode(collection)
            elif choice == '9':
                run_movie_demonstrations(collection)
            elif choice == '10':
                show_movie_help()
            elif choice == '11':
                print("\nThank you for using the Movie Recommendation System!")
                print("   Enjoy your next movie night!")
                break
            else:
                print("Invalid choice. Please enter 1-11.")
                
        except KeyboardInterrupt:
            print("\n\nSession ended. Goodbye!")
            break


# Just make sure these are included at the end:
def perform_basic_movie_search(collection):
    print("\nBASIC SIMILARITY SEARCH")
    print("-" * 35)
    query = input("Describe the movie you're looking for: ").strip()
    if not query:
        print("Please enter a description!")
        return
    
    print(f"\nSearching for movies like: '{query}'...")
    results = perform_similarity_search(collection, query, n_results=5)
    display_movie_results(results, "Basic Search Results")

def perform_genre_filtered_search(collection):
    print("\nGENRE-FILTERED SEARCH")
    print("-" * 35)
    
    popular_genres = ["Action", "Comedy", "Drama", "Romance", "Thriller", 
                      "Horror", "Sci-Fi", "Animation", "Adventure", "Fantasy", 
                      "Crime", "Mystery", "Documentary", "Family"]
    
    print("Popular Genres:")
    for i, genre in enumerate(popular_genres, 1):
        print(f"  {i:2}. {genre}")
    
    query = input("\nWhat kind of movie? (e.g., 'funny adventure'): ").strip()
    genre_input = input("Choose genre key (e.g., '0' for Action Movies): ").strip()
    
    if not query:
        print("Please describe a movie!")
        return
    
    genre = None
    if genre_input.isdigit():
        idx = int(genre_input) - 1
        if 0 <= idx < len(popular_genres):
            genre = popular_genres[idx]
    else:
        # Match partial name
        for g in popular_genres:
            if genre_input.lower() in g.lower() or g.lower() in genre_input.lower():
                genre = g
                break
    
    if not genre:
        print("Invalid genre. Using search without genre filter.")
    
    print(f"\nSearching for '{query}' in {genre or 'any'} genre...")
    results = perform_filtered_similarity_search(
        collection, query, genre=genre, n_results=5
    )
    display_movie_results(results, f"Genre: {genre or 'All Genres'}")

def perform_country_filtered_search(collection):
    print("\nCOUNTRY/REGION SEARCH")
    print("-" * 35)
    
    countries = ["United States", "India", "South Korea", "Japan", "France", 
                 "United Kingdom", "Germany", "Italy", "Spain", "China"]
    codes = ["US", "IN", "KR", "JP", "FR", "GB", "DE", "IT", "ES", "CN"]
    
    print("Popular Regions:")
    for i, (name, code) in enumerate(zip(countries, codes), 1):
        print(f"  {i}. {name} ({code})")
    
    query = input("\nMovie mood/description: ").strip()
    country_input = input("Choose country (number/name/code): ").strip().upper()
    
    if not query:
        return
    
    country_code = None
    if country_input.isdigit():
        idx = int(country_input) - 1
        if 0 <= idx < len(codes):
            country_code = codes[idx]
    else:
        country_code = country_input
    
    country_name = dict(zip(codes, countries)).get(country_code, country_code)
    
    print(f"\nSearching in {country_name} cinema...")
    results = perform_filtered_similarity_search(
        collection, query, country=country_code, n_results=10
    )
    display_movie_results(results, f"Region: {country_name}")

def perform_language_filtered_search(collection):
    print("\nLANGUAGE SEARCH")
    print("-" * 35)
    langs = ["English (en)", "Hindi (hi)", "Korean (ko)", "Japanese (ja)", 
             "French (fr)", "Spanish (es)", "German (de)", "Tamil (ta)"]
    codes = ["en", "hi", "ko", "ja", "fr", "es", "de", "ta"]
    
    for i, lang in enumerate(langs, 1):
        print(f"  {i}. {lang}")
    
    query = input("\nMovie description: ").strip()
    lang_input = input("Choose language (number or code): ").strip().lower()
    
    if not query:
        return
    
    lang_code = None
    if lang_input.isdigit():
        idx = int(lang_input) - 1
        if 0 <= idx < len(codes):
            lang_code = codes[idx]
    else:
        lang_code = lang_input
    
    lang_name = dict(zip(codes, langs)).get(lang_code, lang_code.upper())
    
    results = perform_filtered_similarity_search(
        collection, query, language=lang_code, n_results=10
    )
    display_movie_results(results, f"Language: {lang_name}")

def perform_high_rated_search(collection):
    print("\nHIGH-RATED MOVIES SEARCH")
    print("-" * 35)
    query = input("What are you in the mood for? ").strip()
    min_rating = input("Minimum rating (e.g., 7.5, default 7.0): ").strip()
    
    rating = 7.0
    if min_rating.replace('.', '').isdigit():
        rating = float(min_rating)
    
    print(f"\nFinding highly rated movies (>= {rating})...")
    results = perform_filtered_similarity_search(
        collection, query, min_rating=rating, n_results=10
    )
    display_movie_results(results, f"Highly Rated (≥ {rating}/10)")

def perform_year_filtered_search(collection):
    print("\nYEAR RANGE SEARCH")
    print("-" * 35)
    query = input("Movie type/description: ").strip()
    start = input("From year (e.g., 2010, or blank): ").strip()
    end = input("To year (e.g., 2024, or blank): ").strip()
    
    year_range = None
    if start or end:
        year_range = (start or None, end or None)
    
    results = perform_filtered_similarity_search(
        collection, query, year_range=year_range, n_results=10
    )
    period = f"{start}-{end}" if start and end else "Custom Period"
    display_movie_results(results, f"Year Range: {period}")

def perform_combined_advanced_search(collection):
    print("\nCOMBINED ADVANCED FILTERS")
    print("-" * 35)
    
    query = input("Describe your ideal movie: ").strip()
    if not query:
        return
    
    genre = input("Genre (optional): ").strip()
    country = input("Country code (e.g., IN, US): ").strip().upper() or None
    lang = input("Language code (e.g., hi, en): ").strip().lower() or None
    rating_input = input("Min rating (e.g., 8.0): ").strip()
    year_start = input("From year: ").strip() or None
    year_end = input("To year: ").strip() or None
    
    min_rating = float(rating_input) if rating_input.replace('.', '').isdigit() else None
    year_range = (year_start, year_end) if year_start or year_end else None
    
    filters = []
    if genre: filters.append(f"Genre: {genre}")
    if country: filters.append(f"Country: {country}")
    if lang: filters.append(f"Lang: {lang}")
    if min_rating: filters.append(f"Rating ≥ {min_rating}")
    if year_range: filters.append(f"Years: {year_start or '...'}–{year_end or '...'}")
    
    filter_text = ", ".join(filters) if filters else "No filters"
    
    print(f"\nSearching with: {filter_text}...")
    
    results = perform_filtered_similarity_search(
        collection, query,
        genre=genre or None,
        country=country,
        language=lang,
        min_rating=min_rating,
        year_range=year_range,
        n_results=10
    )
    display_movie_results(results, "Advanced Combined Results")

def perform_live_chat_mode(collection):
    print("\nLIVE RECOMMENDATION CHAT")
    print("Type 'back' to return to menu\n")
    while True:
        query = input("What movie mood are you in? ").strip()
        if query.lower() in ['back', 'exit', 'menu']:
            break
        if not query:
            continue
        
        results = perform_similarity_search(collection, query, n_results=6)
        display_movie_results(results, f"Recommendations for: '{query}'", compact=True)
        print("\n" + "-"*60)

def run_movie_demonstrations(collection):
    print("\nDEMONSTRATION MODE")
    print("=" * 50)
    
    demos = [
        ("Bollywood Romance", "emotional family love story", "Romance", "IN"),
        ("Korean Thriller", "dark suspense psychological", "Thriller", "KR"),
        ("Animated Family Fun", "fun adventure for kids", "Animation", None, 7.0),
        ("Classic 90s Action", "explosive hero saves world", "Action", "US", None, (1990, 1999)),
        ("Award-Worthy Drama", "deep emotional character study", "Drama", None, 8.0),
    ]
    
    for i, (title, query, genre, country, rating, years) in enumerate(demos, 1):
        print(f"\n{i}. {title}")
        print(f"   → '{query}'")
        
        results = perform_filtered_similarity_search(
            collection, query, genre=genre, country=country,
            min_rating=rating, year_range=years, n_results=5
        )
        display_movie_results(results, title, compact=True)
        
        input("\nPress Enter for next demo...")
    
    print("\nDemo complete!")

def display_movie_results(results, title, compact=False):
    print(f"\n{title.upper()}")
    print("=" * 70)
    
    if not results:
        print("No movies found. Try broadening your search!")
        return
    
    for i, m in enumerate(results, 1):
        score = m['similarity_score'] * 100
        year = m['release_year'] or "?"
        rating = m['vote_average']
        genres = m['genres'][:50] + "..." if len(m['genres']) > 50 else m['genres']
        
        if compact:
            print(f"   {i}. {m['title']} ({year}) • {rating:.1f} stars • {score:.0f}% match")
        else:
            print(f"\n{i}. {m['title']} ({year})")
            print(f"   Match: {score:.1f}%  |  Rating: {rating:.1f}/10  |  {m['country_names'] or 'Global'}")
            print(f"   Genres: {genres}")
            overview = m['overview'][:200] + "..." if len(m['overview']) > 200 else m['overview']
            print(f"   {overview}")
    
    print("=" * 70)

def show_movie_help():
    print("\nSEARCH TIPS & EXAMPLES")
    print("=" * 50)
    print("• Be descriptive: 'funny heist with friends'")
    print("• Try moods: 'feel good', 'dark and twisted', 'epic adventure'")
    print("• Regional: 'Bollywood family drama', 'Korean revenge thriller'")
    print("• Era: '80s action', 'recent animated'")
    print("• Combine: 'French romantic comedy'")
    print("\nPopular queries:")
    print("  • 'superhero team movie'")
    print("  • 'emotional Indian family story'")
    print("  • 'mind-bending sci-fi'")
    print("  • 'cute animated kids movie'")
    print("  • 'intense courtroom drama'")

if __name__ == "__main__":
    main()