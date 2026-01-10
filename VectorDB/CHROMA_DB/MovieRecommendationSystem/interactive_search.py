from shared_functions import *  # This now imports your updated movie functions

# Global variable to store loaded movie items
movie_items = []
def main():
    """Main function for interactive CLI movie recommendation system"""
    try:
        print("🎬  Interactive Movie Recommendation System")
        print("=" * 60)
        print("Loading movie database (TMDB ~1.3M movies)...")
        
        # Load and process movie data
        global movie_items
        movie_data = pd.read_csv('./Data/TMDB_movie_dataset_v11.csv')
        movie_items = load_movie_data(movie_data.head(50))
        
        print(f"Processed {len(movie_items):,} raw movie entries")
        
        # SMART FILTERING: Keep movies that are useful for recommendation
        filtered_movies = []
        for m in movie_items:
            # Must have a title
            if not m['title'] or m['title'] == "Untitled":
                continue
            # Prefer movies with overview, but allow if we have genres + keywords
            has_content = bool(m['overview']) or bool(m['genres']) or bool(m['keywords'])
            if not has_content:
                continue
            filtered_movies.append(m)
        
        movie_items = filtered_movies
        
        # Optional: Further reduce size for faster testing (remove when ready for full)
        # movie_items = movie_items[:100000]  # Uncomment for quick testing
        
        if len(movie_items) == 0:
            print("❌ No valid movies remained after filtering!")
            return
            
        print(f"✅ Loaded and processed {len(movie_items):,} valid movies successfully")
        
        # Create and populate ChromaDB collection
        collection = create_similarity_search_collection(
            "interactive_movie_search",
            {'description': 'A collection for interactive movie similarity search'}
        )
        populate_similarity_collection(collection, movie_items)
        print("✅ Search engine ready!")
        
        # Start interactive chatbot
        interactive_movie_chatbot(collection)
        
    except Exception as error:
        print(f"❌ Error initializing system: {error}")
        import traceback
        traceback.print_exc()  # This will show full error details


def interactive_movie_chatbot(collection):
    """Interactive CLI chatbot for movie recommendations"""
    print("\n" + "="*60)
    print("🤖 INTERACTIVE MOVIE RECOMMENDATION CHATBOT")
    print("="*60)
    print("Commands:")
    print("  • Type any movie idea, genre, mood, or description")
    print("  • Examples below!")
    print("  • 'help' - Show tips and examples")
    print("  • 'quit' or 'exit' - Exit the system")
    print("  • Ctrl+C - Emergency exit")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\n🎥 What kind of movie are you in the mood for? ").strip()
            
            if not user_input:
                print("   Please describe a movie, genre, or mood!")
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thank you for using the Movie Recommendation System!")
                print("   Enjoy your movie night! 🍿")
                break
            
            elif user_input.lower() in ['help', 'h']:
                show_help_menu()
            
            else:
                handle_movie_search(collection, user_input)
                
        except KeyboardInterrupt:
            print("\n\n👋 Session ended. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def show_help_menu():
    """Display helpful search tips and examples"""
    print("\n📖 SEARCH TIPS & EXAMPLES")
    print("-" * 40)
    print("Try describing what you want:")
    print("  • 'superhero movie with humor and action'")
    print("  • 'romantic comedy set in Paris'")
    print("  • 'scary horror movie about ghosts'")
    print("  • 'Bollywood drama with family and emotions'")
    print("  • 'animated movie for kids'")
    print("  • 'classic sci-fi from the 80s'")
    print("  • 'thriller with twists like Inception'")
    print("  • 'feel-good movie to relax'")
    print("\nYou can also filter by:")
    print("  • Genre: 'Action', 'Romance', 'Horror', 'Animation'")
    print("  • Country: 'India', 'Korea', 'France', 'Japan'")
    print("  • Language: 'Hindi', 'Korean', 'French'")
    print("  • Rating: 'highly rated', 'critically acclaimed'")
    print("  • Year: '90s movies', 'recent blockbusters'")

def handle_movie_search(collection, query):
    """Perform movie search and display beautiful results"""
    print(f"\n🔍 Searching for movies like: '{query}'")
    print("   Finding the best matches... Please wait...")
    
    # First try basic similarity search
    results = perform_similarity_search(collection, query, n_results=10)
    
    # If few results, try loosening filters or suggest refinements
    if len(results) < 3:
        print("   Not many direct matches. Trying broader search...")
        # You can enhance this later with keyword boosting
        results = perform_similarity_search(collection, query, n_results=15)
    
    if not results:
        print("❌ No movies found matching your description.")
        print("💡 Try broader terms like:")
        print("   • 'action movie'")
        print("   • 'romantic story'")
        print("   • 'funny comedy'")
        print("   • 'Indian family drama'")
        return
    
    print(f"\n✅ Here are {len(results)} movie recommendations for you:")
    print("=" * 80)
    
    for i, movie in enumerate(results, 1):
        score_percent = movie['similarity_score'] * 100
        year = movie['release_year'] or "Unknown"
        rating = movie['vote_average']
        lang = movie['original_language'].upper()
        countries = movie['country_names'] or "International"
        
        print(f"\n{i}. 🎬 {movie['title']} ({year})")
        print(f"   📊 Match Score: {score_percent:.1f}%  |  ⭐ Rating: {rating:.1f}/10")
        print(f"   🌍 Language: {lang}  |  🏳️ Countries: {countries}")
        print(f"   🎭 Genres: {movie['genres']}")
        
        if movie['overview']:
            # Truncate long overviews
            overview = movie['overview'][:300]
            if len(movie['overview']) > 300:
                overview += "..."
            print(f"   📖 Plot: {overview}")
        
        if i < len(results):
            print("   " + "-" * 70)
    
    print("=" * 80)
    
    # Suggest next searches
    suggest_related_searches(results)

def suggest_related_searches(results):
    """Suggest follow-up searches based on top results"""
    if not results:
        return
    
    print("\n💡 You might also like:")
    
    # Extract common genres
    all_genres = []
    for r in results[:5]:
        if r['genres']:
            all_genres.extend(r['genres'].split(", "))
    common_genres = list(set(all_genres))
    
    for genre in common_genres[:3]:
        print(f"   • '{genre} movies'")
    
    # Suggest by country/language
    countries = set()
    languages = set()
    for r in results[:5]:
        if r['country_names']:
            countries.add(r['country_names'].split(", ")[0])
        languages.add(r['original_language'].upper())
    
    if "IN" in [r['country_names'] for r in results if r['country_names']] or "hi" in [r['original_language'] for r in results]:
        print("   • 'Bollywood movies'")
    if "KO" in languages or "KR" in countries:
        print("   • 'Korean drama or thriller'")
    
    # Rating-based suggestion
    avg_rating = sum(r['vote_average'] for r in results) / len(results)
    if avg_rating > 7.5:
        print("   • 'award-winning classics'")
    else:
        print("   • 'highly rated hidden gems'")

if __name__ == "__main__":
    main()