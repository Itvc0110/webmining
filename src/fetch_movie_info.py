import requests

from src.config import TMDB_API_KEY

TMDB_API_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w200"

def fetch_movie_info(title: str, year: int = None) -> dict:
    """
    Search TMDB for movie info by title (and optional year)
    Returns dict with keys: poster_url, overview, user_score
    """
    params = {
        "api_key": TMDB_API_KEY,
        "query": title,
        "include_adult": False,
    }
    if year:
        params["year"] = year

    try:
        resp = requests.get(TMDB_API_URL, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if results:
            movie = results[0]
            poster_path = movie.get("poster_path")
            poster_url = TMDB_IMAGE_BASE + poster_path if poster_path else "https://via.placeholder.com/200x300?text=No+Image"
            overview = movie.get("overview", "No internet, Overview is not available!")
            user_score = movie.get("vote_average", 0.0)  # điểm từ 0 -> 10
            return {
                "poster_url": poster_url,
                "overview": overview,
                "user_score": user_score
            }
    except Exception as e:
        print("TMDB fetch error:", e)

    return {
        "poster_url": "https://via.placeholder.com/200x300?text=No+Image",
        "overview": "No overview available",
        "user_score": 0.0
    }