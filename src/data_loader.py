import pandas as pd

class MovieLens1MLoader:
    """
    Load MovieLens 1M dataset:
    - ratings.dat
    - movies.dat

    ratings.dat format:
        UserID::MovieID::Rating::Timestamp

    movies.dat format:
        MovieID::Title (Year)::Genres
    """

    def __init__(
        self,
        ratings_path: str,
        movies_path: str,
        min_rating: float = 4.0,
    ):
        self.ratings_path = ratings_path
        self.movies_path = movies_path
        self.min_rating = min_rating

        self._load_movies()
        self._load_ratings()

    # =====================
    # Load movies
    # =====================
    def _load_movies(self):
        self.movies = pd.read_csv(
            self.movies_path,
            sep="::",
            engine="python",
            names=["movie_id", "title_year", "genres"],
            encoding="latin-1",
        )

        self.movies["year"] = self.movies["title_year"].str.extract(r"\((\d{4})\)")
        self.movies["title"] = self.movies["title_year"].str.replace(
            r"\s*\(\d{4}\)", "", regex=True
        )

        self.movies["movie_id"] = self.movies["movie_id"].astype(int)

        self.movie_id_to_title = dict(
            zip(self.movies.movie_id, self.movies.title)
        )

    # =====================
    # Load ratings
    # =====================
    def _load_ratings(self):
        ratings = pd.read_csv(
            self.ratings_path,
            sep="::",
            engine="python",
            names=["user_id", "movie_id", "rating", "timestamp"],
        )

        ratings["user_id"] = ratings["user_id"].astype(int)
        ratings["movie_id"] = ratings["movie_id"].astype(int)

        # Binary target (implicit feedback)
        ratings["label"] = (ratings["rating"] >= self.min_rating).astype(int)

        self.ratings = ratings[["user_id", "movie_id", "label"]]

    # =====================
    # Public APIs
    # =====================
    def get_ratings(self) -> pd.DataFrame:
        """Return DataFrame: user_id, movie_id, label"""
        return self.ratings.copy()

    def get_movies(self) -> pd.DataFrame:
        """Return DataFrame: movie_id, title, year, genres"""
        return self.movies[["movie_id", "title", "year", "genres"]].copy()

    def get_movie_title(self, movie_id: int) -> str:
        return self.movie_id_to_title.get(movie_id, "Unknown")

    def get_movie_options(self):
        """
        For UI dropdown:
        ['1 - Toy Story', '2 - Jumanji', ...]
        """
        return [
            f"{mid} - {self.movie_id_to_title.get(mid, 'Unknown')}"
            for mid in sorted(self.movie_id_to_title.keys())
        ]
