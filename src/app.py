# src/app.py
import torch
import numpy as np
from taipy.gui import Gui, State, notify
from taipy.gui.builder import Page, layout, part, text, selector, table, button
import re

from src.data_utils import load_dataset, get_all_watched_movies
from src.model_loader import load_model_dcnv3
from src.infer import dcnv3_predict
from src.fetch_movie_info import fetch_movie_info

# =====================
# Init
# =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_PREDICT = 2

dataset = load_dataset()
model, device = load_model_dcnv3(
    dataset,
    r"src\checkpoints\dcnv3.pth",
    device=DEVICE
)

# =====================
# User list (user_idx)
# =====================
user_ids = sorted(np.unique(dataset.user_ids).tolist())

# =====================
# UI State initial values
# =====================
selected_user = user_ids[40]
liked_movies = get_all_watched_movies(dataset, selected_user)
recommendations = []

# =====================
# Utils
# =====================
def split_title_year(title: str):
    """
    "Toy Story (1995)" -> ("Toy Story", 1995)
    """
    m = re.search(r"\((\d{4})\)$", title)
    if m:
        year = int(m.group(1))
        clean = title[:m.start()].strip()
        return clean, year
    return title, None


# =====================
# Callbacks
# =====================
def on_user_change(state: State):
    user_id = int(state.selected_user)
    state.selected_user = user_id
    state.liked_movies = get_all_watched_movies(dataset, user_id)
    state.recommendations = []


def on_predict(state: State):
    user_id = int(state.selected_user)

    scores = dcnv3_predict(
        model=model,
        dataset=dataset,
        user_id=user_id,
        top_k=NUM_PREDICT
    )

    if not scores:
        state.recommendations = []
        notify(state, "warning", "No recommendation.")
        return

    recs = []
    for mid, score in scores.items():
        raw_title = dataset.movie_titles.get(int(mid), "Unknown")
        title, year = split_title_year(raw_title)

        movie_info = fetch_movie_info(title, year)  # dict với poster, overview, vote
        poster_url = movie_info["poster_url"]
        overview = movie_info["overview"]
        user_score = movie_info["user_score"]

        recs.append({
            "title": title,
            "year": year,
            "score": round(float(score), 4),
            "poster_url": poster_url,
            "overview": overview,
            "user_score": user_score
        })

    print(recs)
    for r in recs:
        print(r["title"])

    state.recommendations = recs
    notify(state, "success", "Prediction done!")



# =====================
# UI Layout
# =====================
with Page() as page:
    with part(class_name="topbar"):
        text(value="The Movie Recommendation System", class_name="topbar-text")

    with layout("4 3 1 16 8"):
        with part():
            pass

        with part(class_name="selector-box"):
            selector(
                label="Select User (user_idx)",
                value="{selected_user}",
                lov=user_ids,
                dropdown=True,
                on_change=on_user_change
            )
            
        with part():
            pass
        
        with part():
            text("### Movies user watched", mode="md")
            table("{liked_movies}", height="260px")

            button("Recommend Movies", on_action=on_predict)

            # Chỉ render khi recommendations có dữ liệu
            with part(render="{recommendations and len(recommendations) > 0}"):
                for idx in range(NUM_PREDICT):
                    movie = f"recommendations[{idx}]"
                    with part(class_name="predict"):
                        with layout("1 3"):
                            text(
                                value=f"![Poster]({{{movie}['poster_url']}})",
                                class_name="predict-image",
                                mode="md"
                            )
                            with part(class_name="predict-info"):
                                text(value=f"{{{movie}['title']}}", class_name="predict-title")
                                text(value=f"{{{movie}['year']}}", class_name="predict-year")
                                text(value=f"Score: {{{movie}['score']}}", class_name="predict-score")
                                text(value=f"TMDB Rating: {{{movie}['user_score'] }} / 10", class_name="predict-vote")
                                text(value=f"{{{movie}['overview']}}", class_name="predict-overview", mode="md")
        with part():
            pass

# =====================
# Run
# =====================
Gui(page, css_file=r"assets\style.css").run(
    title="MoviRecommendation Demo",
    port=5001,
    dark_mode=False
)
