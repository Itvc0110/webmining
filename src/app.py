# src/app.py
import torch
import numpy as np
from taipy.gui import Gui, State, notify
from taipy.gui.builder import Page, layout, part, text, selector, table, button

from src.data_utils import load_dataset, get_all_watched_movies
from src.model_loader import load_model_dcnv3
from src.infer import dcnv3_predict


# =====================
# Init
# =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
selected_user = user_ids[0]
liked_movies = get_all_watched_movies(dataset, selected_user)
recommendations = ""

# =====================
# Callbacks
# =====================
def on_user_change(state: State):
    state.selected_user = int(state.selected_user)
    state.liked_movies = get_all_watched_movies(dataset, state.selected_user)
    state.recommendations = ""


def on_predict(state: State):
    user_id = int(state.selected_user)
    
    scores = dcnv3_predict(
        model=model,
        dataset=dataset,
        user_id=user_id,
        top_k=5
    )
    print("Scores:", scores)
    if not scores:
        state.recommendations = "_No recommendation._"
        return

    lines = []
    for rank, (mid, score) in enumerate(scores.items(), start=1):
        title = dataset.movie_titles.get(int(mid), "Unknown")
        lines.append(
            f"**{rank}. {title}**  \nScore: `{score:.4f}`"
        )

    state.recommendations = "\n\n".join(lines)

    notify(state, "success", "Prediction done!")


# def fetch_poster(movie_id):
    

# =====================
# UI Layout
# =====================
with Page() as page:
    with layout("1 3 1"):
        with part():
            pass
        text("# 🎬 DCNv3 Movie Recommender", mode="md")
        with part():
            pass

    with layout("1 1 6 2"):
        with part():
            pass
        
        selector(
                label="Select User (user_idx)",
                value="{selected_user}",
                lov=user_ids,
                dropdown=True,
                on_change=on_user_change
            )
        
        with part():
            text("### Movies user watched", mode="md")
            table("{liked_movies}", height="260px")

            button("🔮 Predict Top-5", on_action=on_predict)

            text("## 🚀 Recommended movies", mode="md")
            text("{recommendations}")

        with part():
            pass


# =====================
# Run
# =====================
Gui(page).run(
    title="DCNv3 Recommendation Demo",
    port=5001,
    dark_mode=False
)
