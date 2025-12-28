# ui.py
import torch
from taipy.gui.builder import Page, part, layout, text, table, input, selector, button
from taipy.gui import Gui, notify, State

from ui.data_loader import MovieLens1MLoader
# from ui.model_loader import load_model

from models.deepfm import DeepFM
from models.dcnv3 import DCNv3
from models.mlp import MLP

# =====================
# Config
# =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RATINGS_PATH = "data_source/ml-1m/ratings.dat"
MOVIES_PATH = "data_source/ml-1m/movies.dat"

CHECKPOINTS = {
    "DeepFM": "./checkpoints/deepfm.pt",
    "DCNv3": "./checkpoints/deepfm_autofis.pt",
    "MLP": "./checkpoints/mlp.pt",
}



# =====================
# Load data
# =====================
loader = MovieLens1MLoader(
    ratings_path=RATINGS_PATH,
    movies_path=MOVIES_PATH,
)

ratings_df = loader.get_ratings()

num_users = ratings_df["user_id"].nunique()
num_items = ratings_df["movie_id"].nunique()

user_list = sorted(ratings_df["user_id"].unique().tolist())
movie_options = loader.get_movie_options()  # "id - title"




# =====================
# Load model
# =====================
# def load_model(model_name: str):
#     if model_name == "DeepFM":
#         model = DeepFM(
#             num_users=num_users + 1,
#             num_items=num_items + 1,
#             embed_dim=16,
#             hidden_layers=[64, 32],
#         )

#     model.load_state_dict(
#         torch.load(CHECKPOINTS[model_name], map_location=DEVICE)
#     )
#     model.to(DEVICE)
#     model.eval()
#     return model




# =====================
# UI State
# =====================
selected_user = user_list[0]
selected_movie = movie_options[0]   # "movie_id - title"

prediction = "—"

model_cache = {}

selected_model = "DeepFM"
model_options = ["DeepFM", "DCNv3", "MLP"]

selected_user = "{user_list[0]}"
user_options = "{user_list}"

selected_movie = "{movie_options[0]}"
movie_options = "{movie_options}"





# =====================
# Predict function
# =====================
def predict(state):
    global model_cache

    # if state.selected_model not in model_cache:
    #     model_cache[state.selected_model] = load_model(state.selected_model)

    model = model_cache[state.selected_model]

    # Parse movie_id from "id - title"
    movie_id = int(state.selected_movie.split(" - ")[0])
    user_id = int(state.selected_user)

    user_tensor = torch.tensor([user_id], device=DEVICE)
    movie_tensor = torch.tensor([movie_id], device=DEVICE)

    with torch.no_grad():
        logit = model(user_tensor, movie_tensor)
        prob = torch.sigmoid(logit).item()

    state.prediction = f"{prob:.4f}"





# =====================
# UI Layout
# =====================
with Page() as page:
    with layout("1 2 1"):
        with part():
            pass
        
        with part():
            text(value="# 🎬 MovieLens Recommender", mode="md")
            
            selector(label="Model", value="{selected_model}", lov="{model_options}", dropdown=True)
            
            part(class_name="spacer")
            selector(label="User", value="{selected_user}", lov="{user_options}", dropdown=True)

            text("## Movie")
            selector(label="Movie", value="{selected_movie}", lov="{movie_options}", dropdown=True)

            button("Predict", on_action=predict)

        with part():
            text("---")

            
Gui(page, css_file=r"ui\assets\style.css").run(
    title="Movie Recommendation",
    use_reloader=True,
    dark_mode=False
)
