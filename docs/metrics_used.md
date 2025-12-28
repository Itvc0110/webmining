# Evaluation Metrics

## Classification Metrics
Let $\{(y_i, \hat{p}_i)\}_{i=1}^{N}$ denote the ground-truth binary interaction labels and predicted probabilities on the test set, where $y_i \in \{0,1\}$ and $\hat{p}_i \in [0,1]$.

*   **LogLoss (Binary Cross-Entropy).**
$$
\mathrm{LogLoss} =
-\frac{1}{N}\sum_{i=1}^{N}
\left[
y_i \log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)
\right].
$$
    LogLoss measures how well predicted probabilities align with true interaction outcomes. Lower values indicate better calibrated predictions and are particularly important for CTR estimation tasks.

*   **Area Under the ROC Curve (AUC).**
$$
\mathrm{AUC} = \mathbb{P}\left(\hat{p}_{i^+} > \hat{p}_{i^-}\right),
$$
    where $i^+$ and $i^-$ denote positive and negative samples, respectively.  
    AUC evaluates the model’s ability to correctly rank positive interactions above negative ones, independent of a specific decision threshold. In recommender systems, a higher AUC implies stronger global discrimination between preferred and non-preferred items.

*   **Accuracy.**
$$
\mathrm{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}.
$$
    Accuracy measures the proportion of correctly classified interactions. While intuitive, it can be misleading under class imbalance and is therefore reported together with other metrics.

*   **Precision.**
$$
\mathrm{Precision} = \frac{TP}{TP + FP}.
$$
    Precision quantifies the reliability of positive predictions, indicating how many recommended or predicted interactions are truly relevant.

*   **Recall.**
$$
\mathrm{Recall} = \frac{TP}{TP + FN}.
$$
    Recall measures the model’s ability to detect relevant items among the entire item space. In recommendation settings, high recall indicates strong coverage of user-preferred items.

*   **F1-score.**
$$
\mathrm{F1} = \frac{2 \cdot \mathrm{Precision} \cdot \mathrm{Recall}}
{\mathrm{Precision} + \mathrm{Recall}}.
$$
    The F1-score balances precision and recall, providing a single measure that reflects the trade-off between recommendation accuracy and coverage.

*   **Precision--Recall AUC (PR-AUC).**
    PR-AUC summarizes the trade-off between precision and recall across different thresholds and is especially informative in imbalanced datasets, where negative interactions dominate.

*   **Matthews Correlation Coefficient (MCC).**
$$
\mathrm{MCC} =
\frac{TP \cdot TN - FP \cdot FN}
{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}.
$$
    MCC provides a balanced evaluation of binary classification performance, taking all four confusion matrix components into account and remaining robust under class imbalance.

## Ranking Metrics (Top-$K$).
Ranking metrics evaluate how effectively a model retrieves relevant items within the top-$K$ positions of a ranked recommendation list. For each user $u$, let $\mathcal{R}_u^K$ denote the ranked list of top-$K$ recommended items, and let $\mathcal{T}_u$ be the set of relevant (positive) test items.

*   **Precision@K.**
$$
\mathrm{Precision@K} =
\frac{1}{K}
\sum_{i \in \mathcal{R}_u^K} \mathbb{I}(i \in \mathcal{T}_u).
$$
    Precision@K measures the proportion of recommended items in the top-$K$ list that are relevant, reflecting recommendation accuracy at the presentation level.

*   **Recall@K.**
$$
\mathrm{Recall@K} =
\frac{|\mathcal{R}_u^K \cap \mathcal{T}_u|}
{|\mathcal{T}_u|}.
$$
    Recall@K evaluates how well the model recovers a user’s preferred items from the entire item space. Higher recall indicates stronger retrieval capability.

*   **Hit Rate@K.**
$$
\mathrm{Hit@K} =
\mathbb{I}\left(|\mathcal{R}_u^K \cap \mathcal{T}_u| > 0\right).
$$
    Hit Rate@K measures whether at least one relevant item appears in the top-$K$ recommendations, capturing a minimal success criterion for recommendation quality.

*   **Mean Reciprocal Rank (MRR@K).**
$$
\mathrm{MRR@K} =
\frac{1}{|\mathcal{U}|}
\sum_{u \in \mathcal{U}}
\frac{1}{\mathrm{rank}_u},
$$
    where $\mathrm{rank}_u$ is the position of the first relevant item in $\mathcal{R}_u^K$.  
    MRR emphasizes early ranking of relevant items, which is crucial for user satisfaction in real-world systems.

*   **Mean Average Precision (MAP@K).**
$$
\mathrm{AP@K}(u) =
\frac{1}{\min(|\mathcal{T}_u|,K)}
\sum_{k=1}^{K}
\mathrm{Precision@k} \cdot \mathbb{I}(i_k \in \mathcal{T}_u),
$$
$$
\mathrm{MAP@K} =
\frac{1}{|\mathcal{U}|}
\sum_{u \in \mathcal{U}} \mathrm{AP@K}(u).
$$
    MAP@K captures both ranking correctness and position sensitivity across all relevant items, providing a comprehensive ranking quality measure.

*   **Normalized Discounted Cumulative Gain (NDCG@K).**
$$
\mathrm{DCG@K} = \sum_{k=1}^{K} \frac{2^{\mathrm{rel}_k}-1}{\log_2(k+1)}, \quad
\mathrm{IDCG@K} = \sum_{k=1}^{\min(|\mathcal{T}_u|, K)} \frac{2^{\mathrm{rel}_k^{ideal}}-1}{\log_2(k+1)}
$$
$$
\mathrm{NDCG@K} = \frac{\mathrm{DCG@K}}{\mathrm{IDCG@K}}
$$
    NDCG@K accounts for both the relevance and rank position of recommended items, where $\mathrm{rel}_k^{ideal}$ represents the relevance of items in the perfect ranking (sorted by relevance in descending order).
