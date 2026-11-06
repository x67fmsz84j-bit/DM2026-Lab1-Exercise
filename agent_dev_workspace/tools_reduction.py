from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool

def make_tools(session):
    """
    Factory function that creates and returns dimensionality reduction and label binarization tools
    bound to the given session.
    """

    @tool
    def reduce_dimensions_tool(
        method: str = "pca",
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        random_state: int = 42
    ) -> dict:
        """
        Reduces the dimensionality of the document-term matrix to 2D using PCA, t-SNE, or UMAP,
        and generates a scatter plot colored by category.

        Args:
            method: Dimensionality reduction method ('pca', 'tsne', or 'umap'). Default is 'pca'.
            perplexity: Perplexity parameter for t-SNE. Default is 30.0.
            n_neighbors: Number of neighbors parameter for UMAP. Default is 15.
            random_state: Random seed for reproducibility. Default is 42.

        Returns:
            Dict containing result_id, status, method, n_components, n_samples, settings,
            and explained_variance_ratio (for PCA).
        """
        if session.feature_matrix is None:
            return {"status": "error", "message": "No feature matrix found. Please run build_dtm_tool first."}

        # Check matrix size before converting sparse matrix to dense array
        n_rows, n_cols = session.feature_matrix.shape
        estimated_elements = n_rows * n_cols
        # Guard against extremely large dense arrays (e.g. > 100 million elements / ~800MB)
        if estimated_elements > 100_000_000:
            return {
                "status": "error",
                "message": f"Feature matrix is too large to convert to dense ({n_rows}x{n_cols} = {estimated_elements} elements)."
            }

        X = session.feature_matrix.toarray() if hasattr(session.feature_matrix, "toarray") else session.feature_matrix

        method_lower = method.lower()
        explained_variance_ratio = None

        all_settings = {
            "method": method_lower,
            "n_components": 2,
            "perplexity": perplexity,
            "n_neighbors": n_neighbors,
            "random_state": random_state,
        }

        if method_lower == "pca":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=random_state)
            coords = reducer.fit_transform(X)
            explained_variance_ratio = reducer.explained_variance_ratio_.tolist()
        elif method_lower == "tsne":
            from sklearn.manifold import TSNE
            eff_perplexity = min(perplexity, max(1.0, float(X.shape[0] - 1)))
            reducer = TSNE(n_components=2, perplexity=eff_perplexity, random_state=random_state)
            coords = reducer.fit_transform(X)
            all_settings["effective_perplexity"] = eff_perplexity
        elif method_lower == "umap":
            import umap
            eff_neighbors = min(n_neighbors, max(2, X.shape[0] - 1))
            reducer = umap.UMAP(n_components=2, n_neighbors=eff_neighbors, random_state=random_state)
            coords = reducer.fit_transform(X)
            all_settings["effective_neighbors"] = eff_neighbors
        else:
            return {"status": "error", "message": f"Unsupported method '{method}'. Use 'pca', 'tsne', or 'umap'."}

        # Store reduced coordinates in session.artifacts
        session.artifacts["reduced_coordinates"] = coords
        session.artifacts[f"reduced_coords_{method_lower}"] = coords

        # Generate 2D scatter plot colored by category as required by contract
        fig, ax = plt.subplots(figsize=(8, 6))
        labels = session.labels
        if labels is None and session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].values

        if labels is not None and len(labels) == coords.shape[0]:
            sns.scatterplot(
                x=coords[:, 0],
                y=coords[:, 1],
                hue=labels,
                style=labels,
                s=100,
                ax=ax
            )
        else:
            sns.scatterplot(x=coords[:, 0], y=coords[:, 1], s=100, ax=ax)

        ax.set_title(f"Dimensionality Reduction ({method_lower.upper()})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        plt.tight_layout()

        # Store figure in session.pending_figure
        session.pending_figure = fig

        result_id = session.next_result_id("reduction")

        summary = {
            "result_id": result_id,
            "status": "success",
            "method": method_lower,
            "n_components": 2,
            "n_samples": coords.shape[0],
            "settings": all_settings,
        }
        if explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = explained_variance_ratio

        session.store_result(
            "reduce_dimensions_tool",
            {"method": method, "random_state": random_state},
            summary,
            full_report=pd.DataFrame(coords, columns=["dim_1", "dim_2"])
        )

        return summary

    @tool
    def binarize_labels_tool() -> dict:
        """
        One-hot encodes the category labels for all documents in the active dataset using pd.get_dummies,
        storing the resulting binary indicator matrix in session.artifacts['binarized_labels'].

        Returns:
            Dict containing result_id, list of categories, n_samples, and n_categories.
        """
        if session.labels is None and (session.dataframe is None or "category_name" not in session.dataframe.columns):
            return {"status": "error", "message": "No document labels found in session state. Load a dataset first."}

        labels = session.labels if session.labels is not None else session.dataframe["category_name"].values

        # Use pd.get_dummies to convert category labels into one-hot indicator columns (dtype int)
        binarized_df = pd.get_dummies(labels, dtype=int)
        categories = list(binarized_df.columns)

        # Store in session.artifacts
        session.artifacts["binarized_labels"] = binarized_df.values
        session.artifacts["binarized_categories"] = categories

        result_id = session.next_result_id("binarize_labels")

        summary = {
            "result_id": result_id,
            "status": "success",
            "n_samples": len(labels),
            "n_categories": len(categories),
            "categories": categories,
        }

        session.store_result(
            "binarize_labels_tool",
            {},
            summary,
            full_report=binarized_df
        )

        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
