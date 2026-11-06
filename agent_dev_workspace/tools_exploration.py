import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool

def make_tools(session):
    """
    Factory function that creates and returns exploration tools bound to the given session.
    """

    @tool
    def cosine_similarity_tool(
        doc_index_1: int,
        doc_index_2: int,
    ) -> dict:
        """
        Calculates the cosine similarity between two documents in the active dataset using
        the fitted CountVectorizer stored in session.artifacts['count_vectorizer'].

        Args:
            doc_index_1: Index of the first document (0-indexed integer).
            doc_index_2: Index of the second document (0-indexed integer).

        Returns:
            Dict containing result_id, doc_index_1, doc_index_2, cosine_similarity score,
            and document text previews.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"status": "error", "message": "No active dataset found in session state. Load a dataset first."}

        df = session.dataframe
        n_docs = len(df)

        if doc_index_1 < 0 or doc_index_1 >= n_docs:
            return {"status": "error", "message": f"doc_index_1 ({doc_index_1}) out of range (0 to {n_docs - 1})."}

        if doc_index_2 < 0 or doc_index_2 >= n_docs:
            return {"status": "error", "message": f"doc_index_2 ({doc_index_2}) out of range (0 to {n_docs - 1})."}

        # Extract text for both documents, treating NaN/None/missing values as empty string ""
        val1 = df.iloc[doc_index_1]["text"]
        val2 = df.iloc[doc_index_2]["text"]

        text1 = "" if (pd.isna(val1) or val1 is None) else str(val1)
        text2 = "" if (pd.isna(val2) or val2 is None) else str(val2)

        vectorizer = session.artifacts.get("count_vectorizer")
        if vectorizer is None:
            # Fall back to session.feature_matrix row vectors if count_vectorizer artifact is missing
            if session.feature_matrix is None:
                return {"status": "error", "message": "No count_vectorizer or feature_matrix found in session state. Run build_dtm_tool first."}
            vec1 = session.feature_matrix[doc_index_1].toarray().ravel()
            vec2 = session.feature_matrix[doc_index_2].toarray().ravel()
        else:
            # Transform extracted text using fitted vectorizer
            vec1 = vectorizer.transform([text1]).toarray().ravel()
            vec2 = vectorizer.transform([text2]).toarray().ravel()

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        # Handle zero vector case (e.g. empty text document)
        if norm1 == 0 or norm2 == 0:
            similarity = 0.0
        else:
            similarity = float(np.dot(vec1, vec2) / (norm1 * norm2))

        result_id = session.next_result_id("cosine_similarity")

        summary = {
            "result_id": result_id,
            "status": "success",
            "doc_index_1": doc_index_1,
            "doc_index_2": doc_index_2,
            "cosine_similarity": round(similarity, 4),
            "doc_1_preview": text1[:100],
            "doc_2_preview": text2[:100],
        }

        session.store_result(
            "cosine_similarity_tool",
            {"doc_index_1": doc_index_1, "doc_index_2": doc_index_2},
            summary
        )

        return summary

    @tool
    def feature_correlation_matrix_tool() -> dict:
        """
        Computes the Pearson correlation matrix for the top 20 terms by variance in the global DTM,
        generating an annotated heatmap figure.

        Returns:
            Dict containing result_id, top feature names, and correlation matrix values.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No Document-Term Matrix (DTM) found in session state. Run build_dtm_tool first."}

        matrix = session.feature_matrix
        feature_names = session.feature_names

        # Compute column variances across sparse feature matrix: var = E[X^2] - (E[X])^2
        mean = np.asarray(matrix.mean(axis=0)).ravel()
        mean_sq = np.asarray(matrix.power(2).mean(axis=0)).ravel()
        variances = mean_sq - (mean ** 2)

        # Take top 20 terms by variance (or all terms if fewer than 20 exist)
        top_indices = np.argsort(variances)[::-1][:20]
        top_terms = [feature_names[i] for i in top_indices]

        # Extract submatrix for top terms (dense representation for corrcoef)
        submatrix = matrix[:, top_indices].toarray()

        # Compute Pearson correlation matrix across columns (rowvar=False)
        corr_matrix = np.corrcoef(submatrix, rowvar=False)

        # Handle edge case where corrcoef returns a scalar for 1 term
        if corr_matrix.ndim == 0:
            corr_matrix = np.array([[float(corr_matrix)]])

        # Replace any potential NaNs (from zero-variance terms) with 0.0
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # Generate heatmap plot and store on session.pending_figure
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            xticklabels=top_terms,
            yticklabels=top_terms,
            ax=ax,
            cbar=True
        )
        ax.set_title("Feature Correlation Matrix (Top Terms by Variance)")
        plt.tight_layout()

        session.pending_figure = fig

        # Format correlation matrix into rounded nested list for JSON serialization
        corr_matrix_list = [[round(float(val), 4) for val in row] for row in corr_matrix]

        result_id = session.next_result_id("feature_correlation_matrix")

        summary = {
            "result_id": result_id,
            "status": "success",
            "n_terms": len(top_terms),
            "terms": top_terms,
            "correlation_matrix": corr_matrix_list,
        }

        # Store result with DataFrame as full_report
        corr_df = pd.DataFrame(corr_matrix_list, index=top_terms, columns=top_terms)
        session.store_result(
            "feature_correlation_matrix_tool",
            {},
            summary,
            full_report=corr_df
        )

        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
