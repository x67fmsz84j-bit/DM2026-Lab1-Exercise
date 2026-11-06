from typing import Union
import numpy as np
import pandas as pd
from scipy.stats import rankdata, pearsonr
from langchain_core.tools import tool

def make_tools(session):
    """
    Factory function that creates and returns filtering tools bound to the given session.
    """

    @tool
    def variance_filter_tool(
        threshold: Union[int, float] = 0.0,
    ) -> dict:
        """
        Calculates variance for each term across all documents in session.feature_matrix,
        and identifies terms above and below the given variance threshold.

        Args:
            threshold: Variance threshold (float). Terms with variance > threshold are kept. Default is 0.0.

        Returns:
            Dict containing result_id, total terms, count/previews of kept and removed terms,
            and summary metrics.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No Document-Term Matrix (DTM) found in session state. Run build_dtm_tool first."}

        # Work directly with the sparse matrix to calculate column variances across all documents
        # var = E[X^2] - (E[X])^2
        matrix = session.feature_matrix
        feature_names = session.feature_names

        # Compute column means and column means of squares using sparse representation
        mean = np.asarray(matrix.mean(axis=0)).ravel()
        mean_sq = np.asarray(matrix.power(2).mean(axis=0)).ravel()
        variances = mean_sq - (mean ** 2)

        # Flag terms using strict inequality: variance > threshold
        kept_mask = variances > threshold

        # Build full report DataFrame with term (1st), variance (2nd), and keep (3rd), sorted by variance descending
        report_df = pd.DataFrame({
            "term": feature_names,
            "variance": variances,
            "keep": kept_mask
        }).sort_values(by="variance", ascending=False).reset_index(drop=True)

        # Extract ordered lists of kept and removed terms from the sorted report
        kept_terms = report_df[report_df["keep"]]["term"].tolist()
        removed_terms = report_df[~report_df["keep"]]["term"].tolist()

        result_id = session.next_result_id("variance_filter")

        summary = {
            "result_id": result_id,
            "status": "success",
            "threshold": threshold,
            "total_terms": len(feature_names),
            "n_kept": len(kept_terms),
            "n_removed": len(removed_terms),
            "kept_terms_preview": kept_terms[:10],
            "removed_terms_preview": removed_terms[:10],
        }

        # Store result with full_report DataFrame for visualize_result_tool support
        session.store_result(
            "variance_filter_tool",
            {"threshold": threshold},
            summary,
            full_report=report_df
        )

        return summary

    @tool
    def pearson_filter_tool(
        target_class: str,
    ) -> dict:
        """
        Calculates Pearson correlation between each term's frequency and a binary target class indicator.

        Args:
            target_class: String label of the target category to correlate against (one-vs-rest).

        Returns:
            Dict containing result_id, target_class, total terms, top correlations preview,
            and summary metrics.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No Document-Term Matrix (DTM) found in session state. Run build_dtm_tool first."}

        if session.labels is None:
            return {"status": "error", "message": "No document labels found in session state."}

        # Build binary target vector (1 for target_class, 0 otherwise)
        y = (np.array(session.labels) == target_class).astype(float)

        if np.all(y == 0):
            return {"status": "error", "message": f"Target class '{target_class}' not found in document labels."}

        matrix = session.feature_matrix
        feature_names = session.feature_names

        # Compute sparse column means, standard deviations, and cross-products against y vector directly
        # r_i = cov(X_i, y) / (std(X_i) * std(y))
        # cov(X_i, y) = E[X_i * y] - E[X_i] * E[y]
        n_docs = matrix.shape[0]
        y_mean = np.mean(y)
        y_std = np.std(y)

        if y_std == 0:
            pearson_rs = np.zeros(matrix.shape[1])
        else:
            # E[X_i] per column
            x_means = np.asarray(matrix.mean(axis=0)).ravel()

            # E[X_i^2] per column to get std(X_i)
            x_means_sq = np.asarray(matrix.power(2).mean(axis=0)).ravel()
            x_stds = np.sqrt(np.maximum(0, x_means_sq - (x_means ** 2)))

            # Matrix-vector multiplication for sparse E[X_i * y] = (X^T @ y) / n_docs
            x_y_means = np.asarray(matrix.T.dot(y)).ravel() / n_docs

            covariances = x_y_means - (x_means * y_mean)

            # Avoid division by zero for zero-variance terms
            valid_std_mask = x_stds > 0
            pearson_rs = np.zeros(matrix.shape[1])
            pearson_rs[valid_std_mask] = covariances[valid_std_mask] / (x_stds[valid_std_mask] * y_std)

        # Build full report DataFrame with term (1st) and pearson_r (2nd)
        report_df = pd.DataFrame({
            "term": feature_names,
            "pearson_r": pearson_rs,
            "abs_pearson_r": np.abs(pearson_rs)
        }).sort_values(by="abs_pearson_r", ascending=False).reset_index(drop=True)

        # Drop helper column from full_report so 1st is 'term' and 2nd is 'pearson_r'
        full_report_df = report_df[["term", "pearson_r"]]

        result_id = session.next_result_id("pearson_filter")

        # Compact preview of top correlated terms sorted by magnitude
        top_terms_preview = [
            {"term": row["term"], "pearson_r": round(float(row["pearson_r"]), 4)}
            for _, row in full_report_df.head(10).iterrows()
        ]

        summary = {
            "result_id": result_id,
            "status": "success",
            "target_class": target_class,
            "total_terms": len(feature_names),
            "top_correlations_preview": top_terms_preview,
        }

        # Store result with full_report DataFrame for visualize_result_tool support
        session.store_result(
            "pearson_filter_tool",
            {"target_class": target_class},
            summary,
            full_report=full_report_df
        )

        return summary

    @tool
    def spearman_filter_tool(
        target_class: str,
    ) -> dict:
        """
        Calculates Spearman rank correlation between each term's frequency and a binary target class indicator.

        Args:
            target_class: String label of the target category to correlate against (one-vs-rest).

        Returns:
            Dict containing result_id, target_class, total terms, top correlations preview,
            and summary metrics.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No Document-Term Matrix (DTM) found in session state. Run build_dtm_tool first."}

        if session.labels is None:
            return {"status": "error", "message": "No document labels found in session state."}

        # Build binary target vector (1 for target_class, 0 otherwise)
        y = (np.array(session.labels) == target_class).astype(float)

        if np.all(y == 0):
            return {"status": "error", "message": f"Target class '{target_class}' not found in document labels."}

        matrix = session.feature_matrix
        feature_names = session.feature_names

        # Calculate rank vector for binary target y (handles tied ranks automatically)
        y_ranked = rankdata(y)

        spearman_rs = []
        # Process each column (term) individually, ranking counts (including zero counts and ties)
        for col_idx in range(matrix.shape[1]):
            x_col = matrix[:, col_idx].toarray().ravel()
            x_ranked = rankdata(x_col)

            # Pearson correlation on the ranks equals Spearman correlation
            if np.std(x_ranked) == 0 or np.std(y_ranked) == 0:
                r = 0.0
            else:
                r, _ = pearsonr(x_ranked, y_ranked)
                if np.isnan(r):
                    r = 0.0

            spearman_rs.append(r)

        spearman_rs = np.array(spearman_rs)

        # Build full report DataFrame with term (1st) and spearman_r (2nd)
        report_df = pd.DataFrame({
            "term": feature_names,
            "spearman_r": spearman_rs,
            "abs_spearman_r": np.abs(spearman_rs)
        }).sort_values(by="abs_spearman_r", ascending=False).reset_index(drop=True)

        # Drop helper column so 1st is 'term' and 2nd is 'spearman_r'
        full_report_df = report_df[["term", "spearman_r"]]

        result_id = session.next_result_id("spearman_filter")

        # Compact preview of top correlated terms sorted by absolute correlation
        top_terms_preview = [
            {"term": row["term"], "spearman_r": round(float(row["spearman_r"]), 4)}
            for _, row in full_report_df.head(10).iterrows()
        ]

        summary = {
            "result_id": result_id,
            "status": "success",
            "target_class": target_class,
            "total_terms": len(feature_names),
            "top_correlations_preview": top_terms_preview,
        }

        # Store result with full_report DataFrame for visualize_result_tool support
        session.store_result(
            "spearman_filter_tool",
            {"target_class": target_class},
            summary,
            full_report=full_report_df
        )

        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
