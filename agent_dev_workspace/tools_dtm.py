from typing import List, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool

def make_tools(session):
    """
    Factory function that creates and returns DTM tools bound to the given session.
    """

    @tool
    def build_dtm_tool(
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0,
        ngram_range: List[int] = [1, 1],
    ) -> dict:
        """
        Builds a Document-Term Matrix (DTM) from document text stored in session.dataframe.

        Args:
            min_df: Minimum document frequency threshold (int for doc count, float for proportion). Default is 1.
            max_df: Maximum document frequency threshold (int for doc count, float for proportion). Default is 1.0.
            ngram_range: Lower and upper boundary for n-gram extraction as [min_n, max_n]. Default is [1, 1].

        Returns:
            Dict containing matrix shape, non-zero element count, total elements, sparsity percentage,
            result_id, and recorded parameters.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded in session state."}

        if "text" not in session.dataframe.columns:
            return {"status": "error", "message": "'text' column missing from session.dataframe."}

        # Treat missing text as empty strings so all rows are preserved
        text_series = session.dataframe["text"].fillna("").astype(str)

        # Convert ngram_range list to tuple for CountVectorizer
        ngram_tuple = tuple(ngram_range)

        # Initialize CountVectorizer with standard parameters (no stop words removal)
        vectorizer = CountVectorizer(
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_tuple,
        )

        matrix = vectorizer.fit_transform(text_series)
        feature_names = vectorizer.get_feature_names_out().tolist()

        # Update session state attributes
        session.feature_matrix = matrix
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].values)

        # Calculate sparsity metrics
        n_rows, n_cols = matrix.shape
        non_zero = int(matrix.nnz)
        total_elements = int(n_rows * n_cols)
        sparsity_pct = float(100.0 * (1.0 - non_zero / total_elements)) if total_elements > 0 else 0.0

        result_id = session.next_result_id("dtm")

        summary = {
            "result_id": result_id,
            "status": "success",
            "matrix_shape": [n_rows, n_cols],
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
            "parameters": {
                "min_df": min_df,
                "max_df": max_df,
                "ngram_range": list(ngram_range),
            },
        }

        session.store_result("build_dtm_tool", {"min_df": min_df, "max_df": max_df, "ngram_range": ngram_range}, summary)

        return summary

    @tool
    def term_frequency_tool() -> dict:
        """
        Aggregates term frequencies across all documents in session.feature_matrix.

        Returns:
            Dict containing result_id, total_terms, total_word_count, and top_terms_preview.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No Document-Term Matrix (DTM) found in session state. Run build_dtm_tool first."}

        matrix = session.feature_matrix
        feature_names = session.feature_names

        # Sum term frequencies across all document rows
        counts = np.asarray(matrix.sum(axis=0)).ravel()

        report_df = pd.DataFrame({
            "term": feature_names,
            "frequency": counts
        }).sort_values(by="frequency", ascending=False).reset_index(drop=True)

        result_id = session.next_result_id("term_freq")

        top_terms_preview = [
            {"term": row["term"], "frequency": int(row["frequency"])}
            for _, row in report_df.head(10).iterrows()
        ]

        summary = {
            "result_id": result_id,
            "status": "success",
            "total_terms": len(feature_names),
            "total_word_count": int(np.sum(counts)),
            "top_terms_preview": top_terms_preview,
        }

        session.store_result("term_frequency_tool", {}, summary, full_report=report_df)

        return summary

    @tool
    def dtm_heatmap_tool(
        n_terms: int = 20,
        n_documents: int = 20,
    ) -> dict:
        """
        Generates a heatmap visualization for a positional slice of the global DTM.

        Args:
            n_terms: Number of vocabulary terms (columns) to slice from the start. Default is 20.
            n_documents: Number of documents (rows) to slice from the start. Default is 20.

        Returns:
            Dict containing result_id, n_documents, n_terms, document_indices, term_labels,
            and matrix_slice.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No Document-Term Matrix (DTM) found in session state. Run build_dtm_tool first."}

        matrix = session.feature_matrix
        feature_names = session.feature_names

        n_rows, n_cols = matrix.shape
        actual_docs = min(n_documents, n_rows)
        actual_terms = min(n_terms, n_cols)

        # Slice DTM positional window
        matrix_slice = matrix[0:actual_docs, 0:actual_terms].toarray()
        slice_terms = feature_names[0:actual_terms]
        doc_indices = list(range(actual_docs))

        # Create plot and assign to session.pending_figure
        fig, ax = plt.subplots(figsize=(max(6, actual_terms * 0.8), max(4, actual_docs * 0.5)))
        sns.heatmap(
            matrix_slice,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=slice_terms,
            yticklabels=[f"Doc {i}" for i in doc_indices],
            ax=ax,
            cbar=True,
        )
        ax.set_title(f"DTM Slice ({actual_docs} Docs x {actual_terms} Terms)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("dtm_heatmap")

        summary = {
            "result_id": result_id,
            "status": "success",
            "n_documents": actual_docs,
            "n_terms": actual_terms,
            "document_indices": doc_indices,
            "term_labels": slice_terms,
            "matrix_slice": matrix_slice.tolist(),
        }

        session.store_result(
            "dtm_heatmap_tool",
            {"n_terms": n_terms, "n_documents": n_documents},
            summary
        )

        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
