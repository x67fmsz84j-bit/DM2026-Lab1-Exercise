from typing import Optional, Union
import os
from pathlib import Path
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool

def make_tools(session):
    """
    Factory function that creates and returns data tools bound to the given session.
    """

    @tool
    def describe_data_tool() -> dict:
        """
        Computes summary statistics (.describe()) on document text length (character count) overall
        and broken out by category_name. Counts spaces as characters and missing/NaN values as 0.
        Generates a box plot of text length by category and sets it on session.pending_figure.
        Preserves original session.dataframe unchanged.

        Returns:
            Dict containing result_id, status, overall_stats, and per_category_stats.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded in session state."}

        if "text" not in session.dataframe.columns:
            return {"status": "error", "message": "'text' column missing from session.dataframe."}

        # Work on a copy so original session.dataframe stays unchanged
        df = session.dataframe.copy()

        # Compute character length including spaces; count only missing/NaN values as 0
        df["text_length"] = df["text"].apply(lambda x: len(str(x)) if pd.notna(x) else 0)

        def _clean_stats(desc_dict):
            cleaned = {}
            for k, v in desc_dict.items():
                if pd.isna(v):
                    cleaned[k] = None
                else:
                    cleaned[k] = round(float(v), 4)
            return cleaned

        # 1. Overall stats
        overall_desc = df["text_length"].describe().to_dict()
        overall_stats = _clean_stats(overall_desc)

        # 2. Per-category stats
        per_category_stats = {}
        category_col = "category_name" if "category_name" in df.columns else ("label" if "label" in df.columns else None)

        if category_col:
            grouped = df.groupby(category_col)["text_length"].describe()
            for cat, row in grouped.iterrows():
                per_category_stats[str(cat)] = _clean_stats(row.to_dict())

        # 3. Build box plot of text_length by category
        fig, ax = plt.subplots(figsize=(7, 5))
        if category_col:
            sns.boxplot(data=df, x=category_col, y="text_length", ax=ax, palette="Set2")
            ax.set_title("Document Text Length by Category")
            ax.set_xlabel("Category")
        else:
            sns.boxplot(y=df["text_length"], ax=ax, color="skyblue")
            ax.set_title("Document Text Length Overall")

        ax.set_ylabel("Text Length (characters)")
        fig.tight_layout()

        # Set figure on session.pending_figure
        session.pending_figure = fig

        result_id = session.next_result_id("describe_data")

        summary = {
            "result_id": result_id,
            "status": "success",
            "overall_stats": overall_stats,
            "per_category_stats": per_category_stats,
        }

        session.store_result(
            "describe_data_tool",
            {},
            summary,
            full_report=df[["text", category_col, "text_length"]] if category_col else df[["text", "text_length"]]
        )

        return summary

    @tool
    def inspect_data_tool(n: int = 5) -> dict:
        """
        Peeks at the top rows of session.dataframe, displaying row index, text, and category.
        Caps the preview at a maximum of 5 rows and strictly caps text output at 100 characters max total length.

        Args:
            n: Number of head rows to inspect (1 to 5, default 5).

        Returns:
            Dict containing result_id, status, total_rows, total_columns, columns, and rows preview.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded in session state."}

        df = session.dataframe
        total_rows = len(df)
        total_columns = len(df.columns)
        columns = df.columns.tolist()

        # Enforce maximum cap of 5 rows
        n_effective = min(max(1, n), 5, total_rows)
        head_df = df.head(n_effective)

        # Build rows preview with row index, text, and category (strictly <= 100 chars total)
        rows_preview = []
        for idx, row in head_df.iterrows():
            raw_text = str(row.get("text", "")) if pd.notna(row.get("text")) else ""
            if len(raw_text) > 100:
                truncated_text = raw_text[:97] + "..."
            else:
                truncated_text = raw_text

            cat_val = row.get("category_name", row.get("label", row.get("category", None)))
            cat_str = str(cat_val) if pd.notna(cat_val) else "N/A"

            rows_preview.append({
                "row_index": int(idx),
                "text": truncated_text,
                "category": cat_str,
            })

        result_id = session.next_result_id("inspect_data")

        summary = {
            "result_id": result_id,
            "status": "success",
            "total_rows": total_rows,
            "total_columns": total_columns,
            "columns": columns,
            "n_inspected": n_effective,
            "rows_preview": rows_preview,
        }

        report_df = pd.DataFrame(rows_preview)
        session.store_result("inspect_data_tool", {"n": n}, summary, full_report=report_df)

        return summary

    @tool
    def list_files_tool(subdirectory: Optional[str] = None) -> dict:
        """
        Lists files and subdirectories within a given folder in the repository, starting from the repo root.
        Sorts directory and file names alphabetically. Prevents access outside the repository root.

        Args:
            subdirectory: Relative path from repository root to list. Defaults to root ("" or None).

        Returns:
            Dict containing result_id, status, subdirectory, files, and directories.
        """
        repo_root = Path.cwd().resolve()

        if subdirectory:
            target_path = (repo_root / subdirectory).resolve()
        else:
            target_path = repo_root

        # Security check: ensure target_path stays inside repo_root
        try:
            target_path.relative_to(repo_root)
        except ValueError:
            return {
                "status": "error",
                "message": f"Path '{subdirectory}' is outside the repository root."
            }

        if not target_path.exists():
            return {
                "status": "error",
                "message": f"Directory '{subdirectory}' does not exist."
            }

        if not target_path.is_dir():
            return {
                "status": "error",
                "message": f"Path '{subdirectory}' is not a directory."
            }

        directories = []
        files = []

        # List only immediate contents of target_path (including hidden files/folders), sorted alphabetically
        for item in sorted(target_path.iterdir(), key=lambda p: p.name.lower()):
            if item.is_dir():
                directories.append(item.name)
            elif item.is_file():
                files.append(item.name)

        result_id = session.next_result_id("list_files")

        summary = {
            "result_id": result_id,
            "status": "success",
            "subdirectory": subdirectory or "",
            "directories": sorted(directories),
            "files": sorted(files),
        }

        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)

        return summary

    @tool
    def tokenize_tool() -> dict:
        """
        Tokenizes document text into unigram word tokens using NLTK's word_tokenize on lowercased text.
        Preserves stopwords and repeated words without altering original text.
        Stores token lists in session.dataframe['unigrams'] and session.artifacts['tokenized_data'].

        Returns:
            Dict containing result_id, status, total_documents, total_tokens, avg_tokens_per_doc, and vocab_size.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded in session state."}

        if "text" not in session.dataframe.columns:
            return {"status": "error", "message": "'text' column missing from session.dataframe."}

        df = session.dataframe

        def _tokenize(val):
            if pd.isna(val) or not str(val).strip():
                return []
            return nltk.word_tokenize(str(val).lower())

        unigrams = df["text"].apply(_tokenize).tolist()

        # Update session state dataframe and artifacts
        session.dataframe["unigrams"] = unigrams
        session.artifacts["tokenized_data"] = unigrams

        total_documents = len(df)
        all_tokens = [tok for doc in unigrams for tok in doc]
        total_tokens = len(all_tokens)
        vocab_size = len(set(all_tokens))
        avg_tokens_per_doc = round(total_tokens / total_documents, 4) if total_documents > 0 else 0.0

        result_id = session.next_result_id("tokenize")

        summary = {
            "result_id": result_id,
            "status": "success",
            "total_documents": total_documents,
            "total_tokens": total_tokens,
            "avg_tokens_per_doc": avg_tokens_per_doc,
            "vocab_size": vocab_size,
        }

        session.store_result("tokenize_tool", {}, summary)

        return summary

    @tool
    def check_missing_tool() -> dict:
        """
        Checks session.dataframe for missing (NaN/None) or empty/whitespace-only text values.

        Returns:
            Dict containing result_id, total_rows, n_missing, has_missing boolean,
            and a preview of up to 10 missing row indices.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded in session state."}

        if "text" not in session.dataframe.columns:
            return {"status": "error", "message": "'text' column missing from session.dataframe."}

        df = session.dataframe

        # Check for NaN/None OR empty/whitespace-only strings
        is_na = df["text"].isna()
        is_empty_str = df["text"].astype(str).str.strip() == ""
        missing_mask = is_na | is_empty_str

        missing_indices = df[missing_mask].index.tolist()
        n_missing = len(missing_indices)
        total_rows = len(df)

        # Full report DataFrame containing all missing rows and metadata
        report_df = pd.DataFrame({
            "row_index": missing_indices,
            "text": df.loc[missing_indices, "text"].tolist()
        }) if n_missing > 0 else pd.DataFrame(columns=["row_index", "text"])

        result_id = session.next_result_id("check_missing")

        summary = {
            "result_id": result_id,
            "status": "success",
            "total_rows": total_rows,
            "n_missing": n_missing,
            "has_missing": n_missing > 0,
            "missing_indices_preview": missing_indices[:10],
        }

        session.store_result("check_missing_tool", {}, summary, full_report=report_df)

        return summary

    @tool
    def check_duplicates_tool(
        drop: bool = False,
    ) -> dict:
        """
        Checks for (and optionally drops) duplicate documents across full rows in session.dataframe.

        Args:
            drop: If True, drops all copies of duplicated rows (keep=False) from session.dataframe. Default is False.

        Returns:
            Dict containing result_id, total_rows_before, total_rows_after, n_duplicates_found,
            and dropped boolean.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded in session state."}

        df = session.dataframe
        total_rows_before = len(df)

        # Identify duplicate full rows (matching Master's X.duplicated(keep="first"))
        dup_mask = df.duplicated(keep="first")
        duplicate_indices = df[dup_mask].index.tolist()
        n_duplicates = len(duplicate_indices)

        if drop and n_duplicates > 0:
            # Drop all copies of duplicated rows using keep=False (matching Master's X.drop_duplicates(keep=False))
            session.dataframe = df.drop_duplicates(keep=False).reset_index(drop=True)
            total_rows_after = len(session.dataframe)

            # Update labels in session state
            if "category_name" in session.dataframe.columns:
                session.set_labels(session.dataframe["category_name"].values)

            # Clear outdated DTM matrix, feature names, and count_vectorizer artifact
            session.feature_matrix = None
            session.feature_names = None
            session.artifacts.pop("count_vectorizer", None)
        else:
            total_rows_after = total_rows_before

        result_id = session.next_result_id("check_duplicates")

        summary = {
            "result_id": result_id,
            "status": "success",
            "total_rows_before": total_rows_before,
            "total_rows_after": total_rows_after,
            "n_duplicates_found": n_duplicates,
            "dropped": drop,
            "duplicate_indices_preview": duplicate_indices[:10],
        }

        report_df = pd.DataFrame({
            "row_index": duplicate_indices
        }) if n_duplicates > 0 else pd.DataFrame(columns=["row_index"])

        session.store_result("check_duplicates_tool", {"drop": drop}, summary, full_report=report_df)

        return summary

    @tool
    def sample_data_tool(
        n: Optional[int] = None,
        frac: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> dict:
        """
        Subsamples rows from session.dataframe while preserving original row indices.

        Args:
            n: Number of items to sample.
            frac: Fraction of items to sample (e.g. 0.5 for 50%).
            random_state: Seed for reproducible random sampling.

        Returns:
            Dict containing result_id, total_rows_before, total_rows_after, sampled_indices_preview,
            and parameters.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded in session state."}

        df = session.dataframe
        total_rows_before = len(df)

        if n is None and frac is None:
            n = min(5, total_rows_before)

        # Sample rows while keeping original row indices intact (no reset_index)
        sampled_df = df.sample(n=n, frac=frac, random_state=random_state)
        sampled_indices = sampled_df.index.tolist()

        # Update session state dataframe
        session.dataframe = sampled_df

        # Update category labels in session state
        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].values)

        # Clear outdated DTM matrix, feature names, and count_vectorizer artifact
        session.feature_matrix = None
        session.feature_names = None
        session.artifacts.pop("count_vectorizer", None)

        result_id = session.next_result_id("sample_data")

        summary = {
            "result_id": result_id,
            "status": "success",
            "total_rows_before": total_rows_before,
            "total_rows_after": len(sampled_df),
            "sampled_indices_preview": sampled_indices[:10],
            "parameters": {
                "n": n,
                "frac": frac,
                "random_state": random_state,
            },
        }

        session.store_result(
            "sample_data_tool",
            {"n": n, "frac": frac, "random_state": random_state},
            summary,
            full_report=sampled_df
        )

        return summary

    return [describe_data_tool, inspect_data_tool, list_files_tool, tokenize_tool, check_missing_tool, check_duplicates_tool, sample_data_tool]
