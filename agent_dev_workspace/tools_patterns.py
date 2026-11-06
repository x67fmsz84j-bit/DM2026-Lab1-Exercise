import os
import tempfile
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool

def make_tools(session):
    """
    Factory function that creates and returns pattern mining tools bound to the given session.
    """

    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: str = "tfidf",
        algorithm: str = "fpgrowth",
        min_sup: Optional[int] = 1,
        k: Optional[int] = 10
    ) -> dict:
        """
        Mines frequent patterns for a single document category using PAMI (FPGrowth, Top-K, or MaxFPGrowth).

        Args:
            category_name: Name of the category to filter and mine.
            filtering_method: Feature selection approach ('variance', 'tfidf', or 'term_frequency').
            algorithm: PAMI pattern mining algorithm ('fpgrowth', 'topk', or 'maxfpgrowth').
            min_sup: Minimum support threshold (required for 'fpgrowth' and 'maxfpgrowth').
            k: Top-K patterns count (required for 'topk').

        Returns:
            Dict containing result_id, category, algorithm, filtering_method, total pattern count,
            settings, and up to 10 top mined patterns.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataframe loaded in session state."}

        df_cat = session.dataframe[session.dataframe["category_name"] == category_name].copy()
        if df_cat.empty:
            return {"status": "error", "message": f"No documents found for category '{category_name}'."}

        settings = {
            "category_name": category_name,
            "filtering_method": filtering_method.lower(),
            "algorithm": algorithm.lower(),
            "min_sup": min_sup,
            "k": k
        }

        # Fill missing text values with empty string
        texts = df_cat["text"].fillna("").astype(str).tolist()

        # Step 1: Fresh category-local CountVectorizer with English stopwords
        vectorizer = CountVectorizer(stop_words="english")
        try:
            X_cat = vectorizer.fit_transform(texts)
        except ValueError:
            # Handle empty vocabulary case after stopword removal
            result_id = session.next_result_id("patterns")
            summary = {
                "result_id": result_id,
                "status": "success",
                "category": category_name,
                "filtering_method": filtering_method.lower(),
                "algorithm": algorithm.lower(),
                "settings": settings,
                "n_patterns": 0,
                "patterns": [],
                "note": "no terms survived stopword removal"
            }
            session.artifacts[f"patterns_{category_name}_{filtering_method.lower()}_{algorithm.lower()}"] = []
            session.store_result("mine_patterns_tool", settings, summary)
            return summary

        terms = np.array(vectorizer.get_feature_names_out())
        X_dense = X_cat.toarray()

        # Step 2: Apply vocabulary filtering
        f_method = filtering_method.lower()
        if f_method == "variance":
            scores = np.var(X_dense, axis=0)
            low_p, high_p = np.percentile(scores, [5, 95])
            kept_indices = np.where((scores >= low_p) & (scores <= high_p))[0]
        elif f_method == "term_frequency":
            scores = np.sum(X_dense, axis=0)
            low_p, high_p = np.percentile(scores, [5, 95])
            kept_indices = np.where((scores >= low_p) & (scores <= high_p))[0]
        elif f_method == "tfidf":
            transformer = TfidfTransformer()
            tfidf_matrix = transformer.fit_transform(X_cat).toarray()
            scores = np.mean(tfidf_matrix, axis=0)
            p20 = np.percentile(scores, 20)
            kept_indices = np.where(scores >= p20)[0]
        else:
            return {"status": "error", "message": f"Unsupported filtering_method '{filtering_method}'."}

        # Step 3: Handle zero terms kept
        if len(kept_indices) == 0:
            result_id = session.next_result_id("patterns")
            summary = {
                "result_id": result_id,
                "status": "success",
                "category": category_name,
                "filtering_method": f_method,
                "algorithm": algorithm.lower(),
                "settings": settings,
                "n_patterns": 0,
                "patterns": [],
                "note": "no terms survived filtering"
            }
            session.artifacts[f"patterns_{category_name}_{f_method}_{algorithm.lower()}"] = []
            session.store_result("mine_patterns_tool", settings, summary)
            return summary

        kept_terms = terms[kept_indices]
        X_filtered = X_dense[:, kept_indices]

        # Step 4: Build transactional database and mine with PAMI
        df_trans = pd.DataFrame(X_filtered, columns=kept_terms)

        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".csv") as tmp:
            tmp_path = tmp.name

        patterns = []
        try:
            from PAMI.extras.convert import DF2DB
            conv = DF2DB.DF2DB(df_trans)
            conv.convert2TransactionalDatabase(tmp_path, condition=">=", thresholdValue=1)

            algo_lower = algorithm.lower()
            if algo_lower == "fpgrowth":
                from PAMI.frequentPattern.basic import FPGrowth
                miner = FPGrowth.FPGrowth(iFile=tmp_path, minSup=min_sup, sep="\t")
            elif algo_lower == "topk":
                from PAMI.frequentPattern.topk import FAE
                miner = FAE.FAE(iFile=tmp_path, k=k if k is not None else 10, sep="\t")
            elif algo_lower == "maxfpgrowth":
                from PAMI.frequentPattern.maximal import MaxFPGrowth
                miner = MaxFPGrowth.MaxFPGrowth(iFile=tmp_path, minSup=min_sup, sep="\t")
            else:
                return {"status": "error", "message": f"Unsupported algorithm '{algorithm}'."}

            miner.startMine()
            patterns_df = miner.getPatternsAsDataFrame()

            if patterns_df is not None and not patterns_df.empty:
                for _, row in patterns_df.iterrows():
                    pat_str = str(row.iloc[0])
                    sup_val = int(row.iloc[1])
                    patterns.append({"pattern": pat_str, "support": sup_val})
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # Save all patterns into session.artifacts and full_report
        artifact_key = f"patterns_{category_name}_{f_method}_{algorithm.lower()}"
        session.artifacts[artifact_key] = patterns

        result_id = session.next_result_id("patterns")
        summary = {
            "result_id": result_id,
            "status": "success",
            "category": category_name,
            "filtering_method": f_method,
            "algorithm": algorithm.lower(),
            "settings": settings,
            "n_patterns": len(patterns),
            "patterns": patterns[:10]  # Only show top 10 patterns in summary
        }

        session.store_result(
            "mine_patterns_tool",
            settings,
            summary,
            full_report=pd.DataFrame(patterns) if patterns else pd.DataFrame(columns=["pattern", "support"])
        )

        return summary

    return [mine_patterns_tool]
