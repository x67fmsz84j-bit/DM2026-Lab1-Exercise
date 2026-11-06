import pytest
import pandas as pd
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools as make_patterns_tools

def _setup_session():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    return session

def test_mine_patterns_fpgrowth_tfidf():
    # Known answer #17
    session = _setup_session()
    patterns_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in patterns_tools if t.name == "mine_patterns_tool")

    # catA -> alpha (support 3)
    res_a = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_a["status"] == "success"
    assert res_a["n_patterns"] == 1
    assert res_a["patterns"][0]["pattern"] == "alpha"
    assert res_a["patterns"][0]["support"] == 3
    assert res_a["settings"]["min_sup"] == 1
    assert session.artifacts["patterns_catA_tfidf_fpgrowth"] == res_a["patterns"]

    # Verify full report for catA
    stored_a = session.results_store[res_a["result_id"]]
    assert isinstance(stored_a["full_report"], pd.DataFrame)
    assert stored_a["full_report"].to_dict(orient="records") == res_a["patterns"]

    # catB -> gamma (support 4)
    res_b = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_b["status"] == "success"
    assert res_b["n_patterns"] == 1
    assert res_b["patterns"][0]["pattern"] == "gamma"
    assert res_b["patterns"][0]["support"] == 4
    assert res_b["settings"]["min_sup"] == 1
    assert session.artifacts["patterns_catB_tfidf_fpgrowth"] == res_b["patterns"]

    # Verify min_sup=4 threshold edge case for catA (0 patterns) and catB (1 pattern)
    res_a_sup4 = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4
    })
    assert res_a_sup4["n_patterns"] == 0

    res_b_sup4 = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4
    })
    assert res_b_sup4["n_patterns"] == 1
    assert res_b_sup4["patterns"][0]["pattern"] == "gamma"
    assert res_b_sup4["patterns"][0]["support"] == 4

def test_mine_patterns_zero_terms_kept():
    # Known answer #17: variance and term_frequency keep zero terms for both categories
    session = _setup_session()
    patterns_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in patterns_tools if t.name == "mine_patterns_tool")

    for cat in ["catA", "catB"]:
        for f_method in ["variance", "term_frequency"]:
            res = mine_patterns_tool.invoke({
                "category_name": cat,
                "filtering_method": f_method,
                "algorithm": "fpgrowth",
                "min_sup": 1
            })
            assert res["status"] == "success"
            assert res["n_patterns"] == 0
            assert res["patterns"] == []
            assert res["note"] == "no terms survived filtering"
            assert session.artifacts[f"patterns_{cat}_{f_method}_fpgrowth"] == []

def test_mine_patterns_topk_and_maxfpgrowth():
    # Known answer #20
    session = _setup_session()
    patterns_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in patterns_tools if t.name == "mine_patterns_tool")

    # topk for catA (k=1)
    res_topk_a = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert res_topk_a["status"] == "success"
    assert res_topk_a["n_patterns"] == 1
    assert res_topk_a["patterns"][0]["pattern"] == "alpha"
    assert res_topk_a["patterns"][0]["support"] == 3
    assert res_topk_a["settings"]["k"] == 1

    # topk for catB (k=2)
    res_topk_b = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 2
    })
    assert res_topk_b["status"] == "success"
    assert res_topk_b["n_patterns"] == 1
    assert res_topk_b["patterns"][0]["pattern"] == "gamma"
    assert res_topk_b["patterns"][0]["support"] == 4
    assert res_topk_b["settings"]["k"] == 2

    # maxfpgrowth returns zero patterns for both catA and catB at min_sup=1 and min_sup=4
    for cat in ["catA", "catB"]:
        for sup in [1, 4]:
            res_max = mine_patterns_tool.invoke({
                "category_name": cat,
                "filtering_method": "tfidf",
                "algorithm": "maxfpgrowth",
                "min_sup": sup
            })
            assert res_max["status"] == "success"
            assert res_max["n_patterns"] == 0
            assert res_max["patterns"] == []
            assert session.artifacts[f"patterns_{cat}_tfidf_maxfpgrowth"] == []

            stored = session.results_store[res_max["result_id"]]
            assert isinstance(stored["full_report"], pd.DataFrame)
            assert stored["full_report"].empty
