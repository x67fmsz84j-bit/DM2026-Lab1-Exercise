import pytest
import numpy as np
import pandas as pd
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools

def test_variance_filter_tool():
    # Single session instance used throughout the test
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM first
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Save a copy of the original sparse matrix values and feature names to verify DTM remains unchanged
    initial_matrix_copy = session.feature_matrix.copy()
    initial_features = list(session.feature_names)

    # Load variance_filter_tool
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    # 1. Test with threshold=0.15 (Known Answer 5)
    res_015 = variance_filter_tool.invoke({"threshold": 0.15})

    assert res_015["status"] == "success"
    assert res_015["total_terms"] == 5
    assert res_015["n_kept"] == 4
    assert res_015["n_removed"] == 1
    assert set(res_015["kept_terms_preview"]) == {"alpha", "gamma", "delta", "beta"}
    assert res_015["removed_terms_preview"] == ["always"]

    # Verify report structure, columns, sorting, and keep flags in session store
    stored_result = session.results_store[res_015["result_id"]]
    report_df = stored_result["full_report"]

    assert list(report_df.columns) == ["term", "variance", "keep"]

    # Check sorting: variance must be strictly descending
    assert report_df["variance"].is_monotonic_decreasing

    # Verify exact variances against Known Answer 5 using pytest.approx
    var_dict = dict(zip(report_df["term"], report_df["variance"]))
    assert var_dict["alpha"] == pytest.approx(1.1875, abs=1e-3)
    assert var_dict["gamma"] == pytest.approx(1.109375, abs=1e-3)
    assert var_dict["delta"] == pytest.approx(0.25, abs=1e-3)
    assert var_dict["beta"] == pytest.approx(0.234375, abs=1e-3)
    assert var_dict["always"] == pytest.approx(0.109375, abs=1e-3)

    # Check keep flags for threshold 0.15
    keep_dict = dict(zip(report_df["term"], report_df["keep"]))
    assert keep_dict["alpha"] is True or keep_dict["alpha"] == True
    assert keep_dict["gamma"] is True or keep_dict["gamma"] == True
    assert keep_dict["delta"] is True or keep_dict["delta"] == True
    assert keep_dict["beta"] is True or keep_dict["beta"] == True
    assert keep_dict["always"] is False or keep_dict["always"] == False

    # Confirm DTM matrix values and feature names remain identical after 1st call
    assert np.array_equal(session.feature_matrix.toarray(), initial_matrix_copy.toarray())
    assert session.feature_names == initial_features

    # 2. Test with threshold=0.25 (Boundary check for > threshold)
    # Since delta has variance exactly 0.25, strict inequality variance > 0.25 must REMOVE delta
    res_025 = variance_filter_tool.invoke({"threshold": 0.25})
    assert res_025["status"] == "success"
    assert res_025["n_kept"] == 2
    assert res_025["n_removed"] == 3
    assert set(res_025["kept_terms_preview"]) == {"alpha", "gamma"}
    assert "delta" in res_025["removed_terms_preview"]

    # Confirm DTM matrix values and feature names remain identical after 2nd call
    assert np.array_equal(session.feature_matrix.toarray(), initial_matrix_copy.toarray())
    assert session.feature_names == initial_features


def test_pearson_filter_tool():
    # Fresh session instance per test function
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Save initial sparse matrix copy and feature names
    initial_matrix_copy = session.feature_matrix.copy()
    initial_features = list(session.feature_names)

    # Load pearson_filter_tool
    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    # Run pearson_filter_tool with target_class="catB"
    res = pearson_filter_tool.invoke({"target_class": "catB"})

    assert res["status"] == "success"
    assert res["target_class"] == "catB"
    assert res["total_terms"] == 5

    # Retrieve stored full report
    stored_result = session.results_store[res["result_id"]]
    report_df = stored_result["full_report"]

    # Check columns
    assert list(report_df.columns) == ["term", "pearson_r"]

    # Verify sorting by absolute correlation magnitude descending
    abs_rs = np.abs(report_df["pearson_r"].values)
    assert np.all(np.diff(abs_rs) <= 1e-9)

    # Expected order by absolute correlation: delta (1.0), gamma (0.8307), beta (-0.7746), alpha (-0.6882), always (0.3780)
    expected_term_order = ["delta", "gamma", "beta", "alpha", "always"]
    assert list(report_df["term"]) == expected_term_order

    # Verify Pearson r values and signs against Known Answer 6 using pytest.approx
    r_dict = dict(zip(report_df["term"], report_df["pearson_r"]))
    assert r_dict["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert r_dict["gamma"] == pytest.approx(0.8307, abs=1e-3)
    assert r_dict["beta"] == pytest.approx(-0.7746, abs=1e-3)
    assert r_dict["alpha"] == pytest.approx(-0.6882, abs=1e-3)
    assert r_dict["always"] == pytest.approx(0.3780, abs=1e-3)

    # Confirm DTM matrix values and feature names remain completely identical after Pearson filter call
    assert np.array_equal(session.feature_matrix.toarray(), initial_matrix_copy.toarray())
    assert session.feature_names == initial_features


def test_spearman_filter_tool():
    # Fresh session instance per test function
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Save initial sparse matrix copy and feature names
    initial_matrix_copy = session.feature_matrix.copy()
    initial_features = list(session.feature_names)

    # Load spearman_filter_tool
    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    # Run spearman_filter_tool with target_class="catB"
    res = spearman_filter_tool.invoke({"target_class": "catB"})

    assert res["status"] == "success"
    assert res["target_class"] == "catB"
    assert res["total_terms"] == 5

    # Retrieve stored full report
    stored_result = session.results_store[res["result_id"]]
    report_df = stored_result["full_report"]

    # Check columns
    assert list(report_df.columns) == ["term", "spearman_r"]

    # Verify sorting by absolute correlation magnitude descending
    abs_rs = np.abs(report_df["spearman_r"].values)
    assert np.all(np.diff(abs_rs) <= 1e-9)

    # Expected order by absolute Spearman correlation: delta (1.0), gamma (0.9363), beta (-0.7746), alpha (-0.7500), always (0.3780)
    expected_term_order = ["delta", "gamma", "beta", "alpha", "always"]
    assert list(report_df["term"]) == expected_term_order

    # Verify Spearman r values and signs against Known Answer 6 using pytest.approx
    r_dict = dict(zip(report_df["term"], report_df["spearman_r"]))
    assert r_dict["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert r_dict["gamma"] == pytest.approx(0.9363, abs=1e-3)
    assert r_dict["beta"] == pytest.approx(-0.7746, abs=1e-3)
    assert r_dict["alpha"] == pytest.approx(-0.7500, abs=1e-3)
    assert r_dict["always"] == pytest.approx(0.3780, abs=1e-3)

    # Confirm DTM matrix values and feature names remain completely identical after Spearman filter call
    assert np.array_equal(session.feature_matrix.toarray(), initial_matrix_copy.toarray())
    assert session.feature_names == initial_features
