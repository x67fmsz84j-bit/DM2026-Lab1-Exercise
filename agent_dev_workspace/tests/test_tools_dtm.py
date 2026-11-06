import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools

def test_build_dtm_tool():
    # Single session instance used throughout the test
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load sample_fixture.csv dataset
    load_res = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    assert load_res["n_documents"] == 8

    # Instantiate build_dtm_tool from tools_dtm
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    # Run build_dtm_tool with default parameters
    result = build_dtm_tool.invoke({})

    # Check returned summary fields
    assert result["status"] == "success"
    assert result["matrix_shape"] == [8, 5]
    assert result["non_zero"] == 21
    assert result["total_elements"] == 40
    assert result["sparsity_pct"] == pytest.approx(47.5, abs=1e-3)

    # Check session state attributes
    expected_vocab = ["alpha", "always", "beta", "delta", "gamma"]
    assert session.feature_names == expected_vocab
    assert session.feature_matrix is not None
    assert session.feature_matrix.shape == (8, 5)

    # Confirm Row 7 (the empty text document) is preserved as a row of all zeros
    dense_matrix = session.feature_matrix.toarray()
    assert np.all(dense_matrix[7, :] == 0)

    # Verify CountVectorizer artifact is saved in session
    assert "count_vectorizer" in session.artifacts
    assert isinstance(session.artifacts["count_vectorizer"], CountVectorizer)


def test_term_frequency_tool():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")

    build_dtm_tool.invoke({})

    # Copy initial sparse matrix and feature names to verify immutability
    initial_matrix_copy = session.feature_matrix.copy()
    initial_features = list(session.feature_names)

    # Invoke term_frequency_tool
    res = term_frequency_tool.invoke({})

    assert res["status"] == "success"
    assert res["total_terms"] == 5
    assert res["total_word_count"] == 27

    # Check stored report
    stored_result = session.results_store[res["result_id"]]
    report_df = stored_result["full_report"]

    assert list(report_df.columns) == ["term", "frequency"]
    assert report_df["frequency"].is_monotonic_decreasing

    # Verify exact term frequencies against Known Answer 14
    freq_dict = dict(zip(report_df["term"], report_df["frequency"]))
    assert freq_dict["alpha"] == 6
    assert freq_dict["always"] == 7
    assert freq_dict["beta"] == 3
    assert freq_dict["delta"] == 4
    assert freq_dict["gamma"] == 7

    # Verify DTM matrix values and feature names remain identical
    assert np.array_equal(session.feature_matrix.toarray(), initial_matrix_copy.toarray())
    assert session.feature_names == initial_features


def test_dtm_heatmap_tool():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")

    build_dtm_tool.invoke({})

    initial_matrix_copy = session.feature_matrix.copy()
    initial_features = list(session.feature_names)

    # 1. Default invocation (Known Answer 21, entire 8x5 matrix)
    res_default = dtm_heatmap_tool.invoke({})

    assert res_default["status"] == "success"
    assert res_default["n_documents"] == 8
    assert res_default["n_terms"] == 5
    assert res_default["term_labels"] == ["alpha", "always", "beta", "delta", "gamma"]

    # Verify full 8x5 matrix slice against Known Answer 21
    expected_full_matrix = [
        [3, 1, 1, 0, 0],  # Doc 0
        [2, 1, 1, 0, 0],  # Doc 1
        [1, 1, 1, 0, 0],  # Doc 2
        [0, 1, 0, 1, 3],  # Doc 3
        [0, 1, 0, 1, 2],  # Doc 4
        [0, 1, 0, 1, 1],  # Doc 5
        [0, 1, 0, 1, 1],  # Doc 6
        [0, 0, 0, 0, 0],  # Doc 7
    ]
    assert res_default["matrix_slice"] == expected_full_matrix

    # Check heatmap figure was assigned to session.pending_figure
    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, plt.Figure)

    # 2. Sliced invocation (3 docs x 2 terms corner slice from Known Answer 21)
    res_sliced = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})

    assert res_sliced["status"] == "success"
    assert res_sliced["n_documents"] == 3
    assert res_sliced["n_terms"] == 2
    assert res_sliced["term_labels"] == ["alpha", "always"]
    assert res_sliced["matrix_slice"] == [[3, 1], [2, 1], [1, 1]]

    # Verify DTM matrix values and feature names remain identical
    assert np.array_equal(session.feature_matrix.toarray(), initial_matrix_copy.toarray())
    assert session.feature_names == initial_features
