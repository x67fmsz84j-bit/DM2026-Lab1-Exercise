import pytest
import numpy as np
import matplotlib.pyplot as plt
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools

def test_cosine_similarity_tool():
    # Single session instance
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

    # Build DTM (populates count_vectorizer artifact and feature_matrix)
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Load cosine_similarity_tool
    exploration_tools = make_exploration_tools(session)
    cosine_similarity_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    # 1. Doc 0 vs 1 (both catA): 0.9847
    res_0_1 = cosine_similarity_tool.invoke({"doc_index_1": 0, "doc_index_2": 1})
    assert res_0_1["status"] == "success"
    assert res_0_1["cosine_similarity"] == pytest.approx(0.9847, abs=1e-3)

    # 2. Doc 0 vs 3 (catA vs catB): 0.0909
    res_0_3 = cosine_similarity_tool.invoke({"doc_index_1": 0, "doc_index_2": 3})
    assert res_0_3["status"] == "success"
    assert res_0_3["cosine_similarity"] == pytest.approx(0.0909, abs=1e-3)

    # 3. Doc 3 vs 4 (both catB): 0.9847
    res_3_4 = cosine_similarity_tool.invoke({"doc_index_1": 3, "doc_index_2": 4})
    assert res_3_4["status"] == "success"
    assert res_3_4["cosine_similarity"] == pytest.approx(0.9847, abs=1e-3)

    # 4. Doc 0 vs 7 (catA vs empty text row): 0.0
    res_0_7 = cosine_similarity_tool.invoke({"doc_index_1": 0, "doc_index_2": 7})
    assert res_0_7["status"] == "success"
    assert res_0_7["cosine_similarity"] == pytest.approx(0.0, abs=1e-3)


def test_feature_correlation_matrix_tool():
    # Fresh session instance
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

    # Load feature_correlation_matrix_tool
    exploration_tools = make_exploration_tools(session)
    feature_corr_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")

    # Reset pending_figure to ensure tool sets it
    session.pending_figure = None

    # Invoke tool
    res = feature_corr_tool.invoke({})

    assert res["status"] == "success"
    assert res["n_terms"] == 5

    # Check terms order by variance descending (Known Answer 19)
    expected_terms = ["alpha", "gamma", "delta", "beta", "always"]
    assert res["terms"] == expected_terms

    # Full 5x5 expected correlation matrix from Known Answer 19
    # Terms order: alpha, gamma, delta, beta, always
    expected_matrix = np.array([
        [ 1.0000, -0.5718, -0.6882,  0.8885,  0.2601],
        [-0.5718,  1.0000,  0.8307, -0.6435,  0.3140],
        [-0.6882,  0.8307,  1.0000, -0.7746,  0.3780],
        [ 0.8885, -0.6435, -0.7746,  1.0000,  0.2928],
        [ 0.2601,  0.3140,  0.3780,  0.2928,  1.0000],
    ])

    actual_matrix = np.array(res["correlation_matrix"])
    assert actual_matrix.shape == (5, 5)

    # Assert entire 5x5 matrix matches Known Answer 19 within 0.001 tolerance using np.allclose
    assert np.allclose(actual_matrix, expected_matrix, atol=0.001)

    # Inspect heatmap figure on session.pending_figure
    fig = session.pending_figure
    assert fig is not None
    assert isinstance(fig, plt.Figure)

    # Inspect the axes, ticks, and drawn heatmap data values
    ax = fig.axes[0]
    xticklabels = [label.get_text() for label in ax.get_xticklabels()]
    yticklabels = [label.get_text() for label in ax.get_yticklabels()]

    assert xticklabels == expected_terms
    assert yticklabels == expected_terms

    # Extract the image array drawn by seaborn heatmap in ax.collections[0]
    heatmap_array = ax.collections[0].get_array().reshape((5, 5))
    assert np.allclose(heatmap_array, expected_matrix, atol=0.001)
