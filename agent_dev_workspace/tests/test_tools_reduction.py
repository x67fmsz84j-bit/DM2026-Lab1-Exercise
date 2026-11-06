import pytest
import numpy as np
import pandas as pd
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools

def test_binarize_labels_tool():
    # Single session instance used throughout the test function
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

    # Load binarize_labels_tool
    reduction_tools = make_reduction_tools(session)
    binarize_labels_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")

    # Invoke tool
    res = binarize_labels_tool.invoke({})

    assert res["status"] == "success"
    assert res["n_samples"] == 8
    assert res["n_categories"] == 2
    assert res["categories"] == ["catA", "catB"]

    # Check session artifacts
    binarized_matrix = session.artifacts["binarized_labels"]
    assert binarized_matrix.shape == (8, 2)

    # CatA rows (0, 1, 2, 7) should be [1, 0]
    expected_cata = [1, 0]
    for idx in [0, 1, 2, 7]:
        assert list(binarized_matrix[idx]) == expected_cata

    # CatB rows (3, 4, 5, 6) should be [0, 1]
    expected_catb = [0, 1]
    for idx in [3, 4, 5, 6]:
        assert list(binarized_matrix[idx]) == expected_catb

    # Check full_report in session results_store
    stored_result = session.results_store[res["result_id"]]
    report_df = stored_result["full_report"]
    assert list(report_df.columns) == ["catA", "catB"]


def test_binarize_labels_tool_single_category():
    # Test single-category edge case
    session = SessionState()

    # Manually set dataframe and labels with only 1 category
    session.dataframe = pd.DataFrame({
        "text": ["hello", "world"],
        "category_name": ["catA", "catA"]
    })
    session.labels = np.array(["catA", "catA"])

    reduction_tools = make_reduction_tools(session)
    binarize_labels_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")

    res = binarize_labels_tool.invoke({})

    assert res["status"] == "success"
    assert res["n_samples"] == 2
    assert res["n_categories"] == 1
    assert res["categories"] == ["catA"]

    # Single category column should be all ones
    binarized_matrix = session.artifacts["binarized_labels"]
    assert binarized_matrix.shape == (2, 1)
    assert np.array_equal(binarized_matrix, np.array([[1], [1]]))


def _setup_session_with_dtm():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    return session


def _verify_plot_points_and_labels(fig, coords, expected_labels):
    ax = fig.axes[0]
    collections = ax.collections
    assert len(collections) > 0, "Plot has no point collections"

    # Extract all plotted offset points
    plotted_points = []
    for c in collections:
        offsets = c.get_offsets()
        if len(offsets) > 0:
            plotted_points.append(np.asarray(offsets))

    assert len(plotted_points) > 0, "No offsets found in plot collections"
    all_plotted_points = np.vstack(plotted_points)

    # Sort both arrays to compare point sets directly
    sorted_coords = coords[np.lexsort((coords[:, 1], coords[:, 0]))]
    sorted_plotted = all_plotted_points[np.lexsort((all_plotted_points[:, 1], all_plotted_points[:, 0]))]

    assert sorted_plotted.shape == sorted_coords.shape, f"Plotted points shape {sorted_plotted.shape} != expected {sorted_coords.shape}"
    np.testing.assert_allclose(sorted_plotted, sorted_coords, atol=1e-3)

    # Strictly require legend to exist
    legend = ax.get_legend()
    assert legend is not None, "Plot legend is missing"

    legend_texts = [t.get_text() for t in legend.get_texts()]
    for label in set(expected_labels):
        assert label in legend_texts, f"Category '{label}' missing from plot legend"


def test_reduce_dimensions_pca():
    session = _setup_session_with_dtm()
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_dimensions_tool.invoke({"method": "pca", "random_state": 42})

    assert res["status"] == "success"
    assert res["method"] == "pca"
    assert res["n_components"] == 2

    # Expected PCA coordinates from TEST_FIXTURE.md #7
    expected_coords = np.array([
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822]
    ])

    actual_coords = session.artifacts["reduced_coordinates"]
    np.testing.assert_allclose(actual_coords, expected_coords, atol=1e-3)

    # Explained variance ratio from TEST_FIXTURE.md #7
    expected_evr = [0.7532, 0.1965]
    np.testing.assert_allclose(res["explained_variance_ratio"], expected_evr, atol=1e-3)

    # Verify plot
    assert session.pending_figure is not None
    _verify_plot_points_and_labels(session.pending_figure, actual_coords, session.labels)


def test_reduce_dimensions_tsne():
    session = _setup_session_with_dtm()
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_dimensions_tool.invoke({"method": "tsne", "perplexity": 2, "random_state": 42})

    assert res["status"] == "success"
    assert res["method"] == "tsne"

    # Expected t-SNE coordinates from TEST_FIXTURE.md #8
    expected_coords = np.array([
        [1330.9991, 32.1039],
        [951.2459, 393.5588],
        [530.8209, 682.9791],
        [-1558.6448, -314.1625],
        [-915.5153, -346.3907],
        [-280.0313, -1261.2115],
        [-280.0313, -1261.2115],
        [-701.3674, 1465.3973]
    ])

    actual_coords = session.artifacts["reduced_coordinates"]
    np.testing.assert_allclose(actual_coords, expected_coords, atol=1e-2)

    # Verify plot
    assert session.pending_figure is not None
    _verify_plot_points_and_labels(session.pending_figure, actual_coords, session.labels)


def test_reduce_dimensions_umap():
    session = _setup_session_with_dtm()
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_dimensions_tool.invoke({"method": "umap", "n_neighbors": 3, "random_state": 42})

    assert res["status"] == "success"
    assert res["method"] == "umap"

    # Expected UMAP coordinates from TEST_FIXTURE.md #9
    expected_coords = np.array([
        [6.6811, -6.8218],
        [6.9907, -7.2245],
        [7.7022, -7.3712],
        [10.2228, -6.2310],
        [9.8759, -5.6537],
        [9.3831, -6.5748],
        [9.2153, -5.8297],
        [8.5860, -7.2622]
    ])

    actual_coords = session.artifacts["reduced_coordinates"]
    np.testing.assert_allclose(actual_coords, expected_coords, atol=1e-2)

    # Verify plot
    assert session.pending_figure is not None
    _verify_plot_points_and_labels(session.pending_figure, actual_coords, session.labels)
