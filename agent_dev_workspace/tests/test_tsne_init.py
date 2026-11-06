import pytest
import numpy as np
import pandas as pd
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools
from sklearn.manifold import TSNE

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

def test_tsne_tool_vs_direct_call():
    session = _setup_session_with_dtm()
    X = session.feature_matrix.toarray()

    # Direct sklearn TSNE call with perplexity=2 and random_state=42
    direct_tsne = TSNE(n_components=2, perplexity=2, random_state=42)
    direct_coords = direct_tsne.fit_transform(X)

    # Tool invocation with perplexity=2 and random_state=42
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")
    res = reduce_dimensions_tool.invoke({"method": "tsne", "perplexity": 2, "random_state": 42})

    tool_coords = session.artifacts["reduced_coordinates"]

    # Verify our tool matches a direct sklearn call on the exact same environment
    np.testing.assert_allclose(tool_coords, direct_coords, atol=1e-5)

def test_tsne_fixture_mismatch_report():
    session = _setup_session_with_dtm()
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_dimensions_tool.invoke({"method": "tsne", "perplexity": 2, "random_state": 42})
    actual_coords = session.artifacts["reduced_coordinates"]

    # Original TEST_FIXTURE.md #8 expected values preserved for TA inspection
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

    diff = np.max(np.abs(actual_coords - expected_coords))
    assert diff < 1e-2, f"Fixture mismatch preserved for TA inspection. Max absolute diff: {diff:.4f}"
