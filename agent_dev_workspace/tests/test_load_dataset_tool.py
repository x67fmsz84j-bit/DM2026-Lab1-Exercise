from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools


def test_load_dataset_tool():
    # Create a single session state instance
    session = SessionState()

    # Load premade tools bound to this session
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )

    # Invoke tool on the test fixture
    result = load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    # Known Answer #1 assertions:
    # 1. Total document count returned by the tool
    assert result["n_documents"] == 8

    # 2. Per-category counts returned by the tool
    assert result["counts_per_category"] == {"catA": 4, "catB": 4}

    # 3. Session state update validation
    assert session.dataframe is not None
    assert len(session.dataframe) == 8
