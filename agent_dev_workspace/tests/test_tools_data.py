import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools as make_data_tools
from tools_dtm import make_tools as make_dtm_tools

def test_check_missing_tool():
    # Single session instance used throughout the test
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load sample_fixture.csv
    load_res = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    assert load_res["n_documents"] == 8

    # Load check_missing_tool from tools_data
    data_tools = make_data_tools(session)
    check_missing_tool = next(t for t in data_tools if t.name == "check_missing_tool")

    # Run check_missing_tool
    result = check_missing_tool.invoke({})

    # Assert against Known Answer 2 in TEST_FIXTURE.md
    assert result["status"] == "success"
    assert result["total_rows"] == 8
    assert result["n_missing"] == 1
    assert result["has_missing"] is True
    assert result["missing_indices_preview"] == [7]

    # Check stored full report DataFrame
    stored_result = session.results_store[result["result_id"]]
    report_df = stored_result["full_report"]

    assert list(report_df.columns) == ["row_index", "text"]
    assert report_df["row_index"].tolist() == [7]


def test_check_duplicates_tool():
    # Fresh session instance per test function
    session = SessionState()

    # Load dataset
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM first to populate feature_matrix, feature_names, and vectorizer artifact
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Save exact copies of original dataframe, matrix, features, and labels
    initial_df_copy = session.dataframe.copy()
    initial_matrix_copy = session.feature_matrix.copy()
    initial_features = list(session.feature_names)
    initial_labels_copy = list(session.labels)

    # Load check_duplicates_tool
    data_tools = make_data_tools(session)
    check_duplicates_tool = next(t for t in data_tools if t.name == "check_duplicates_tool")

    # 1. Test drop=False (Known Answer 3)
    res_check = check_duplicates_tool.invoke({"drop": False})

    assert res_check["status"] == "success"
    assert res_check["total_rows_before"] == 8
    assert res_check["total_rows_after"] == 8
    assert res_check["n_duplicates_found"] == 1
    assert res_check["dropped"] is False

    # Confirm session.dataframe and DTM remain completely unchanged after drop=False
    pd.testing.assert_frame_equal(session.dataframe, initial_df_copy)
    assert np.array_equal(session.feature_matrix.toarray(), initial_matrix_copy.toarray())
    assert session.feature_names == initial_features
    assert list(session.labels) == initial_labels_copy

    # 2. Test drop=True (Known Answer 3 dropping with keep=False)
    res_drop = check_duplicates_tool.invoke({"drop": True})

    assert res_drop["status"] == "success"
    assert res_drop["total_rows_before"] == 8
    assert res_drop["total_rows_after"] == 6  # Both Row 5 and Row 6 removed because keep=False
    assert res_drop["n_duplicates_found"] == 1
    assert res_drop["dropped"] is True
    assert len(session.dataframe) == 6

    # Verify remaining texts and category_names match original rows [0, 1, 2, 3, 4, 7]
    expected_indices = [0, 1, 2, 3, 4, 7]
    expected_texts = initial_df_copy.loc[expected_indices, "text"].tolist()
    expected_categories = initial_df_copy.loc[expected_indices, "category_name"].tolist()

    assert session.dataframe["text"].tolist() == expected_texts
    assert session.dataframe["category_name"].tolist() == expected_categories
    assert list(session.labels) == expected_categories

    # Verify outdated DTM, feature names, and vectorizer artifact were cleared
    assert session.feature_matrix is None
    assert session.feature_names is None
    assert "count_vectorizer" not in session.artifacts


def test_sample_data_tool():
    # Fresh session instance per test function
    session = SessionState()

    # Load dataset
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM first to populate feature_matrix, feature_names, and vectorizer artifact
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Save initial dataframe copy before sampling
    initial_df_copy = session.dataframe.copy()

    # Load sample_data_tool
    data_tools = make_data_tools(session)
    sample_data_tool = next(t for t in data_tools if t.name == "sample_data_tool")

    # Run sample_data_tool with n=4, random_state=42 (Known Answer 13)
    res = sample_data_tool.invoke({"n": 4, "random_state": 42})

    # Assert returned dict structure and numbers against Known Answer 13
    assert res["status"] == "success"
    assert res["total_rows_before"] == 8
    assert res["total_rows_after"] == 4
    expected_indices = [1, 5, 0, 7]
    assert res["sampled_indices_preview"] == expected_indices

    # Compare actual session.dataframe, labels, and full report against expected rows [1, 5, 0, 7]
    expected_df = initial_df_copy.loc[expected_indices]
    pd.testing.assert_frame_equal(session.dataframe, expected_df)

    expected_labels = initial_df_copy.loc[expected_indices, "category_name"].tolist()
    assert list(session.labels) == expected_labels

    stored_result = session.results_store[res["result_id"]]
    full_report_df = stored_result["full_report"]
    pd.testing.assert_frame_equal(full_report_df, expected_df)

    # Verify outdated DTM matrix, feature names, and vectorizer artifact were cleared
    assert session.feature_matrix is None
    assert session.feature_names is None
    assert "count_vectorizer" not in session.artifacts


def test_tokenize_tool():
    # Fresh session instance per test function
    session = SessionState()

    # Load dataset
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Save copy of initial dataframe for verifying text, row order, and indices remain unchanged
    initial_df_copy = session.dataframe.copy()

    # Load tokenize_tool
    data_tools = make_data_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")

    # Run tokenize_tool
    res = tokenize_tool.invoke({})

    # Assert return dict values against Known Answer 10 in TEST_FIXTURE.md
    assert res["status"] == "success"
    assert res["total_documents"] == 8
    assert res["total_tokens"] == 27
    assert res["avg_tokens_per_doc"] == pytest.approx(3.375, abs=1e-4)
    assert res["vocab_size"] == 5

    # Known Answer 10 expected token lists per document
    expected_unigrams = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        [],
    ]

    # Verify session.dataframe["unigrams"] and session.artifacts["tokenized_data"]
    assert session.dataframe["unigrams"].tolist() == expected_unigrams
    assert session.artifacts["tokenized_data"] == expected_unigrams

    # Verify original text, row order, and indices remain identical
    pd.testing.assert_series_equal(session.dataframe["text"], initial_df_copy["text"])
    assert session.dataframe.index.tolist() == initial_df_copy.index.tolist()
    assert list(session.dataframe.columns) == list(initial_df_copy.columns) + ["unigrams"]


def test_list_files_tool():
    session = SessionState()
    data_tools = make_data_tools(session)
    list_files_tool = next(t for t in data_tools if t.name == "list_files_tool")

    # 1. Known Answer 11: list_files_tool(subdirectory="newdataset")
    res = list_files_tool.invoke({"subdirectory": "newdataset"})

    assert res["status"] == "success"
    assert res["subdirectory"] == "newdataset"
    assert res["directories"] == [".ipynb_checkpoints"]
    expected_files = sorted([
        "Reddit-stock-sentiment.csv",
        "Untitled.ipynb",
        "Untitled1.ipynb",
        "Untitled2.ipynb",
    ])
    assert sorted(res["files"]) == expected_files

    # 2. Test missing folder handling
    res_missing = list_files_tool.invoke({"subdirectory": "non_existent_folder_xyz"})
    assert res_missing["status"] == "error"
    assert "does not exist" in res_missing["message"]

    # 3. Test path outside repository root security check
    res_outside = list_files_tool.invoke({"subdirectory": "../../"})
    assert res_outside["status"] == "error"
    assert "outside the repository root" in res_outside["message"]


def test_inspect_data_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Append one row with long text (> 100 characters) to test truncation cutoff
    long_text = "word " * 30  # 150 characters
    new_row = pd.DataFrame([{
        "text": long_text,
        "category": 0,
        "category_name": "catA"
    }])
    session.dataframe = pd.concat([session.dataframe, new_row], ignore_index=True)

    initial_df_copy = session.dataframe.copy()

    data_tools = make_data_tools(session)
    inspect_data_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    # 1. Test requesting n=10 (should be capped at 5)
    res_cap = inspect_data_tool.invoke({"n": 10})

    assert res_cap["status"] == "success"
    assert res_cap["n_inspected"] == 5
    assert len(res_cap["rows_preview"]) == 5

    # Check actual rows 0 through 4 previewed content
    expected_preview = [
        {"row_index": 0, "text": "always alpha alpha alpha beta", "category": "catA"},
        {"row_index": 1, "text": "always alpha alpha beta", "category": "catA"},
        {"row_index": 2, "text": "always alpha beta", "category": "catA"},
        {"row_index": 3, "text": "always gamma gamma gamma delta", "category": "catB"},
        {"row_index": 4, "text": "always gamma gamma delta", "category": "catB"},
    ]
    assert res_cap["rows_preview"] == expected_preview

    # 2. Test inspecting row 8 (which contains long_text > 100 chars)
    # Re-order dataframe so long text row is at index 0 to inspect it via head(1)
    session.dataframe = pd.concat([new_row, initial_df_copy], ignore_index=True)
    res_long = inspect_data_tool.invoke({"n": 1})

    preview_text = res_long["rows_preview"][0]["text"]
    assert len(preview_text) == 100
    assert preview_text.endswith("...")
    assert preview_text == long_text[:97] + "..."

    # 3. Verify session.dataframe data, row order, and indices remain completely unchanged after inspection
    session.dataframe = initial_df_copy.copy()
    inspect_data_tool.invoke({"n": 5})
    pd.testing.assert_frame_equal(session.dataframe, initial_df_copy)


def test_describe_data_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    initial_df_copy = session.dataframe.copy()

    data_tools = make_data_tools(session)
    describe_data_tool = next(t for t in data_tools if t.name == "describe_data_tool")

    res = describe_data_tool.invoke({})

    assert res["status"] == "success"

    # Known Answer 18 values from TEST_FIXTURE.md
    # Overall: count=8, mean=19.875, std=9.4330, min=0, 25%=17.75, 50%=20.5, 75%=25.25, max=30
    overall = res["overall_stats"]
    assert overall["count"] == 8.0
    assert overall["mean"] == pytest.approx(19.875, abs=1e-3)
    assert overall["std"] == pytest.approx(9.4330, abs=1e-3)
    assert overall["min"] == 0.0
    assert overall["25%"] == pytest.approx(17.75, abs=1e-3)
    assert overall["50%"] == pytest.approx(20.5, abs=1e-3)
    assert overall["75%"] == pytest.approx(25.25, abs=1e-3)
    assert overall["max"] == 30.0

    # catA: count=4, mean=17.25, std=12.5, min=0, 25%=12.75, 50%=20.0, 75%=24.5, max=29
    catA = res["per_category_stats"]["catA"]
    assert catA["count"] == 4.0
    assert catA["mean"] == pytest.approx(17.25, abs=1e-3)
    assert catA["std"] == pytest.approx(12.5, abs=1e-3)
    assert catA["min"] == 0.0
    assert catA["25%"] == pytest.approx(12.75, abs=1e-3)
    assert catA["50%"] == pytest.approx(20.0, abs=1e-3)
    assert catA["75%"] == pytest.approx(24.5, abs=1e-3)
    assert catA["max"] == 29.0

    # catB: count=4, mean=22.5, std=5.7446, min=18, 25%=18.0, 50%=21.0, 75%=25.5, max=30
    catB = res["per_category_stats"]["catB"]
    assert catB["count"] == 4.0
    assert catB["mean"] == pytest.approx(22.5, abs=1e-3)
    assert catB["std"] == pytest.approx(5.7446, abs=1e-3)
    assert catB["min"] == 18.0
    assert catB["25%"] == pytest.approx(18.0, abs=1e-3)
    assert catB["50%"] == pytest.approx(21.0, abs=1e-3)
    assert catB["75%"] == pytest.approx(25.5, abs=1e-3)
    assert catB["max"] == 30.0

    # Verify session.pending_figure is a Matplotlib figure with "catA" and "catB" x-axis labels
    fig = session.pending_figure
    assert isinstance(fig, plt.Figure)

    ax = fig.axes[0]
    x_tick_labels = [label.get_text() for label in ax.get_xticklabels()]
    assert set(x_tick_labels) == {"catA", "catB"}

    # Verify session.dataframe remains unchanged
    pd.testing.assert_frame_equal(session.dataframe, initial_df_copy)
