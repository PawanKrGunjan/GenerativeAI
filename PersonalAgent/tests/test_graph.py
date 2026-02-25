import pytest
from Agent.graph import build_graph


@pytest.mark.skip(reason="LangGraph checkpointer integration not unit-test friendly")
def test_build_graph_does_not_crash(temp_db_conn, monkeypatch):
    """
    Unit test for build_graph.
    External dependencies mocked.
    """

    def fake_semantic_search(conn, query, top_k=7):
        return [
            "Source: test.txt\n\nPawan joined Metro Infrasys in August 2024.",
            "Source: test.txt\n\nPawan joined Geeksforgeeks in December 2022.",
        ]

    monkeypatch.setattr("Agent.graph.semantic_search", fake_semantic_search)

    graph = build_graph(temp_db_conn, None)

    assert graph is not None
