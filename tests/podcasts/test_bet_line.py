from engine_common.podcasts.render import EngineEdge, bet_line, edge_text, nearest_edge


def test_bet_line_reads_name_record_selection_price_edge() -> None:
    line = bet_line("Joel Meyer", "12-9 graded", "MTL ML", 120, "engine +3.1%")
    assert line == "<li><b>Joel Meyer</b> (12-9 graded). <b>MTL ML +120</b> (engine +3.1%)</li>"
    assert "-111</b>" in bet_line("A", "1-0", "MIN ML", -111, "no engine edge")
    assert "MIN ML</b>" in bet_line("A", "1-0", "MIN ML", None, "no engine edge")


def test_edge_is_at_the_hosts_number_or_names_the_nearest() -> None:
    priced = [(-1.5, 0.02), (1.5, -0.01)]
    assert nearest_edge(-1.5, priced) == EngineEdge(0.02)
    assert nearest_edge(-2.0, priced) == EngineEdge(0.02, -1.5)
    assert nearest_edge(None, [(None, 0.04)]) == EngineEdge(0.04)
    assert nearest_edge(5.5, []) is None
    assert edge_text(EngineEdge(0.02, -1.5), "game_pl") == "engine +2.0% at -1.5"
    assert edge_text(EngineEdge(-0.013, 6.0), "game_total") == "engine -1.3% at 6"
    assert edge_text(None, "game_ml", "my number is -140") == "my number is -140; no engine edge"
