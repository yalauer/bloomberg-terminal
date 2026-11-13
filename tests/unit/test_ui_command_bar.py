"""Tests des Kommandozeilen-Parsers. Reine Funktionen, kein Dash noetig."""

from __future__ import annotations

import pytest

from btlite.ui.command_bar import (
    MNEMONICS,
    Command,
    CommandHistory,
    ParseError,
    format_command,
    parse_command,
    split_tickers,
    suggest_mnemonic,
)


class TestSplitTickers:
    def test_single(self):
        assert split_tickers("aapl") == ["AAPL"]

    def test_comma_separated(self):
        assert split_tickers("aapl,msft,nvda") == ["AAPL", "MSFT", "NVDA"]

    def test_semicolon_and_spaces(self):
        assert split_tickers("aapl; msft  nvda") == ["AAPL", "MSFT", "NVDA"]

    def test_duplicates_removed_stably(self):
        assert split_tickers("AAPL,MSFT,aapl") == ["AAPL", "MSFT"]

    def test_empty(self):
        assert split_tickers("") == []
        assert split_tickers("  ,  ; ") == []

    def test_keeps_dots_and_dashes(self):
        assert split_tickers("brk-b,bmw.de") == ["BRK-B", "BMW.DE"]


class TestParseCommand:
    @pytest.mark.parametrize("mnemonic", sorted(MNEMONICS))
    def test_every_mnemonic_parses(self, mnemonic):
        parsed = parse_command(f"AAPL {mnemonic} GO")
        assert isinstance(parsed, Command)
        assert parsed.mnemonic == mnemonic

    def test_with_go(self):
        parsed = parse_command("AAPL FA GO")
        assert parsed == Command(mnemonic="FA", tickers=("AAPL",), args=())

    def test_without_go(self):
        parsed = parse_command("AAPL FA")
        assert parsed == Command(mnemonic="FA", tickers=("AAPL",), args=())

    def test_ticker_list(self):
        parsed = parse_command("AAPL,MSFT,NVDA PORT GO")
        assert isinstance(parsed, Command)
        assert parsed.tickers == ("AAPL", "MSFT", "NVDA")
        assert parsed.mnemonic == "PORT"

    def test_lowercase(self):
        parsed = parse_command("aapl fa go")
        assert isinstance(parsed, Command)
        assert parsed.mnemonic == "FA"
        assert parsed.tickers == ("AAPL",)

    def test_extra_whitespace(self):
        parsed = parse_command("   AAPL    GP    GO   ")
        assert isinstance(parsed, Command)
        assert parsed.mnemonic == "GP"

    def test_arguments_after_mnemonic(self):
        parsed = parse_command("AAPL GP 5y GO")
        assert isinstance(parsed, Command)
        assert parsed.args == ("5y",)

    def test_arguments_without_go(self):
        parsed = parse_command("AAPL BT sma_cross")
        assert isinstance(parsed, Command)
        assert parsed.args == ("sma_cross",)

    def test_eqs_without_ticker(self):
        parsed = parse_command("EQS GO")
        assert isinstance(parsed, Command)
        assert parsed.mnemonic == "EQS"
        assert parsed.tickers == ()

    def test_path(self):
        parsed = parse_command("AAPL FA GO")
        assert isinstance(parsed, Command)
        assert parsed.path == "/fa"


class TestParseErrors:
    def test_empty_input(self):
        result = parse_command("")
        assert isinstance(result, ParseError)

    def test_whitespace_only(self):
        assert isinstance(parse_command("    "), ParseError)

    def test_go_alone(self):
        result = parse_command("GO")
        assert isinstance(result, ParseError)

    def test_unknown_mnemonic_has_suggestion(self):
        result = parse_command("AAPL FAA GO")
        assert isinstance(result, ParseError)
        assert result.suggestion == "FA"
        assert "FA" in result.user_message

    def test_unknown_mnemonic_typo_port(self):
        result = parse_command("AAPL PROT GO")
        assert isinstance(result, ParseError)
        assert result.suggestion == "PORT"

    def test_totally_unknown_lists_options(self):
        result = parse_command("AAPL ZZZZZZZZ GO")
        assert isinstance(result, ParseError)
        assert result.suggestion is None
        assert "DES" in result.user_message

    def test_mnemonic_requiring_ticker(self):
        result = parse_command("FA GO")
        assert isinstance(result, ParseError)
        assert "FA needs at least one ticker" in result.user_message

    def test_no_crash_on_odd_input(self):
        for raw in ("!!!", "1234", ",,,", "GO GO GO", "a" * 200):
            assert isinstance(parse_command(raw), Command | ParseError)


class TestSuggestMnemonic:
    def test_close_match(self):
        assert suggest_mnemonic("DESC") == "DES"

    def test_no_match(self):
        assert suggest_mnemonic("QQQQQQQ") is None

    def test_case_insensitive(self):
        assert suggest_mnemonic("fa") == "FA"


class TestFormatCommand:
    def test_roundtrip(self):
        parsed = parse_command("aapl,msft gp 5y")
        assert isinstance(parsed, Command)
        assert format_command(parsed) == "AAPL,MSFT GP 5y GO"

    def test_without_tickers(self):
        parsed = parse_command("EQS value")
        assert isinstance(parsed, Command)
        assert format_command(parsed) == "EQS value GO"


class TestCommandHistory:
    def test_add_and_previous(self):
        history = CommandHistory()
        history.add("AAPL FA GO")
        history.add("MSFT GP GO")
        assert history.previous() == "MSFT GP GO"
        assert history.previous() == "AAPL FA GO"
        assert history.previous() is None

    def test_next_returns_to_empty(self):
        history = CommandHistory()
        history.add("AAPL FA GO")
        history.add("MSFT GP GO")
        history.previous()
        history.previous()
        assert history.next() == "MSFT GP GO"
        assert history.next() is None

    def test_ignores_blank(self):
        history = CommandHistory()
        history.add("   ")
        assert history.entries == []

    def test_no_consecutive_duplicates(self):
        history = CommandHistory()
        history.add("AAPL FA GO")
        history.add("AAPL FA GO")
        assert history.entries == ["AAPL FA GO"]

    def test_respects_max_entries(self):
        history = CommandHistory(max_entries=3)
        for i in range(10):
            history.add(f"AAPL FA {i}")
        assert len(history.entries) == 3
        assert history.entries[-1] == "AAPL FA 9"

    def test_empty_history(self):
        history = CommandHistory()
        assert history.previous() is None
        assert history.next() is None
