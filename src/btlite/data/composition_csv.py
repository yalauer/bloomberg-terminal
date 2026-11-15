"""Read user-supplied equity fund holdings without network access."""

import base64
import binascii
import csv
import io
from collections import defaultdict

from btlite.core.clock import utc_now
from btlite.models.common import DataSource, Provenance, normalize_ticker
from btlite.models.composition import Constituent, FundComposition


def read_upload(contents: str | None) -> dict[str, FundComposition]:
    """Parse UTF-8 CSV: fund,ticker,name,weight_percent,country,sector.

    Weights are percentages of total fund assets (5 means 5%), never shares
    of a top-ten subset. Only equity constituents are supported by this import.
    """
    if not contents:
        return {}
    if len(contents) > 4_000_000:
        raise ValueError("CSV ist zu groß (maximal ca. 3 MB).")
    try:
        raw = base64.b64decode(contents.split(",", 1)[1], validate=True).decode("utf-8-sig")
    except (ValueError, IndexError, UnicodeError, binascii.Error) as exc:
        raise ValueError("CSV muss eine UTF-8-Datei sein.") from exc
    reader = csv.DictReader(io.StringIO(raw))
    if not {"fund", "ticker", "weight_percent"}.issubset(reader.fieldnames or []):
        raise ValueError("CSV benötigt fund,ticker,weight_percent (Komma als Trennzeichen).")
    grouped: dict[str, list[Constituent]] = defaultdict(list)
    for number, row in enumerate(reader, start=2):
        try:
            fund = normalize_ticker(row["fund"].strip())
            grouped[fund].append(
                Constituent(
                    ticker=row["ticker"].strip(),
                    isin=(row.get("isin") or "").strip() or None,
                    name=(row.get("name") or "").strip() or None,
                    weight=float(row["weight_percent"]) / 100,
                    country=(row.get("country") or "").strip() or None,
                    sector=(row.get("sector") or "").strip() or None,
                )
            )
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError(f"Ungültige CSV-Zeile {number}: {exc}") from exc
    if not grouped:
        raise ValueError("Die CSV enthält keine Positionen.")
    return {
        ticker: FundComposition(
            ticker=ticker,
            holdings=holdings,
            provenance=Provenance(
                source=DataSource.USER,
                fetched_at=utc_now(),
                note="CSV-Import; Zeitpunkt = Importzeit, Bestandsdatum vom Nutzer zu prüfen.",
            ),
        )
        for ticker, holdings in grouped.items()
    }
