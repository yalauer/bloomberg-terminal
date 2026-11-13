# Screenshots

Dieses Verzeichnis hält die Bildschirmfotos der sechs Seiten. Es ist bis auf diese Datei
noch leer.

## Was hier hingehört

Je ein Bild pro Mnemonic, benannt nach dem Mnemonic in Kleinschreibung:

| Datei | Seite | Aufgenommen mit |
|-------|-------|-----------------|
| `des.png` | Description | `AAPL DES GO` |
| `gp.png` | Graph Price | `AAPL,MSFT,NVDA GP 5y GO` |
| `fa.png` | Financial Analysis | `AAPL,MSFT FA annual GO` |
| `eqs.png` | Equity Screening | `EQS value GO` |
| `bt.png` | Backtest | `AAPL BT sma_cross GO` |
| `port.png` | Portfolio & Risk | `AAPL,MSFT,KO,JNJ PORT max_sharpe GO` |

Optional `overview.png`: die Startansicht mit Kommandozeile und Navigation, als erstes
Bild im README.

## Anforderungen an die Aufnahme

- **PNG**, Breite 1600 px, Fenster maximiert. Kein JPEG — die dünnen Linien des Themes
  und der Text auf dunklem Grund werden davon unsauber.
- **Offline-Modus** (der Default). Damit sind die Bilder reproduzierbar und zeigen für
  jeden dieselben Zahlen.
- **Die Herkunftszeile muss sichtbar sein.** Sie weist Quelle, Zeitstempel und Verzug
  aus. Ein Screenshot ohne sie legt nahe, die Zahlen seien aktuell.
- **Der Footer mit dem Haftungshinweis muss sichtbar sein** oder das Bild ist so
  beschnitten, dass kein Eindruck einer Anlageempfehlung entsteht.
- Keine persönlichen Daten im Bild: keine Browser-Lesezeichen, keine anderen Tabs, keine
  Pfade mit Klarnamen in der Adresszeile.

## Aufnahme

```bash
.venv\Scripts\python.exe -m btlite
```

Dann `http://127.0.0.1:8050` öffnen, das jeweilige Kommando in die Zeile tippen und mit
Enter auslösen.
