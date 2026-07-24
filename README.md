# WEK Research

Reprodukowalny projekt badawczy dla wskaźnika WEK, backtestu i walk-forward study na danych Binance zapisanych lokalnie w Parquet.

## Wymagania

- Python 3.11
- lokalne środowisko `.venv`
- dane OHLCV pobrane z publicznego Binance i zapisane w `data/`

## Instalacja

```bash
cd "/Users/edwin/Documents/Inne projekty/wek-research"
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Pobranie danych z Binance

Domyślny downloader pobiera publiczne OHLCV z Binance, uzupełnia cache i zapisuje pliki do `data/{base}_{quote}_{timeframe}.parquet`.

```bash
./.venv/bin/python -m data_pipeline
./.venv/bin/python -m data_pipeline --symbols BTC/USDT ETH/USDT --timeframes 1h 4h
```

Domyślne instrumenty i interwały to `BTC/USDT`, `ETH/USDT`, `SOL/USDT` oraz `1h`, `4h`, `1d`.

## Testy

```bash
./.venv/bin/python -m pytest
```

## Study szybkie i pełne

```bash
./.venv/bin/python run_research.py --quick
./.venv/bin/python run_research.py
```

Opcjonalnie można ograniczyć liczbę datasetów:

```bash
./.venv/bin/python run_research.py --quick --max-datasets 1
```

## Artefakty

Typowy wynik pełnego uruchomienia:

```text
charts/
results/
  aggregate_leaderboard.csv
  ablation.csv
  assumptions_manifest.json
  final_config.json
  ...
report.md
```

`report.md` jest generowany z rzeczywistego uruchomienia badania, a nie składany ręcznie.

## Założenia

- koszty są modelowane jako `fee_rate + slippage_rate` per strona
- `strategy_final.get_signal` czyta domyślnie `results/final_config.json`
- `strategy_final.get_signal` zwraca docelową pozycję dla następnego otwarcia (`next open`), czyli decyzja z `t` jest realizowana na `t+1`
- `strategy_final.get_signal` zależy od pełnego i poprawnego `final_config` oraz od ramek z kolumnami `open`, `high`, `low`, `close`, `volume`
- shorty są syntetycznymi shortami 1x, bez borrow i funding
- walk-forward używa wyłącznie danych z okna treningowego do selekcji parametrów, a sygnały są liczone przy zachowaniu przyczynowości
- dla tego projektu seed badania wynosi `42`

## Uwagi o reprodukowalności

- pełna reprodukcja wymaga tych samych danych wejściowych, tego samego `results/final_config.json` i tego samego seedu
- `quick` to deterministyczny smoke study, nie zamiennik pełnego uruchomienia
- `report.md` i pliki w `results/` odzwierciedlają konkretny przebieg badania, więc mogą się różnić po ponownym uruchomieniu
