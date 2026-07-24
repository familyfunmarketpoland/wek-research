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

## Prerejestrowany workflow H1–H6

Na świeżym, pełnym cache administracyjny split wykonuje się dokładnie raz,
przed obliczeniem jakichkolwiek wyników H1–H6:

```bash
./.venv/bin/python scripts/split_holdout.py --data-dir data --cutoff-months 6
```

Po utworzeniu `data/research/`, `data/holdout/` i manifestu uruchamia się
wyłącznie fazę research:

```bash
./.venv/bin/python run_confirmatory.py research
```

Jeżeli i tylko jeżeli komenda zwróci `WINNER_FROZEN` oraz zapisze zamrożony
artefakt zwycięzcy, wolno wykonać dokładnie jeden przebieg holdoutu:

```bash
./.venv/bin/python run_confirmatory.py holdout
```

Runner sam odrzuca fazę holdout bez poprawnego zamrożonego zwycięzcy.
Administracyjnego splitu nie wolno powtarzać po rozpoczęciu prerejestrowanego przebiegu.

## Artefakty

Typowy wynik prerejestrowanego przebiegu:

```text
report2.md
configs/pre_registered.json
research_lab/                 # reużywalne sygnały, silnik, statystyka i guard danych
results2/
  candidates.csv
  hypothesis_summary.csv
  folds.csv
  permutation_best_sharpe.csv
  study_decision.json
```

`report2.md` jest generowany z rzeczywistego uruchomienia badania, a nie składany ręcznie.

## Testy

```bash
./.venv/bin/python -m pytest
```

## Ostrzeżenia

- Nie uruchamiaj ponownie `run_confirmatory.py holdout`; holdout jest warunkowy i dokładnie jednorazowy.
- Nie kieruj starego ani nowego kodu researchowego na `data/holdout/`; research czyta wyłącznie `data/research/` przez guard.
- Nie mieszaj artefaktów z poprzedniego `results/` ani `report.md` z nowym prerejestrowanym przebiegiem.

## Założenia

- koszty są modelowane jako `fee_rate + slippage_rate` per strona
- `strategy_final.get_signal` czyta domyślnie `results/final_config.json`
- `strategy_final.get_signal` zwraca docelową pozycję dla następnego otwarcia (`next open`), czyli decyzja z `t` jest realizowana na `t+1`
- `strategy_final.get_signal` zależy od pełnego i poprawnego `final_config` oraz od ramek z kolumnami `open`, `high`, `low`, `close`, `volume`
- shorty są syntetycznymi shortami 1x, bez borrow i funding
- confirmatory walk-forward ocenia zamrożone kandydatury na zszytym OOS 12m/3m/3m, bez adaptacyjnego rozszerzania siatek
- dla tego projektu seed badania wynosi `42`

## Uwagi o reprodukowalności

- pełna reprodukcja fazy research wymaga tego samego manifestu danych, `configs/pre_registered.json`, kodu i seedu
- `report2.md` oraz `results2/` dotyczą badania H1–H6; `report.md` i `results/` należą do wcześniejszego badania WEK
- holdout nie jest przebiegiem reprodukcyjnym: claim jednorazowego dostępu jest spalany przed odczytem
