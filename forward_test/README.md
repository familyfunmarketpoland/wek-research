# Forward Test

Ten katalog zawiera publiczny, papierowy forward test zamrozonego kandydata z
[`prereg_forward.json`](./prereg_forward.json). Workflow nie ma prawa czytac
ani listowac `data/holdout/`, nie sklada live orderow i uzywa tylko publicznego
OHLCV Binance pobieranego przez runner.

## Artefakty stanu

Aktywny workflow moze mutowac tylko te pliki: runner zmienia pierwsze trzy,
a renderer dashboardu zmienia czwarty:

- `forward_test/state.json`
- `forward_test/ledger.jsonl`
- `forward_test/head.sha256`
- `forward_test/dashboard/index.html`

Workflow commituje tylko powyzsza allowliste. Nie ma force-pusha, rebase ani
automatycznego rozwiazywania wyscigu: jezeli zdalny head zmieni sie po starcie
joba, push ma zakonczyc sie bledem.

## Kontrakt dashboardu

`forward_test/dashboard.py` renderuje statyczny HTML bez zewnetrznego JS. Moduł:

```bash
python -m forward_test.dashboard
```

Czyta zawsze `prereg_forward.json`, a `state.json` traktuje jako opcjonalny. To
pozwala utrzymac strone GitHub Pages jeszcze przed pierwszym eligible candle.

Renderer obsluguje dwa stany:

1. Stan poczatkowy: brak `state.json`, status `PRE_REGISTERED_NOT_STARTED`,
   licznik `0/30`, brak equity curve i brak trades.
2. Stan aktywny lub terminalny: `state.json` obecny, status `RUNNING`, `PASS`,
   `FAIL` albo `UNDERPOWERED`, plus wykres, benchmark buy-and-hold, lista
   zamknietych transakcji oraz sekcja hashy/audytu.

Minimalny kontrakt `state.json` oczekiwany przez dashboard:

```json
{
  "status": "RUNNING",
  "updated_at_utc": "2026-07-25T04:17:00Z",
  "first_eligible_open_utc": "2026-07-25T00:00:00Z",
  "underpowered_deadline_utc": "2027-07-25T00:00:00Z",
  "last_processed_open_utc": "2026-07-26T08:00:00Z",
  "closed_trades_count": 3,
  "position": -1,
  "max_drawdown": 0.0312,
  "performance": {
    "initial_equity": 1.0,
    "net_return": 0.0241,
    "per_trade_sharpe": 0.42,
    "equity_curve": [
      {"open_utc": "2026-07-25T00:00:00Z", "equity": 1.0},
      {"open_utc": "2026-07-25T04:00:00Z", "equity": 1.0031}
    ]
  },
  "benchmark": {
    "name": "SOL/USDT buy-and-hold",
    "equity": 1.0110,
    "net_return": 0.0110,
    "equity_curve": [
      {"open_utc": "2026-07-25T00:00:00Z", "equity": 0.9985},
      {"open_utc": "2026-07-25T04:00:00Z", "equity": 1.0012}
    ]
  },
  "closed_trades": [
    {
      "entry_open_utc": "2026-07-25T00:00:00Z",
      "exit_open_utc": "2026-07-26T08:00:00Z",
      "side": "short",
      "bars": 8,
      "net_return": 0.0134,
      "exit_reason": "hold_expiry"
    }
  ],
  "hashes": {
    "ledger_head": "...",
    "prereg_sha256": "...",
    "parameter_sha256": "..."
  },
  "audit": {
    "latest_ledger_event_sha256": "..."
  }
}
```

Dashboard toleruje rowniez plaski wariant z polami takimi jak
`current_equity`, `buy_hold_equity`, `net_return` albo `equity_curve`, jesli
runner zachowa ta sama semantyke. Domyslnym kontraktem integracyjnym powinien
jednak pozostac powyzszy ksztalt z `performance`, `benchmark` i `hashes`.

## Workflow GitHub Actions

Workflow `.github/workflows/forward.yml`:

- startuje co 4h o `17 */4 * * *` UTC i z `workflow_dispatch`
- serializuje wykonanie przez `concurrency` bez anulowania runu w toku
- sprawdza czysty worktree i zapamietuje zdalny head przed runnerem
- uruchamia `python -m forward_test.runner`
- renderuje `python -m forward_test.dashboard`
- commituje tylko allowliste plikow forward testu
- publikuje `forward_test/dashboard/` na GitHub Pages

## Uruchomienie repo

1. Skonfiguruj remote GitHub, wypchnij branch zawierajacy workflow i ustaw go
   jako default branch. Dla nowego remote wykonaj:

   ```bash
   git remote add origin git@github.com:OWNER/REPO.git
   git push -u origin BRANCH
   ```

   Jezeli `origin` juz istnieje, zweryfikuj go przez `git remote -v` zamiast
   dodawac ponownie. Workflow odmawia pracy na refie typu tag.
2. W `Settings -> Actions -> General -> Workflow permissions` wybierz
   `Read and write permissions`, aby izolowany job `commit` mogl wypchnac
   allowlistowane artefakty stanu. Job `trade` pozostaje tylko do odczytu.
3. W `Settings -> Pages -> Build and deployment -> Source` wybierz
   `GitHub Actions`.
4. Uruchom workflow `Forward Test` recznie przez `Actions -> Forward Test ->
   Run workflow` na branchu albo poczekaj na harmonogram `17 */4 * * *` UTC.

Job `trade` przed uruchomieniem runnera sprawdza czysty checkout i zapamietuje
head zdalnego brancha. Izolowany job `commit` ponownie pobiera remote i odmawia
zapisu, jezeli head zmienil sie w czasie przebiegu. Workflow nie wykonuje
force-pusha ani rebase.

## Sprawdzanie statusu

Status operator sprawdza w dwoch miejscach:

- opublikowana strona GitHub Pages pokazuje status `RUNNING`, `PASS`, `FAIL`
  albo `UNDERPOWERED`, licznik zamknietych transakcji i dane audytowe
- kanoniczna wartosc jest w polu `status` pliku `forward_test/state.json` na
  zdalnym branchu; Pages jest tylko statyczna prezentacja tego stanu

Po pierwszym uruchomieniu sprawdz takze wynik jobow `trade`, `commit` i `deploy`
w zakladce Actions. Kazdy poprawny przebieg ze statusem `RUNNING` aktualizuje
audit timestamp dashboardu i tworzy commit, nawet gdy nie zamknieto nowej
transakcji. Brak nowego commitu jest oczekiwany dopiero po terminalnym no-op
(`PASS`, `FAIL` albo `UNDERPOWERED`) albo gdy job zakonczyl sie bledem przed
commitem.

## NIE WOLNO

- Nie zmieniaj `forward_test/prereg_forward.json` po rozpoczeciu testu.
- Nie zmieniaj zamrozonych parametrow kandydata ani ich hashy.
- Nie edytuj, nie usuwaj i nie odtwarzaj recznie `forward_test/state.json`,
  `forward_test/ledger.jsonl` ani `forward_test/head.sha256`.
- Nie zmieniaj kodu zamrozonej strategii uzywanego przez aktywny forward test.
- Nie czytaj ani nie listuj `data/holdout/` zadnym narzedziem forward testu,
  researchowym ani diagnostycznym.
- Nie skladaj live orderow i nie dodawaj sekretow exchange; runner ma korzystac
  tylko z publicznego OHLCV i papierowego PnL.
- Nie restartuj, nie resetuj i nie rerunuj testu po terminalnym `PASS`, `FAIL`
  albo `UNDERPOWERED`; terminalny stan pozostaje niezmienny.
