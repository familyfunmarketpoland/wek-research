# Raport potwierdzajacy H1-H6

## Decyzja
Wynik: NO_EDGE. Zaden kandydat nie spelnil kompletu regul prerejestracji. Holdout nie zostal odczytany i nie wolno go odczytywac bez zamrozonego zwyciezcy.

## Zakres prerejestracji
- Commit prerejestracji: `96bebdf`
- Fingerprint konfiguracji: `7e77ff5103e7b8e82a2d3ae3d16db0afebc9cfa3c105985ddf3fe35b1fd84990`
- Fingerprint manifestu danych: `9f964de5ee113ff3ab2d097edecfe3874411c867e1b6693d43548a179c093bb3`
- Liczba kandydatow: `186`
- Minimalna liczba transakcji OOS: `30`
- Rodzina testow wielokrotnych: `186` kandydatow, `500` permutacji
- WFO: kazdy kandydat jest staly, bez adaptacyjnego wyboru parametrow; scoring dotyczy stitched OOS.
- Egzekucja: syntetyczna pozycja 1x; 0.15% kosztu na strone; bez funding, borrow i market impact.
- H1 caveat: sygnal wolumenowy korzysta z proxy wolumenu dostepnego w OHLCV, bez danych order-book/tape.

## Zamrozone uzasadnienia ekonomiczne
- H1 — Volume Shock Mean Reversion: Capitulation-style down moves on abnormal volume may overshoot and mean revert over short horizons.
- H2 — Realized-Volatility Compression Donchian Breakout: Breakouts from realized-volatility compression regimes may persist once price escapes a 20-bar channel.
- H3 — Session Boundary Drift: Systematic order-flow transitions around major session boundaries may induce directional drift from one session open to the matching session close.
- H4 — Run-Length Continuation Or Reversal: Consecutive same-sign returns can indicate either momentum continuation or exhaustion reversal, depending on regime.
- H5 — NR7 Delayed Breakout Confirmation: Narrow-range compression can precede expansion, but requiring next-bar close confirmation reduces false triggers.
- H6 — Low-Entropy Realized-Volatility Compression Donchian Breakout: Breakouts may be strongest when realized-volatility compression is paired with unusually concentrated volume states, indicating latent imbalance before expansion.

## Hipotezy
| Hipoteza | Verdict | candidate_id | symbol | timeframe | params | OOS total return | annualized Sharpe | DSR | familywise permutation p | trades | benchmark return | annualized benchmark Sharpe |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| H1 | FAIL | H1\|btcusdt\|4h\|hold_bars-6 | BTC/USDT | 4h | hold_bars=6.0 | 0.123657 | 0.611469 | 2.30164e-08 | 1 | 35 | 0.354992 | 0.667268 |
| H2 | FAIL | H2\|solusdt\|4h\|side-short\|hold_bars-20 | SOL/USDT | 4h | side=short, hold_bars=20.0 | 0.299059 | 0.79919 | 1.26949e-07 | 0.994012 | 23 | -0.265569 | 0.158883 |
| H3 | FAIL | H3\|btcusdt\|1h\|side-long\|session_utc-Asia | BTC/USDT | 1h | side=long, session_utc=Asia | -0.775689 | -4.01186 | NA | 1 | 549 | 0.356347 | 0.66706 |
| H4 | FAIL | H4\|ethusdt\|4h\|mode-continuation\|streak_length-5 | ETH/USDT | 4h | mode=continuation, hold_bars=1.0, streak_length=5.0 | 0.0393592 | 0.260468 | 1.31829e-09 | 1 | 93 | -0.143408 | 0.205167 |
| H5 | FAIL | H5\|btcusdt\|1h\|side-long\|hold_bars-1 | BTC/USDT | 1h | side=long, hold_bars=1.0 | -0.8844 | -15.1485 | NA | 1 | 680 | 0.356347 | 0.66706 |
| H6 | FAIL | H6\|btcusdt\|4h\|side-long\|hold_bars-10 | BTC/USDT | 4h | side=long, hold_bars=10.0 | 0.0470721 | 0.366727 | 6.54734e-09 | 1 | 21 | 0.354992 | 0.667268 |

- H1: FAIL; kandydaci=18, eligible=15, passing=0, najblizszy=`H1|btcusdt|4h|hold_bars-6`.
- H2: FAIL; kandydaci=36, eligible=23, passing=0, najblizszy=`H2|solusdt|4h|side-short|hold_bars-20`.
- H3: FAIL; kandydaci=12, eligible=12, passing=0, najblizszy=`H3|btcusdt|1h|side-long|session_utc-Asia`.
- H4: FAIL; kandydaci=48, eligible=48, passing=0, najblizszy=`H4|ethusdt|4h|mode-continuation|streak_length-5`.
- H5: FAIL; kandydaci=36, eligible=36, passing=0, najblizszy=`H5|btcusdt|1h|side-long|hold_bars-1`.
- H6: FAIL; kandydaci=36, eligible=16, passing=0, najblizszy=`H6|btcusdt|4h|side-long|hold_bars-10`.

## Wnioski z fazy research
- Pelny zestaw regul przeszedl: `0/186` kandydatow.
- Dodatni zwrot netto OOS mialo `18/186`; prog mocy >=30 transakcji spelnilo `150/186`.
- DSR > 0.95 spelnilo `0/186`; oba warunki familywise permutation spelnilo `0/186`.
- Wedlug zamrozonego rankingu DSR najblizej byl `H2|solusdt|4h|side-short|hold_bars-20`: OOS return=0.299059, annualized Sharpe=0.79919, DSR=1.26949e-07, familywise p=0.994012, trades=23.
- Werdykt konfirmacyjny: NO_EDGE; siatki nie sa rozszerzane, a holdout pozostaje nieodczytany.

## Najblizszy kandydat i powody porazki
Najblizszy kandydat: `H2|solusdt|4h|side-short|hold_bars-20`.
Powody niespelnienia regul:
- minimum_oos_trades
- dsr_pass
- permutation_q95_pass
- permutation_p_pass

## Ostrzezenia mocy
- 36/186 kandydatow nie osiagnelo progu 30 transakcji OOS i zostalo automatycznie odrzuconych.

## Reguly statystyczne
DSR liczony jest dla calej rodziny 186 kandydatow; niezdefiniowane Sharpe w rodzinie sa mapowane na 0 tylko na potrzeby korekty rodzinnej. Test permutacyjny raportuje najlepszy Sharpe rodziny dla kazdej z 500 niezaleznych przesuniec cyklicznych sygnalu w obrebie foldow.
Zamrozona miara selekcyjna DSR/permutacji to nieannualizowany Sharpe per-bar wspolny dla 1h i 4h. Annualizowane Sharpe w tabeli sa opisowe i nie uczestnicza w selekcji; mieszanie czestotliwosci w jednej rodzinie jest ograniczeniem interpretacyjnym prerejestracji.
Regula przejscia: net OOS return > 0, transakcje OOS >= 30, przewaga nad cost-matched buy-and-hold w zwrocie i Sharpe, DSR > 0.95, observed Sharpe > q95 permutacji oraz familywise empirical p <= 0.05.
Nota zrodla/formuly: Deflated Sharpe Ratio wedlug Bailey i Lopez de Prado, DOI 10.2139/ssrn.2460551.

## Holdout
Holdout: zapieczetowany 6-miesieczny zbior finalny; nieodczytany i nietkniety w tej fazie.
