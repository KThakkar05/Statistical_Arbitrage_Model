# data/

Empty by default. Price data is fetched at runtime via `src.cointegration.load_prices`
(yfinance) rather than checked into the repo. Use this folder for optional local
caches (e.g. CSV snapshots) if you want reproducible runs without hitting the
network — `.gitignore` excludes everything here except this file.
