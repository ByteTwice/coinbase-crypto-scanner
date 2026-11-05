# Free Android browser test — Coinbase breakout scanner

This version uses **Streamlit Community Cloud** to host the dashboard for free (subject to Streamlit's current availability and limits). No paid Render service, Coinbase account key, Telegram bot or credit card is needed for the test.

## Deploy using Android Chrome
1. Create/sign in to a GitHub account at https://github.com . Create a new repository, preferably private if your Streamlit account supports deploying private repos.
2. From the extracted `coinbase_free_test` folder upload **scanner.py** and **requirements.txt** to the repository root (GitHub > Add file > Upload files). Do not upload credentials.
3. Sign in to https://share.streamlit.io with GitHub. Create app > choose the repository and branch; set main file path to `scanner.py`; deploy. Free availability and resource limits can change.
4. Open the resulting Streamlit app URL on Android Chrome. In the sidebar choose **5–10 pairs** and **35–60 days** to start. Tap **Scan and backtest**. The first scan may take minutes; if rate limited, try fewer pairs later.
5. Inspect historical sample count, holdout results, probability calibration, ranked candidates and charts. A blank qualifying list is a valid result.
6. Chrome menu > Add to Home screen for convenient access.

## What this free test does NOT do
- It does **not** scan while you are away, send Telegram notifications, or monitor stops/targets continuously. You must open the dashboard and tap Scan.
- It does **not** execute trades or connect to your Coinbase account.
- It does **not** establish that a 75% predicted probability is reliable. Backtests have limitations (hourly bars, no realistic execution costs, possible market-regime changes, selection/survivorship bias).
- The local CSV alert log may be lost on Streamlit Community Cloud restarts; do not depend on it for durable alerts.

## Optional local computer test
`pip install -r requirements.txt` then `streamlit run scanner.py`.

Do not pay for 24/7 hosting until the model's holdout performance has been evaluated across enough independent examples. No live scan was performed when packaging this test.
