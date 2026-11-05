"""Coinbase public-market, alert-only 10% / 5% / 24h breakout scanner.
Run: streamlit run scanner.py
"""
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = 'https://api.coinbase.com/api/v3/brokerage/market'
HORIZON = 24  # hourly bars
TAKE = 0.10
STOP = 0.05
FEE_SIDE = 0.006  # adjustable approximation; price barriers are gross
FEATURES = ['rsi', 'vol_ratio', 'ret_6h', 'ret_24h', 'breakout', 'ema_gap', 'range_pct', 'btc_ret_24h']
STATE = Path.home() / '.coinbase_breakout_alerts.csv'
SESSION = requests.Session()
SESSION.headers.update({'User-Agent': 'ResearchBreakoutScanner/1.0'})


def api(path, params=None):
    for attempt in range(4):
        try:
            r = SESSION.get(BASE + path, params=params, timeout=20)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(1.5 * (attempt + 1)); continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == 3: raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError('Coinbase API temporarily unavailable')


@st.cache_data(ttl=3600)
def products(max_pairs=35):
    result = []
    offset = 0
    while offset < 1500:
        data = api('/products', {'limit': 250, 'offset': offset, 'product_type': 'SPOT'})
        batch = data.get('products', [])
        if not batch: break
        for p in batch:
            pid = p.get('product_id', '')
            if not pid.endswith('-USD') or p.get('trading_disabled') or p.get('is_disabled'):
                continue
            if p.get('status') not in (None, 'online', 'ONLINE'): continue
            try: volume = float(p.get('volume_24h') or 0) * float(p.get('price') or 0)
            except (TypeError, ValueError): volume = 0
            if volume >= 1000000:
                result.append((pid, volume))
        offset += len(batch)
        if len(batch) < 250: break
    result.sort(key=lambda x: x[1], reverse=True)
    return [p for p, _ in result[:max_pairs]]


@st.cache_data(ttl=900)
def candles(pid, days=75):
    now = int(time.time() // 3600 * 3600)
    start = now - days * 86400
    rows = []
    for a in range(start, now, 300 * 3600):
        b = min(a + 300 * 3600, now)
        data = api(f'/products/{pid}/candles', {'start': str(a), 'end': str(b),
                      'granularity': 'ONE_HOUR', 'limit': 350})
        rows.extend(data.get('candles', []))
        time.sleep(0.08)
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows)
    for c in ['start', 'open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna().drop_duplicates('start').sort_values('start').set_index('start')
    # Missing hours invalidate contiguous lookback/forward windows; don't invent OHLC.
    df = df.reindex(np.arange(start, now, 3600))
    df.index = pd.to_datetime(df.index, unit='s', utc=True)
    return df


def indicators(df, btc):
    x = df.copy()
    c = x.close
    change = c.diff()
    gain = change.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss = (-change.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    x['rsi'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    x['vol_ratio'] = x.volume / x.volume.shift(1).rolling(24, min_periods=24).mean()
    x['ret_6h'] = c / c.shift(6) - 1
    x['ret_24h'] = c / c.shift(24) - 1
    x['breakout'] = c / x.high.shift(1).rolling(24, min_periods=24).max() - 1
    x['ema_gap'] = c.ewm(span=12, adjust=False, min_periods=12).mean() / c.ewm(span=26, adjust=False, min_periods=26).mean() - 1
    x['range_pct'] = (x.high - x.low).rolling(12, min_periods=12).mean() / c
    x['btc_ret_24h'] = (btc.close / btc.close.shift(24) - 1).reindex(x.index)
    # Signal is computed at closed candle t; simulated entry at next candle open.
    x['setup'] = ((x.vol_ratio >= 1.5) & (x.rsi.between(48, 78)) &
                  (x.ema_gap > 0) & (x.breakout > -0.025) &
                  (x.ret_6h > 0) & (x.btc_ret_24h > -0.06))
    return x


def label_outcomes(x):
    """Conservative: if target and stop touch same hourly candle, stop wins."""
    y = np.full(len(x), np.nan)
    entry = np.full(len(x), np.nan)
    outcome = [''] * len(x)
    for i in range(48, len(x) - HORIZON - 1):
        if not bool(x.setup.iloc[i]): continue
        price = x.open.iloc[i + 1]
        if not np.isfinite(price) or price <= 0: continue
        future = x.iloc[i+1:i+1+HORIZON]
        if future[['high', 'low', 'close']].isna().any().any(): continue
        entry[i] = price
        result = 0
        why = 'TIMEOUT'
        for _, bar in future.iterrows():
            if bar.low <= price * (1 - STOP):
                result = 0; why = 'STOP'; break
            if bar.high >= price * (1 + TAKE):
                result = 1; why = 'TARGET'; break
        y[i] = result
        outcome[i] = why
    x = x.copy()
    x['y'] = y
    x['entry_backtest'] = entry
    x['outcome'] = outcome
    return x


def fit_and_score(dataset, current):
    """Chronological holdout, with 24-hour purge and isotonic calibration."""
    train = dataset.dropna(subset=FEATURES + ['y']).sort_values('timestamp')
    if len(train) < 120 or train.y.nunique() < 2:
        return None, 'Need at least 120 completed historical signals and both outcomes.'
    cut1, cut2 = int(len(train) * .60), int(len(train) * .80)
    t1, t2 = train.timestamp.iloc[cut1], train.timestamp.iloc[cut2]
    # Purge 24h around boundaries to avoid overlapping forward labels.
    fit = train[train.timestamp < t1 - pd.Timedelta(hours=24)]
    cal = train[(train.timestamp >= t1) & (train.timestamp < t2 - pd.Timedelta(hours=24))]
    test = train[train.timestamp >= t2]
    if min(len(fit), len(cal), len(test)) < 25 or fit.y.nunique() < 2:
        return None, 'Not enough chronologically independent samples for calibration.'
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight=None))
    model.fit(fit[FEATURES], fit.y.astype(int))
    p_cal = model.predict_proba(cal[FEATURES])[:, 1]
    if len(np.unique(p_cal)) < 2:
        return None, 'Insufficient variation to calibrate predictions.'
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    iso.fit(p_cal, cal.y)
    p_test = iso.predict(model.predict_proba(test[FEATURES])[:, 1])
    test = test.copy(); test['probability'] = p_test
    qualified = test[test.probability >= .75]
    # Final model deliberately not refitted: honest chronological evaluation.
    current = current.copy()
    current['probability'] = iso.predict(model.predict_proba(current[FEATURES])[:, 1]) if len(current) else []
    stats = {'samples': len(train), 'holdout': len(test), 'base_win': float(test.y.mean()),
             'qualified_count': len(qualified), 'qualified_win': float(qualified.y.mean()) if len(qualified) else None,
             'holdout_brier': float(np.mean((p_test - test.y.to_numpy())**2)),
             'holdout': test}
    return (current, stats), None


def chart(x, pid):
    d = x.tail(120)
    fig = go.Figure(data=[go.Candlestick(x=d.index, open=d.open, high=d.high,
                                        low=d.low, close=d.close, name=pid)])
    if len(d) and np.isfinite(d.close.iloc[-1]):
        last = float(d.close.iloc[-1])
        fig.add_hline(y=last*1.10, line_dash='dash', line_color='green', annotation_text='+10% indicative')
        fig.add_hline(y=last*.95, line_dash='dash', line_color='red', annotation_text='-5% indicative')
    fig.update_layout(title=pid + ' • hourly candles', xaxis_rangeslider_visible=False,
                      height=480, margin=dict(l=10,r=10,t=50,b=10))
    return fig


def alert_new(rows):
    old = pd.read_csv(STATE) if STATE.exists() else pd.DataFrame(columns=['id', 'timestamp', 'message'])
    known = set(old.id.astype(str))
    new = []
    for r in rows:
        key = f"{r['coin']}:{r['timestamp']}"
        if key not in known:
            new.append({'id': key, 'timestamp': datetime.now(timezone.utc).isoformat(),
                        'message': f"ENTRY WATCH {r['coin']} calibrated p={r['probability']:.1%}, reference={r['price']:.8g}, +10% / -5% / 24h"})
    if new:
        pd.concat([old, pd.DataFrame(new)], ignore_index=True).to_csv(STATE, index=False)
    return new


def dashboard():
    st.set_page_config(page_title='Coinbase Breakout Scanner', layout='wide')
    st.title('Coinbase +10% / -5% / 24h scanner')
    st.caption('Research and alerts only • no trade execution • historical estimates are not guarantees')
    with st.sidebar:
        n = st.slider('Number of high-volume USD pairs', 5, 100, 30, 5)
        days = st.slider('History per pair (days)', 35, 180, 90, 5)
        st.write('Minimum 24h USD volume: $1 million')
        run = st.button('Scan and backtest', type='primary')
        st.caption('Refresh manually. For ongoing alerts, leave dashboard open and rescan.')

    if run:
        with st.spinner('Fetching Coinbase historical candles and evaluating signals...'):
            try:
                ids = products(n)
                if 'BTC-USD' not in ids: ids.insert(0, 'BTC-USD')
                btc = candles('BTC-USD', days)
                historical, latest, histories, errors = [], [], {}, []
                progress = st.progress(0)
                for j, pid in enumerate(ids):
                    try:
                        x = label_outcomes(indicators(candles(pid, days), btc))
                        histories[pid] = x
                        hist = x[x.y.notna()].copy()
                        if len(hist):
                            hist['coin'] = pid; hist['timestamp'] = hist.index
                            historical.append(hist)
                        # Only closed hourly candles are used; index ends at last hour boundary minus 1h.
                        now_hour = pd.Timestamp.now(tz='UTC').floor('h')
                        eligible = x[x.index < now_hour].dropna(subset=FEATURES + ['close'])
                        if len(eligible) and bool(eligible.setup.iloc[-1]) and eligible.index[-1] >= now_hour - pd.Timedelta(hours=2):
                            row = eligible.iloc[[-1]].copy()
                            row['coin'] = pid; row['timestamp'] = row.index
                            latest.append(row)
                    except Exception as e:
                        errors.append(f'{pid}: {str(e)[:140]}')
                    progress.progress((j+1)/len(ids))
                if not historical:
                    st.error('No usable historical breakout signals; extend history or adjust universe.')
                else:
                    dataset = pd.concat(historical)
                    current = pd.concat(latest) if latest else dataset.iloc[:0].copy()
                    result, error = fit_and_score(dataset, current)
                    if error: st.warning(error)
                    else:
                        scored, stats = result
                        a,b,c,d = st.columns(4)
                        a.metric('Historical signals', stats['samples'])
                        b.metric('Holdout signals', stats['holdout'])
                        c.metric('Holdout win rate', f"{stats['base_win']:.1%}")
                        d.metric('75%+ signals / win rate', f"{stats['qualified_count']} / " +
                                 (f"{stats['qualified_win']:.1%}" if stats['qualified_win'] is not None else 'N/A'))
                        st.caption('Holdout: later unseen chronological period. High-confidence performance may be unavailable if too few trades qualify.')
                        if len(scored):
                            scored = scored.sort_values('probability', ascending=False)
                            table = []
                            for _, r in scored.iterrows():
                                table.append({'coin': r.coin, 'timestamp': r.timestamp.isoformat(),
                                              'price': float(r.close), 'probability': float(r.probability),
                                              'target': float(r.close)*1.10, 'stop': float(r.close)*.95,
                                              'status': 'WATCH' if r.probability >= .75 else 'BELOW THRESHOLD'})
                            st.subheader('Current ranked candidates')
                            st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
                            eligible_alerts = [r for r in table if r['probability'] >= .75]
                            fresh = alert_new(eligible_alerts)
                            for x in fresh: st.success(x['message'])
                        else:
                            st.info('No current breakout setups. This is a valid outcome: do not force a trade.')
                        st.subheader('Historical holdout calibration')
                        hold = stats['holdout'].copy()
                        hold['bucket'] = pd.cut(hold.probability, bins=[0,.2,.4,.6,.75,1], include_lowest=True)
                        calibration = hold.groupby('bucket', observed=True).agg(predicted=('probability','mean'), observed=('y','mean'), n=('y','size'))
                        st.dataframe(calibration, use_container_width=True)
                        st.caption(f"Brier score: {stats['holdout_brier']:.4f} (lower is better). Historical tests exclude trading costs from gross price-barrier hits.")
                        chosen = st.selectbox('Chart', list(histories), index=0)
                        st.plotly_chart(chart(histories[chosen], chosen), use_container_width=True)
                        st.download_button('Download ranked candidates CSV', pd.DataFrame(table if len(scored) else []).to_csv(index=False), 'coinbase_candidates.csv')
                if errors:
                    with st.expander(f'API/data warnings ({len(errors)})'):
                        st.write('\n'.join(errors))
            except Exception as e:
                st.error(f'Scan failed: {e}')
    else:
        st.info('Press “Scan and backtest” to download public Coinbase data and produce rankings.')
    st.divider()
    st.subheader('Exit-alert reference')
    st.write('For a watched entry price E, target = 1.10 × E, stop = 0.95 × E, and time exit = entry + 24h. This dashboard issues new entry-watch alerts; exit levels are reference values, not continuously monitored orders.')

if __name__ == "__main__":
    dashboard()
