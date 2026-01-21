import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==========================================
# 0. 設定・定数定義
# ==========================================
st.set_page_config(page_title="デイトレ運用エージェント", layout="wide")

# 監視対象リスト（コードのみ定義）
# ※本番ではここに日経225全銘柄を入れてください
NIKKEI_225_SAMPLE = [
    "7203.T", "9984.T", "8035.T", "6758.T", "6861.T", 
    "6098.T", "6920.T", "4063.T", "7741.T", "8058.T",
    "5401.T", "8306.T", "9432.T", "7011.T", "6501.T"
]

# スコア計算ルール
SCORE_RULES = {
    "volume_accel": 2, # 出来高加速
    "gap": 1,          # ギャップ
    "price_range": 1,  # 価格帯
    "prev_vol": 1,     # 前日ボラ
    "vwap_loc": 1      # VWAP位置
}

# ==========================================
# 1. データ取得・ロジック関数
# ==========================================

@st.cache_data(ttl=86400) # 銘柄名は24時間キャッシュ（遅いため）
def fetch_ticker_names(tickers):
    """yfinanceから銘柄名を取得する"""
    name_map = {}
    # プログレスバー表示（銘柄数が多いと時間がかかるため）
    progress_text = "銘柄情報を取得中..."
    my_bar = st.progress(0, text=progress_text)
    
    total = len(tickers)
    for i, t in enumerate(tickers):
        try:
            # yfinanceのTickerオブジェクトから情報取得
            ticker_info = yf.Ticker(t).info
            # 日本語の省略名(shortName)があればそれを、なければlongName、なければコード
            name = ticker_info.get('shortName', ticker_info.get('longName', t))
            name_map[t] = name
        except Exception:
            name_map[t] = t
        
        # プログレスバー更新
        my_bar.progress((i + 1) / total, text=f"{progress_text} ({i+1}/{total})")
    
    my_bar.empty() # バーを消す
    return name_map

@st.cache_data(ttl=60) 
def fetch_market_data(tickers):
    """株価データの取得"""
    if not tickers:
        return None, None
    
    # 日足（5日分）
    daily_data = yf.download(
        tickers, period="5d", interval="1d", 
        group_by='ticker', auto_adjust=True, progress=False, threads=True
    )
    
    # 分足（5日分：前日VWAP計算のため）
    intraday_data = yf.download(
        tickers, period="5d", interval="1m", 
        group_by='ticker', auto_adjust=True, progress=False, threads=True
    )
    
    return daily_data, intraday_data

def get_prev_vwap(df_m, prev_date_str):
    """前日の分足データからVWAPを計算"""
    try:
        prev_day_data = df_m.loc[prev_date_str]
        if prev_day_data.empty:
            return 0
        v = prev_day_data['Volume']
        p = prev_day_data['Close']
        vwap = (p * v).sum() / v.sum()
        return vwap
    except:
        return 0

def calculate_scores(tickers, names_map, daily_data, intraday_data):
    """スクリーニングと各種数値の計算"""
    results = []
    
    for t in tickers:
        try:
            # データ切り出し
            if len(tickers) > 1:
                df_d = daily_data[t]
                df_m = intraday_data[t] if t in intraday_data.columns.levels[0] else pd.DataFrame()
            else:
                df_d = daily_data
                df_m = intraday_data

            if len(df_d) < 2: continue

            today = df_d.iloc[-1]
            prev = df_d.iloc[-2]
            prev_date = df_d.index[-2].strftime('%Y-%m-%d')
            
            # --- 数値計算 ---
            prev_vol = prev['Volume']
            avg_vol_5d = df_d['Volume'].iloc[-6:-1].mean()
            if pd.isna(avg_vol_5d): avg_vol_5d = prev_vol

            prev_vwap = get_prev_vwap(df_m, prev_date)
            if prev_vwap == 0:
                prev_vwap = (prev['High'] + prev['Low'] + prev['Close']) / 3

            # --- スコア判定 ---
            score = 0
            reasons = []

            # A. 出来高加速
            if prev_vol >= avg_vol_5d * 1.2:
                score += SCORE_RULES['volume_accel']
                reasons.append("出来高増")

            # B. ギャップ
            gap_rate = (today['Open'] - prev['Close']) / prev['Close']
            if abs(gap_rate) >= 0.007:
                score += SCORE_RULES['gap']
                reasons.append("ギャップ")

            # C. 価格帯
            if 300 <= today['Close'] <= 3000:
                score += SCORE_RULES['price_range']
                reasons.append("価格適正")
            
            # D. 前日ボラ
            prev_range = (prev['High'] - prev['Low']) / prev['Close']
            if prev_range >= 0.02:
                score += SCORE_RULES['prev_vol']
                reasons.append("高ボラ")

            # E. 当日VWAP位置
            vwap_val = 0
            if not df_m.empty:
                today_date_str = df_d.index[-1].strftime('%Y-%m-%d')
                try:
                    df_m_today = df_m.loc[today_date_str]
                    if not df_m_today.empty:
                        cum_vol = df_m_today['Volume'].cumsum()
                        cum_pv = (df_m_today['Close'] * df_m_today['Volume']).cumsum()
                        vwap_val = (cum_pv / cum_vol).iloc[-1]
                        
                        if today['Close'] > vwap_val:
                            score += SCORE_RULES['vwap_loc']
                            reasons.append("VWAP上")
                        elif today['Close'] < vwap_val:
                            score += SCORE_RULES['vwap_loc']
                            reasons.append("VWAP下")
                except:
                    pass

            # マップから名称取得
            name = names_map.get(t, t)

            results.append({
                "Ticker": t,
                "Name": name,
                "Score": score,
                "Price": today['Close'],
                "Change%": (today['Close'] - prev['Close']) / prev['Close'] * 100,
                "Volume": today['Volume'],
                "Reasons": ", ".join(reasons),
                # CSV出力用データ
                "PrevVol": prev_vol,
                "AvgVol5d": avg_vol_5d,
                "PrevClose": prev['Close'],
                "PrevHigh": prev['High'],
                "PrevLow": prev['Low'],
                "PrevVWAP": prev_vwap
            })
            
        except Exception as e:
            continue
            
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by=["Score", "Volume"], ascending=[False, False])
    return df

def draw_candle_chart(ticker, name, df_m):
    """Plotlyでチャート描画"""
    if df_m.empty:
        st.warning("分足データがありません。")
        return

    last_date = df_m.index[-1].date()
    df_plot = df_m[df_m.index.date == last_date]

    fig = go.Figure(data=[go.Candlestick(
        x=df_plot.index,
        open=df_plot['Open'], high=df_plot['High'],
        low=df_plot['Low'], close=df_plot['Close'],
        name=ticker
    )])
    
    # VWAP
    cum_vol = df_plot['Volume'].cumsum()
    cum_pv = (df_plot['Close'] * df_plot['Volume']).cumsum()
    vwap = cum_pv / cum_vol
    
    fig.add_trace(go.Scatter(
        x=df_plot.index, y=vwap, mode='lines', name='VWAP', line=dict(color='orange', width=1.5)
    ))

    fig.update_layout(
        title=f"{ticker} {name} 本日の推移",
        xaxis_title="Time", yaxis_title="Price", height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

def generate_csv_string(row):
    """指定フォーマットのCSV文字列を生成"""
    return (f"{row['Ticker']}, {row['Name']}, {int(row['PrevVol'])}, {int(row['AvgVol5d'])}, "
            f"{int(row['Price'])}, {int(row['PrevClose'])}, {int(row['PrevHigh'])}, "
            f"{int(row['PrevLow'])}, {int(row['PrevVWAP'])}")

# ==========================================
# 2. メインUI構成
# ==========================================

st.title("📊 デイトレ運用エージェント v1.2")
st.markdown("---")

# サイドバー
with st.sidebar:
    st.header("⚙️ 設定")
    capital = st.number_input("元手資金 (円)", value=400000, step=10000)
    risk_val = st.number_input("1回許容損失 (円)", value=4000, step=500)
    
    if st.button("データ更新 / 再計算", type="primary"):
        st.cache_data.clear()
        st.rerun()
    
    st.caption("※初回は銘柄名取得のため時間がかかります")

# データ取得プロセス
with st.spinner('市場データを取得中...'):
    # 1. 銘柄名取得（キャッシュあり）
    names_map = fetch_ticker_names(NIKKEI_225_SAMPLE)
    # 2. 株価データ取得
    daily, intraday = fetch_market_data(NIKKEI_225_SAMPLE)

if daily is None:
    st.error("データ取得に失敗しました。")
    st.stop()

# スクリーニング計算
df_result = calculate_scores(NIKKEI_225_SAMPLE, names_map, daily, intraday)

# --- UIタブ ---
tab1, tab2, tab3 = st.tabs(["🔥 監視ボード & 出力", "📋 全体リスト", "🧮 資金管理"])

# ----------------------------------------------------
# TAB 1: 監視ボード & データ出力
# ----------------------------------------------------
with tab1:
    if df_result.empty:
        st.warning("該当銘柄なし")
    else:
        st.subheader("Today's Top Picks")
        top3 = df_result.head(3)
        
        cols = st.columns(3)
        for i, (index, row) in enumerate(top3.iterrows()):
            with cols[i]:
                st.markdown(f"""
                <div style="border:1px solid #555; padding:15px; border-radius:10px; background-color:#262730; margin-bottom:10px;">
                    <div style="font-size:0.9em; color:#ccc;">{row['Ticker']}</div>
                    <div style="font-size:1.2em; font-weight:bold;">{row['Name']}</div>
                    <div style="color:#00FFAA; font-size:1.5em; font-weight:bold;">¥{row['Price']:.0f}</div>
                    <div style="color:#FFDD00;">Score: {row['Score']}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # === 銘柄選択・CSV出力 ===
        st.subheader("📋 データ出力 (CSV Copy)")
        
        # 選択肢作成
        options = df_result.apply(lambda x: f"{x['Ticker']} {x['Name']} (Score:{x['Score']})", axis=1).tolist()
        selected_option = st.selectbox("詳細表示・出力する銘柄を選択:", options)
        
        # データ抽出
        selected_ticker = selected_option.split(" ")[0]
        sel_row = df_result[df_result['Ticker'] == selected_ticker].iloc[0]
        
        # CSV生成
        csv_text = generate_csv_string(sel_row)
        
        st.caption("以下のテキストをコピーしてください（右上のアイコンでコピー可）")
        st.code(csv_text, language="csv")
        st.info("順序: コード, 名称, 前日出来高, 5日平均, 現在値, 前日終値, 前日高値, 前日安値, 前日VWAP")

        # チャート描画
        st.markdown("---")
        st.subheader(f"📈 {selected_ticker} {sel_row['Name']} チャート")
        
        target_df = pd.DataFrame()
        if len(NIKKEI_225_SAMPLE) > 1:
            if selected_ticker in intraday.columns.levels[0]:
                target_df = intraday[selected_ticker]
        else:
            target_df = intraday
            
        draw_candle_chart(selected_ticker, sel_row['Name'], target_df)

# ----------------------------------------------------
# TAB 2: 全体リスト
# ----------------------------------------------------
with tab2:
    st.header("全スクリーニング結果")
    disp_cols = ["Ticker", "Name", "Score", "Price", "Change%", "Volume", "Reasons"]
    st.dataframe(
        df_result[disp_cols].style.format({
            "Price": "{:.0f}", "Change%": "{:.2f}%", "Volume": "{:,.0f}"
        }), 
        use_container_width=True, height=600
    )

# ----------------------------------------------------
# TAB 3: 資金管理
# ----------------------------------------------------
with tab3:
    st.header("🧮 エントリー計算機")
    c1, c2 = st.columns(2)
    with c1:
        calc_ticker_raw = st.selectbox("計算対象", options)
        calc_ticker = calc_ticker_raw.split(" ")[0]
        row_data = df_result[df_result['Ticker']==calc_ticker].iloc[0]
        entry_price = st.number_input("エントリー価格", value=float(row_data['Price']), step=1.0)
    with c2:
        sl_pct = st.slider("損切り幅 (%)", 0.1, 2.0, 0.6, 0.1)
        st.metric("許容リスク額", f"{risk_val:,} 円")

    if entry_price > 0:
        sl_price = int(entry_price * (1 - sl_pct/100))
        loss_per_share = entry_price - sl_price
        if loss_per_share > 0:
            max_shares = int(risk_val / loss_per_share)
            shares = (max_shares // 100) * 100
            if shares == 0: shares = 100
            total_risk = loss_per_share * shares
            tp_2r = int(entry_price + (loss_per_share * 2))
            
            res1, res2, res3 = st.columns(3)
            res1.error(f"損切り (SL)\n# {sl_price} 円\n(-{total_risk:,}円)")
            res2.info(f"適正株数\n# {shares} 株\n(約 {int(entry_price*shares/10000)}万円)")
            res3.success(f"利確 (TP)\n# {tp_2r} 円\n(+{int(total_risk*2):,}円)")
