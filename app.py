import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==========================================
# 0. 設定・定数定義
# ==========================================
st.set_page_config(page_title="デイトレ運用エージェント", layout="wide")

# サンプルとして主要銘柄のみ記載。
# 本番運用時はここに日経225全銘柄のコード（末尾に.T）を記述してください。
NIKKEI_225_SAMPLE = [
    "7203.T", "9984.T", "8035.T", "6758.T", "6861.T", 
    "6098.T", "6920.T", "4063.T", "7741.T", "8058.T",
    "5401.T", "8306.T", "9432.T", "7011.T", "6501.T"
]

# スコア計算ルール（変更可能）
SCORE_RULES = {
    "volume_accel": 2, # 出来高加速
    "gap": 1,          # ギャップ
    "price_range": 1,  # 価格帯(300-3000)
    "prev_vol": 1,     # 前日ボラ
    "vwap_loc": 1      # VWAP位置
}

# ==========================================
# 1. データ取得・ロジック関数
# ==========================================

@st.cache_data(ttl=60) # 1分間キャッシュしてAPI負荷軽減
def fetch_market_data(tickers):
    """yfinanceからデータ取得（日足5日分、分足1日分）"""
    if not tickers:
        return None, None
    
    # 日足（前日比較用）
    daily_data = yf.download(
        tickers, period="5d", interval="1d", 
        group_by='ticker', auto_adjust=True, progress=False, threads=True
    )
    
    # 分足（当日監視用）
    # ※yfinanceの制約：日本株の分足は取得できない場合や遅延が大きい場合があります
    intraday_data = yf.download(
        tickers, period="1d", interval="1m", 
        group_by='ticker', auto_adjust=True, progress=False, threads=True
    )
    
    return daily_data, intraday_data

def calculate_scores(tickers, daily_data, intraday_data):
    """スクリーニングとスコア計算を実行"""
    results = []
    
    for t in tickers:
        try:
            # データ切り出し（MultiIndex対応）
            # 単一銘柄指定などの場合で構造が変わるため調整
            if len(tickers) > 1:
                df_d = daily_data[t]
                # 分足が存在しない場合のハンドリング
                df_m = intraday_data[t] if t in intraday_data.columns.levels[0] else pd.DataFrame()
            else:
                df_d = daily_data
                df_m = intraday_data

            if len(df_d) < 2: continue

            today = df_d.iloc[-1]
            prev = df_d.iloc[-2]
            
            # --- スコア判定ロジック ---
            score = 0
            reasons = []

            # 1. 出来高加速
            avg_vol_5d = df_d['Volume'].tail(5).mean()
            if prev['Volume'] >= avg_vol_5d * 1.2:
                score += SCORE_RULES['volume_accel']
                reasons.append("出来高増")

            # 2. ギャップ (始値 vs 前日終値)
            gap_rate = (today['Open'] - prev['Close']) / prev['Close']
            if abs(gap_rate) >= 0.007:
                score += SCORE_RULES['gap']
                reasons.append("ギャップ")

            # 3. 価格帯
            if 300 <= today['Close'] <= 3000:
                score += SCORE_RULES['price_range']
                reasons.append("価格適正")
            
            # 4. 前日ボラティリティ
            prev_range = (prev['High'] - prev['Low']) / prev['Close']
            if prev_range >= 0.02:
                score += SCORE_RULES['prev_vol']
                reasons.append("高ボラ")

            # 5. VWAP位置 (分足がある場合のみ)
            vwap_val = 0
            if not df_m.empty:
                # VWAP計算
                cum_vol = df_m['Volume'].cumsum()
                cum_pv = (df_m['Close'] * df_m['Volume']).cumsum()
                vwap_series = cum_pv / cum_vol
                vwap_val = vwap_series.iloc[-1]
                
                if today['Close'] > vwap_val:
                    score += SCORE_RULES['vwap_loc']
                    reasons.append("VWAP上")
                elif today['Close'] < vwap_val:
                    score += SCORE_RULES['vwap_loc']
                    reasons.append("VWAP下")

            results.append({
                "Ticker": t,
                "Score": score,
                "Price": f"{today['Close']:.0f}",
                "Change%": f"{(today['Close'] - prev['Close']) / prev['Close'] * 100:.2f}%",
                "Volume": f"{today['Volume']:,}",
                "Reasons": ", ".join(reasons),
                "RawPrice": today['Close'], # ソート用
                "RawVol": today['Volume']   # ソート用
            })
            
        except Exception as e:
            continue
            
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by=["Score", "RawVol"], ascending=[False, False])
    return df

def draw_candle_chart(ticker, df_m):
    """Plotlyでローソク足チャートを描画"""
    if df_m.empty:
        st.warning("分足データがありません。")
        return

    fig = go.Figure(data=[go.Candlestick(
        x=df_m.index,
        open=df_m['Open'],
        high=df_m['High'],
        low=df_m['Low'],
        close=df_m['Close'],
        name=ticker
    )])
    
    # VWAP追加
    cum_vol = df_m['Volume'].cumsum()
    cum_pv = (df_m['Close'] * df_m['Volume']).cumsum()
    vwap = cum_pv / cum_vol
    
    fig.add_trace(go.Scatter(
        x=df_m.index, y=vwap, mode='lines', name='VWAP', line=dict(color='orange', width=1.5)
    ))

    fig.update_layout(
        title=f"{ticker} 1分足 + VWAP",
        xaxis_title="Time",
        yaxis_title="Price",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 2. メインUI構成
# ==========================================

st.title("📊 デイトレード運用エージェント")
st.markdown("---")

# サイドバー設定
with st.sidebar:
    st.header("⚙️ 設定・入力")
    capital = st.number_input("元手資金 (円)", value=400000, step=10000)
    risk_val = st.number_input("1回あたり許容損失 (円)", value=4000, step=500)
    
    st.markdown("---")
    if st.button("データ更新 / スクリーニング実行"):
        st.cache_data.clear() # キャッシュクリアして再取得
        st.rerun()
        
    st.info("※ yfinanceのデータは15-20分遅延します。発注は必ず証券会社のツールで行ってください。")

# データロード
with st.spinner('市場データを取得中...'):
    daily, intraday = fetch_market_data(NIKKEI_225_SAMPLE)

if daily is None:
    st.error("データ取得に失敗しました。")
    st.stop()

# スクリーニング実行
df_result = calculate_scores(NIKKEI_225_SAMPLE, daily, intraday)

# --- タブ構成 ---
tab1, tab2, tab3 = st.tabs(["🔥 監視ダッシュボード", "📋 全体ランキング", "🧮 資金管理・計算機"])

# ----------------------------------------------------
# TAB 1: 監視ダッシュボード (上位3銘柄)
# ----------------------------------------------------
with tab1:
    st.header("Today's Top Picks (上位3銘柄)")
    
    if df_result.empty:
        st.warning("該当銘柄がありません。")
    else:
        top3 = df_result.head(3)
        
        # 3カラムでカード表示
        cols = st.columns(3)
        for i, (index, row) in enumerate(top3.iterrows()):
            with cols[i]:
                # カード風デザイン
                st.markdown(f"""
                <div style="border:1px solid #444; padding:15px; border-radius:10px; background-color:#262730;">
                    <h3 style="margin:0;">{row['Ticker']}</h3>
                    <h2 style="color:#00FFAA; margin:0;">¥{row['Price']}</h2>
                    <p style="color:#FFDD00;">Score: {row['Score']}点</p>
                    <p>前日比: {row['Change%']}</p>
                    <small>{row['Reasons']}</small>
                </div>
                """, unsafe_allow_html=True)
                
                # 個別チャート表示ボタン
                if st.button(f"詳細チャート: {row['Ticker']}", key=f"btn_{i}"):
                    st.session_state['selected_ticker'] = row['Ticker']

        st.markdown("---")
        
        # 詳細チャートエリア（ボタンで選択された銘柄を表示）
        if 'selected_ticker' in st.session_state:
            sel_t = st.session_state['selected_ticker']
            st.subheader(f"📈 {sel_t} リアルタイム分析")
            
            # 分足データがあるか確認して描画
            target_df = pd.DataFrame()
            if len(NIKKEI_225_SAMPLE) > 1:
                if sel_t in intraday.columns.levels[0]:
                    target_df = intraday[sel_t]
            else:
                target_df = intraday
                
            draw_candle_chart(sel_t, target_df)

# ----------------------------------------------------
# TAB 2: 全体ランキング
# ----------------------------------------------------
with tab2:
    st.header("スクリーニング結果一覧")
    # 表示用カラムに絞る
    display_df = df_result[["Ticker", "Score", "Price", "Change%", "Volume", "Reasons"]]
    st.dataframe(display_df, use_container_width=True, height=500)

# ----------------------------------------------------
# TAB 3: 資金管理・計算機
# ----------------------------------------------------
with tab3:
    st.header("🧮 エントリープラン計算機")
    
    c1, c2 = st.columns(2)
    with c1:
        calc_ticker = st.selectbox("銘柄選択", df_result['Ticker'].tolist())
        # 選択銘柄の現在値をデフォルトに
        curr_price_val = float(df_result[df_result['Ticker']==calc_ticker]['RawPrice'].values[0])
        entry_price = st.number_input("エントリー価格", value=curr_price_val, step=1.0)
        
    with c2:
        sl_pct = st.slider("損切り幅 (%)", 0.1, 2.0, 0.6, 0.1)
        risk_money = st.number_input("許容リスク額 (自動反映)", value=risk_val, disabled=True)

    st.markdown("### 📋 トレード計画")
    
    if entry_price > 0:
        # 計算ロジック
        sl_price = int(entry_price * (1 - sl_pct/100))
        loss_per_share = entry_price - sl_price
        
        if loss_per_share > 0:
            # 枚数計算 (許容リスク ÷ 1株あたり損失)
            max_shares = int(risk_money / loss_per_share)
            # 単元(100株)で丸め
            shares = (max_shares // 100) * 100
            if shares == 0: shares = 100 # 最低1単元
            
            total_risk = loss_per_share * shares
            tp_2r = int(entry_price + (loss_per_share * 2))
            tp_3r = int(entry_price + (loss_per_share * 3))
            
            # 結果表示
            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                st.error(f"損切り(SL)\n# {sl_price} 円")
                st.caption(f"損失額: -{total_risk:,} 円")
            with res_col2:
                st.info(f"適正株数\n# {shares} 株")
                st.caption(f"建玉額: {int(entry_price * shares):,} 円")
            with res_col3:
                st.success(f"利確(TP)\n# 2R: {tp_2r} 円\n# 3R: {tp_3r} 円")
        else:
            st.warning("損切り幅が小さすぎます（1Tick以下）")