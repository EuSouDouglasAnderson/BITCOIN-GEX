import sqlite3
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ============================================================
# CONFIGURAÇÃO
# ============================================================
BINANCE_API = "https://data-api.binance.vision"
SYMBOL = "BTCUSDT"
INTERVAL = "5m"
DATABASE = "btc_trader_v2.db"
AUTO_REFRESH_SECONDS = 10
KLINE_LIMIT = 500
ATR_STOP_MULTIPLIER = 1.0
ATR_TARGET_MULTIPLIER = 2.0

# ============================================================
# PÁGINA / ESTILO
# ============================================================
st.set_page_config(
    page_title="BTC Quant Trader v2",
    page_icon="₿",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: #0b0f14; color: #f5f7fa; }
    [data-testid="stMetricValue"] { font-size: 1.55rem; }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    div[data-testid="stDataFrame"] { border: 1px solid #222a33; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# BANCO DE DADOS
# Novo banco: btc_trader_v2.db
# ============================================================
def get_conn():
    conn = sqlite3.connect(DATABASE, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def criar_banco():
    conn = get_conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_time TEXT NOT NULL,
            entry_price REAL NOT NULL,
            stop_price REAL NOT NULL,
            target_price REAL NOT NULL,
            exit_time TEXT,
            exit_price REAL,
            pnl_pct REAL,
            result TEXT,
            exit_reason TEXT,
            score REAL,
            regime TEXT,
            notes TEXT,
            signal_time TEXT,

            -- Snapshot completo dos indicadores no momento da entrada
            ema20 REAL,
            ema50 REAL,
            ema200 REAL,
            rsi REAL,
            atr REAL,
            ret_1 REAL,
            ret_3 REAL,
            ret_12 REAL,
            ret_48 REAL,
            volatility REAL,
            vol_z REAL,
            volume_ratio REAL,
            volume_z REAL,
            vwap REAL,
            cvd_delta REAL,
            orderbook_imbalance REAL,
            score_compra REAL,
            score_venda REAL,
            signal TEXT,
            entry_reason TEXT
        )
        """
    )
    # Compatibilidade caso o banco v2 já tenha sido criado antes desta atualização.
    colunas = {row[1] for row in conn.execute("PRAGMA table_info(trades)").fetchall()}
    if "exit_reason" not in colunas:
        conn.execute("ALTER TABLE trades ADD COLUMN exit_reason TEXT")

    conn.commit()
    conn.close()


criar_banco()

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================
def agora():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def fmt_num(value, casas=2):
    if value is None or not np.isfinite(float(value)):
        return "-"
    return f"{float(value):,.{casas}f}"


def buscar_operacoes_abertas():
    conn = get_conn()
    df = pd.read_sql_query(
        """
        SELECT
            id, symbol, side, entry_time, entry_price, stop_price, target_price,
            exit_time, exit_price, pnl_pct, result, exit_reason, score, regime, notes,
            signal_time, ema20, ema50, ema200, rsi, atr, ret_1, ret_3, ret_12,
            ret_48, volatility, vol_z, volume_ratio, volume_z, vwap, cvd_delta,
            orderbook_imbalance, score_compra, score_venda, signal, entry_reason
        FROM trades
        WHERE exit_time IS NULL
        ORDER BY id DESC
        """,
        conn,
    )
    conn.close()
    return df


def buscar_historico(limite=100):
    conn = get_conn()
    df = pd.read_sql_query(
        """
        SELECT
            id, side, entry_time, entry_price, stop_price, target_price,
            exit_time, exit_price, pnl_pct, result, exit_reason, score, regime,
            signal_time, signal, entry_reason
        FROM trades
        ORDER BY id DESC
        LIMIT ?
        """,
        conn,
        params=(limite,),
    )
    conn.close()
    return df


def entrada_ja_registrada(signal_time):
    conn = get_conn()
    cur = conn.execute(
        "SELECT COUNT(*) FROM trades WHERE signal_time = ?",
        (str(signal_time),),
    )
    existe = cur.fetchone()[0] > 0
    conn.close()
    return existe


def validar_plano(side, entrada, stop, target):
    """Validação de segurança antes de gravar a operação."""
    entrada = float(entrada)
    stop = float(stop)
    target = float(target)

    if not all(np.isfinite([entrada, stop, target])):
        return False, "Preço de entrada, stop ou alvo inválido."

    if side == "COMPRA":
        if not (stop < entrada < target):
            return False, (
                f"Plano inválido para COMPRA: esperado STOP < ENTRADA < ALVO, "
                f"mas recebeu {stop:.2f} < {entrada:.2f} < {target:.2f}."
            )
    elif side == "VENDA":
        if not (target < entrada < stop):
            return False, (
                f"Plano inválido para VENDA: esperado ALVO < ENTRADA < STOP, "
                f"mas recebeu {target:.2f} < {entrada:.2f} < {stop:.2f}."
            )
    else:
        return False, f"Lado inválido: {side}"

    return True, "Plano válido."


def registrar_trade(
    side,
    entry_price,
    stop_price,
    target_price,
    score,
    regime,
    signal_time,
    row,
    imbalance,
    score_compra,
    score_venda,
    signal,
    entry_reason,
    notes="",
):
    # 3) Segurança: nunca grava uma operação com stop/alvo invertidos.
    valido, mensagem = validar_plano(side, entry_price, stop_price, target_price)
    if not valido:
        raise ValueError(mensagem)

    conn = get_conn()
    conn.execute(
        """
        INSERT INTO trades (
            symbol, side, entry_time, entry_price, stop_price, target_price,
            exit_time, exit_price, pnl_pct, result, score, regime, notes,
            signal_time,
            ema20, ema50, ema200, rsi, atr, ret_1, ret_3, ret_12, ret_48,
            volatility, vol_z, volume_ratio, volume_z, vwap, cvd_delta,
            orderbook_imbalance, score_compra, score_venda, signal, entry_reason
        ) VALUES (
            ?, ?, ?, ?, ?, ?,
            NULL, NULL, NULL, NULL, ?, ?, ?,
            ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?
        )
        """,
        (
            SYMBOL,
            side,
            agora(),
            float(entry_price),
            float(stop_price),
            float(target_price),
            float(score),
            regime,
            notes,
            str(signal_time),
            float(row["EMA20"]),
            float(row["EMA50"]),
            float(row["EMA200"]),
            float(row["RSI"]),
            float(row["ATR"]),
            float(row["ret_1"]),
            float(row["ret_3"]),
            float(row["ret_12"]),
            float(row["ret_48"]),
            float(row["volatility"]),
            float(row["vol_z"]),
            float(row["volume_ratio"]),
            float(row["volume_z"]),
            float(row["VWAP"]),
            float(row["cvd_delta"]),
            float(imbalance),
            float(score_compra),
            float(score_venda),
            signal,
            entry_reason,
        ),
    )
    conn.commit()
    trade_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return trade_id


def fechar_trade(trade_id, exit_price, exit_reason):
    conn = get_conn()
    row = conn.execute(
        "SELECT side, entry_price FROM trades WHERE id = ?",
        (int(trade_id),),
    ).fetchone()

    if row is None:
        conn.close()
        return

    side, entry_price = row
    entry_price = float(entry_price)
    exit_price = float(exit_price)

    if side == "COMPRA":
        pnl_pct = ((exit_price / entry_price) - 1) * 100
    else:
        pnl_pct = ((entry_price / exit_price) - 1) * 100

    if pnl_pct > 0:
        result = "GANHO"
    elif pnl_pct < 0:
        result = "PERDA"
    else:
        result = "EMPATE"

    conn.execute(
        """
        UPDATE trades
        SET exit_time = ?, exit_price = ?, pnl_pct = ?, result = ?, exit_reason = ?
        WHERE id = ?
        """,
        (agora(), exit_price, pnl_pct, result, exit_reason, int(trade_id)),
    )
    conn.commit()
    conn.close()


def monitorar_operacoes(preco_atual, tempo_maximo):
    abertas = buscar_operacoes_abertas()
    if abertas.empty:
        return []

    fechadas = []
    agora_dt = datetime.now()

    for _, trade in abertas.iterrows():
        trade_id = int(trade["id"])
        side = trade["side"]
        entry_time = pd.to_datetime(trade["entry_time"])
        minutos_aberto = (agora_dt - entry_time.to_pydatetime()).total_seconds() / 60

        exit_reason = None

        if side == "COMPRA":
            if preco_atual <= float(trade["stop_price"]):
                exit_reason = "STOP"
            elif preco_atual >= float(trade["target_price"]):
                exit_reason = "ALVO"
        elif side == "VENDA":
            if preco_atual >= float(trade["stop_price"]):
                exit_reason = "STOP"
            elif preco_atual <= float(trade["target_price"]):
                exit_reason = "ALVO"

        if exit_reason is None and minutos_aberto >= tempo_maximo:
            exit_reason = "TIMEOUT"

        if exit_reason:
            fechar_trade(trade_id, preco_atual, exit_reason)
            # O resultado GANHO/PERDA é calculado pelo P&L real no fechamento.
            fechadas.append((trade_id, exit_reason))

    return fechadas

# ============================================================
# BINANCE
# ============================================================
@st.cache_data(ttl=4, show_spinner=False)
def buscar_klines():
    url = f"{BINANCE_API}/api/v3/klines"
    params = {"symbol": SYMBOL, "interval": INTERVAL, "limit": KLINE_LIMIT}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    cols = [
        "open_time", "Open", "High", "Low", "Close", "Volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(data, columns=cols)

    numeric_cols = ["Open", "High", "Low", "Close", "Volume", "quote_volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    return df


@st.cache_data(ttl=2, show_spinner=False)
def buscar_preco():
    url = f"{BINANCE_API}/api/v3/ticker/price"
    r = requests.get(url, params={"symbol": SYMBOL}, timeout=10)
    r.raise_for_status()
    return float(r.json()["price"])


@st.cache_data(ttl=2, show_spinner=False)
def buscar_orderbook():
    url = f"{BINANCE_API}/api/v3/depth"
    r = requests.get(url, params={"symbol": SYMBOL, "limit": 50}, timeout=10)
    r.raise_for_status()
    data = r.json()

    bids = sum(float(price) * float(qty) for price, qty in data.get("bids", []))
    asks = sum(float(price) * float(qty) for price, qty in data.get("asks", []))

    total = bids + asks
    imbalance = (bids - asks) / total if total > 0 else 0.0

    return {
        "bids": bids,
        "asks": asks,
        "imbalance": imbalance,
    }

# ============================================================
# INDICADORES
# ============================================================
def calcular_indicadores(df):
    df = df.copy()

    df["ret_1"] = df["Close"].pct_change(1)
    df["ret_3"] = df["Close"].pct_change(3)
    df["ret_12"] = df["Close"].pct_change(12)
    df["ret_48"] = df["Close"].pct_change(48)

    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    df["volatility"] = df["ret_1"].rolling(48).std()
    vol_mean = df["Volume"].rolling(100).mean()
    vol_std = df["Volume"].rolling(100).std()
    df["vol_z"] = (df["Volume"] - vol_mean) / vol_std.replace(0, np.nan)

    df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    vr_mean = df["volume_ratio"].rolling(20).mean()
    vr_std = df["volume_ratio"].rolling(20).std()
    df["volume_z"] = (df["volume_ratio"] - vr_mean) / vr_std.replace(0, np.nan)

    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    df["VWAP"] = (
        (typical_price * df["Volume"]).rolling(48).sum()
        / df["Volume"].rolling(48).sum()
    )

    direction = np.sign(df["Close"].diff()).fillna(0)
    df["cvd"] = (direction * df["Volume"]).cumsum()
    df["cvd_delta"] = df["cvd"].diff()

    return df

# ============================================================
# SCORE / SINAL
# ============================================================
def gerar_score(row, imbalance):
    compra = 0
    venda = 0
    fatores_compra = []
    fatores_venda = []

    def add_compra(pontos, texto):
        nonlocal compra
        compra += pontos
        fatores_compra.append(f"{texto} (+{pontos})")

    def add_venda(pontos, texto):
        nonlocal venda
        venda += pontos
        fatores_venda.append(f"{texto} (+{pontos})")

    if row["EMA20"] > row["EMA50"]:
        add_compra(15, "EMA20 > EMA50")
    else:
        add_venda(15, "EMA20 <= EMA50")

    if row["Close"] > row["EMA200"]:
        add_compra(10, "Preço > EMA200")
    else:
        add_venda(10, "Preço <= EMA200")

    if row["ret_12"] > 0:
        add_compra(10, "Retorno 12 candles > 0")
    else:
        add_venda(10, "Retorno 12 candles <= 0")

    if row["ret_48"] > 0:
        add_compra(10, "Retorno 48 candles > 0")
    else:
        add_venda(10, "Retorno 48 candles <= 0")

    if 50 <= row["RSI"] <= 70:
        add_compra(10, "RSI entre 50 e 70")
    elif 30 <= row["RSI"] < 45:
        add_venda(10, "RSI entre 30 e 45")
    elif 45 <= row["RSI"] < 50:
        add_venda(5, "RSI entre 45 e 50")

    if row["volume_z"] > 1 and row["ret_1"] > 0:
        add_compra(10, "Volume Z > 1 com retorno positivo")
    elif row["volume_z"] > 1 and row["ret_1"] < 0:
        add_venda(10, "Volume Z > 1 com retorno negativo")

    if row["Close"] > row["VWAP"]:
        add_compra(10, "Preço > VWAP")
    else:
        add_venda(10, "Preço <= VWAP")

    if row["cvd_delta"] > 0:
        add_compra(10, "CVD delta > 0")
    elif row["cvd_delta"] < 0:
        add_venda(10, "CVD delta < 0")

    if imbalance > 0.10:
        add_compra(10, "Order book imbalance > 0,10")
    elif imbalance < -0.10:
        add_venda(10, "Order book imbalance < -0,10")

    if compra >= 65 and compra > venda + 10:
        sinal = "COMPRA"
    elif venda >= 65 and venda > compra + 10:
        sinal = "VENDA"
    else:
        sinal = "AGUARDAR"

    if row["EMA20"] > row["EMA50"] and row["Close"] > row["EMA200"]:
        regime = "ALTA"
    elif row["EMA20"] < row["EMA50"] and row["Close"] < row["EMA200"]:
        regime = "BAIXA"
    else:
        regime = "LATERAL"

    return (
        compra,
        venda,
        sinal,
        regime,
        fatores_compra,
        fatores_venda,
    )


def montar_motivo_entrada(
    side,
    score_compra,
    score_venda,
    regime,
    fatores_compra,
    fatores_venda,
):
    fatores = fatores_compra if side == "COMPRA" else fatores_venda
    score = score_compra if side == "COMPRA" else score_venda

    partes = [
        side,
        f"Score {score:.0f}",
        f"Regime {regime}",
    ]
    partes.extend(fatores)
    return " | ".join(partes)


def calcular_plano(side, preco, atr):
    if side == "COMPRA":
        stop = preco - atr * ATR_STOP_MULTIPLIER
        target = preco + atr * ATR_TARGET_MULTIPLIER
    elif side == "VENDA":
        stop = preco + atr * ATR_STOP_MULTIPLIER
        target = preco - atr * ATR_TARGET_MULTIPLIER
    else:
        raise ValueError("Side inválido para cálculo do plano.")

    valido, mensagem = validar_plano(side, preco, stop, target)
    if not valido:
        raise ValueError(mensagem)

    return stop, target

# ============================================================
# SIMULAÇÃO DE BANCA
# ============================================================
def simular_banca(historico, banca_inicial, percentual_entrada):
    if historico.empty:
        return pd.DataFrame(), float(banca_inicial)

    df = historico.copy()
    df = df[df["pnl_pct"].notna()].copy()
    if df.empty:
        return pd.DataFrame(), float(banca_inicial)

    df["entry_time_sort"] = pd.to_datetime(df["entry_time"], errors="coerce")
    df = df.sort_values(["entry_time_sort", "id"]).reset_index(drop=True)

    banca = float(banca_inicial)
    registros = []
    taxa = float(percentual_entrada) / 100.0

    for _, trade in df.iterrows():
        banca_antes = banca
        valor_entrada = banca_antes * taxa
        pnl_pct = float(trade["pnl_pct"])
        resultado_rs = valor_entrada * (pnl_pct / 100.0)
        banca = banca_antes + resultado_rs

        registros.append({
            "#": int(trade["id"]),
            "Lado": trade["side"],
            "Entrada": trade["entry_time"],
            "Saída": trade["exit_time"],
            "P&L mercado %": pnl_pct,
            "Banca antes": banca_antes,
            "Entrada 1%": valor_entrada,
            "Resultado R$": resultado_rs,
            "Banca depois": banca,
            "Resultado": trade["result"] if pd.notna(trade["result"]) else ("GANHO" if pnl_pct > 0 else "PERDA" if pnl_pct < 0 else "EMPATE"),
            "Saída por": trade["exit_reason"] if pd.notna(trade.get("exit_reason")) else "-",
        })

    return pd.DataFrame(registros), banca


# ============================================================
# INTERFACE
# ============================================================
st.title("₿ BTC Quant Trader — Paper Trading v2")
st.caption(
    "BTC/USDT • candles de 5 minutos • dados públicos Binance • sem ordens reais • banco: btc_trader_v2.db"
)

# Sidebar
with st.sidebar:
    st.header("Configuração")
    max_operacoes = st.number_input(
        "Máximo de operações abertas",
        min_value=1,
        max_value=50,
        value=5,
        step=1,
    )
    tempo_maximo = st.number_input(
        "Tempo máximo por operação (min)",
        min_value=5,
        max_value=1440,
        value=60,
        step=5,
    )
    banca_inicial = st.number_input(
        "Banca inicial da simulação (R$)",
        min_value=1.0,
        value=1000.0,
        step=100.0,
    )
    percentual_entrada = st.number_input(
        "Entrada por operação (% da banca)",
        min_value=0.1,
        max_value=100.0,
        value=1.0,
        step=0.1,
    )
    st.divider()
    st.write(f"**Stop:** {ATR_STOP_MULTIPLIER:.1f} × ATR")
    st.write(f"**Alvo:** {ATR_TARGET_MULTIPLIER:.1f} × ATR")
    st.write(f"**Refresh:** {AUTO_REFRESH_SECONDS}s")
    st.write("**Entrada:** somente sinal confirmado")
    st.write("**Banco:** `btc_trader_v2.db`")

# ============================================================
# MONITORAMENTO
# ============================================================
@st.fragment(run_every=AUTO_REFRESH_SECONDS)
def monitor():
    try:
        df = buscar_klines()
        preco_atual = buscar_preco()
        orderbook = buscar_orderbook()

        df = calcular_indicadores(df)

        # Último candle FECHADO. O último registro pode ainda estar em formação.
        row = df.iloc[-2]
        signal_time = row["close_time"]

        (
            score_compra,
            score_venda,
            sinal,
            regime,
            fatores_compra,
            fatores_venda,
        ) = gerar_score(row, orderbook["imbalance"])

        # 4) Auditoria: tudo usado para gerar o sinal é salvo na entrada.
        motivo_entrada = montar_motivo_entrada(
            sinal,
            score_compra,
            score_venda,
            regime,
            fatores_compra,
            fatores_venda,
        )

        # Monitora operações já abertas antes de procurar nova entrada.
        fechadas = monitorar_operacoes(preco_atual, tempo_maximo)
        if fechadas:
            st.toast(
                " | ".join(f"#{trade_id}: {resultado}" for trade_id, resultado in fechadas)
            )

        abertas = buscar_operacoes_abertas()
        quantidade_abertas = len(abertas)

        # Entrada automática
        if quantidade_abertas < int(max_operacoes):
            if sinal in ["COMPRA", "VENDA"]:
                if not entrada_ja_registrada(signal_time):
                    atr = float(row["ATR"])

                    if np.isfinite(atr) and atr > 0:
                        entrada = float(preco_atual)
                        stop, alvo = calcular_plano(sinal, entrada, atr)
                        score = score_compra if sinal == "COMPRA" else score_venda

                        trade_id = registrar_trade(
                            side=sinal,
                            entry_price=entrada,
                            stop_price=stop,
                            target_price=alvo,
                            score=score,
                            regime=regime,
                            signal_time=signal_time,
                            row=row,
                            imbalance=orderbook["imbalance"],
                            score_compra=score_compra,
                            score_venda=score_venda,
                            signal=sinal,
                            entry_reason=motivo_entrada,
                            notes="Entrada automática por sinal confirmado. Snapshot completo salvo para auditoria.",
                        )

                        st.success(
                            f"Nova operação #{trade_id}: {sinal} | "
                            f"Entrada ${entrada:,.2f} | Score {score:.0f}"
                        )

                        # Recarrega para refletir a nova operação imediatamente.
                        abertas = buscar_operacoes_abertas()
                        quantidade_abertas = len(abertas)

        # --------------------------------------------------------
        # Painel principal
        # --------------------------------------------------------
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("BTC", f"${preco_atual:,.2f}")
        c2.metric("Sinal", sinal)
        c3.metric("Score COMPRA", f"{score_compra:.0f}")
        c4.metric("Score VENDA", f"{score_venda:.0f}")
        c5.metric("Regime", regime)

        st.divider()

        # Sinal e plano
        left, right = st.columns([1.15, 1])

        with left:
            st.subheader("Sinal atual")
            if sinal == "COMPRA":
                st.success("🟢 COMPRA")
            elif sinal == "VENDA":
                st.error("🔴 VENDA")
            else:
                st.info("🟡 AGUARDAR")

            st.write(f"**Candle analisado:** {signal_time}")
            st.write(f"**Preço do candle:** ${float(row['Close']):,.2f}")
            st.write(f"**Preço atual:** ${preco_atual:,.2f}")
            st.write(f"**Order book imbalance:** {orderbook['imbalance']:+.4f}")

            if sinal in ["COMPRA", "VENDA"]:
                try:
                    stop_view, alvo_view = calcular_plano(sinal, preco_atual, float(row["ATR"]))
                    st.write(f"**Stop:** ${stop_view:,.2f}")
                    st.write(f"**Alvo:** ${alvo_view:,.2f}")
                except ValueError as exc:
                    st.warning(str(exc))

        with right:
            st.subheader("Fatores do sinal")
            if sinal == "COMPRA":
                for fator in fatores_compra:
                    st.write("•", fator)
            elif sinal == "VENDA":
                for fator in fatores_venda:
                    st.write("•", fator)
            else:
                st.write("Nenhum lado atingiu os critérios de entrada.")

        st.divider()

        # Operações abertas
        st.subheader(f"Operações abertas ({quantidade_abertas}/{int(max_operacoes)})")

        if abertas.empty:
            st.info("Nenhuma operação aberta.")
        else:
            exibicao = abertas.copy()
            exibicao["entry_price"] = exibicao["entry_price"].map(lambda x: f"${x:,.2f}")
            exibicao["stop_price"] = exibicao["stop_price"].map(lambda x: f"${x:,.2f}")
            exibicao["target_price"] = exibicao["target_price"].map(lambda x: f"${x:,.2f}")

            def pnl_atual(row_trade):
                if row_trade["side"] == "COMPRA":
                    return ((preco_atual / float(row_trade["entry_price"].replace('$', '').replace(',', ''))) - 1) * 100
                return ((float(row_trade["entry_price"].replace('$', '').replace(',', '')) / preco_atual) - 1) * 100

            # Mantém P&L atual separado para não alterar a coluna original do banco.
            exibicao["P&L atual %"] = [
                pnl_atual(exibicao.iloc[i]) for i in range(len(exibicao))
            ]
            exibicao["P&L atual %"] = exibicao["P&L atual %"].map(lambda x: f"{x:+.2f}%")

            exibicao = exibicao.rename(
                columns={
                    "id": "#",
                    "side": "Lado",
                    "entry_time": "Entrada",
                    "entry_price": "Preço entrada",
                    "stop_price": "Stop",
                    "target_price": "Alvo",
                    "score": "Score",
                    "regime": "Regime",
                }
            )
            colunas = [
                "#", "Lado", "Entrada", "Preço entrada", "Stop", "Alvo",
                "P&L atual %", "Score", "Regime",
            ]
            st.dataframe(exibicao[colunas], use_container_width=True, hide_index=True)

        # --------------------------------------------------------
        # Auditoria da última leitura
        # --------------------------------------------------------
        with st.expander("🔎 Snapshot do sinal atual / auditoria"):
            audit = pd.DataFrame(
                {
                    "Indicador": [
                        "EMA20", "EMA50", "EMA200", "RSI", "ATR",
                        "ret_1", "ret_3", "ret_12", "ret_48", "volatility",
                        "vol_z", "volume_ratio", "volume_z", "VWAP",
                        "cvd_delta", "orderbook_imbalance", "score_compra",
                        "score_venda", "regime", "signal", "signal_time",
                    ],
                    "Valor": [
                        row["EMA20"], row["EMA50"], row["EMA200"], row["RSI"], row["ATR"],
                        row["ret_1"], row["ret_3"], row["ret_12"], row["ret_48"], row["volatility"],
                        row["vol_z"], row["volume_ratio"], row["volume_z"], row["VWAP"],
                        row["cvd_delta"], orderbook["imbalance"], score_compra,
                        score_venda, regime, sinal, str(signal_time),
                    ],
                }
            )
            st.dataframe(audit, use_container_width=True, hide_index=True)
            if sinal in ["COMPRA", "VENDA"]:
                st.markdown("**Motivo que será salvo se houver entrada:**")
                st.code(motivo_entrada, language="text")

        # --------------------------------------------------------
        # Histórico
        # --------------------------------------------------------
        st.subheader("Histórico")
        historico = buscar_historico(100)
        if historico.empty:
            st.info("Ainda não existem operações no novo banco.")
        else:
            hist = historico.copy()
            hist = hist.rename(
                columns={
                    "id": "#",
                    "side": "Lado",
                    "entry_time": "Entrada",
                    "entry_price": "Preço entrada",
                    "stop_price": "Stop",
                    "target_price": "Alvo",
                    "exit_time": "Saída",
                    "exit_price": "Preço saída",
                    "pnl_pct": "P&L %",
                    "result": "Resultado",
                    "exit_reason": "Saída por",
                    "score": "Score",
                    "regime": "Regime",
                    "signal": "Sinal",
                }
            )
            hist["Resultado"] = hist["Resultado"].fillna("ABERTA")
            hist["Saída por"] = hist["Saída por"].fillna("-")
            hist["P&L %"] = hist["P&L %"].map(
                lambda x: "-" if pd.isna(x) else f"{x:+.2f}%"
            )
            for col in ["Preço entrada", "Stop", "Alvo", "Preço saída"]:
                if col in hist.columns:
                    hist[col] = hist[col].map(
                        lambda x: "-" if pd.isna(x) else f"${float(x):,.2f}"
                    )

            hist_cols = [
                "#", "Lado", "Entrada", "Preço entrada", "Stop", "Alvo",
                "Saída", "Preço saída", "P&L %", "Resultado", "Saída por",
                "Score", "Regime", "Sinal",
            ]
            st.dataframe(hist[hist_cols], use_container_width=True, hide_index=True)

            # Simulação de banca: 1% da banca em cada operação.
            sim, banca_final = simular_banca(historico, banca_inicial, percentual_entrada)
            st.markdown("### 💰 Simulação da banca")
            st.caption(
                f"Banca inicial: R$ {banca_inicial:,.2f} • "
                f"Entrada: {percentual_entrada:.1f}% da banca em cada operação • "
                "O lucro/prejuízo da operação é aplicado proporcionalmente ao valor da entrada."
            )

            if sim.empty:
                st.info("A simulação aparecerá quando houver pelo menos uma operação fechada.")
            else:
                ganhos_sim = int((sim["Resultado"] == "GANHO").sum())
                perdas_sim = int((sim["Resultado"] == "PERDA").sum())
                banca_inicial_sim = float(banca_inicial)
                lucro_total_rs = banca_final - banca_inicial_sim
                retorno_banca = (banca_final / banca_inicial_sim - 1) * 100 if banca_inicial_sim else 0

                b1, b2, b3, b4 = st.columns(4)
                b1.metric("Banca atual", f"R$ {banca_final:,.2f}")
                b2.metric("Lucro / prejuízo", f"R$ {lucro_total_rs:+,.2f}")
                b3.metric("Retorno da banca", f"{retorno_banca:+.2f}%")
                b4.metric("Ganhas / Perdidas", f"{ganhos_sim} / {perdas_sim}")

                sim_exib = sim.copy()
                sim_exib["P&L mercado %"] = sim_exib["P&L mercado %"].map(lambda x: f"{x:+.2f}%")
                for col in ["Banca antes", "Entrada 1%", "Resultado R$", "Banca depois"]:
                    sim_exib[col] = sim_exib[col].map(lambda x: f"R$ {x:,.2f}")
                sim_exib["Resultado"] = sim_exib["Resultado"].fillna("-")
                st.dataframe(
                    sim_exib[[
                        "#", "Lado", "Entrada", "Saída", "P&L mercado %",
                        "Banca antes", "Entrada 1%", "Resultado R$",
                        "Banca depois", "Resultado", "Saída por",
                    ]],
                    use_container_width=True,
                    hide_index=True,
                )

            ultimo = historico.iloc[0]
            with st.expander("📋 Motivo da última operação registrada"):
                st.write(f"**Operação #{int(ultimo['id'])} — {ultimo['side']}**")
                st.write(f"**Resultado:** {ultimo['result'] if pd.notna(ultimo['result']) else 'ABERTA'}")
                if pd.notna(ultimo.get("exit_reason")):
                    st.write(f"**Saída por:** {ultimo['exit_reason']}")
                st.code(str(ultimo["entry_reason"]), language="text")

        # --------------------------------------------------------
        # Estatísticas
        # --------------------------------------------------------
        conn = get_conn()
        stats = pd.read_sql_query(
            "SELECT COUNT(*) AS total, SUM(CASE WHEN result = 'GANHO' THEN 1 ELSE 0 END) AS ganhos, COALESCE(SUM(pnl_pct), 0) AS pnl_total FROM trades WHERE result IS NOT NULL",
            conn,
        )
        conn.close()

        total = int(stats.iloc[0]["total"] or 0)
        ganhos = int(stats.iloc[0]["ganhos"] or 0)
        pnl_total = float(stats.iloc[0]["pnl_total"] or 0)
        wr = (ganhos / total * 100) if total else 0

        st.divider()
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Operações fechadas", total)
        s2.metric("Ganhas", ganhos)
        s3.metric("WR", f"{wr:.2f}%")
        s4.metric("P&L acumulado", f"{pnl_total:+.2f}%")

    except requests.RequestException as exc:
        st.error(f"Erro ao consultar Binance: {exc}")
    except Exception as exc:
        st.error(f"Erro no monitoramento: {exc}")


monitor()
