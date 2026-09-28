import sqlite3
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# CONFIGURAÇÕES
# ============================================================

BINANCE_API = "https://data-api.binance.vision"
SYMBOL = "BTCUSDT"
INTERVAL = "5m"

DATABASE = "btc_trader.db"

AUTO_REFRESH_SECONDS = 10

ATR_STOP_MULTIPLIER = 1.0
ATR_TARGET_MULTIPLIER = 2.0

KLINE_LIMIT = 500


# ============================================================
# STREAMLIT
# ============================================================

st.set_page_config(
    page_title="BTC Quant Trader",
    page_icon="₿",
    layout="wide"
)


# ============================================================
# BANCO DE DADOS
# ============================================================

def conectar():

    return sqlite3.connect(
        DATABASE,
        check_same_thread=False
    )


def criar_banco():

    conn = conectar()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            symbol TEXT,
            side TEXT,

            entry_time TEXT,
            entry_price REAL,

            stop_price REAL,
            target_price REAL,

            exit_time TEXT,
            exit_price REAL,

            pnl_pct REAL,
            result TEXT,

            score REAL,
            regime TEXT,

            notes TEXT,
            signal_time TEXT
        )
    """)

    conn.commit()

    # Verifica colunas existentes
    colunas_existentes = {
        row[1]
        for row in conn.execute(
            "PRAGMA table_info(trades)"
        ).fetchall()
    }

    # Migração para bancos antigos
    if "signal_time" not in colunas_existentes:

        conn.execute("""
            ALTER TABLE trades
            ADD COLUMN signal_time TEXT
        """)

    conn.commit()
    conn.close()


# ============================================================
# BUSCAR TRADE ABERTO
# ============================================================

def buscar_trade_aberto():

    conn = conectar()

    cursor = conn.execute(
        """
        SELECT
            id,
            symbol,
            side,
            entry_time,
            entry_price,
            stop_price,
            target_price,
            exit_time,
            exit_price,
            pnl_pct,
            result,
            score,
            regime,
            notes,
            signal_time
        FROM trades
        WHERE exit_time IS NULL
        ORDER BY id DESC
        LIMIT 1
        """
    )

    row = cursor.fetchone()

    conn.close()

    if row is None:
        return None

    colunas = [
        "id",
        "symbol",
        "side",
        "entry_time",
        "entry_price",
        "stop_price",
        "target_price",
        "exit_time",
        "exit_price",
        "pnl_pct",
        "result",
        "score",
        "regime",
        "notes",
        "signal_time"
    ]

    return dict(zip(colunas, row))


# ============================================================
# VERIFICAR SE SINAL JÁ FOI REGISTRADO
# ============================================================

def entrada_ja_registrada(signal_time):

    conn = conectar()

    resultado = conn.execute(
        """
        SELECT COUNT(*)
        FROM trades
        WHERE signal_time = ?
        """,
        (signal_time,)
    ).fetchone()

    conn.close()

    if resultado is None:
        return False

    return resultado[0] > 0


# ============================================================
# REGISTRAR TRADE
# ============================================================

def registrar_trade(
    side,
    entry_price,
    stop_price,
    target_price,
    score,
    regime,
    signal_time,
    notes=""
):

    conn = conectar()

    entry_time = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    conn.execute(
        """
        INSERT INTO trades (
            symbol,
            side,
            entry_time,
            entry_price,
            stop_price,
            target_price,
            exit_time,
            exit_price,
            pnl_pct,
            result,
            score,
            regime,
            notes,
            signal_time
        )
        VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, ?, ?, ?, ?)
        """,
        (
            SYMBOL,
            side,
            entry_time,
            float(entry_price),
            float(stop_price),
            float(target_price),
            float(score),
            regime,
            notes,
            signal_time
        )
    )

    conn.commit()
    conn.close()


# ============================================================
# FECHAR TRADE
# ============================================================

def fechar_trade(
    trade_id,
    exit_price,
    result
):

    conn = conectar()

    trade = conn.execute(
        """
        SELECT
            side,
            entry_price
        FROM trades
        WHERE id = ?
        """,
        (trade_id,)
    ).fetchone()

    if trade is None:

        conn.close()
        return

    side = trade[0]
    entry_price = float(trade[1])
    exit_price = float(exit_price)

    # Cálculo correto de P&L
    if side == "LONG":

        pnl_pct = (
            (exit_price / entry_price) - 1
        ) * 100

    else:

        pnl_pct = (
            (entry_price / exit_price) - 1
        ) * 100

    exit_time = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    conn.execute(
        """
        UPDATE trades
        SET
            exit_time = ?,
            exit_price = ?,
            pnl_pct = ?,
            result = ?
        WHERE id = ?
        """,
        (
            exit_time,
            exit_price,
            pnl_pct,
            result,
            trade_id
        )
    )

    conn.commit()
    conn.close()


# ============================================================
# BINANCE - KLINES
# ============================================================

def buscar_klines():

    url = f"{BINANCE_API}/api/v3/klines"

    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "limit": KLINE_LIMIT
    }

    response = requests.get(
        url,
        params=params,
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    colunas = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore"
    ]

    df = pd.DataFrame(
        data,
        columns=colunas
    )

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume"
    ]

    for col in numeric_cols:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        unit="ms"
    )

    df["close_time"] = pd.to_datetime(
        df["close_time"],
        unit="ms"
    )

    return df


# ============================================================
# PREÇO ATUAL
# ============================================================

def buscar_preco():

    url = f"{BINANCE_API}/api/v3/ticker/price"

    params = {
        "symbol": SYMBOL
    }

    response = requests.get(
        url,
        params=params,
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    return float(data["price"])


# ============================================================
# ORDER BOOK
# ============================================================

def buscar_orderbook():

    url = f"{BINANCE_API}/api/v3/depth"

    params = {
        "symbol": SYMBOL,
        "limit": 50
    }

    response = requests.get(
        url,
        params=params,
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    bids = sum(
        float(item[1])
        for item in data["bids"]
    )

    asks = sum(
        float(item[1])
        for item in data["asks"]
    )

    total = bids + asks

    if total == 0:

        imbalance = 0

    else:

        imbalance = (
            (bids - asks) / total
        )

    return {
        "bids": bids,
        "asks": asks,
        "imbalance": imbalance
    }


# ============================================================
# INDICADORES
# ============================================================

def calcular_indicadores(df):

    df = df.copy()

    # Retornos
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_12"] = df["close"].pct_change(12)
    df["ret_48"] = df["close"].pct_change(48)

    # EMAs
    df["EMA20"] = (
        df["close"]
        .ewm(span=20, adjust=False)
        .mean()
    )

    df["EMA50"] = (
        df["close"]
        .ewm(span=50, adjust=False)
        .mean()
    )

    df["EMA200"] = (
        df["close"]
        .ewm(span=200, adjust=False)
        .mean()
    )

    # RSI
    delta = df["close"].diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = (
        gain
        .rolling(14)
        .mean()
    )

    avg_loss = (
        loss
        .rolling(14)
        .mean()
    )

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    df["RSI"] = (
        100 -
        (100 / (1 + rs))
    )

    # ATR
    high_low = (
        df["high"] -
        df["low"]
    )

    high_close = (
        df["high"] -
        df["close"].shift()
    ).abs()

    low_close = (
        df["low"] -
        df["close"].shift()
    ).abs()

    true_range = pd.concat(
        [
            high_low,
            high_close,
            low_close
        ],
        axis=1
    ).max(axis=1)

    df["ATR"] = (
        true_range
        .rolling(14)
        .mean()
    )

    # Volatilidade
    df["volatility"] = (
        df["ret_1"]
        .rolling(48)
        .std()
    )

    vol_mean = (
        df["volatility"]
        .rolling(100)
        .mean()
    )

    vol_std = (
        df["volatility"]
        .rolling(100)
        .std()
    )

    df["vol_z"] = (
        (df["volatility"] - vol_mean) /
        vol_std.replace(0, np.nan)
    )

    # Volume
    volume_mean = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    volume_std = (
        df["volume"]
        .rolling(20)
        .std()
    )

    df["volume_ratio"] = (
        df["volume"] /
        volume_mean.replace(0, np.nan)
    )

    df["volume_z"] = (
        (df["volume"] - volume_mean) /
        volume_std.replace(0, np.nan)
    )

    # VWAP
    typical_price = (
        df["high"] +
        df["low"] +
        df["close"]
    ) / 3

    df["VWAP"] = (
        typical_price * df["volume"]
    ).rolling(48).sum() / (
        df["volume"]
        .rolling(48)
        .sum()
        .replace(0, np.nan)
    )

    # CVD aproximado
    direction = np.sign(
        df["close"].diff()
    )

    df["cvd"] = (
        direction *
        df["volume"]
    ).cumsum()

    df["cvd_delta"] = (
        df["cvd"]
        .diff()
    )

    return df


# ============================================================
# SCORE
# ============================================================

def gerar_score(
    row,
    imbalance
):

    long_score = 0
    short_score = 0

    fatores_long = []
    fatores_short = []

    # --------------------------------------------------------
    # EMA 20 / 50
    # --------------------------------------------------------

    if row["EMA20"] > row["EMA50"]:

        long_score += 15

        fatores_long.append(
            ("EMA20 > EMA50", 15)
        )

    else:

        short_score += 15

        fatores_short.append(
            ("EMA20 < EMA50", 15)
        )

    # --------------------------------------------------------
    # EMA 200
    # --------------------------------------------------------

    if row["close"] > row["EMA200"]:

        long_score += 10

        fatores_long.append(
            ("Preço > EMA200", 10)
        )

    else:

        short_score += 10

        fatores_short.append(
            ("Preço < EMA200", 10)
        )

    # --------------------------------------------------------
    # Retorno 12 candles
    # --------------------------------------------------------

    if row["ret_12"] > 0:

        long_score += 10

        fatores_long.append(
            ("Momentum 12 períodos", 10)
        )

    else:

        short_score += 10

        fatores_short.append(
            ("Momentum 12 períodos", 10)
        )

    # --------------------------------------------------------
    # Retorno 48 candles
    # --------------------------------------------------------

    if row["ret_48"] > 0:

        long_score += 10

        fatores_long.append(
            ("Momentum 48 períodos", 10)
        )

    else:

        short_score += 10

        fatores_short.append(
            ("Momentum 48 períodos", 10)
        )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    rsi = row["RSI"]

    if 50 <= rsi <= 70:

        long_score += 10

        fatores_long.append(
            (f"RSI favorável ({rsi:.1f})", 10)
        )

    elif 30 <= rsi <= 45:

        short_score += 10

        fatores_short.append(
            (f"RSI favorável ({rsi:.1f})", 10)
        )

    elif 45 < rsi < 50:

        short_score += 5

        fatores_short.append(
            (f"RSI levemente vendedor ({rsi:.1f})", 5)
        )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    if (
        row["volume_z"] > 1 and
        row["ret_1"] > 0
    ):

        long_score += 10

        fatores_long.append(
            ("Volume forte + candle positivo", 10)
        )

    elif (
        row["volume_z"] > 1 and
        row["ret_1"] < 0
    ):

        short_score += 10

        fatores_short.append(
            ("Volume forte + candle negativo", 10)
        )

    # --------------------------------------------------------
    # VWAP
    # --------------------------------------------------------

    if row["close"] > row["VWAP"]:

        long_score += 10

        fatores_long.append(
            ("Preço > VWAP", 10)
        )

    else:

        short_score += 10

        fatores_short.append(
            ("Preço < VWAP", 10)
        )

    # --------------------------------------------------------
    # CVD
    # --------------------------------------------------------

    if row["cvd_delta"] > 0:

        long_score += 10

        fatores_long.append(
            ("CVD positivo", 10)
        )

    elif row["cvd_delta"] < 0:

        short_score += 10

        fatores_short.append(
            ("CVD negativo", 10)
        )

    # --------------------------------------------------------
    # Order Book
    # --------------------------------------------------------

    if imbalance > 0.10:

        long_score += 10

        fatores_long.append(
            (
                f"Order Book comprador ({imbalance:.2f})",
                10
            )
        )

    elif imbalance < -0.10:

        short_score += 10

        fatores_short.append(
            (
                f"Order Book vendedor ({imbalance:.2f})",
                10
            )
        )

    # --------------------------------------------------------
    # REGIME
    # --------------------------------------------------------

    if (
        row["EMA20"] > row["EMA50"] and
        row["close"] > row["EMA200"]
    ):

        regime = "ALTA"

    elif (
        row["EMA20"] < row["EMA50"] and
        row["close"] < row["EMA200"]
    ):

        regime = "BAIXA"

    else:

        regime = "LATERAL"

    # --------------------------------------------------------
    # SINAL
    # --------------------------------------------------------

    if (
        long_score >= 65 and
        long_score > short_score + 10
    ):

        sinal = "LONG"

    elif (
        short_score >= 65 and
        short_score > long_score + 10
    ):

        sinal = "SHORT"

    else:

        sinal = "NEUTRO"

    return (
        long_score,
        short_score,
        sinal,
        regime,
        fatores_long,
        fatores_short
    )


# ============================================================
# PLANO DE TRADE
# ============================================================

def calcular_plano(
    side,
    preco,
    atr
):

    if side == "LONG":

        stop = (
            preco -
            atr * ATR_STOP_MULTIPLIER
        )

        target = (
            preco +
            atr * ATR_TARGET_MULTIPLIER
        )

    else:

        stop = (
            preco +
            atr * ATR_STOP_MULTIPLIER
        )

        target = (
            preco -
            atr * ATR_TARGET_MULTIPLIER
        )

    return stop, target


# ============================================================
# MONITORAR TRADE ABERTO
# ============================================================

def monitorar_trade_aberto(
    trade,
    preco_atual
):

    if trade is None:

        return

    side = trade["side"]

    stop = float(
        trade["stop_price"]
    )

    target = float(
        trade["target_price"]
    )

    trade_id = trade["id"]

    # LONG
    if side == "LONG":

        if preco_atual <= stop:

            fechar_trade(
                trade_id,
                preco_atual,
                "LOSS"
            )

            st.session_state["ultimo_evento"] = (
                f"LOSS LONG em ${preco_atual:,.2f}"
            )

        elif preco_atual >= target:

            fechar_trade(
                trade_id,
                preco_atual,
                "GAIN"
            )

            st.session_state["ultimo_evento"] = (
                f"GAIN LONG em ${preco_atual:,.2f}"
            )

    # SHORT
    elif side == "SHORT":

        if preco_atual >= stop:

            fechar_trade(
                trade_id,
                preco_atual,
                "LOSS"
            )

            st.session_state["ultimo_evento"] = (
                f"LOSS SHORT em ${preco_atual:,.2f}"
            )

        elif preco_atual <= target:

            fechar_trade(
                trade_id,
                preco_atual,
                "GAIN"
            )

            st.session_state["ultimo_evento"] = (
                f"GAIN SHORT em ${preco_atual:,.2f}"
            )


# ============================================================
# PERFORMANCE
# ============================================================

def buscar_performance():

    conn = conectar()

    df = pd.read_sql_query(
        """
        SELECT *
        FROM trades
        ORDER BY id DESC
        """,
        conn
    )

    conn.close()

    if df.empty:

        return {
            "trades": 0,
            "gains": 0,
            "losses": 0,
            "wr": 0,
            "pnl": 0
        }

    fechados = df[
        df["exit_time"].notna()
    ].copy()

    if fechados.empty:

        return {
            "trades": 0,
            "gains": 0,
            "losses": 0,
            "wr": 0,
            "pnl": 0
        }

    ganhos = (
        fechados["result"] == "GAIN"
    ).sum()

    perdas = (
        fechados["result"] == "LOSS"
    ).sum()

    total = len(fechados)

    wr = (
        ganhos / total * 100
        if total > 0
        else 0
    )

    pnl = (
        fechados["pnl_pct"]
        .fillna(0)
        .sum()
    )

    return {
        "trades": total,
        "gains": int(ganhos),
        "losses": int(perdas),
        "wr": wr,
        "pnl": pnl
    }


# ============================================================
# INICIALIZAÇÃO
# ============================================================

criar_banco()

if "ultimo_evento" not in st.session_state:

    st.session_state["ultimo_evento"] = ""


# ============================================================
# CABEÇALHO
# ============================================================

st.title("₿ BTC Quant Trader")

st.caption(
    "Monitoramento automático • "
    "BTCUSDT • 5 minutos • Paper Trading"
)

st.info(
    "🤖 ROBÔ EM MODO PAPER TRADING — "
    "nenhuma ordem real é enviada para a Binance."
)


# ============================================================
# MONITORAMENTO AUTOMÁTICO
# ============================================================

@st.fragment(run_every=AUTO_REFRESH_SECONDS)
def monitor():

    try:

        # ----------------------------------------------------
        # DADOS
        # ----------------------------------------------------

        df = buscar_klines()

        preco_atual = buscar_preco()

        orderbook = buscar_orderbook()

        df = calcular_indicadores(df)

        # ----------------------------------------------------
        # ÚLTIMO CANDLE FECHADO
        # ----------------------------------------------------

        if len(df) < 250:

            st.warning(
                "Aguardando histórico suficiente..."
            )

            return

        # O último candle pode ainda estar aberto.
        # Usamos o anterior.
        row = df.iloc[-2]

        candle_time = row["close_time"]

        signal_time = candle_time.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # ----------------------------------------------------
        # SCORE
        # ----------------------------------------------------

        (
            long_score,
            short_score,
            sinal,
            regime,
            fatores_long,
            fatores_short
        ) = gerar_score(
            row,
            orderbook["imbalance"]
        )

        # ----------------------------------------------------
        # TRADE ABERTO
        # ----------------------------------------------------

        trade_aberto = buscar_trade_aberto()

        # ----------------------------------------------------
        # MONITORAR TRADE
        # ----------------------------------------------------

        if trade_aberto is not None:

            monitorar_trade_aberto(
                trade_aberto,
                preco_atual
            )

            # Atualiza após possível fechamento
            trade_aberto = buscar_trade_aberto()

        # ----------------------------------------------------
        # ENTRADA AUTOMÁTICA
        # ----------------------------------------------------

        if trade_aberto is None:

            if sinal in ["LONG", "SHORT"]:

                if not entrada_ja_registrada(
                    signal_time
                ):

                    atr = float(row["ATR"])

                    if (
                        np.isfinite(atr) and
                        atr > 0
                    ):

                        side = sinal

                        entry_price = preco_atual

                        stop, target = (
                            calcular_plano(
                                side,
                                entry_price,
                                atr
                            )
                        )

                        score = (
                            long_score
                            if side == "LONG"
                            else short_score
                        )

                        registrar_trade(
                            side=side,
                            entry_price=entry_price,
                            stop_price=stop,
                            target_price=target,
                            score=score,
                            regime=regime,
                            signal_time=signal_time,
                            notes=(
                                "Entrada automática "
                                "por sinal fechado."
                            )
                        )

                        trade_aberto = (
                            buscar_trade_aberto()
                        )

                        st.session_state[
                            "ultimo_evento"
                        ] = (
                            f"ENTRADA AUTOMÁTICA "
                            f"{side} em "
                            f"${entry_price:,.2f}"
                        )

        # ====================================================
        # PAINEL PRINCIPAL
        # ====================================================

        st.divider()

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "BTC",
                f"${preco_atual:,.2f}"
            )

        with col2:

            st.metric(
                "LONG",
                f"{long_score}/100"
            )

        with col3:

            st.metric(
                "SHORT",
                f"{short_score}/100"
            )

        with col4:

            st.metric(
                "REGIME",
                regime
            )

        # ====================================================
        # DECISÃO
        # ====================================================

        st.subheader(
            "🎯 DECISÃO DO ROBÔ"
        )

        if sinal == "LONG":

            st.success(
                f"🟢 COMPRAR BTC — "
                f"Score LONG {long_score}/100"
            )

        elif sinal == "SHORT":

            st.error(
                f"🔴 VENDER / SHORT BTC — "
                f"Score SHORT {short_score}/100"
            )

        else:

            st.warning(
                f"🟡 AGUARDAR — "
                f"Long {long_score} | "
                f"Short {short_score}"
            )

        # ====================================================
        # ÚLTIMO EVENTO
        # ====================================================

        if st.session_state["ultimo_evento"]:

            st.info(
                "📌 " +
                st.session_state["ultimo_evento"]
            )

        # ====================================================
        # TRADE ABERTO
        # ====================================================

        if trade_aberto is not None:

            st.subheader(
                "📌 TRADE EM ANDAMENTO"
            )

            side = trade_aberto["side"]

            entry = float(
                trade_aberto["entry_price"]
            )

            stop = float(
                trade_aberto["stop_price"]
            )

            target = float(
                trade_aberto["target_price"]
            )

            if side == "LONG":

                pnl_atual = (
                    (preco_atual / entry) - 1
                ) * 100

            else:

                pnl_atual = (
                    (entry / preco_atual) - 1
                ) * 100

            a, b, c, d, e = st.columns(5)

            with a:

                st.metric(
                    "Direção",
                    side
                )

            with b:

                st.metric(
                    "Entrada",
                    f"${entry:,.2f}"
                )

            with c:

                st.metric(
                    "Preço atual",
                    f"${preco_atual:,.2f}"
                )

            with d:

                st.metric(
                    "P&L",
                    f"{pnl_atual:.2f}%"
                )

            with e:

                st.metric(
                    "ID",
                    trade_aberto["id"]
                )

            x, y, z = st.columns(3)

            with x:

                st.write("**Entrada**")
                st.write(
                    f"${entry:,.2f}"
                )

            with y:

                st.write("**Stop**")
                st.write(
                    f"${stop:,.2f}"
                )

            with z:

                st.write("**Target**")
                st.write(
                    f"${target:,.2f}"
                )

        # ====================================================
        # PLANO DO SINAL
        # ====================================================

        if sinal in ["LONG", "SHORT"]:

            atr = float(row["ATR"])

            if np.isfinite(atr) and atr > 0:

                plano_stop, plano_target = (
                    calcular_plano(
                        sinal,
                        preco_atual,
                        atr
                    )
                )

                risco = abs(
                    preco_atual -
                    plano_stop
                )

                recompensa = abs(
                    plano_target -
                    preco_atual
                )

                rr = (
                    recompensa / risco
                    if risco > 0
                    else 0
                )

                st.subheader(
                    "📋 PLANO DO SINAL"
                )

                p1, p2, p3, p4 = st.columns(4)

                with p1:

                    st.metric(
                        "Entrada",
                        f"${preco_atual:,.2f}"
                    )

                with p2:

                    st.metric(
                        "Stop",
                        f"${plano_stop:,.2f}"
                    )

                with p3:

                    st.metric(
                        "Target",
                        f"${plano_target:,.2f}"
                    )

                with p4:

                    st.metric(
                        "Risco / Retorno",
                        f"1 : {rr:.2f}"
                    )

        # ====================================================
        # FATORES
        # ====================================================

        st.subheader(
            "🧠 POR QUE O ROBÔ DECIDIU?"
        )

        f1, f2 = st.columns(2)

        with f1:

            st.markdown(
                "### 🟢 LONG"
            )

            if fatores_long:

                tabela_long = pd.DataFrame(
                    fatores_long,
                    columns=[
                        "Indicador",
                        "Pontos"
                    ]
                )

                st.dataframe(
                    tabela_long,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.write(
                    "Nenhum fator comprador."
                )

        with f2:

            st.markdown(
                "### 🔴 SHORT"
            )

            if fatores_short:

                tabela_short = pd.DataFrame(
                    fatores_short,
                    columns=[
                        "Indicador",
                        "Pontos"
                    ]
                )

                st.dataframe(
                    tabela_short,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.write(
                    "Nenhum fator vendedor."
                )

        # ====================================================
        # INDICADORES
        # ====================================================

        st.subheader(
            "📊 INDICADORES"
        )

        i1, i2, i3, i4, i5 = st.columns(5)

        with i1:

            st.metric(
                "RSI",
                f"{row['RSI']:.2f}"
            )

        with i2:

            st.metric(
                "ATR",
                f"${row['ATR']:,.2f}"
            )

        with i3:

            st.metric(
                "EMA20",
                f"${row['EMA20']:,.2f}"
            )

        with i4:

            st.metric(
                "EMA50",
                f"${row['EMA50']:,.2f}"
            )

        with i5:

            st.metric(
                "EMA200",
                f"${row['EMA200']:,.2f}"
            )

        # ====================================================
        # ORDER BOOK
        # ====================================================

        st.subheader(
            "📚 ORDER BOOK"
        )

        ob1, ob2, ob3 = st.columns(3)

        with ob1:

            st.metric(
                "Bids",
                f"{orderbook['bids']:,.2f}"
            )

        with ob2:

            st.metric(
                "Asks",
                f"{orderbook['asks']:,.2f}"
            )

        with ob3:

            st.metric(
                "Imbalance",
                f"{orderbook['imbalance']:.3f}"
            )

        # ====================================================
        # GRÁFICO
        # ====================================================

        st.subheader(
            "📈 BTC"
        )

        chart_df = df.tail(150)

        fig = go.Figure()

        fig.add_trace(
            go.Candlestick(
                x=chart_df["close_time"],
                open=chart_df["open"],
                high=chart_df["high"],
                low=chart_df["low"],
                close=chart_df["close"],
                name="BTC"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=chart_df["close_time"],
                y=chart_df["EMA20"],
                name="EMA20"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=chart_df["close_time"],
                y=chart_df["EMA50"],
                name="EMA50"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=chart_df["close_time"],
                y=chart_df["EMA200"],
                name="EMA200"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=chart_df["close_time"],
                y=chart_df["VWAP"],
                name="VWAP"
            )
        )

        fig.update_layout(
            height=600,
            xaxis_rangeslider_visible=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ====================================================
        # PERFORMANCE
        # ====================================================

        st.subheader(
            "📊 PERFORMANCE DO ROBÔ"
        )

        perf = buscar_performance()

        p1, p2, p3, p4, p5 = st.columns(5)

        with p1:

            st.metric(
                "Trades",
                perf["trades"]
            )

        with p2:

            st.metric(
                "Gains",
                perf["gains"]
            )

        with p3:

            st.metric(
                "Losses",
                perf["losses"]
            )

        with p4:

            st.metric(
                "Win Rate",
                f"{perf['wr']:.2f}%"
            )

        with p5:

            st.metric(
                "P&L acumulado",
                f"{perf['pnl']:.2f}%"
            )

        # ====================================================
        # HISTÓRICO
        # ====================================================

        st.subheader(
            "📚 HISTÓRICO"
        )

        conn = conectar()

        historico = pd.read_sql_query(
            """
            SELECT
                id,
                side,
                entry_time,
                entry_price,
                stop_price,
                target_price,
                exit_time,
                exit_price,
                pnl_pct,
                result,
                score,
                regime
            FROM trades
            ORDER BY id DESC
            LIMIT 100
            """,
            conn
        )

        conn.close()

        if not historico.empty:

            st.dataframe(
                historico,
                use_container_width=True,
                hide_index=True
            )

        # ====================================================
        # STATUS
        # ====================================================

        st.caption(
            f"🤖 Monitoramento automático ativo | "
            f"Atualização a cada {AUTO_REFRESH_SECONDS}s | "
            f"Último candle analisado: {signal_time}"
        )

    except Exception as e:

        st.error(
            f"Erro no monitoramento: {e}"
        )


# ============================================================
# EXECUTAR MONITOR
# ============================================================

monitor()
