import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# CONFIGURAÇÃO
# ============================================================

BINANCE_API = "https://data-api.binance.vision"
SYMBOL = "BTCUSDT"
DATABASE = "btc_trader.db"

AUTO_REFRESH_SECONDS = 10

st.set_page_config(
    page_title="BTC Quant Trader V2",
    page_icon="₿",
    layout="wide"
)


# ============================================================
# BANCO DE DADOS
# ============================================================

def conectar():
    return sqlite3.connect(DATABASE)


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

    # --------------------------------------------------------
    # MIGRAÇÃO DO BANCO ANTIGO
    # --------------------------------------------------------

    colunas = [
        row[1]
        for row in conn.execute(
            "PRAGMA table_info(trades)"
        ).fetchall()
    ]

    if "signal_time" not in colunas:

        conn.execute("""
            ALTER TABLE trades
            ADD COLUMN signal_time TEXT
        """)

    conn.commit()
    conn.close()


criar_banco()


# ============================================================
# API BINANCE
# ============================================================

@st.cache_data(ttl=5)
def buscar_preco():

    url = f"{BINANCE_API}/api/v3/ticker/price"

    resposta = requests.get(
        url,
        params={
            "symbol": SYMBOL
        },
        timeout=10
    )

    resposta.raise_for_status()

    return float(
        resposta.json()["price"]
    )


@st.cache_data(ttl=10)
def buscar_candles(
    symbol="BTCUSDT",
    interval="5m",
    limit=500
):

    url = f"{BINANCE_API}/api/v3/klines"

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    resposta = requests.get(
        url,
        params=params,
        timeout=10
    )

    resposta.raise_for_status()

    dados = resposta.json()

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
        dados,
        columns=colunas
    )

    colunas_numericas = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote"
    ]

    for coluna in colunas_numericas:

        df[coluna] = pd.to_numeric(
            df[coluna],
            errors="coerce"
        )

    df["time"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
        utc=True
    )

    df.set_index(
        "time",
        inplace=True
    )

    return df


@st.cache_data(ttl=5)
def buscar_orderbook():

    url = f"{BINANCE_API}/api/v3/depth"

    resposta = requests.get(
        url,
        params={
            "symbol": SYMBOL,
            "limit": 20
        },
        timeout=10
    )

    resposta.raise_for_status()

    return resposta.json()


# ============================================================
# INDICADORES
# ============================================================

def calcular_indicadores(df):

    df = df.copy()

    # --------------------------------------------------------
    # RETORNOS
    # --------------------------------------------------------

    df["ret_1"] = df["close"].pct_change(1)

    df["ret_3"] = df["close"].pct_change(3)

    df["ret_12"] = df["close"].pct_change(12)

    df["ret_48"] = df["close"].pct_change(48)

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    df["ema20"] = (
        df["close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    df["ema50"] = (
        df["close"]
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )

    df["ema200"] = (
        df["close"]
        .ewm(
            span=200,
            adjust=False
        )
        .mean()
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    delta = df["close"].diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = (
        gain
        .ewm(
            alpha=1 / 14,
            adjust=False
        )
        .mean()
    )

    avg_loss = (
        loss
        .ewm(
            alpha=1 / 14,
            adjust=False
        )
        .mean()
    )

    rs = (
        avg_gain /
        avg_loss.replace(
            0,
            np.nan
        )
    )

    df["rsi"] = (
        100 -
        (
            100 /
            (1 + rs)
        )
    )

    df["rsi"] = df["rsi"].fillna(50)

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    fechamento_anterior = (
        df["close"].shift(1)
    )

    tr = pd.concat(
        [
            df["high"] - df["low"],

            (
                df["high"] -
                fechamento_anterior
            ).abs(),

            (
                df["low"] -
                fechamento_anterior
            ).abs()
        ],
        axis=1
    ).max(axis=1)

    df["atr"] = (
        tr
        .rolling(14)
        .mean()
    )

    # --------------------------------------------------------
    # VOLATILIDADE
    # --------------------------------------------------------

    df["volatilidade"] = (
        df["ret_1"]
        .rolling(24)
        .std()
    )

    media_vol = (
        df["volatilidade"]
        .rolling(100)
        .mean()
    )

    desvio_vol = (
        df["volatilidade"]
        .rolling(100)
        .std()
    )

    df["vol_z"] = (
        (
            df["volatilidade"] -
            media_vol
        )
        /
        desvio_vol.replace(
            0,
            np.nan
        )
    )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    media_volume = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    desvio_volume = (
        df["volume"]
        .rolling(50)
        .std()
    )

    df["volume_ratio"] = (
        df["volume"] /
        media_volume.replace(
            0,
            np.nan
        )
    )

    df["volume_z"] = (
        (
            df["volume"] -
            df["volume"].rolling(50).mean()
        )
        /
        desvio_volume.replace(
            0,
            np.nan
        )
    )

    # --------------------------------------------------------
    # VWAP
    # --------------------------------------------------------

    preco_tipico = (
        df["high"] +
        df["low"] +
        df["close"]
    ) / 3

    df["vwap"] = (
        (
            preco_tipico *
            df["volume"]
        )
        .rolling(50)
        .sum()
        /
        df["volume"]
        .rolling(50)
        .sum()
    )

    # --------------------------------------------------------
    # CVD APROXIMADO
    # --------------------------------------------------------

    volume_assinado = np.where(

        df["close"] > df["open"],

        df["volume"],

        np.where(
            df["close"] < df["open"],
            -df["volume"],
            0
        )
    )

    df["cvd"] = pd.Series(
        volume_assinado,
        index=df.index
    ).cumsum()

    df["cvd_delta"] = (
        df["cvd"]
        .diff(12)
    )

    return df


# ============================================================
# ORDER BOOK
# ============================================================

def calcular_imbalance(orderbook):

    bids = sum(
        float(qtd)
        for preco, qtd in orderbook["bids"]
    )

    asks = sum(
        float(qtd)
        for preco, qtd in orderbook["asks"]
    )

    total = bids + asks

    if total == 0:
        return 0

    return (
        (bids - asks) /
        total
    )


# ============================================================
# SCORE QUANTITATIVO
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

    if row["ema20"] > row["ema50"]:

        long_score += 15

        fatores_long.append(
            ("EMA 20 > EMA 50", 15)
        )

    else:

        short_score += 15

        fatores_short.append(
            ("EMA 20 < EMA 50", 15)
        )

    # --------------------------------------------------------
    # EMA 200
    # --------------------------------------------------------

    if row["close"] > row["ema200"]:

        long_score += 10

        fatores_long.append(
            ("Preço acima da EMA 200", 10)
        )

    else:

        short_score += 10

        fatores_short.append(
            ("Preço abaixo da EMA 200", 10)
        )

    # --------------------------------------------------------
    # MOMENTUM 1H
    # --------------------------------------------------------

    if row["ret_12"] > 0:

        long_score += 10

        fatores_long.append(
            ("Momentum 1H positivo", 10)
        )

    else:

        short_score += 10

        fatores_short.append(
            ("Momentum 1H negativo", 10)
        )

    # --------------------------------------------------------
    # MOMENTUM 4H
    # --------------------------------------------------------

    if row["ret_48"] > 0:

        long_score += 10

        fatores_long.append(
            ("Momentum 4H positivo", 10)
        )

    else:

        short_score += 10

        fatores_short.append(
            ("Momentum 4H negativo", 10)
        )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    if 50 <= row["rsi"] <= 70:

        long_score += 10

        fatores_long.append(
            (
                f"RSI favorável ({row['rsi']:.1f})",
                10
            )
        )

    elif 30 <= row["rsi"] < 50:

        short_score += 5

        fatores_short.append(
            (
                f"RSI enfraquecido ({row['rsi']:.1f})",
                5
            )
        )

    if 30 <= row["rsi"] <= 45:

        short_score += 10

        fatores_short.append(
            (
                f"RSI favorece SHORT ({row['rsi']:.1f})",
                10
            )
        )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    if row["volume_z"] > 1:

        if row["ret_1"] > 0:

            long_score += 10

            fatores_long.append(
                (
                    f"Volume forte + candle positivo "
                    f"(Z={row['volume_z']:.2f})",
                    10
                )
            )

        else:

            short_score += 10

            fatores_short.append(
                (
                    f"Volume forte + candle negativo "
                    f"(Z={row['volume_z']:.2f})",
                    10
                )
            )

    # --------------------------------------------------------
    # VWAP
    # --------------------------------------------------------

    if row["close"] > row["vwap"]:

        long_score += 10

        fatores_long.append(
            ("Preço acima da VWAP", 10)
        )

    else:

        short_score += 10

        fatores_short.append(
            ("Preço abaixo da VWAP", 10)
        )

    # --------------------------------------------------------
    # CVD
    # --------------------------------------------------------

    if row["cvd_delta"] > 0:

        long_score += 10

        fatores_long.append(
            ("CVD comprador", 10)
        )

    elif row["cvd_delta"] < 0:

        short_score += 10

        fatores_short.append(
            ("CVD vendedor", 10)
        )

    # --------------------------------------------------------
    # ORDER BOOK
    # --------------------------------------------------------

    if imbalance > 0.10:

        long_score += 10

        fatores_long.append(
            (
                f"Order Book comprador "
                f"({imbalance:+.2%})",
                10
            )
        )

    elif imbalance < -0.10:

        short_score += 10

        fatores_short.append(
            (
                f"Order Book vendedor "
                f"({imbalance:+.2%})",
                10
            )
        )

    # --------------------------------------------------------
    # REGIME
    # --------------------------------------------------------

    if row["vol_z"] > 1.5:

        regime = "EXPANSÃO"

    elif row["vol_z"] < -0.8:

        regime = "BAIXA VOLATILIDADE"

    elif (
        row["ema20"] > row["ema50"]
        and
        row["close"] > row["ema200"]
    ):

        regime = "TENDÊNCIA ALTA"

    elif (
        row["ema20"] < row["ema50"]
        and
        row["close"] < row["ema200"]
    ):

        regime = "TENDÊNCIA BAIXA"

    else:

        regime = "NEUTRO"

    # --------------------------------------------------------
    # SINAL
    # --------------------------------------------------------

    if (
        long_score >= 65
        and
        long_score > short_score + 10
    ):

        sinal = "LONG"

    elif (
        short_score >= 65
        and
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
# TRADES
# ============================================================

def consultar_trades():

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

    return df


def buscar_trade_aberto():

    conn = conectar()

    trade = conn.execute(
        """
        SELECT *
        FROM trades
        WHERE exit_time IS NULL
        ORDER BY id DESC
        LIMIT 1
        """
    ).fetchone()

    colunas = [
        desc[0]
        for desc in conn.execute(
            "PRAGMA table_info(trades)"
        ).fetchall()
    ]

    conn.close()

    if trade is None:
        return None

    return dict(
        zip(colunas, trade)
    )


def registrar_trade(
    side,
    entrada,
    stop,
    alvo,
    score,
    regime,
    observacao,
    signal_time
):

    conn = conectar()

    agora = (
        datetime
        .now(timezone.utc)
        .isoformat()
    )

    conn.execute(
        """
        INSERT INTO trades
        (
            symbol,
            side,
            entry_time,
            entry_price,
            stop_price,
            target_price,
            score,
            regime,
            notes,
            signal_time
        )

        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,

        (
            SYMBOL,
            side,
            agora,
            entrada,
            stop,
            alvo,
            score,
            regime,
            observacao,
            signal_time
        )
    )

    conn.commit()
    conn.close()


def fechar_trade(
    trade_id,
    preco_saida
):

    conn = conectar()

    trade = conn.execute(
        """
        SELECT
            side,
            entry_price

        FROM trades

        WHERE id = ?
        AND exit_time IS NULL
        """,

        (trade_id,)
    ).fetchone()

    if not trade:

        conn.close()
        return

    side, entrada = trade

    if side == "LONG":

        pnl = (
            preco_saida /
            entrada -
            1
        ) * 100

    else:

        pnl = (
            entrada /
            preco_saida -
            1
        ) * 100

    if pnl > 0:

        resultado = "GAIN"

    elif pnl < 0:

        resultado = "LOSS"

    else:

        resultado = "ZERO"

    agora = (
        datetime
        .now(timezone.utc)
        .isoformat()
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
            agora,
            preco_saida,
            pnl,
            resultado,
            trade_id
        )
    )

    conn.commit()
    conn.close()


# ============================================================
# VERIFICAR SE JÁ HOUVE ENTRADA NA VELA
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
    ).fetchone()[0]

    conn.close()

    return resultado > 0


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Configuração do Robô")

timeframe = st.sidebar.selectbox(
    "Timeframe",
    [
        "1m",
        "5m",
        "15m",
        "30m",
        "1h",
        "4h"
    ],
    index=1
)

quantidade = st.sidebar.slider(
    "Candles",
    200,
    1000,
    500,
    100
)

atr_stop = st.sidebar.number_input(
    "Stop ATR",
    0.1,
    10.0,
    1.5,
    0.1
)

atr_target = st.sidebar.number_input(
    "Target ATR",
    0.1,
    20.0,
    3.0,
    0.1
)

st.sidebar.divider()

st.sidebar.success(
    f"🤖 ROBÔ ATIVO\n\n"
    f"Atualização: {AUTO_REFRESH_SECONDS}s"
)

if st.sidebar.button(
    "🔄 Atualizar agora"
):

    st.cache_data.clear()
    st.rerun()


# ============================================================
# FUNÇÃO PRINCIPAL DO MONITOR
# ============================================================

def executar_monitor():

    try:

        # ----------------------------------------------------
        # DADOS
        # ----------------------------------------------------

        df = buscar_candles(
            SYMBOL,
            timeframe,
            quantidade
        )

        df = calcular_indicadores(df)

        preco = buscar_preco()

        orderbook = buscar_orderbook()

        imbalance = calcular_imbalance(
            orderbook
        )

        # ----------------------------------------------------
        # IMPORTANTE:
        # usar a última vela FECHADA para o sinal
        # ----------------------------------------------------

        ultima = df.iloc[-2]

        signal_time = str(
            df.index[-2]
        )

        (
            long_score,
            short_score,
            sinal,
            regime,
            fatores_long,
            fatores_short
        ) = gerar_score(
            ultima,
            imbalance
        )

        # ----------------------------------------------------
        # ATR
        # ----------------------------------------------------

        atr = float(
            ultima["atr"]
        )

        # ----------------------------------------------------
        # PLANO DA OPERAÇÃO
        # ----------------------------------------------------

        if sinal == "LONG":

            stop = (
                preco -
                atr * atr_stop
            )

            alvo = (
                preco +
                atr * atr_target
            )

            score_operacao = long_score

        elif sinal == "SHORT":

            stop = (
                preco +
                atr * atr_stop
            )

            alvo = (
                preco -
                atr * atr_target
            )

            score_operacao = short_score

        else:

            stop = np.nan
            alvo = np.nan

            score_operacao = max(
                long_score,
                short_score
            )

        # ----------------------------------------------------
        # OPERAÇÃO ABERTA
        # ----------------------------------------------------

        trade_aberto = buscar_trade_aberto()

        # ----------------------------------------------------
        # MONITORAR TRADE EXISTENTE
        # ----------------------------------------------------

        if trade_aberto is not None:

            side = trade_aberto["side"]

            entrada = float(
                trade_aberto["entry_price"]
            )

            stop_aberto = float(
                trade_aberto["stop_price"]
            )

            alvo_aberto = float(
                trade_aberto["target_price"]
            )

            trade_id = int(
                trade_aberto["id"]
            )

            # ----------------------------------------------
            # LONG
            # ----------------------------------------------

            if side == "LONG":

                # STOP
                if preco <= stop_aberto:

                    fechar_trade(
                        trade_id,
                        stop_aberto
                    )

                    st.session_state["ultimo_evento"] = (
                        f"🛑 STOP atingido no LONG #{trade_id}"
                    )

                # TARGET
                elif preco >= alvo_aberto:

                    fechar_trade(
                        trade_id,
                        alvo_aberto
                    )

                    st.session_state["ultimo_evento"] = (
                        f"🎯 TARGET atingido no LONG #{trade_id}"
                    )

            # ----------------------------------------------
            # SHORT
            # ----------------------------------------------

            elif side == "SHORT":

                # STOP
                if preco >= stop_aberto:

                    fechar_trade(
                        trade_id,
                        stop_aberto
                    )

                    st.session_state["ultimo_evento"] = (
                        f"🛑 STOP atingido no SHORT #{trade_id}"
                    )

                # TARGET
                elif preco <= alvo_aberto:

                    fechar_trade(
                        trade_id,
                        alvo_aberto
                    )

                    st.session_state["ultimo_evento"] = (
                        f"🎯 TARGET atingido no SHORT #{trade_id}"
                    )

        # ----------------------------------------------------
        # VERIFICAR NOVA ENTRADA
        # ----------------------------------------------------

        trade_aberto = buscar_trade_aberto()

        if (
            trade_aberto is None
            and
            sinal in ["LONG", "SHORT"]
            and
            not entrada_ja_registrada(
                signal_time
            )
        ):

            if sinal == "LONG":

                motivos = "; ".join(
                    [
                        f"{nome} +{pontos}"
                        for nome, pontos
                        in fatores_long
                    ]
                )

            else:

                motivos = "; ".join(
                    [
                        f"{nome} +{pontos}"
                        for nome, pontos
                        in fatores_short
                    ]
                )

            observacao = (
                f"Sinal={sinal}; "
                f"Long={long_score}; "
                f"Short={short_score}; "
                f"Regime={regime}; "
                f"Motivos={motivos}"
            )

            registrar_trade(
                sinal,
                preco,
                stop,
                alvo,
                score_operacao,
                regime,
                observacao,
                signal_time
            )

            st.session_state["ultimo_evento"] = (
                f"🚨 NOVA ENTRADA AUTOMÁTICA: "
                f"{sinal}"
            )

        return {
            "df": df,
            "preco": preco,
            "ultima": ultima,
            "imbalance": imbalance,
            "long_score": long_score,
            "short_score": short_score,
            "sinal": sinal,
            "regime": regime,
            "fatores_long": fatores_long,
            "fatores_short": fatores_short,
            "atr": atr,
            "stop": stop,
            "alvo": alvo,
            "signal_time": signal_time
        }

    except Exception as erro:

        st.error(
            f"Erro no monitoramento: {erro}"
        )

        return None


# ============================================================
# EXECUTAR MONITOR
# ============================================================

dados = executar_monitor()

if dados is None:
    st.stop()


df = dados["df"]
preco = dados["preco"]
ultima = dados["ultima"]

imbalance = dados["imbalance"]

long_score = dados["long_score"]
short_score = dados["short_score"]

sinal = dados["sinal"]
regime = dados["regime"]

fatores_long = dados["fatores_long"]
fatores_short = dados["fatores_short"]

atr = dados["atr"]
stop = dados["stop"]
alvo = dados["alvo"]

signal_time = dados["signal_time"]


# ============================================================
# CABEÇALHO
# ============================================================

st.title(
    "₿ BTC Quant Trader V2"
)

st.caption(
    "Robô de monitoramento + paper trading automático"
)


# ============================================================
# STATUS DO ROBÔ
# ============================================================

agora = datetime.now(
    timezone.utc
)

c1, c2, c3 = st.columns(3)

c1.success(
    "🤖 ROBÔ MONITORANDO"
)

c2.metric(
    "Atualização",
    f"{AUTO_REFRESH_SECONDS}s"
)

c3.write(
    f"Última verificação:\n"
    f"{agora.strftime('%H:%M:%S')} UTC"
)


# ============================================================
# ÚLTIMO EVENTO
# ============================================================

if "ultimo_evento" in st.session_state:

    st.info(
        st.session_state["ultimo_evento"]
    )


# ============================================================
# DASHBOARD PRINCIPAL
# ============================================================

st.subheader(
    "📊 Mercado"
)

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "BTC",
    f"${preco:,.2f}"
)

c2.metric(
    "LONG",
    f"{long_score}/100"
)

c3.metric(
    "SHORT",
    f"{short_score}/100"
)

c4.metric(
    "REGIME",
    regime
)

c5.metric(
    "BOOK",
    f"{imbalance:+.2%}"
)


# ============================================================
# DECISÃO
# ============================================================

st.subheader(
    "🎯 DECISÃO DO ROBÔ"
)

trade_aberto = buscar_trade_aberto()

if trade_aberto is not None:

    side = trade_aberto["side"]

    entrada_aberta = float(
        trade_aberto["entry_price"]
    )

    stop_aberto = float(
        trade_aberto["stop_price"]
    )

    alvo_aberto = float(
        trade_aberto["target_price"]
    )

    if side == "LONG":

        pnl_atual = (
            preco /
            entrada_aberta -
            1
        ) * 100

        st.success(
            f"🟢 LONG EM ANDAMENTO — "
            f"Operação #{trade_aberto['id']}"
        )

    else:

        pnl_atual = (
            entrada_aberta /
            preco -
            1
        ) * 100

        st.error(
            f"🔴 SHORT EM ANDAMENTO — "
            f"Operação #{trade_aberto['id']}"
        )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Entrada",
        f"${entrada_aberta:,.2f}"
    )

    c2.metric(
        "Preço atual",
        f"${preco:,.2f}"
    )

    c3.metric(
        "Stop",
        f"${stop_aberto:,.2f}"
    )

    c4.metric(
        "Target",
        f"${alvo_aberto:,.2f}"
    )

    c5.metric(
        "P&L atual",
        f"{pnl_atual:+.2f}%"
    )

else:

    if sinal == "LONG":

        st.success(
            f"🟢 COMPRAR BTC\n\n"
            f"Score LONG: {long_score}/100"
        )

    elif sinal == "SHORT":

        st.error(
            f"🔴 VENDER / SHORT BTC\n\n"
            f"Score SHORT: {short_score}/100"
        )

    else:

        st.warning(
            f"🟡 AGUARDAR — NÃO OPERAR\n\n"
            f"Long {long_score} | "
            f"Short {short_score}"
        )


# ============================================================
# PLANO DE OPERAÇÃO
# ============================================================

if trade_aberto is not None:

    entrada_plano = entrada_aberta
    stop_plano = stop_aberto
    alvo_plano = alvo_aberto
    lado_plano = side

elif sinal in ["LONG", "SHORT"]:

    entrada_plano = preco
    stop_plano = stop
    alvo_plano = alvo
    lado_plano = sinal

else:

    entrada_plano = np.nan
    stop_plano = np.nan
    alvo_plano = np.nan
    lado_plano = None


if lado_plano is not None:

    distancia_stop = abs(
        entrada_plano -
        stop_plano
    )

    distancia_alvo = abs(
        alvo_plano -
        entrada_plano
    )

    risco_retorno = (
        distancia_alvo /
        distancia_stop
        if distancia_stop > 0
        else 0
    )

    st.subheader(
        "📋 Plano da operação"
    )

    if lado_plano == "LONG":

        st.success(
            "🟢 OPERAÇÃO DE COMPRA"
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "📥 ENTRADA",
            f"${entrada_plano:,.2f}"
        )

        c2.metric(
            "🛑 STOP",
            f"${stop_plano:,.2f}"
        )

        c3.metric(
            "🎯 TARGET",
            f"${alvo_plano:,.2f}"
        )

        c4.metric(
            "Risco / Retorno",
            f"1:{risco_retorno:.2f}"
        )

        st.write(
            f"👉 **COMPRAR** em "
            f"${entrada_plano:,.2f}"
        )

        st.write(
            f"🛑 Sair com perda se atingir "
            f"${stop_plano:,.2f}"
        )

        st.write(
            f"🎯 Realizar lucro se atingir "
            f"${alvo_plano:,.2f}"
        )

    else:

        st.error(
            "🔴 OPERAÇÃO DE VENDA / SHORT"
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "📤 ENTRADA",
            f"${entrada_plano:,.2f}"
        )

        c2.metric(
            "🛑 STOP",
            f"${stop_plano:,.2f}"
        )

        c3.metric(
            "🎯 TARGET",
            f"${alvo_plano:,.2f}"
        )

        c4.metric(
            "Risco / Retorno",
            f"1:{risco_retorno:.2f}"
        )

        st.write(
            f"👉 **VENDER / SHORT** em "
            f"${entrada_plano:,.2f}"
        )

        st.write(
            f"🛑 Sair com perda se atingir "
            f"${stop_plano:,.2f}"
        )

        st.write(
            f"🎯 Realizar lucro se atingir "
            f"${alvo_plano:,.2f}"
        )


# ============================================================
# INDICADORES QUE DERAM PONTOS
# ============================================================

st.subheader(
    "🧠 Por que o robô chegou nessa decisão?"
)

col_long, col_short = st.columns(2)


with col_long:

    st.markdown(
        "### 🟢 Fatores LONG"
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
            "Nenhum indicador favorecendo LONG."
        )


with col_short:

    st.markdown(
        "### 🔴 Fatores SHORT"
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
            "Nenhum indicador favorecendo SHORT."
        )


# ============================================================
# VARIÁVEIS
# ============================================================

st.subheader(
    "📊 Variáveis quantitativas"
)

dados_variaveis = pd.DataFrame({

    "Variável": [

        "EMA 20",
        "EMA 50",
        "EMA 200",
        "RSI",
        "Retorno 1 candle",
        "Retorno 1h",
        "Retorno 4h",
        "Volatilidade Z",
        "Volume Ratio",
        "Volume Z",
        "VWAP",
        "CVD Delta",
        "ATR",
        "Order Book"
    ],

    "Valor": [

        ultima["ema20"],
        ultima["ema50"],
        ultima["ema200"],
        ultima["rsi"],
        ultima["ret_1"],
        ultima["ret_12"],
        ultima["ret_48"],
        ultima["vol_z"],
        ultima["volume_ratio"],
        ultima["volume_z"],
        ultima["vwap"],
        ultima["cvd_delta"],
        ultima["atr"],
        imbalance
    ]
})

st.dataframe(
    dados_variaveis,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# GRÁFICO
# ============================================================

st.subheader(
    "📈 Gráfico BTC"
)

grafico = go.Figure()

grafico.add_trace(
    go.Candlestick(

        x=df.index,

        open=df["open"],

        high=df["high"],

        low=df["low"],

        close=df["close"],

        name="BTC"
    )
)

grafico.add_trace(
    go.Scatter(

        x=df.index,

        y=df["ema20"],

        name="EMA 20"
    )
)

grafico.add_trace(
    go.Scatter(

        x=df.index,

        y=df["ema50"],

        name="EMA 50"
    )
)

grafico.add_trace(
    go.Scatter(

        x=df.index,

        y=df["ema200"],

        name="EMA 200"
    )
)

grafico.add_trace(
    go.Scatter(

        x=df.index,

        y=df["vwap"],

        name="VWAP"
    )
)

# ------------------------------------------------------------
# MARCAR ENTRADA
# ------------------------------------------------------------

if trade_aberto is not None:

    grafico.add_hline(
        y=entrada_aberta,
        annotation_text="ENTRADA"
    )

    grafico.add_hline(
        y=stop_aberto,
        annotation_text="STOP"
    )

    grafico.add_hline(
        y=alvo_aberto,
        annotation_text="TARGET"
    )

grafico.update_layout(

    height=650,

    xaxis_rangeslider_visible=False
)

st.plotly_chart(
    grafico,
    use_container_width=True
)


# ============================================================
# PERFORMANCE
# ============================================================

st.subheader(
    "💰 Performance do robô"
)

trades = consultar_trades()

fechadas = trades[
    trades["exit_time"].notna()
].copy()

if len(fechadas) == 0:

    st.info(
        "O robô ainda não possui operações encerradas."
    )

else:

    total = len(fechadas)

    gains = fechadas[
        fechadas["pnl_pct"] > 0
    ]

    losses = fechadas[
        fechadas["pnl_pct"] < 0
    ]

    wr = (
        len(gains) /
        total *
        100
    )

    avg_gain = (

        gains["pnl_pct"].mean()

        if len(gains)

        else 0
    )

    avg_loss = (

        losses["pnl_pct"].mean()

        if len(losses)

        else 0
    )

    expectancy = (

        (wr / 100) *
        avg_gain

        +

        ((100 - wr) / 100) *
        avg_loss
    )

    if len(losses) > 0:

        profit_factor = (

            gains["pnl_pct"].sum()
            /
            abs(
                losses["pnl_pct"].sum()
            )
        )

    else:

        profit_factor = np.nan

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Trades",
        total
    )

    c2.metric(
        "Win Rate",
        f"{wr:.2f}%"
    )

    c3.metric(
        "Gain Médio",
        f"{avg_gain:.2f}%"
    )

    c4.metric(
        "Loss Médio",
        f"{avg_loss:.2f}%"
    )

    c5.metric(
        "Expectancy",
        f"{expectancy:.3f}%"
    )

    st.metric(

        "Profit Factor",

        "—"

        if np.isnan(
            profit_factor
        )

        else f"{profit_factor:.2f}"
    )

    # --------------------------------------------------------
    # EQUITY
    # --------------------------------------------------------

    fechadas = fechadas.sort_values(
        "id"
    )

    fechadas["equity"] = (
        fechadas["pnl_pct"]
        .cumsum()
    )

    curva = go.Figure()

    curva.add_trace(
        go.Scatter(

            x=list(
                range(
                    1,
                    len(fechadas) + 1
                )
            ),

            y=fechadas["equity"],

            mode="lines+markers",

            name="Equity"
        )
    )

    curva.update_layout(

        title="Curva de P&L acumulado",

        xaxis_title="Operações",

        yaxis_title="P&L (%)",

        height=400
    )

    st.plotly_chart(
        curva,
        use_container_width=True
    )


# ============================================================
# OPERAÇÕES ABERTAS
# ============================================================

st.subheader(
    "📌 Operação atualmente monitorada"
)

aberta = buscar_trade_aberto()

if aberta is None:

    st.info(
        "Nenhuma operação aberta."
    )

else:

    lado = aberta["side"]

    entrada = float(
        aberta["entry_price"]
    )

    stop_trade = float(
        aberta["stop_price"]
    )

    alvo_trade = float(
        aberta["target_price"]
    )

    if lado == "LONG":

        pnl = (
            preco /
            entrada -
            1
        ) * 100

    else:

        pnl = (
            entrada /
            preco -
            1
        ) * 100

    st.write(
        f"### #{int(aberta['id'])} — {lado}"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Entrada",
        f"${entrada:,.2f}"
    )

    c2.metric(
        "Atual",
        f"${preco:,.2f}"
    )

    c3.metric(
        "Stop",
        f"${stop_trade:,.2f}"
    )

    c4.metric(
        "Target",
        f"${alvo_trade:,.2f}"
    )

    st.metric(
        "P&L atual",
        f"{pnl:+.2f}%"
    )


# ============================================================
# HISTÓRICO
# ============================================================

st.subheader(
    "🗂️ Histórico de operações"
)

if len(trades):

    st.dataframe(

        trades,

        use_container_width=True,

        hide_index=True
    )

else:

    st.info(
        "Nenhuma operação registrada."
    )


# ============================================================
# AUTO REFRESH
# ============================================================

st.caption(
    f"🤖 Monitoramento automático ativo — "
    f"atualização a cada {AUTO_REFRESH_SECONDS} segundos."
)

# Streamlit moderno
try:

    st.fragment(
        run_every=AUTO_REFRESH_SECONDS
    )

except Exception:

    pass


# ============================================================
# AVISO
# ============================================================

st.caption(
    "Paper Trading: este sistema não envia ordens reais. "
    "As entradas e saídas são simuladas com dados públicos "
    "da Binance."
)
