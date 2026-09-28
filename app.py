import sqlite3
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

KLINE_LIMIT = 500

ATR_STOP_MULTIPLIER = 1.0
ATR_TARGET_MULTIPLIER = 2.0


# ============================================================
# CONFIGURAÇÃO STREAMLIT
# ============================================================

st.set_page_config(
    page_title="BTC Quant Trader",
    page_icon="₿",
    layout="wide"
)


# ============================================================
# TEMA ESCURO
# ============================================================

st.markdown(
    """
    <style>

    /* Fundo geral */
    .stApp {
        background-color: #080808;
        color: #F5F5F5;
    }

    /* Cabeçalho */
    header[data-testid="stHeader"] {
        background-color: #080808;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0D0D0D;
    }

    /* Texto */
    .stApp p,
    .stApp label,
    .stApp span,
    .stApp div {
        color: #F1F1F1;
    }

    /* Cards */
    div[data-testid="metric-container"] {
        background-color: #111111;
        border: 1px solid #252525;
        border-radius: 12px;
        padding: 12px;
    }

    /* Dataframes */
    div[data-testid="stDataFrame"] {
        background-color: #111111;
        border-radius: 10px;
    }

    /* Inputs */
    input,
    textarea,
    select {
        background-color: #151515 !important;
        color: #FFFFFF !important;
    }

    /* Botões */
    .stButton > button {
        background-color: #1A1A1A;
        color: #FFFFFF;
        border: 1px solid #333333;
        border-radius: 8px;
    }

    .stButton > button:hover {
        border-color: #666666;
        color: #FFFFFF;
    }

    /* Divisórias */
    hr {
        border-color: #292929;
    }

    /* Alertas */
    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    /* Títulos */
    h1, h2, h3 {
        color: #FFFFFF !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# BANCO
# ============================================================

def conectar():

    return sqlite3.connect(
        DATABASE,
        check_same_thread=False
    )


def criar_banco():

    conn = conectar()

    conn.execute(
        """
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
        """
    )

    conn.commit()

    colunas_existentes = {
        row[1]
        for row in conn.execute(
            "PRAGMA table_info(trades)"
        ).fetchall()
    }

    if "signal_time" not in colunas_existentes:

        conn.execute(
            """
            ALTER TABLE trades
            ADD COLUMN signal_time TEXT
            """
        )

    conn.commit()
    conn.close()


# ============================================================
# OPERAÇÕES ABERTAS
# ============================================================

def buscar_operacoes_abertas():

    conn = conectar()

    df = pd.read_sql_query(
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
        ORDER BY id ASC
        """,
        conn
    )

    conn.close()

    return df


# ============================================================
# DUPLICIDADE
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
# REGISTRAR OPERAÇÃO
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

        VALUES (
            ?, ?,
            ?, ?,
            ?, ?,
            NULL, NULL,
            NULL, NULL,
            ?, ?,
            ?, ?
        )
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
# FECHAR OPERAÇÃO
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

    entry_price = float(
        trade[1]
    )

    exit_price = float(
        exit_price
    )

    if side == "COMPRA":

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
# BINANCE
# ============================================================

def buscar_klines():

    url = (
        f"{BINANCE_API}/api/v3/klines"
    )

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


def buscar_preco():

    url = (
        f"{BINANCE_API}/api/v3/ticker/price"
    )

    response = requests.get(
        url,
        params={"symbol": SYMBOL},
        timeout=10
    )

    response.raise_for_status()

    return float(
        response.json()["price"]
    )


def buscar_orderbook():

    url = (
        f"{BINANCE_API}/api/v3/depth"
    )

    response = requests.get(
        url,
        params={
            "symbol": SYMBOL,
            "limit": 50
        },
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

    df["ret_1"] = (
        df["close"].pct_change(1)
    )

    df["ret_3"] = (
        df["close"].pct_change(3)
    )

    df["ret_12"] = (
        df["close"].pct_change(12)
    )

    df["ret_48"] = (
        df["close"].pct_change(48)
    )

    df["EMA20"] = (
        df["close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    df["EMA50"] = (
        df["close"]
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )

    df["EMA200"] = (
        df["close"]
        .ewm(
            span=200,
            adjust=False
        )
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

    rs = (
        avg_gain /
        avg_loss.replace(
            0,
            np.nan
        )
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

    tr = pd.concat(
        [
            high_low,
            high_close,
            low_close
        ],
        axis=1
    ).max(axis=1)

    df["ATR"] = (
        tr
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
        (
            df["volatility"] -
            vol_mean
        ) /
        vol_std.replace(
            0,
            np.nan
        )
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
        volume_mean.replace(
            0,
            np.nan
        )
    )

    df["volume_z"] = (
        (
            df["volume"] -
            volume_mean
        ) /
        volume_std.replace(
            0,
            np.nan
        )
    )

    # VWAP
    typical_price = (
        df["high"] +
        df["low"] +
        df["close"]
    ) / 3

    df["VWAP"] = (
        (
            typical_price *
            df["volume"]
        )
        .rolling(48)
        .sum()
        /
        df["volume"]
        .rolling(48)
        .sum()
        .replace(
            0,
            np.nan
        )
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
        df["cvd"].diff()
    )

    return df


# ============================================================
# SCORE
# ============================================================

def gerar_score(
    row,
    imbalance
):

    compra = 0
    venda = 0

    fatores_compra = []
    fatores_venda = []

    # EMA 20 / 50
    if row["EMA20"] > row["EMA50"]:

        compra += 15

        fatores_compra.append(
            ("EMA20 acima da EMA50", 15)
        )

    else:

        venda += 15

        fatores_venda.append(
            ("EMA20 abaixo da EMA50", 15)
        )

    # EMA 200
    if row["close"] > row["EMA200"]:

        compra += 10

        fatores_compra.append(
            ("Preço acima da EMA200", 10)
        )

    else:

        venda += 10

        fatores_venda.append(
            ("Preço abaixo da EMA200", 10)
        )

    # Momentum 12
    if row["ret_12"] > 0:

        compra += 10

        fatores_compra.append(
            ("Momentum positivo - 12 períodos", 10)
        )

    else:

        venda += 10

        fatores_venda.append(
            ("Momentum negativo - 12 períodos", 10)
        )

    # Momentum 48
    if row["ret_48"] > 0:

        compra += 10

        fatores_compra.append(
            ("Momentum positivo - 48 períodos", 10)
        )

    else:

        venda += 10

        fatores_venda.append(
            ("Momentum negativo - 48 períodos", 10)
        )

    # RSI
    rsi = row["RSI"]

    if 50 <= rsi <= 70:

        compra += 10

        fatores_compra.append(
            (
                f"RSI favorável ({rsi:.1f})",
                10
            )
        )

    elif 30 <= rsi <= 45:

        venda += 10

        fatores_venda.append(
            (
                f"RSI favorável ({rsi:.1f})",
                10
            )
        )

    elif 45 < rsi < 50:

        venda += 5

        fatores_venda.append(
            (
                f"RSI vendedor ({rsi:.1f})",
                5
            )
        )

    # Volume
    if (
        row["volume_z"] > 1
        and row["ret_1"] > 0
    ):

        compra += 10

        fatores_compra.append(
            (
                "Volume forte + candle comprador",
                10
            )
        )

    elif (
        row["volume_z"] > 1
        and row["ret_1"] < 0
    ):

        venda += 10

        fatores_venda.append(
            (
                "Volume forte + candle vendedor",
                10
            )
        )

    # VWAP
    if row["close"] > row["VWAP"]:

        compra += 10

        fatores_compra.append(
            ("Preço acima da VWAP", 10)
        )

    else:

        venda += 10

        fatores_venda.append(
            ("Preço abaixo da VWAP", 10)
        )

    # CVD
    if row["cvd_delta"] > 0:

        compra += 10

        fatores_compra.append(
            ("CVD positivo", 10)
        )

    elif row["cvd_delta"] < 0:

        venda += 10

        fatores_venda.append(
            ("CVD negativo", 10)
        )

    # Order Book
    if imbalance > 0.10:

        compra += 10

        fatores_compra.append(
            (
                f"Order Book comprador ({imbalance:.2f})",
                10
            )
        )

    elif imbalance < -0.10:

        venda += 10

        fatores_venda.append(
            (
                f"Order Book vendedor ({imbalance:.2f})",
                10
            )
        )

    # Regime
    if (
        row["EMA20"] > row["EMA50"]
        and row["close"] > row["EMA200"]
    ):

        regime = "ALTA"

    elif (
        row["EMA20"] < row["EMA50"]
        and row["close"] < row["EMA200"]
    ):

        regime = "BAIXA"

    else:

        regime = "LATERAL"

    # Sinal
    if (
        compra >= 65
        and compra > venda + 10
    ):

        sinal = "COMPRA"

    elif (
        venda >= 65
        and venda > compra + 10
    ):

        sinal = "VENDA"

    else:

        sinal = "AGUARDAR"

    return (
        compra,
        venda,
        sinal,
        regime,
        fatores_compra,
        fatores_venda
    )


# ============================================================
# STOP / ALVO
# ============================================================

def calcular_plano(
    side,
    preco,
    atr
):

    if side == "COMPRA":

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
# MONITORAMENTO DAS OPERAÇÕES
# ============================================================

def monitorar_operacoes(
    preco_atual,
    tempo_maximo
):

    abertas = buscar_operacoes_abertas()

    eventos = []

    agora = datetime.now()

    for _, trade in abertas.iterrows():

        trade_id = int(
            trade["id"]
        )

        side = trade["side"]

        entrada = float(
            trade["entry_price"]
        )

        stop = float(
            trade["stop_price"]
        )

        alvo = float(
            trade["target_price"]
        )

        # ----------------------------------------------------
        # TEMPO DA OPERAÇÃO
        # ----------------------------------------------------

        try:

            entrada_data = pd.to_datetime(
                trade["entry_time"]
            ).to_pydatetime()

            minutos_aberto = (
                agora -
                entrada_data
            ).total_seconds() / 60

        except Exception:

            minutos_aberto = 0

        # ----------------------------------------------------
        # STOP / ALVO
        # ----------------------------------------------------

        resultado = None

        if side == "COMPRA":

            if preco_atual <= stop:

                resultado = "PERDA"

            elif preco_atual >= alvo:

                resultado = "GANHO"

        else:

            if preco_atual >= stop:

                resultado = "PERDA"

            elif preco_atual <= alvo:

                resultado = "GANHO"

        # ----------------------------------------------------
        # TEMPO MÁXIMO
        # ----------------------------------------------------

        if (
            resultado is None
            and minutos_aberto >= tempo_maximo
        ):

            resultado = "TEMPO ESGOTADO"

        # ----------------------------------------------------
        # FECHAMENTO
        # ----------------------------------------------------

        if resultado is not None:

            fechar_trade(
                trade_id,
                preco_atual,
                resultado
            )

            eventos.append(
                f"Operação #{trade_id}: "
                f"{resultado}"
            )

    return eventos


# ============================================================
# PERFORMANCE
# ============================================================

def buscar_performance():

    conn = conectar()

    df = pd.read_sql_query(
        """
        SELECT *
        FROM trades
        WHERE exit_time IS NOT NULL
        """,
        conn
    )

    conn.close()

    if df.empty:

        return {
            "total": 0,
            "ganhos": 0,
            "perdas": 0,
            "tempo": 0,
            "wr": 0,
            "pnl": 0
        }

    ganhos = (
        df["result"] == "GANHO"
    ).sum()

    perdas = (
        df["result"] == "PERDA"
    ).sum()

    tempo = (
        df["result"] == "TEMPO ESGOTADO"
    ).sum()

    total = len(df)

    wr = (
        ganhos / total * 100
        if total > 0
        else 0
    )

    pnl = (
        df["pnl_pct"]
        .fillna(0)
        .sum()
    )

    return {
        "total": total,
        "ganhos": int(ganhos),
        "perdas": int(perdas),
        "tempo": int(tempo),
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
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Configuração")

    max_operacoes = st.number_input(
        "Máximo de operações abertas",
        min_value=1,
        max_value=20,
        value=5,
        step=1
    )

    tempo_maximo = st.number_input(
        "Tempo máximo da operação (minutos)",
        min_value=5,
        max_value=1440,
        value=60,
        step=5
    )

    st.divider()

    st.write(
        "### Estratégia"
    )

    st.write(
        f"Intervalo: **{INTERVAL}**"
    )

    st.write(
        f"Atualização: **{AUTO_REFRESH_SECONDS}s**"
    )

    st.write(
        "Modo: **PAPER TRADING**"
    )

    st.divider()

    st.caption(
        "O robô não envia ordens reais."
    )


# ============================================================
# TÍTULO
# ============================================================

st.title(
    "₿ BTC Quant Trader"
)

st.caption(
    "Monitoramento quantitativo automático • "
    "BTC/USDT • Paper Trading"
)

st.success(
    "🤖 ROBÔ ATIVO — MODO PAPER TRADING"
)


# ============================================================
# MONITORAMENTO
# ============================================================

@st.fragment(
    run_every=AUTO_REFRESH_SECONDS
)
def monitor():

    try:

        # ----------------------------------------------------
        # DADOS
        # ----------------------------------------------------

        df = buscar_klines()

        preco_atual = buscar_preco()

        orderbook = buscar_orderbook()

        df = calcular_indicadores(df)

        if len(df) < 250:

            st.warning(
                "Aguardando histórico suficiente..."
            )

            return

        # ----------------------------------------------------
        # ÚLTIMO CANDLE FECHADO
        # ----------------------------------------------------

        row = df.iloc[-2]

        signal_time = (
            row["close_time"]
            .strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

        # ----------------------------------------------------
        # SCORE
        # ----------------------------------------------------

        (
            score_compra,
            score_venda,
            sinal,
            regime,
            fatores_compra,
            fatores_venda
        ) = gerar_score(
            row,
            orderbook["imbalance"]
        )

        # ----------------------------------------------------
        # MONITORAR OPERAÇÕES
        # ----------------------------------------------------

        eventos = monitorar_operacoes(
            preco_atual,
            tempo_maximo
        )

        if eventos:

            st.session_state[
                "ultimo_evento"
            ] = " | ".join(eventos)

        # ----------------------------------------------------
        # OPERAÇÕES ABERTAS
        # ----------------------------------------------------

        abertas = (
            buscar_operacoes_abertas()
        )

        quantidade_abertas = len(
            abertas
        )

        # ----------------------------------------------------
        # ENTRADA AUTOMÁTICA
        # ----------------------------------------------------

        if (
            quantidade_abertas
            < max_operacoes
        ):

            if sinal in [
                "COMPRA",
                "VENDA"
            ]:

                if not entrada_ja_registrada(
                    signal_time
                ):

                    atr = float(
                        row["ATR"]
                    )

                    if (
                        np.isfinite(atr)
                        and atr > 0
                    ):

                        entrada = (
                            preco_atual
                        )

                        stop, alvo = (
                            calcular_plano(
                                sinal,
                                entrada,
                                atr
                            )
                        )

                        score = (
                            score_compra
                            if sinal == "COMPRA"
                            else score_venda
                        )

                        registrar_trade(
                            side=sinal,
                            entry_price=entrada,
                            stop_price=stop,
                            target_price=alvo,
                            score=score,
                            regime=regime,
                            signal_time=signal_time,
                            notes=(
                                "Entrada automática "
                                "por sinal confirmado."
                            )
                        )

                        st.session_state[
                            "ultimo_evento"
                        ] = (
                            f"Nova operação: "
                            f"{sinal} "
                            f"#{quantidade_abertas + 1}"
                        )

                        abertas = (
                            buscar_operacoes_abertas()
                        )

                        quantidade_abertas = len(
                            abertas
                        )

        # ====================================================
        # STATUS PRINCIPAL
        # ====================================================

        st.divider()

        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:

            st.metric(
                "Preço BTC",
                f"${preco_atual:,.2f}"
            )

        with c2:

            st.metric(
                "Score Compra",
                f"{score_compra}/100"
            )

        with c3:

            st.metric(
                "Score Venda",
                f"{score_venda}/100"
            )

        with c4:

            st.metric(
                "Regime",
                regime
            )

        with c5:

            st.metric(
                "Operações",
                f"{quantidade_abertas}/{max_operacoes}"
            )

        # ====================================================
        # DECISÃO
        # ====================================================

        st.subheader(
            "🎯 DECISÃO DO ROBÔ"
        )

        if sinal == "COMPRA":

            st.success(
                f"🟢 COMPRA BTC — "
                f"Score {score_compra}/100"
            )

        elif sinal == "VENDA":

            st.error(
                f"🔴 VENDA BTC — "
                f"Score {score_venda}/100"
            )

        else:

            st.warning(
                f"🟡 AGUARDAR — "
                f"Compra {score_compra} | "
                f"Venda {score_venda}"
            )

        # ====================================================
        # EVENTO
        # ====================================================

        if st.session_state[
            "ultimo_evento"
        ]:

            st.info(
                "📌 " +
                st.session_state[
                    "ultimo_evento"
                ]
            )

        # ====================================================
        # OPERAÇÕES ABERTAS
        # ====================================================

        st.subheader(
            f"📌 OPERAÇÕES ABERTAS "
            f"({quantidade_abertas}/{max_operacoes})"
        )

        if not abertas.empty:

            tabela = []

            for _, trade in abertas.iterrows():

                entrada = float(
                    trade["entry_price"]
                )

                stop = float(
                    trade["stop_price"]
                )

                alvo = float(
                    trade["target_price"]
                )

                side = trade["side"]

                if side == "COMPRA":

                    pnl = (
                        (
                            preco_atual /
                            entrada
                        ) - 1
                    ) * 100

                else:

                    pnl = (
                        (
                            entrada /
                            preco_atual
                        ) - 1
                    ) * 100

                try:

                    entrada_data = pd.to_datetime(
                        trade["entry_time"]
                    )

                    minutos = int(
                        (
                            pd.Timestamp.now()
                            - entrada_data
                        ).total_seconds()
                        / 60
                    )

                except Exception:

                    minutos = 0

                tabela.append(
                    {
                        "ID": int(
                            trade["id"]
                        ),

                        "Operação":
                            (
                                "🟢 COMPRA"
                                if side == "COMPRA"
                                else "🔴 VENDA"
                            ),

                        "Data/Hora Entrada":
                            trade["entry_time"],

                        "Preço de Entrada":
                            f"${entrada:,.2f}",

                        "Preço Atual":
                            f"${preco_atual:,.2f}",

                        "Stop":
                            f"${stop:,.2f}",

                        "Preço Alvo":
                            f"${alvo:,.2f}",

                        "Tempo":
                            f"{minutos} min",

                        "P&L":
                            f"{pnl:.2f}%",

                        "Score":
                            int(
                                trade["score"]
                            ),

                        "Regime":
                            trade["regime"]
                    }
                )

            st.dataframe(
                pd.DataFrame(tabela),
                use_container_width=True,
                hide_index=True
            )

        else:

            st.write(
                "Nenhuma operação aberta."
            )

        # ====================================================
        # PLANO DO NOVO SINAL
        # ====================================================

        if sinal in [
            "COMPRA",
            "VENDA"
        ]:

            atr = float(
                row["ATR"]
            )

            if (
                np.isfinite(atr)
                and atr > 0
            ):

                stop, alvo = (
                    calcular_plano(
                        sinal,
                        preco_atual,
                        atr
                    )
                )

                risco = abs(
                    preco_atual - stop
                )

                recompensa = abs(
                    alvo - preco_atual
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
                        "Preço de Entrada",
                        f"${preco_atual:,.2f}"
                    )

                with p2:

                    st.metric(
                        "Stop",
                        f"${stop:,.2f}"
                    )

                with p3:

                    st.metric(
                        "Preço Alvo",
                        f"${alvo:,.2f}"
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
                "### 🟢 COMPRA"
            )

            if fatores_compra:

                tabela_compra = pd.DataFrame(
                    fatores_compra,
                    columns=[
                        "Indicador",
                        "Pontos"
                    ]
                )

                st.dataframe(
                    tabela_compra,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.write(
                    "Nenhum fator comprador."
                )

        with f2:

            st.markdown(
                "### 🔴 VENDA"
            )

            if fatores_venda:

                tabela_venda = pd.DataFrame(
                    fatores_venda,
                    columns=[
                        "Indicador",
                        "Pontos"
                    ]
                )

                st.dataframe(
                    tabela_venda,
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

        o1, o2, o3 = st.columns(3)

        with o1:

            st.metric(
                "Compradores",
                f"{orderbook['bids']:,.2f}"
            )

        with o2:

            st.metric(
                "Vendedores",
                f"{orderbook['asks']:,.2f}"
            )

        with o3:

            st.metric(
                "Desequilíbrio",
                f"{orderbook['imbalance']:.3f}"
            )

        # ====================================================
        # GRÁFICO
        # ====================================================

        st.subheader(
            "📈 GRÁFICO BTC"
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
            template="plotly_dark",
            height=600,
            xaxis_rangeslider_visible=False,
            paper_bgcolor="#080808",
            plot_bgcolor="#080808"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ====================================================
        # PERFORMANCE
        # ====================================================

        st.subheader(
            "📊 PERFORMANCE"
        )

        perf = buscar_performance()

        a, b, c, d, e, f = st.columns(6)

        with a:

            st.metric(
                "Operações",
                perf["total"]
            )

        with b:

            st.metric(
                "Ganhos",
                perf["ganhos"]
            )

        with c:

            st.metric(
                "Perdas",
                perf["perdas"]
            )

        with d:

            st.metric(
                "Tempo Esgotado",
                perf["tempo"]
            )

        with e:

            st.metric(
                "Win Rate",
                f"{perf['wr']:.2f}%"
            )

        with f:

            st.metric(
                "P&L",
                f"{perf['pnl']:.2f}%"
            )

        # ====================================================
        # HISTÓRICO
        # ====================================================

        st.subheader(
            "📚 HISTÓRICO DE OPERAÇÕES"
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
            LIMIT 200
            """,
            conn
        )

        conn.close()

        if not historico.empty:

            historico["Operação"] = (
                historico["side"]
                .map(
                    {
                        "COMPRA": "🟢 COMPRA",
                        "VENDA": "🔴 VENDA"
                    }
                )
            )

            historico["Data/Hora Entrada"] = (
                historico["entry_time"]
            )

            historico["Preço de Entrada"] = (
                historico["entry_price"]
                .apply(
                    lambda x:
                    f"${x:,.2f}"
                )
            )

            historico["Stop"] = (
                historico["stop_price"]
                .apply(
                    lambda x:
                    f"${x:,.2f}"
                )
            )

            historico["Preço Alvo"] = (
                historico["target_price"]
                .apply(
                    lambda x:
                    f"${x:,.2f}"
                )
            )

            historico["Data/Hora Saída"] = (
                historico["exit_time"]
                .fillna("EM ABERTO")
            )

            historico["Preço de Saída"] = (
                historico["exit_price"]
                .apply(
                    lambda x:
                    "—"
                    if pd.isna(x)
                    else f"${x:,.2f}"
                )
            )

            historico["Resultado"] = (
                historico["result"]
                .fillna("ABERTA")
            )

            historico["P&L"] = (
                historico["pnl_pct"]
                .apply(
                    lambda x:
                    "—"
                    if pd.isna(x)
                    else f"{x:.2f}%"
                )
            )

            historico["Score"] = (
                historico["score"]
            )

            historico["Regime"] = (
                historico["regime"]
            )

            historico_final = historico[
                [
                    "id",
                    "Operação",
                    "Data/Hora Entrada",
                    "Preço de Entrada",
                    "Stop",
                    "Preço Alvo",
                    "Data/Hora Saída",
                    "Preço de Saída",
                    "Resultado",
                    "P&L",
                    "Score",
                    "Regime"
                ]
            ].rename(
                columns={
                    "id": "ID"
                }
            )

            st.dataframe(
                historico_final,
                use_container_width=True,
                hide_index=True
            )

        # ====================================================
        # RODAPÉ
        # ====================================================

        st.caption(
            f"🤖 Monitoramento automático ativo | "
            f"Atualização a cada {AUTO_REFRESH_SECONDS}s | "
            f"Máximo: {max_operacoes} operações | "
            f"Tempo máximo: {tempo_maximo} minutos | "
            f"Último candle analisado: {signal_time}"
        )

    except Exception as e:

        st.error(
            f"Erro no monitoramento: {e}"
        )


# ============================================================
# EXECUTAR
# ============================================================

monitor()
