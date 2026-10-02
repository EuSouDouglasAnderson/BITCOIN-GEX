import sqlite3
import math
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import uuid

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ============================================================
# CONFIGURAÇÃO
# ============================================================
BINANCE_API = "https://data-api.binance.vision"
DERIBIT_API = "https://www.deribit.com/api/v2"
SYMBOL = "BTCUSDT"
INTERVAL = "5m"
DATABASE = "btc_trader_v2.db"
AUTO_REFRESH_SECONDS = 10
KLINE_LIMIT = 500
ATR_STOP_MULTIPLIER = 1.0
ATR_TARGET_MULTIPLIER = 2.0

# Estratégia D: proxy de GEX baseado no open interest e gamma publicados
# pela Deribit. O sinal de GEX é uma PROXY, porque o posicionamento do dealer
# não é observável diretamente pelo open interest público.
D_MAX_DTE_HOURS = 24.0
D_ATM_BAND_PCT = 0.01
D_ATM_ELEVATED_PERCENTILE = 75.0
D_GEX_NEGATIVE_THRESHOLD = 0.0

# Estratégia E: ratio Brent / WTI (Crude)
BRENT_YAHOO = "BZ=F"
WTI_YAHOO = "CL=F"
E_RATIO_WINDOW = 100
E_RATIO_Z_THRESHOLD = 2.0
E_RATIO_INTERVAL = "5m"
E_RATIO_RANGE = "5d"

# Estratégia B: reversão à média
B_RSI_LOW = 30.0
B_RSI_HIGH = 70.0
B_BB_WINDOW = 20
B_BB_STD = 2.0
B_MIN_VWAP_DISTANCE = 0.003  # 0,30%

# Estratégia C: rompimento/momentum
C_DONCHIAN_WINDOW = 20
C_VOLUME_Z_MIN = 1.0
C_ATR_MIN_PCT = 0.001

# Estratégia F — GEX Walls / First Touch (Sniper A Seco)
F_BRASILIA_TZ = "America/Sao_Paulo"
F_SESSION_START_HOUR = 5
F_MAX_DTE_HOURS = 48.0
F_ATM_BAND_PCT = 0.05
F_MIN_WALL_DISTANCE_PCT = 0.005

# Estratégia G — GEX Expansion 1.50%
# O alvo de 1,50% é movimento do BTC. A alavancagem 10x NÃO entra no
# cálculo dos indicadores; ela só transforma aproximadamente +1,50% de
# movimento do ativo em +5% sobre a margem, antes de custos.
G_TARGET_PCT = 0.015
G_MIN_VOLUME_Z = 1.0
G_MIN_SCORE = 70.0
G_MAX_WALL_DISTANCE_PCT = 0.015
G_MAX_GAMMA_FLIP_DISTANCE_PCT = 0.015

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
            entry_reason TEXT,
            strategy TEXT DEFAULT 'A',
            cycle_id TEXT,
            gex_proxy REAL,
            gex_calls REAL,
            gex_puts REAL,
            atm_oi REAL,
            atm_oi_total REAL,
            atm_oi_ratio REAL,
            nearest_expiry TEXT,
            dte_hours REAL,
            gex_condition TEXT,
            brent_price REAL,
            wti_price REAL,
            brent_wti_ratio REAL,
            brent_wti_ratio_mean REAL,
            brent_wti_ratio_std REAL,
            brent_wti_z REAL,
            brent_wti_change REAL,
            brent_wti_condition TEXT
        )
        """
    )

    colunas = {row[1] for row in conn.execute("PRAGMA table_info(trades)").fetchall()}
    novas = {
        "exit_reason": "TEXT",
        "entry_reason": "TEXT",
        "strategy": "TEXT DEFAULT 'A'",
        "cycle_id": "TEXT",
        "gex_proxy": "REAL",
        "gex_calls": "REAL",
        "gex_puts": "REAL",
        "atm_oi": "REAL",
        "atm_oi_total": "REAL",
        "atm_oi_ratio": "REAL",
        "nearest_expiry": "TEXT",
        "dte_hours": "REAL",
        "gex_condition": "TEXT",
        "brent_price": "REAL",
        "wti_price": "REAL",
        "brent_wti_ratio": "REAL",
        "brent_wti_ratio_mean": "REAL",
        "brent_wti_ratio_std": "REAL",
        "brent_wti_z": "REAL",
        "brent_wti_change": "REAL",
        "brent_wti_condition": "TEXT",
    }
    for nome, tipo in novas.items():
        if nome not in colunas:
            conn.execute(f"ALTER TABLE trades ADD COLUMN {nome} {tipo}")

    # Estado persistente da Estratégia F: um snapshot de paredes por dia
    # operacional e, principalmente, o bloqueio do segundo toque.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS f_day_state (
            operational_date TEXT PRIMARY KEY,
            call_wall REAL,
            put_wall REAL,
            gamma_flip REAL,
            pin_candidate REAL,
            gamma_centroid REAL,
            first_touch_done INTEGER DEFAULT 0,
            first_touch_time TEXT,
            first_touch_wall TEXT,
            created_at TEXT NOT NULL
        )
        """
    )

    # Tabela separada para configurações persistentes da interface.
    # Não altera nem apaga a tabela de trades existente.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS app_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )

    conn.commit()
    conn.close()


def carregar_configuracoes():
    """Carrega as últimas configurações salvas no SQLite."""
    conn = get_conn()
    try:
        rows = conn.execute("SELECT key, value FROM app_settings").fetchall()
        return {key: value for key, value in rows}
    finally:
        conn.close()


def salvar_configuracoes():
    """Persiste as configurações atuais da sidebar no SQLite."""
    valores = {
        "estrategia": st.session_state.get("cfg_estrategia", "A — Atual"),
        "max_operacoes": int(st.session_state.get("cfg_max_operacoes", 5)),
        "tempo_maximo": int(st.session_state.get("cfg_tempo_maximo", 60)),
        "banca_inicial": float(st.session_state.get("cfg_banca_inicial", 1000.0)),
        "valor_entrada": float(st.session_state.get("cfg_valor_entrada", 100.0)),
        "automatizar_todas": bool(st.session_state.get("cfg_automatizar_todas", True)),
    }
    conn = get_conn()
    try:
        for key, value in valores.items():
            conn.execute(
                """
                INSERT INTO app_settings (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (key, str(value), agora()),
            )
        conn.commit()
    finally:
        conn.close()


def inicializar_configuracoes_session():
    """Inicializa os widgets com os valores persistidos apenas uma vez por sessão."""
    if st.session_state.get("_configs_carregadas", False):
        return

    salvas = carregar_configuracoes()
    estrategias = [
        "A — Atual",
        "B — Reversão Bollinger + RSI + VWAP",
        "C — Rompimento Donchian + Volume",
        "D — GEX + OI ATM + Expiração + Dual",
        "E — Brent/WTI + Sinal BTC",
        "F — GEX Walls / First Touch",
    ]

    estrategia_salva = salvas.get("estrategia", "A — Atual")
    if estrategia_salva not in estrategias:
        estrategia_salva = "A — Atual"

    st.session_state["cfg_estrategia"] = estrategia_salva
    st.session_state["cfg_max_operacoes"] = int(salvas.get("max_operacoes", 5))
    st.session_state["cfg_tempo_maximo"] = int(salvas.get("tempo_maximo", 60))
    st.session_state["cfg_banca_inicial"] = float(salvas.get("banca_inicial", 1000.0))
    st.session_state["cfg_valor_entrada"] = float(salvas.get("valor_entrada", 100.0))
    # Compatibilidade com versões anteriores que salvavam percentual
    if "valor_entrada" not in salvas and "percentual_entrada" in salvas:
        st.session_state["cfg_valor_entrada"] = 100.0
    st.session_state["cfg_automatizar_todas"] = str(salvas.get("automatizar_todas", "True")).lower() in ("1", "true", "sim", "yes")
    st.session_state["_configs_carregadas"] = True

    # Garante que valores antigos/inválidos não quebrem os widgets.
    st.session_state["cfg_max_operacoes"] = min(50, max(1, st.session_state["cfg_max_operacoes"]))
    st.session_state["cfg_tempo_maximo"] = min(1440, max(5, st.session_state["cfg_tempo_maximo"]))
    st.session_state["cfg_banca_inicial"] = max(1.0, st.session_state["cfg_banca_inicial"])
    st.session_state["cfg_valor_entrada"] = max(1.0, float(st.session_state["cfg_valor_entrada"]))


criar_banco()
inicializar_configuracoes_session()

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================
def agora():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def fmt_num(value, casas=2):
    if value is None:
        return "-"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(value):
        return "-"
    return f"{value:,.{casas}f}"


def buscar_operacoes_abertas():
    conn = get_conn()
    df = pd.read_sql_query(
        """
        SELECT * FROM trades
        WHERE exit_time IS NULL
        ORDER BY id DESC
        """,
        conn,
    )
    conn.close()
    return df


def buscar_operacao_f_relevante():
    """Retorna a operação F mais relevante para o painel.

    Prioridade:
      1. operação F ainda aberta;
      2. última operação F registrada, mesmo que já encerrada.

    Isso evita que o painel perca a entrada/stop/alvo depois que o primeiro
    toque diário é consumido e o sinal F passa novamente para AGUARDAR.
    """
    conn = get_conn()
    try:
        row = conn.execute(
            """
            SELECT * FROM trades
            WHERE COALESCE(strategy, 'A') = 'F'
            ORDER BY
                CASE WHEN exit_time IS NULL THEN 0 ELSE 1 END,
                id DESC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            return None
        cols = [desc[0] for desc in conn.execute("SELECT * FROM trades LIMIT 1").description]
        return dict(zip(cols, row))
    finally:
        conn.close()


def buscar_historico(limite=200):
    conn = get_conn()
    df = pd.read_sql_query(
        """
        SELECT * FROM trades
        ORDER BY id DESC
        LIMIT ?
        """,
        conn,
        params=(limite,),
    )
    conn.close()
    return df


def entrada_ja_registrada(signal_time, strategy="A", cycle_id=None, side=None):
    conn = get_conn()
    query = "SELECT COUNT(*) FROM trades WHERE signal_time = ? AND COALESCE(strategy, 'A') = ?"
    params = [str(signal_time), strategy]
    if cycle_id is not None:
        query += " AND cycle_id = ?"
        params.append(cycle_id)
    if side is not None:
        query += " AND side = ?"
        params.append(side)
    cur = conn.execute(query, tuple(params))
    existe = cur.fetchone()[0] > 0
    conn.close()
    return existe


def validar_plano(side, entrada, stop, target):
    entrada = float(entrada)
    stop = float(stop)
    target = float(target)

    if not all(np.isfinite([entrada, stop, target])):
        return False, "Preço de entrada, stop ou alvo inválido."

    if side == "COMPRA" and not (stop < entrada < target):
        return False, "Plano inválido para COMPRA: STOP < ENTRADA < ALVO."
    if side == "VENDA" and not (target < entrada < stop):
        return False, "Plano inválido para VENDA: ALVO < ENTRADA < STOP."
    if side not in ("COMPRA", "VENDA"):
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
    strategy="A",
    cycle_id=None,
    gex_data=None,
    ratio_data=None,
    notes="",
):
    valido, mensagem = validar_plano(side, entry_price, stop_price, target_price)
    if not valido:
        raise ValueError(mensagem)

    gex_data = gex_data or {}
    ratio_data = ratio_data or {}
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO trades (
            symbol, side, entry_time, entry_price, stop_price, target_price,
            exit_time, exit_price, pnl_pct, result, exit_reason, score, regime,
            notes, signal_time,
            ema20, ema50, ema200, rsi, atr, ret_1, ret_3, ret_12, ret_48,
            volatility, vol_z, volume_ratio, volume_z, vwap, cvd_delta,
            orderbook_imbalance, score_compra, score_venda, signal, entry_reason,
            strategy, cycle_id, gex_proxy, gex_calls, gex_puts, atm_oi,
            atm_oi_total, atm_oi_ratio, nearest_expiry, dte_hours, gex_condition,
            brent_price, wti_price, brent_wti_ratio, brent_wti_ratio_mean,
            brent_wti_ratio_std, brent_wti_z, brent_wti_change, brent_wti_condition
        ) VALUES (
            ?, ?, ?, ?, ?, ?,
            NULL, NULL, NULL, NULL, NULL, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?
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
            strategy,
            cycle_id,
            gex_data.get("gex_proxy"),
            gex_data.get("gex_calls"),
            gex_data.get("gex_puts"),
            gex_data.get("atm_oi"),
            gex_data.get("atm_oi_total"),
            gex_data.get("atm_oi_ratio"),
            gex_data.get("nearest_expiry"),
            gex_data.get("dte_hours"),
            gex_data.get("gex_condition"),
            ratio_data.get("brent_price"),
            ratio_data.get("wti_price"),
            ratio_data.get("ratio"),
            ratio_data.get("ratio_mean"),
            ratio_data.get("ratio_std"),
            ratio_data.get("z"),
            ratio_data.get("change"),
            ratio_data.get("condition"),
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

    result = "GANHO" if pnl_pct > 0 else "PERDA" if pnl_pct < 0 else "EMPATE"
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
            fechadas.append((trade_id, exit_reason))

    return fechadas

# ============================================================
# BINANCE
# ============================================================
@st.cache_data(ttl=4, show_spinner=False)
def buscar_klines():
    r = requests.get(
        f"{BINANCE_API}/api/v3/klines",
        params={"symbol": SYMBOL, "interval": INTERVAL, "limit": KLINE_LIMIT},
        timeout=10,
    )
    r.raise_for_status()
    data = r.json()
    cols = [
        "open_time", "Open", "High", "Low", "Close", "Volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(data, columns=cols)
    for col in ["Open", "High", "Low", "Close", "Volume", "quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    return df


@st.cache_data(ttl=2, show_spinner=False)
def buscar_preco():
    r = requests.get(
        f"{BINANCE_API}/api/v3/ticker/price",
        params={"symbol": SYMBOL},
        timeout=10,
    )
    r.raise_for_status()
    return float(r.json()["price"])


@st.cache_data(ttl=2, show_spinner=False)
def buscar_orderbook():
    r = requests.get(
        f"{BINANCE_API}/api/v3/depth",
        params={"symbol": SYMBOL, "limit": 50},
        timeout=10,
    )
    r.raise_for_status()
    data = r.json()
    bids = sum(float(price) * float(qty) for price, qty in data.get("bids", []))
    asks = sum(float(price) * float(qty) for price, qty in data.get("asks", []))
    total = bids + asks
    imbalance = (bids - asks) / total if total > 0 else 0.0
    return {"bids": bids, "asks": asks, "imbalance": imbalance}

# ============================================================
# BRENT / WTI — Estratégia E
# ============================================================
@st.cache_data(ttl=30, show_spinner=False)
def buscar_serie_yahoo(ticker, range_value=E_RATIO_RANGE, interval=E_RATIO_INTERVAL):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    r = requests.get(
        url,
        params={"range": range_value, "interval": interval, "events": "history"},
        timeout=15,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    r.raise_for_status()
    data = r.json().get("chart", {}).get("result")
    if not data:
        raise RuntimeError(f"Yahoo não retornou dados para {ticker}.")
    result = data[0]
    timestamps = result.get("timestamp", [])
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    closes = quote.get("close", [])
    frame = pd.DataFrame({"timestamp": pd.to_datetime(timestamps, unit="s", utc=True), "close": closes})
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["close"]).drop_duplicates("timestamp").sort_values("timestamp")
    if frame.empty:
        raise RuntimeError(f"Série vazia para {ticker}.")
    return frame


@st.cache_data(ttl=30, show_spinner=False)
def calcular_ratio_brent_wti():
    brent = buscar_serie_yahoo(BRENT_YAHOO).rename(columns={"close": "brent"})
    wti = buscar_serie_yahoo(WTI_YAHOO).rename(columns={"close": "wti"})
    df_ratio = pd.merge(brent, wti, on="timestamp", how="inner")
    df_ratio = df_ratio[(df_ratio["brent"] > 0) & (df_ratio["wti"] > 0)].copy()
    df_ratio["ratio"] = df_ratio["brent"] / df_ratio["wti"]
    df_ratio["ratio_mean"] = df_ratio["ratio"].rolling(E_RATIO_WINDOW).mean()
    df_ratio["ratio_std"] = df_ratio["ratio"].rolling(E_RATIO_WINDOW).std()
    df_ratio["z"] = (df_ratio["ratio"] - df_ratio["ratio_mean"]) / df_ratio["ratio_std"].replace(0, np.nan)
    df_ratio["change"] = df_ratio["ratio"].pct_change()
    ultimo = df_ratio.iloc[-1]
    z = float(ultimo["z"]) if pd.notna(ultimo["z"]) else np.nan
    if np.isfinite(z) and z >= E_RATIO_Z_THRESHOLD:
        condition = "RATIO ALTO / EXTREMO"
    elif np.isfinite(z) and z <= -E_RATIO_Z_THRESHOLD:
        condition = "RATIO BAIXO / EXTREMO"
    elif np.isfinite(z):
        condition = "RATIO NORMAL"
    else:
        condition = "RATIO INSUFICIENTE"
    return {
        "brent_price": float(ultimo["brent"]),
        "wti_price": float(ultimo["wti"]),
        "ratio": float(ultimo["ratio"]),
        "ratio_mean": float(ultimo["ratio_mean"]) if pd.notna(ultimo["ratio_mean"]) else None,
        "ratio_std": float(ultimo["ratio_std"]) if pd.notna(ultimo["ratio_std"]) else None,
        "z": z if np.isfinite(z) else None,
        "change": float(ultimo["change"]) if pd.notna(ultimo["change"]) else None,
        "condition": condition,
        "ready": bool(np.isfinite(z)),
    }


def estrategia_e_signal(sinal_tecnico, ratio_data):
    """
    Estratégia E usa Brent/WTI como filtro de contexto, não como previsão
    direcional isolada: só libera o sinal técnico quando o ratio está em
    extremo (|Z| >= 2). A direção continua vindo do modelo técnico BTC.
    """
    if not ratio_data or not ratio_data.get("ready"):
        return "AGUARDAR", "E sem Z-score suficiente para o ratio Brent/WTI."
    z = float(ratio_data["z"])
    if abs(z) < E_RATIO_Z_THRESHOLD:
        return "AGUARDAR", f"E bloqueada: |Z| {abs(z):.2f} < {E_RATIO_Z_THRESHOLD:.1f}."
    if sinal_tecnico not in ("COMPRA", "VENDA"):
        return "AGUARDAR", "E encontrou extremo no ratio, mas o sinal técnico BTC não confirmou direção."
    return sinal_tecnico, (
        f"E confirmada | Brent ${ratio_data['brent_price']:.2f} | WTI ${ratio_data['wti_price']:.2f} | "
        f"Ratio {ratio_data['ratio']:.4f} | Z {z:+.2f} | {ratio_data['condition']} | "
        f"direção BTC confirmada pelo sinal técnico {sinal_tecnico}."
    )


# ============================================================
# DERIBIT / GEX
# ============================================================
@st.cache_data(ttl=30, show_spinner=False)
def buscar_opcoes_deribit():
    """
    Busca opções BTC na Deribit usando os endpoints públicos.

    Importante: get_book_summary_by_currency normalmente fornece OI e mark_iv,
    mas não precisa fornecer o campo `greeks`. Por isso, o GEX desta versão
    calcula o gamma de Black-Scholes a partir de:
        - open_interest
        - mark_iv
        - strike
        - expiration_timestamp
        - preço spot do BTC

    Assim não dependemos de um campo `greeks` que pode não vir no summary.
    """
    r_summary = requests.get(
        f"{DERIBIT_API}/public/get_book_summary_by_currency",
        params={"currency": "BTC", "kind": "option"},
        timeout=15,
    )
    r_summary.raise_for_status()
    payload = r_summary.json()
    if "error" in payload:
        raise RuntimeError(f"Deribit summary: {payload['error'].get('message', payload['error'])}")
    summary = payload.get("result", [])
    if not summary:
        raise RuntimeError("Deribit não retornou resumo de opções BTC.")

    r_instruments = requests.get(
        f"{DERIBIT_API}/public/get_instruments",
        params={"currency": "BTC", "kind": "option", "expired": "false"},
        timeout=15,
    )
    r_instruments.raise_for_status()
    payload_i = r_instruments.json()
    if "error" in payload_i:
        raise RuntimeError(f"Deribit instruments: {payload_i['error'].get('message', payload_i['error'])}")
    instruments = payload_i.get("result", [])

    metadata = {item.get("instrument_name"): item for item in instruments}
    resultado = []
    for item in summary:
        nome = item.get("instrument_name")
        if not nome:
            continue
        meta = metadata.get(nome, {})
        combinado = dict(meta)
        combinado.update(item)
        resultado.append(combinado)

    if not resultado:
        raise RuntimeError("Deribit retornou opções, mas nenhum instrumento pôde ser associado.")

    return resultado


def _normalizar_iv(mark_iv):
    """Converte IV da Deribit para decimal anual."""
    try:
        iv = float(mark_iv)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(iv) or iv <= 0:
        return np.nan
    # A API pode representar IV em pontos percentuais (ex.: 65.0).
    # Se vier como decimal (ex.: 0.65), preservamos.
    return iv / 100.0 if iv > 3.0 else iv


def _gamma_black_scholes(spot, strike, iv, t_years):
    """Gamma aproximado de Black-Scholes para uma opção europeia."""
    if not all(np.isfinite([spot, strike, iv, t_years])):
        return np.nan
    if spot <= 0 or strike <= 0 or iv <= 0 or t_years <= 0:
        return np.nan
    try:
        sigma_sqrt_t = iv * math.sqrt(t_years)
        d1 = (
            math.log(spot / strike)
            + 0.5 * iv * iv * t_years
        ) / sigma_sqrt_t
        pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
        return pdf / (spot * sigma_sqrt_t)
    except (ValueError, ZeroDivisionError, OverflowError):
        return np.nan


def calcular_gex_proxy(opcoes, preco_btc):
    """
    Calcula uma proxy normalizada de GEX usando OI + IV + Black-Scholes gamma.

    Assunção explícita: calls contribuem positivamente e puts negativamente.
    Isso NÃO observa a posição real dos dealers; é uma variável de regime/pesquisa.

    A principal correção em relação à versão anterior é que não exigimos mais
    `greeks.gamma` no retorno da Deribit. O gamma é calculado a partir do mark_iv.
    """
    agora_utc = datetime.now(timezone.utc)
    registros = []
    spot = float(preco_btc)

    for item in opcoes:
        nome = item.get("instrument_name", "")
        if not nome:
            continue

        tipo = str(item.get("option_type", "")).lower()
        if tipo not in ("call", "put"):
            if nome.endswith("-C"):
                tipo = "call"
            elif nome.endswith("-P"):
                tipo = "put"
            else:
                continue

        try:
            strike = float(item.get("strike"))
        except (TypeError, ValueError):
            try:
                partes = nome.split("-")
                strike = float(partes[-2])
            except Exception:
                continue

        try:
            oi = float(item.get("open_interest"))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(oi) or oi <= 0:
            continue

        exp_ms = item.get("expiration_timestamp")
        if exp_ms is None:
            continue
        try:
            expiry = datetime.fromtimestamp(float(exp_ms) / 1000, tz=timezone.utc)
        except Exception:
            continue

        dte_hours = (expiry - agora_utc).total_seconds() / 3600.0
        if dte_hours < -1:
            continue

        iv = _normalizar_iv(item.get("mark_iv"))
        if not np.isfinite(iv):
            continue

        t_years = max(dte_hours, 0.01) / (24.0 * 365.0)
        gamma = _gamma_black_scholes(spot, strike, iv, t_years)
        if not np.isfinite(gamma) or gamma <= 0:
            continue

        # Contract size vem dos metadados quando disponível. Para a proxy,
        # usamos 1.0 como fallback para não descartar a opção.
        try:
            contract_size = float(item.get("contract_size", 1.0))
        except (TypeError, ValueError):
            contract_size = 1.0
        if not np.isfinite(contract_size) or contract_size <= 0:
            contract_size = 1.0

        # Proxy de exposição. O fator 0.01 só reduz a escala visual.
        bruto = oi * contract_size * gamma * (spot ** 2) * 0.01
        assinado = bruto if tipo == "call" else -bruto

        registros.append(
            {
                "instrument_name": nome,
                "type": tipo,
                "strike": strike,
                "oi": oi,
                "iv": iv,
                "gamma": gamma,
                "expiry": expiry,
                "dte_hours": dte_hours,
                "gex": assinado,
            }
        )

    if not registros:
        raise RuntimeError(
            "Deribit retornou opções, mas não encontrei opções com OI + mark_iv + strike + expiração válidos para calcular gamma."
        )

    df = pd.DataFrame(registros)
    expiries = sorted(df["expiry"].unique())
    nearest_expiry = expiries[0]
    nearest = df[df["expiry"] == nearest_expiry].copy()

    gex_calls = float(df.loc[df["type"] == "call", "gex"].sum())
    gex_puts = float(df.loc[df["type"] == "put", "gex"].sum())
    gex_proxy = float(df["gex"].sum())

    # ATM OI na expiração mais próxima: strikes dentro de ±1% do spot.
    atm_mask = (nearest["strike"] / spot - 1).abs() <= D_ATM_BAND_PCT
    atm_oi = float(nearest.loc[atm_mask, "oi"].sum())
    atm_oi_total = float(nearest["oi"].sum())
    atm_oi_ratio = atm_oi / atm_oi_total if atm_oi_total > 0 else 0.0

    # Elevado é relativo às expirações disponíveis.
    atm_por_exp = []
    for _, grupo in df.groupby("expiry"):
        mask = (grupo["strike"] / spot - 1).abs() <= D_ATM_BAND_PCT
        atm_por_exp.append(float(grupo.loc[mask, "oi"].sum()))
    threshold = float(np.percentile(atm_por_exp, D_ATM_ELEVATED_PERCENTILE)) if atm_por_exp else 0.0
    atm_elevado = atm_oi >= threshold and atm_oi > 0

    dte_hours = float((nearest_expiry - agora_utc).total_seconds() / 3600.0)
    gex_negativo = gex_proxy < D_GEX_NEGATIVE_THRESHOLD
    dentro_janela = dte_hours <= D_MAX_DTE_HOURS

    condicoes = ["GEX negativo" if gex_negativo else "GEX não negativo"]
    condicoes.append("OI ATM elevado" if atm_elevado else "OI ATM não elevado")
    condicoes.append("expiração <= 24h" if dentro_janela else "expiração > 24h")

    d_ativa = bool(gex_negativo and atm_elevado and dentro_janela)

    return {
        "gex_proxy": gex_proxy,
        "gex_calls": gex_calls,
        "gex_puts": gex_puts,
        "atm_oi": atm_oi,
        "atm_oi_total": atm_oi_total,
        "atm_oi_ratio": atm_oi_ratio,
        "atm_oi_threshold": threshold,
        "atm_elevado": atm_elevado,
        "nearest_expiry": nearest_expiry.isoformat(),
        "dte_hours": dte_hours,
        "gex_negativo": gex_negativo,
        "dentro_janela": dentro_janela,
        "d_ativa": d_ativa,
        "gex_condition": "D ATIVA" if d_ativa else "D INATIVA",
        "condicoes": condicoes,
        "options_used": len(df),
        "gex_method": "OI + mark_iv + Black-Scholes gamma",
    }

# ============================================================
# ESTRATÉGIA F — GEX WALLS / FIRST TOUCH
# ============================================================
def f_data_operacional():
    """Dia operacional F: começa às 05:00 no horário de Brasília."""
    agora_sp = datetime.now(timezone.utc).astimezone(ZoneInfo(F_BRASILIA_TZ))
    data = agora_sp.date()
    if agora_sp.hour < F_SESSION_START_HOUR:
        data = data - timedelta(days=1)
    return data.isoformat(), agora_sp


def f_buscar_estado():
    """Busca o snapshot persistente do dia operacional atual."""
    operational_date, _ = f_data_operacional()
    conn = get_conn()
    try:
        row = conn.execute(
            """
            SELECT operational_date, call_wall, put_wall, gamma_flip,
                   pin_candidate, gamma_centroid, first_touch_done,
                   first_touch_time, first_touch_wall, created_at
            FROM f_day_state
            WHERE operational_date = ?
            """,
            (operational_date,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None

    cols = [
        "operational_date", "call_wall", "put_wall", "gamma_flip",
        "pin_candidate", "gamma_centroid", "first_touch_done",
        "first_touch_time", "first_touch_wall", "created_at",
    ]
    return dict(zip(cols, row))


def f_criar_snapshot(walls):
    """Cria o snapshot apenas uma vez por dia; nunca substitui walls já fixadas."""
    if not walls:
        return f_buscar_estado()

    operational_date, agora_sp = f_data_operacional()
    existente = f_buscar_estado()
    if existente is not None:
        return existente

    put_wall = float(walls["put_wall"])
    call_wall = float(walls["call_wall"])

    if not np.isfinite(put_wall) or not np.isfinite(call_wall):
        return None
    if put_wall >= call_wall:
        return None

    conn = get_conn()
    try:
        conn.execute(
            """
            INSERT OR IGNORE INTO f_day_state (
                operational_date, call_wall, put_wall, gamma_flip,
                pin_candidate, gamma_centroid, first_touch_done,
                first_touch_time, first_touch_wall, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, 0, NULL, NULL, ?)
            """,
            (
                operational_date,
                call_wall,
                put_wall,
                float(walls.get("gamma_flip")) if walls.get("gamma_flip") is not None else None,
                float(walls.get("pin_candidate")) if walls.get("pin_candidate") is not None else None,
                float(walls.get("gamma_centroid")) if walls.get("gamma_centroid") is not None else None,
                agora_sp.strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    return f_buscar_estado()


def f_marcar_primeiro_toque(wall_name, signal_time):
    """Consome o primeiro toque do dia de forma atômica."""
    operational_date, _ = f_data_operacional()
    wall_name = str(wall_name)
    conn = get_conn()
    try:
        cur = conn.execute(
            """
            UPDATE f_day_state
            SET first_touch_done = 1,
                first_touch_time = ?,
                first_touch_wall = ?
            WHERE operational_date = ?
              AND COALESCE(first_touch_done, 0) = 0
            """,
            (str(signal_time), wall_name, operational_date),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def calcular_gex_walls(opcoes, preco_btc):
    """Calcula as GEX Walls para a estratégia F.

    Regras operacionais:
      - Put Wall: maior put GEX em strike abaixo do spot.
      - Call Wall: maior call GEX em strike acima do spot.
      - Gamma Flip: ponto de mudança de sinal da soma acumulada do GEX.
      - Pin: strike com maior OI.
      - Centróide: média ponderada pelo módulo do GEX.

    O snapshot diário é tratado separadamente por f_criar_snapshot().
    """
    agora_utc = datetime.now(timezone.utc)
    try:
        spot = float(preco_btc)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(spot) or spot <= 0:
        return None

    registros = []
    for item in opcoes or []:
        nome = str(item.get("instrument_name", ""))
        tipo = str(item.get("option_type", "")).lower()
        if tipo not in ("call", "put"):
            if nome.endswith("-C"):
                tipo = "call"
            elif nome.endswith("-P"):
                tipo = "put"
            else:
                continue

        try:
            strike = float(item.get("strike"))
            oi = float(item.get("open_interest"))
            exp_ms = float(item.get("expiration_timestamp"))
        except (TypeError, ValueError):
            continue
        if not all(np.isfinite([strike, oi, exp_ms])) or oi <= 0 or strike <= 0:
            continue

        try:
            expiry = datetime.fromtimestamp(exp_ms / 1000.0, tz=timezone.utc)
        except Exception:
            continue
        dte_hours = (expiry - agora_utc).total_seconds() / 3600.0
        if dte_hours < -1 or dte_hours > F_MAX_DTE_HOURS:
            continue

        iv = _normalizar_iv(item.get("mark_iv"))
        if not np.isfinite(iv):
            continue
        gamma = _gamma_black_scholes(
            spot, strike, iv, max(dte_hours, 0.01) / (24.0 * 365.0)
        )
        if not np.isfinite(gamma) or gamma <= 0:
            continue

        try:
            contract_size = float(item.get("contract_size", 1.0))
        except (TypeError, ValueError):
            contract_size = 1.0
        if not np.isfinite(contract_size) or contract_size <= 0:
            contract_size = 1.0

        bruto = oi * contract_size * gamma * (spot ** 2) * 0.01
        registros.append({
            "type": tipo,
            "strike": strike,
            "oi": oi,
            "gex": bruto if tipo == "call" else -bruto,
            "expiry": expiry,
        })

    if not registros:
        return None

    d = pd.DataFrame(registros)
    d = d[(d["strike"] / spot - 1).abs() <= F_ATM_BAND_PCT].copy()
    if d.empty:
        return None

    por_strike = (
        d.groupby("strike", as_index=False)
        .agg(
            net_gex=("gex", "sum"),
            call_gex=("gex", lambda x: float(x[x > 0].sum())),
            put_gex=("gex", lambda x: float(abs(x[x < 0].sum()))),
            oi=("oi", "sum"),
        )
        .sort_values("strike")
        .reset_index(drop=True)
    )
    if por_strike.empty:
        return None

    # --------------------------------------------------------
    # WALLS: força a geometria Put < Spot < Call sempre que
    # existirem strikes suficientes dos dois lados.
    # --------------------------------------------------------
    distancia_min = max(0.0, float(F_MIN_WALL_DISTANCE_PCT))
    put_preferidos = por_strike[
        (por_strike["strike"] < spot * (1.0 - distancia_min))
        & (por_strike["put_gex"] > 0)
    ].copy()
    call_preferidos = por_strike[
        (por_strike["strike"] > spot * (1.0 + distancia_min))
        & (por_strike["call_gex"] > 0)
    ].copy()

    # Fallback: se não houver distância mínima disponível, usa qualquer
    # strike válido de cada lado do spot.
    if put_preferidos.empty:
        put_preferidos = por_strike[(por_strike["strike"] < spot) & (por_strike["put_gex"] > 0)].copy()
    if call_preferidos.empty:
        call_preferidos = por_strike[(por_strike["strike"] > spot) & (por_strike["call_gex"] > 0)].copy()

    if put_preferidos.empty or call_preferidos.empty:
        return None

    put_row = put_preferidos.sort_values(["put_gex", "strike"], ascending=[False, False]).iloc[0]
    call_row = call_preferidos.sort_values(["call_gex", "strike"], ascending=[False, True]).iloc[0]
    put_wall = float(put_row["strike"])
    call_wall = float(call_row["strike"])

    if not (put_wall < spot < call_wall):
        return None

    # --------------------------------------------------------
    # PIN: maior OI por strike dentro da janela.
    # --------------------------------------------------------
    pin_row = por_strike.sort_values(["oi", "strike"], ascending=[False, True]).iloc[0]
    pin_candidate = float(pin_row["strike"])

    # --------------------------------------------------------
    # CENTRÓIDE GEX: ponderado pelo módulo do net GEX.
    # --------------------------------------------------------
    pesos = por_strike["net_gex"].abs().astype(float)
    if float(pesos.sum()) > 0:
        gamma_centroid = float((por_strike["strike"] * pesos).sum() / pesos.sum())
    else:
        gamma_centroid = float(spot)

    # --------------------------------------------------------
    # GAMMA FLIP: procura cruzamento do GEX acumulado.
    # --------------------------------------------------------
    acumulado = por_strike["net_gex"].cumsum().to_numpy(dtype=float)
    strikes = por_strike["strike"].to_numpy(dtype=float)
    gamma_flip = None
    for i in range(1, len(acumulado)):
        a = acumulado[i - 1]
        b = acumulado[i]
        if a == 0:
            gamma_flip = strikes[i - 1]
            break
        if (a < 0 <= b) or (a > 0 >= b):
            x1, x2 = strikes[i - 1], strikes[i]
            if b != a:
                gamma_flip = float(x1 + (0 - a) * (x2 - x1) / (b - a))
            else:
                gamma_flip = float(x1)
            break
    if gamma_flip is None:
        gamma_flip = float(strikes[int(np.argmin(np.abs(acumulado)))])

    return {
        "spot": spot,
        "put_wall": put_wall,
        "call_wall": call_wall,
        "gamma_flip": gamma_flip,
        "pin_candidate": pin_candidate,
        "gamma_centroid": gamma_centroid,
        "walls_validos": bool(put_wall < spot < call_wall),
        "put_distance_pct": (spot / put_wall - 1.0) * 100.0,
        "call_distance_pct": (call_wall / spot - 1.0) * 100.0,
        "walls_distance_pct": (call_wall - put_wall) / spot * 100.0,
        "walls_spaced": bool(((call_wall - put_wall) / spot) >= 0.03),
        "options_used": len(d),
    }


def estrategia_f_signal(row, f_estado):
    """Sinal F: exclusivamente no primeiro toque diário de uma wall.

    Put Wall tocada + fechamento de confirmação acima -> COMPRA.
    Call Wall tocada + fechamento de confirmação abaixo -> VENDA.
    As duas walls no mesmo candle -> BLOQUEAR.
    O consumo persistente do primeiro toque é feito pelo monitor.
    """
    fatores_compra = []
    fatores_venda = []

    if not f_estado:
        return "AGUARDAR", "F aguardando snapshot diário das GEX Walls.", fatores_compra, fatores_venda

    if int(f_estado.get("first_touch_done") or 0) == 1:
        wall = f_estado.get("first_touch_wall") or "desconhecida"
        return "AGUARDAR", f"F bloqueada: primeiro toque do dia já consumido ({wall}).", fatores_compra, fatores_venda

    try:
        low = float(row["Low"])
        high = float(row["High"])
        close = float(row["Close"])
        put_wall = float(f_estado["put_wall"])
        call_wall = float(f_estado["call_wall"])
    except (TypeError, ValueError, KeyError):
        return "AGUARDAR", "F sem dados válidos de candle/walls.", fatores_compra, fatores_venda

    toque_put = low <= put_wall <= high
    toque_call = low <= call_wall <= high

    if toque_put and toque_call:
        return "BLOQUEAR", (
            f"F BLOQUEAR | candle tocou Put Wall ${put_wall:,.0f} e Call Wall ${call_wall:,.0f} simultaneamente."
        ), fatores_compra, fatores_venda

    if toque_put:
        fatores_compra.extend([
            f"Primeiro toque Put Wall ${put_wall:,.0f}",
            f"Fechamento ${close:,.0f} {'acima' if close >= put_wall else 'abaixo'} da Put Wall",
        ])
        if close >= put_wall:
            return "COMPRA", (
                f"F SNIPER A SECO | primeiro toque PUT WALL ${put_wall:,.0f} | "
                f"close ${close:,.0f} confirmou acima da wall."
            ), fatores_compra, fatores_venda
        return "AGUARDAR", (
            f"F primeiro toque PUT WALL ${put_wall:,.0f}, mas sem confirmação de fechamento acima."
        ), fatores_compra, fatores_venda

    if toque_call:
        fatores_venda.extend([
            f"Primeiro toque Call Wall ${call_wall:,.0f}",
            f"Fechamento ${close:,.0f} {'abaixo' if close <= call_wall else 'acima'} da Call Wall",
        ])
        if close <= call_wall:
            return "VENDA", (
                f"F SNIPER A SECO | primeiro toque CALL WALL ${call_wall:,.0f} | "
                f"close ${close:,.0f} confirmou abaixo da wall."
            ), fatores_compra, fatores_venda
        return "AGUARDAR", (
            f"F primeiro toque CALL WALL ${call_wall:,.0f}, mas sem confirmação de fechamento abaixo."
        ), fatores_compra, fatores_venda

    return "AGUARDAR", (
        f"F aguardando primeiro toque | Put ${put_wall:,.0f} | Call ${call_wall:,.0f} | Close ${close:,.0f}."
    ), fatores_compra, fatores_venda


# ============================================================
# ESTRATÉGIA G — GEX EXPANSION 1.50% (versão com painel)
# ============================================================
import numpy as np

# Constantes (usam o valor do seu módulo se já estiverem definidas; senão, estes padrões).
# AJUSTE os valores padrão abaixo conforme o seu backtest.
G_TARGET_PCT = globals().get("G_TARGET_PCT", 0.015)                          # alvo de 1,50%
G_MAX_GAMMA_FLIP_DISTANCE_PCT = globals().get("G_MAX_GAMMA_FLIP_DISTANCE_PCT", 0.015)
G_MAX_WALL_DISTANCE_PCT = globals().get("G_MAX_WALL_DISTANCE_PCT", 0.015)
G_MIN_VOLUME_Z = globals().get("G_MIN_VOLUME_Z", 0.5)
G_MIN_SCORE = globals().get("G_MIN_SCORE", 60.0)
G_FLIP_DIRECTIONAL = globals().get("G_FLIP_DIRECTIONAL", False)
G_FEE_ROUND_TRIP_PCT = globals().get("G_FEE_ROUND_TRIP_PCT", 0.0010)         # 0,05% taker x 2 lados
G_GEX_NORM_THRESHOLD = globals().get("G_GEX_NORM_THRESHOLD", 0.0)            # sugestão: -0.10
G_MIN_WALL_SPREAD_PCT = globals().get("G_MIN_WALL_SPREAD_PCT", 0.018)        # walls encavaladas < 1,80%
G_LEVERAGE_REF = 10.0


def _detalhes_base(status, **extra):
    """Detalhes mínimos para os retornos antecipados (o painel usa d.get)."""
    d = {
        "status": status,
        "target_pct": G_TARGET_PCT * 100.0,
        "leverage_reference": G_LEVERAGE_REF,
        "roe_target_pct": G_TARGET_PCT * 100.0 * G_LEVERAGE_REF,
        "min_score": G_MIN_SCORE,
    }
    d.update(extra)
    return d


def estrategia_g_signal(row, gex_data, walls_data):
    """GEX Expansion: procura movimento de pelo menos +/-1,50% no BTC.

    10x é usado apenas como referência de ROE, nunca para normalizar
    GEX, Walls ou Volume Z.

    Condições de ativação (todas obrigatórias):
      1. GEX normalizado < G_GEX_NORM_THRESHOLD (regime de expansão)
      2. Spot a no máximo G_MAX_GAMMA_FLIP_DISTANCE_PCT do Gamma Flip
         (se G_FLIP_DIRECTIONAL=True, exige também o lado: LONG acima, SHORT abaixo)
      3. Wall do lado do trade (Call p/ LONG, Put p/ SHORT) entre 0 e
         G_MAX_WALL_DISTANCE_PCT do spot
      4. Volume Z >= G_MIN_VOLUME_Z
      5. Walls não encavaladas (spread >= G_MIN_WALL_SPREAD_PCT)
      6. Score do lado >= G_MIN_SCORE e maior que o score do lado oposto

    Score (0-100): GEX 25 + Flip 20 + Wall 20..30 + Volume 15 + bônus 10
    (bônus quando a wall está além do alvo, ou seja, o caminho até o alvo
    não é barrado por ela).

    Retorna: (sinal, mensagem, fatores_compra, fatores_venda,
              score_compra, score_venda, detalhes)
    """
    fatores_compra, fatores_venda = [], []

    if not gex_data:
        return "AGUARDAR", "G aguardando dados GEX da Deribit.", fatores_compra, fatores_venda, 0.0, 0.0, \
            _detalhes_base("SEM_GEX")
    if not walls_data:
        return "AGUARDAR", "GEX disponível, mas G aguardando Gamma Flip/Walls válidas.", \
            fatores_compra, fatores_venda, 0.0, 0.0, _detalhes_base("SEM_WALLS")

    try:
        spot = float(row["Close"])
        volume_z = float(row["volume_z"])
        gex = float(gex_data["gex_proxy"])
        gex_calls = abs(float(gex_data.get("gex_calls", 0.0)))
        gex_puts = abs(float(gex_data.get("gex_puts", 0.0)))
        gamma_flip = float(walls_data["gamma_flip"])
        put_wall = float(walls_data["put_wall"])
        call_wall = float(walls_data["call_wall"])
    except (TypeError, ValueError, KeyError):
        return "AGUARDAR", "G sem dados válidos para normalização.", fatores_compra, fatores_venda, 0.0, 0.0, \
            _detalhes_base("DADOS_INVALIDOS")

    valores = [spot, volume_z, gex, gex_calls, gex_puts, gamma_flip, put_wall, call_wall]
    if not all(np.isfinite(valores)) or spot <= 0 or gamma_flip <= 0 or put_wall <= 0 or call_wall <= 0:
        return "AGUARDAR", "G sem dados numéricos válidos.", fatores_compra, fatores_venda, 0.0, 0.0, \
            _detalhes_base("DADOS_NAO_NUMERICOS")

    total_abs = gex_calls + gex_puts
    gex_norm = float(np.clip(gex / total_abs, -1.0, 1.0)) if total_abs > 0 else 0.0
    gamma_flip_dist = spot / gamma_flip - 1.0
    put_dist = (spot - put_wall) / spot
    call_dist = (call_wall - spot) / spot
    wall_spread = abs(call_wall - put_wall) / spot
    walls_ok = wall_spread >= G_MIN_WALL_SPREAD_PCT

    def score_lado(lado):
        """Calcula todos os componentes (sem retorno antecipado).

        Retorna (score, fatores, gates). O lado só é ativável se
        all(gates.values()) for True.
        """
        score = 0.0
        fatores = []
        gates = {}

        # 1. GEX
        gex_ok = gex_norm < G_GEX_NORM_THRESHOLD
        gates["GEX"] = gex_ok
        if gex_ok:
            score += 25.0
            fatores.append(f"GEX expansão normalizado {gex_norm:+.2f}")
        else:
            fatores.append(f"GEX não confirmou expansão ({gex_norm:+.2f})")

        # 2. Gamma Flip
        if lado == "COMPRA":
            wall_dist, wall_nome = call_dist, "Call Wall"
            lado_flip_ok = gamma_flip_dist >= 0
        else:
            wall_dist, wall_nome = put_dist, "Put Wall"
            lado_flip_ok = gamma_flip_dist <= 0

        perto_flip = abs(gamma_flip_dist) <= G_MAX_GAMMA_FLIP_DISTANCE_PCT
        flip_ok = perto_flip and (lado_flip_ok if G_FLIP_DIRECTIONAL else True)
        gates["Flip"] = flip_ok
        if flip_ok:
            score += 20.0
            fatores.append(f"Gamma Flip confirmado | distância {gamma_flip_dist:+.2%}")
        elif not perto_flip:
            fatores.append(
                f"Gamma Flip longe ({gamma_flip_dist:+.2%} > {G_MAX_GAMMA_FLIP_DISTANCE_PCT:.2%})"
            )
        else:
            fatores.append(f"Gamma Flip do lado errado para {lado} ({gamma_flip_dist:+.2%})")

        # 3. Wall
        wall_ok = 0 <= wall_dist <= G_MAX_WALL_DISTANCE_PCT
        gates["Wall"] = wall_ok
        if wall_ok:
            proximity = max(0.0, 1.0 - wall_dist / G_MAX_WALL_DISTANCE_PCT)
            score += 20.0 + 10.0 * proximity
            fatores.append(f"{wall_nome} a {wall_dist:+.2%} | alvo {G_TARGET_PCT:.2%}")
        else:
            fatores.append(f"{wall_nome} fora da faixa ({wall_dist:+.2%})")

        # 4. Volume
        vol_ok = volume_z >= G_MIN_VOLUME_Z
        gates["Volume"] = vol_ok
        if vol_ok:
            score += 15.0 * min((volume_z - G_MIN_VOLUME_Z) / 2.0 + 0.5, 1.0)
            fatores.append(f"Volume Z {volume_z:+.2f}")
        else:
            fatores.append(f"Volume Z fraco ({volume_z:+.2f} < {G_MIN_VOLUME_Z:.2f})")

        # 5. Walls encavaladas (gate; não altera o score)
        gates["Spread"] = walls_ok
        if walls_ok:
            fatores.append(f"Spread das walls {wall_spread:.2%}")
        else:
            fatores.append(
                f"Walls encavaladas (spread {wall_spread:.2%} < {G_MIN_WALL_SPREAD_PCT:.2%})"
            )

        # Bônus: wall além do alvo (não barra o movimento antes de 1,50%)
        if wall_ok and wall_dist >= G_TARGET_PCT:
            score += 10.0
            fatores.append("Wall além do alvo (caminho livre)")

        return min(score, 100.0), fatores, gates

    score_compra, fatores_compra, gates_compra = score_lado("COMPRA")
    score_venda, fatores_venda, gates_venda = score_lado("VENDA")
    ok_compra = all(gates_compra.values())
    ok_venda = all(gates_venda.values())

    roe_bruto = G_TARGET_PCT * 100.0 * G_LEVERAGE_REF
    roe_liquido = (G_TARGET_PCT - G_FEE_ROUND_TRIP_PCT) * 100.0 * G_LEVERAGE_REF

    detalhes = _detalhes_base(
        "OK",
        spot=spot,
        gamma_flip=gamma_flip,
        put_wall=put_wall,
        call_wall=call_wall,
        gex_raw=gex,
        gex_norm=gex_norm,
        gamma_flip_dist_pct=gamma_flip_dist * 100.0,
        put_wall_dist_pct=put_dist * 100.0,
        call_wall_dist_pct=call_dist * 100.0,
        wall_spread_pct=wall_spread * 100.0,
        volume_z=volume_z,
        target_up=spot * (1 + G_TARGET_PCT),
        target_down=spot * (1 - G_TARGET_PCT),
        roe_net_est_pct=roe_liquido,
        gates_compra=gates_compra,
        gates_venda=gates_venda,
    )

    if ok_compra and score_compra >= G_MIN_SCORE and score_compra > score_venda:
        return ("COMPRA",
                f"G ATIVA | score {score_compra:.0f} | alvo BTC +{G_TARGET_PCT:.2%} "
                f"(~+{roe_bruto:.1f}% ROE bruto / ~{roe_liquido:.1f}% líquido em 10x).",
                fatores_compra, fatores_venda, score_compra, score_venda, detalhes)
    if ok_venda and score_venda >= G_MIN_SCORE and score_venda > score_compra:
        return ("VENDA",
                f"G ATIVA | score {score_venda:.0f} | alvo BTC -{G_TARGET_PCT:.2%} "
                f"(~+{roe_bruto:.1f}% ROE bruto / ~{roe_liquido:.1f}% líquido em 10x).",
                fatores_compra, fatores_venda, score_compra, score_venda, detalhes)
    return ("AGUARDAR",
            f"G aguardando confirmação | compra {score_compra:.0f} | venda {score_venda:.0f} | mínimo {G_MIN_SCORE:.0f}.",
            fatores_compra, fatores_venda, score_compra, score_venda, detalhes)


def painel_g(sinal, score_c, score_v, d):
    """Texto do painel da estratégia G (use d.get: retornos antecipados têm poucos campos)."""
    ok = lambda b: "✅" if b else "❌"
    titulo = f"G — GEX Expansion {d.get('target_pct', G_TARGET_PCT * 100.0):.2f}%  |  {sinal}"

    if d.get("status") != "OK":
        return "\n".join([
            titulo,
            f"Status: {d.get('status', 'SEM_DADOS')}",
            f"Score COMPRA {score_c:.0f} | VENDA {score_v:.0f} (mín. {d.get('min_score', G_MIN_SCORE):.0f})",
            f"Alvo BTC: ±{d.get('target_pct', 0):.2f}%",
        ])

    linhas = [
        titulo,
        f"Spot: {d['spot']:,.2f}  |  Alvo: ↑ {d['target_up']:,.2f}  ↓ {d['target_down']:,.2f}",
        f"Gamma Flip: {d['gamma_flip']:,.0f} ({d['gamma_flip_dist_pct']:+.2f}%)",
        f"Call Wall: {d['call_wall']:,.0f} ({d['call_wall_dist_pct']:+.2f}%)  |  "
        f"Put Wall: {d['put_wall']:,.0f} ({d['put_wall_dist_pct']:+.2f}%)",
        f"Spread das walls: {d['wall_spread_pct']:.2f}%",
        f"GEX norm: {d['gex_norm']:+.2f} (bruto {d['gex_raw']:,.0f})  |  Volume Z: {d['volume_z']:+.2f}",
        f"ROE 10x: {d['roe_target_pct']:.1f}% bruto / {d['roe_net_est_pct']:.1f}% líquido",
        f"Score COMPRA {score_c:.0f} | VENDA {score_v:.0f} (mín. {d['min_score']:.0f})",
        "COMPRA: " + "  ".join(f"{k} {ok(v)}" for k, v in d["gates_compra"].items()),
        "VENDA:  " + "  ".join(f"{k} {ok(v)}" for k, v in d["gates_venda"].items()),
    ]
    return "\n".join(linhas)


def card_g_campos(sinal, score_c, score_v, d):
    """Lista de (rótulo, valor) para o card, no mesmo estilo do painel atual.

    Use direto no seu card: for rotulo, valor in card_g_campos(...): ...
    """
    ok = lambda b: "✅" if b else "❌"
    campos = [
        ("Score G COMPRA", f"{score_c:.0f}"),
        ("Score G VENDA", f"{score_v:.0f}"),
        ("Alvo BTC", f"±{d.get('target_pct', G_TARGET_PCT * 100.0):.2f}%"),
    ]

    if d.get("status") != "OK":
        campos.append(("Status", str(d.get("status", "SEM_DADOS"))))
        campos.append(("Sinal", sinal))
        return campos

    campos += [
        ("Preço BTC (spot)", f"US$ {d['spot']:,.2f}"),
        ("Alvo ↑ (compra)", f"US$ {d['target_up']:,.2f}"),
        ("Alvo ↓ (venda)", f"US$ {d['target_down']:,.2f}"),
        ("Gamma Flip", f"US$ {d['gamma_flip']:,.0f} ({d['gamma_flip_dist_pct']:+.2f}%)"),
        ("Call Wall", f"US$ {d['call_wall']:,.0f} ({d['call_wall_dist_pct']:+.2f}%)"),
        ("Put Wall", f"US$ {d['put_wall']:,.0f} ({d['put_wall_dist_pct']:+.2f}%)"),
        ("Spread das walls", f"{d['wall_spread_pct']:.2f}%"),
        ("GEX normalizado", f"{d['gex_norm']:+.2f} (bruto {d['gex_raw']:,.0f})"),
        ("Volume Z", f"{d['volume_z']:+.2f} (mín. {G_MIN_VOLUME_Z:.2f})"),
        ("ROE teórico em 10x", f"±{d['roe_target_pct']:.1f}% bruto / ±{d['roe_net_est_pct']:.1f}% líquido"),
        ("Condições COMPRA", "  ".join(f"{k} {ok(v)}" for k, v in d["gates_compra"].items())),
        ("Condições VENDA", "  ".join(f"{k} {ok(v)}" for k, v in d["gates_venda"].items())),
        ("Mínimo para ativar", f"{d['min_score']:.0f}"),
        ("Sinal", sinal),
    ]
    return campos


def card_g_html(sinal, score_c, score_v, d):
    """Card em HTML puro com todas as informações da estratégia G."""
    from html import escape
    cor = {"COMPRA": "#16a34a", "VENDA": "#dc2626"}.get(sinal, "#6b7280")
    linhas = "".join(
        f"<div style='display:flex;justify-content:space-between;gap:16px;"
        f"padding:4px 0;border-bottom:1px solid #2a2f3a'>"
        f"<span style='opacity:.7'>{escape(r)}</span><b>{escape(v)}</b></div>"
        for r, v in card_g_campos(sinal, score_c, score_v, d)
    )
    return (
        f"<div style='font-family:sans-serif;background:#161a23;color:#e5e7eb;"
        f"border-radius:12px;padding:16px;border-left:6px solid {cor};max-width:520px'>"
        f"<div style='font-size:16px;font-weight:700;margin-bottom:8px'>"
        f"G — GEX Expansion {d.get('target_pct', G_TARGET_PCT * 100.0):.2f}% "
        f"<span style='color:{cor}'>• {escape(sinal)}</span></div>{linhas}</div>"
    )


# Exemplo de uso:
# sinal, msg, f_c, f_v, s_c, s_v, det = estrategia_g_signal(row, gex_data, walls_data)
# print(painel_g(sinal, s_c, s_v, det))                            # texto simples
# for rotulo, valor in card_g_campos(sinal, s_c, s_v, det): ...    # no seu card atual
# html = card_g_html(sinal, s_c, s_v, det)                         # card pronto em HTML
# ============================================================
# INDICADORES
# ============================================================
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
    # RSI simétrico: quando não há perdas na janela, o RSI deve ser 100
    # (e não NaN). Quando não há ganhos, deve ser 0. Isso evita que a
    # Estratégia A/B ignore movimentos fortemente unidirecionais de alta.
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
    df["RSI"] = rsi

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

    # Bollinger Bands para a Estratégia B
    df["BB_MID"] = df["Close"].rolling(B_BB_WINDOW).mean()
    bb_std = df["Close"].rolling(B_BB_WINDOW).std()
    df["BB_UPPER"] = df["BB_MID"] + B_BB_STD * bb_std
    df["BB_LOWER"] = df["BB_MID"] - B_BB_STD * bb_std
    df["BB_WIDTH"] = (df["BB_UPPER"] - df["BB_LOWER"]) / df["BB_MID"].replace(0, np.nan)

    # Donchian usa apenas candles anteriores para evitar look-ahead.
    df["DONCHIAN_HIGH"] = df["High"].shift(1).rolling(C_DONCHIAN_WINDOW).max()
    df["DONCHIAN_LOW"] = df["Low"].shift(1).rolling(C_DONCHIAN_WINDOW).min()
    df["ATR_PCT"] = df["ATR"] / df["Close"].replace(0, np.nan)
    return df

# ============================================================
# ESTRATÉGIAS B E C
# ============================================================
def estrategia_b_signal(row):
    """
    B — Reversão à média.
    COMPRA: preço toca/rompe a banda inferior + RSI sobrevendido + distância
    relevante abaixo do VWAP.
    VENDA: espelho para banda superior/RSI sobrecomprado.
    O sinal exige as 3 condições, evitando operar apenas por RSI.
    """
    fatores_compra = []
    fatores_venda = []

    valores = [row.get("RSI"), row.get("BB_LOWER"), row.get("BB_UPPER"), row.get("VWAP")]
    if not all(pd.notna(x) for x in valores):
        return "AGUARDAR", "B sem dados suficientes para RSI/Bollinger/VWAP.", [], []

    close = float(row["Close"])
    rsi = float(row["RSI"])
    vwap = float(row["VWAP"])
    bb_lower = float(row["BB_LOWER"])
    bb_upper = float(row["BB_UPPER"])

    dist_vwap = (close / vwap) - 1 if vwap else np.nan

    compra = close <= bb_lower and rsi <= B_RSI_LOW and dist_vwap <= -B_MIN_VWAP_DISTANCE
    venda = close >= bb_upper and rsi >= B_RSI_HIGH and dist_vwap >= B_MIN_VWAP_DISTANCE

    if close <= bb_lower:
        fatores_compra.append("Preço <= banda inferior Bollinger")
    if rsi <= B_RSI_LOW:
        fatores_compra.append(f"RSI <= {B_RSI_LOW:.0f}")
    if np.isfinite(dist_vwap) and dist_vwap <= -B_MIN_VWAP_DISTANCE:
        fatores_compra.append(f"Preço {abs(dist_vwap)*100:.2f}% abaixo do VWAP")

    if close >= bb_upper:
        fatores_venda.append("Preço >= banda superior Bollinger")
    if rsi >= B_RSI_HIGH:
        fatores_venda.append(f"RSI >= {B_RSI_HIGH:.0f}")
    if np.isfinite(dist_vwap) and dist_vwap >= B_MIN_VWAP_DISTANCE:
        fatores_venda.append(f"Preço {abs(dist_vwap)*100:.2f}% acima do VWAP")

    if compra and not venda:
        return "COMPRA", (
            f"B REVERSÃO | RSI {rsi:.1f} | Close ${close:,.2f} | "
            f"BB inferior ${bb_lower:,.2f} | VWAP ${vwap:,.2f} | "
            f"dist. VWAP {dist_vwap*100:+.2f}%"
        ), fatores_compra, fatores_venda
    if venda and not compra:
        return "VENDA", (
            f"B REVERSÃO | RSI {rsi:.1f} | Close ${close:,.2f} | "
            f"BB superior ${bb_upper:,.2f} | VWAP ${vwap:,.2f} | "
            f"dist. VWAP {dist_vwap*100:+.2f}%"
        ), fatores_compra, fatores_venda
    return "AGUARDAR", (
        f"B sem confirmação | RSI {rsi:.1f} | dist. VWAP {dist_vwap*100:+.2f}%"
        if np.isfinite(dist_vwap) else "B sem confirmação."
    ), fatores_compra, fatores_venda


def estrategia_c_signal(row):
    """
    C — Rompimento/momentum.
    O Donchian é calculado somente com candles anteriores.
    Exige rompimento + tendência EMA20/EMA50 + volume acima da média + ATR
    mínimo relativo ao preço.
    """
    campos = ["DONCHIAN_HIGH", "DONCHIAN_LOW", "EMA20", "EMA50", "volume_z", "ATR_PCT"]
    if not all(pd.notna(row.get(c)) for c in campos):
        return "AGUARDAR", "C sem dados suficientes para Donchian/tendência/volume/ATR.", [], []

    close = float(row["Close"])
    high_break = float(row["DONCHIAN_HIGH"])
    low_break = float(row["DONCHIAN_LOW"])
    volume_z = float(row["volume_z"])
    atr_pct = float(row["ATR_PCT"])

    tendencia_alta = float(row["EMA20"]) > float(row["EMA50"])
    tendencia_baixa = float(row["EMA20"]) < float(row["EMA50"])
    volume_ok = volume_z >= C_VOLUME_Z_MIN
    volatilidade_ok = atr_pct >= C_ATR_MIN_PCT
    rompimento_alta = close > high_break
    rompimento_baixa = close < low_break

    fatores_compra = []
    fatores_venda = []
    if rompimento_alta:
        fatores_compra.append(f"Rompimento Donchian {C_DONCHIAN_WINDOW} acima de ${high_break:,.2f}")
    if tendencia_alta:
        fatores_compra.append("EMA20 > EMA50")
    if volume_ok:
        fatores_compra.append(f"Volume Z {volume_z:+.2f}")
    if volatilidade_ok:
        fatores_compra.append(f"ATR/Preço {atr_pct*100:.2f}%")

    if rompimento_baixa:
        fatores_venda.append(f"Rompimento Donchian {C_DONCHIAN_WINDOW} abaixo de ${low_break:,.2f}")
    if tendencia_baixa:
        fatores_venda.append("EMA20 < EMA50")
    if volume_ok:
        fatores_venda.append(f"Volume Z {volume_z:+.2f}")
    if volatilidade_ok:
        fatores_venda.append(f"ATR/Preço {atr_pct*100:.2f}%")

    compra = rompimento_alta and tendencia_alta and volume_ok and volatilidade_ok
    venda = rompimento_baixa and tendencia_baixa and volume_ok and volatilidade_ok

    if compra and not venda:
        return "COMPRA", (
            f"C ROMPIMENTO | Close ${close:,.2f} > Donchian ${high_break:,.2f} | "
            f"EMA20/50 alta | Volume Z {volume_z:+.2f} | ATR/Preço {atr_pct*100:.2f}%"
        ), fatores_compra, fatores_venda
    if venda and not compra:
        return "VENDA", (
            f"C ROMPIMENTO | Close ${close:,.2f} < Donchian ${low_break:,.2f} | "
            f"EMA20/50 baixa | Volume Z {volume_z:+.2f} | ATR/Preço {atr_pct*100:.2f}%"
        ), fatores_compra, fatores_venda
    return "AGUARDAR", (
        f"C sem confirmação | Close ${close:,.2f} | Donchian H ${high_break:,.2f} / L ${low_break:,.2f} | "
        f"Volume Z {volume_z:+.2f}"
    ), fatores_compra, fatores_venda


def montar_motivo_entrada_bc(strategy_code, side, row, motivo, fatores):
    score = 0
    if strategy_code == "B":
        score = 100 if side in ("COMPRA", "VENDA") else 0
        prefixo = "B — REVERSÃO"
    else:
        score = 100 if side in ("COMPRA", "VENDA") else 0
        prefixo = "C — ROMPIMENTO"
    return f"{prefixo} | {motivo} | Fatores: " + " | ".join(fatores)

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
    # RSI simétrico: evita viés estrutural para COMPRA.
    # Acima de 55 favorece COMPRA; abaixo de 45 favorece VENDA.
    if row["RSI"] >= 55:
        add_compra(10, "RSI >= 55")
    elif row["RSI"] <= 45:
        add_venda(10, "RSI <= 45")
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

    return compra, venda, sinal, regime, fatores_compra, fatores_venda


def montar_motivo_entrada(side, score_compra, score_venda, regime, fatores_compra, fatores_venda, gex_data=None):
    fatores = fatores_compra if side == "COMPRA" else fatores_venda
    score = score_compra if side == "COMPRA" else score_venda
    partes = [side, f"Score {score:.0f}", f"Regime {regime}"]
    partes.extend(fatores)
    if gex_data:
        partes.extend(
            [
                f"GEX proxy {gex_data['gex_proxy']:+,.2f}",
                f"OI ATM {gex_data['atm_oi']:,.2f}",
                f"OI ATM ratio {gex_data['atm_oi_ratio']:.2%}",
                f"DTE {gex_data['dte_hours']:.1f}h",
            ]
        )
    return " | ".join(partes)


def montar_motivo_dual_d(score_compra, score_venda, regime, gex_data):
    return (
        f"D DUAL/REVERSÃO | Score C {score_compra:.0f} | Score V {score_venda:.0f} | "
        f"Regime {regime} | GEX proxy {gex_data['gex_proxy']:+,.2f} | "
        f"OI ATM {gex_data['atm_oi']:,.2f} | OI ATM ratio {gex_data['atm_oi_ratio']:.2%} | "
        f"DTE {gex_data['dte_hours']:.1f}h | "
        + " | ".join(gex_data["condicoes"])
    )


def montar_motivo_entrada_e(side, score_compra, score_venda, regime, fatores_compra, fatores_venda, ratio_data):
    fatores = fatores_compra if side == "COMPRA" else fatores_venda
    score = score_compra if side == "COMPRA" else score_venda
    partes = [f"E BRENT/WTI", side, f"Score {score:.0f}", f"Regime {regime}"]
    partes.extend(fatores)
    partes.extend([
        f"Brent ${ratio_data['brent_price']:.2f}",
        f"WTI ${ratio_data['wti_price']:.2f}",
        f"Ratio {ratio_data['ratio']:.4f}",
        f"Ratio Z {ratio_data['z']:+.2f}",
        ratio_data["condition"],
    ])
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
# SIMULAÇÃO DA BANCA
# ============================================================
def simular_banca(historico, banca_inicial, valor_entrada):
    if historico.empty:
        return pd.DataFrame(), float(banca_inicial)
    df = historico[historico["pnl_pct"].notna()].copy()
    if df.empty:
        return pd.DataFrame(), float(banca_inicial)

    df["entry_time_sort"] = pd.to_datetime(df["entry_time"], errors="coerce")
    df = df.sort_values(["entry_time_sort", "id"]).reset_index(drop=True)
    banca = float(banca_inicial)
    valor_fixo = float(valor_entrada)
    registros = []

    for _, trade in df.iterrows():
        banca_antes = banca
        valor_op = min(valor_fixo, banca_antes)
        pnl_pct = float(trade["pnl_pct"])
        resultado_rs = valor_op * (pnl_pct / 100.0)
        banca += resultado_rs
        registros.append(
            {
                "#": int(trade["id"]),
                "Estratégia": trade.get("strategy", "A") or "A",
                "Ciclo": trade.get("cycle_id", "-") or "-",
                "Lado": trade["side"],
                "Entrada": trade["entry_time"],
                "Saída": trade["exit_time"],
                "P&L mercado %": pnl_pct,
                "Banca antes": banca_antes,
                "Valor entrada": valor_op,
                "Resultado R$": resultado_rs,
                "Banca depois": banca,
                "Resultado": trade["result"] if pd.notna(trade["result"]) else "-",
                "Saída por": trade["exit_reason"] if pd.notna(trade["exit_reason"]) else "-",
            }
        )
    return pd.DataFrame(registros), banca


# ============================================================
# GRÁFICO DE CANDLES
# ============================================================
def criar_grafico_candles(df, janela=100):
    d = df.tail(janela).copy()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=d["close_time"], open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
        name="BTC/USDT"
    ))
    for col, nome in [("EMA20", "EMA20"), ("EMA50", "EMA50"), ("EMA200", "EMA200"), ("VWAP", "VWAP")]:
        if col in d.columns:
            fig.add_trace(go.Scatter(x=d["close_time"], y=d[col], mode="lines", name=nome))
    if "BB_UPPER" in d.columns:
        fig.add_trace(go.Scatter(x=d["close_time"], y=d["BB_UPPER"], mode="lines", name="BB superior", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=d["close_time"], y=d["BB_LOWER"], mode="lines", name="BB inferior", line=dict(dash="dot")))
    if "DONCHIAN_HIGH" in d.columns:
        fig.add_trace(go.Scatter(x=d["close_time"], y=d["DONCHIAN_HIGH"], mode="lines", name="Donchian alta", line=dict(dash="dash")))
        fig.add_trace(go.Scatter(x=d["close_time"], y=d["DONCHIAN_LOW"], mode="lines", name="Donchian baixa", line=dict(dash="dash")))
    fig.update_layout(height=560, margin=dict(l=10,r=10,t=35,b=10), xaxis_rangeslider_visible=False,
                      title="BTC/USDT — candles de 5 minutos")
    return fig


# ============================================================
# INTERFACE
# ============================================================
st.title("₿ BTC Quant Trader — Paper Trading v2")
st.caption(
    "BTC/USDT • 5 minutos • Binance + Deribit • paper trading • sem ordens reais • banco: btc_trader_v2.db"
)

with st.sidebar:
    st.header("Configuração")
    st.caption("💾 As configurações são salvas automaticamente no banco.")
    estrategia = st.selectbox(
        "Estratégia exibida no painel",
        [
            "A — Atual",
            "B — Reversão Bollinger + RSI + VWAP",
            "C — Rompimento Donchian + Volume",
            "D — GEX + OI ATM + Expiração + Dual",
            "E — Brent/WTI + Sinal BTC",
            "F — GEX Walls / First Touch",
            "G — GEX Expansion 1.50%",
        ],
        key="cfg_estrategia",
        on_change=salvar_configuracoes,
        help="A = score técnico; B = reversão; C = rompimento; D = GEX/OI/expiração; E = Brent/WTI; F = GEX Walls com primeiro toque; G = GEX Expansion com alvo BTC de 1,50%.",
    )
    automatizar_todas = st.checkbox(
        "🤖 Automatizar as 7 estratégias",
        key="cfg_automatizar_todas",
        on_change=salvar_configuracoes,
        help="Quando ativado, A, B, C, D, E, F e G são avaliadas a cada candle. A estratégia selecionada acima serve apenas para detalhar o painel.",
    )
    if automatizar_todas:
        st.success("🤖 A + B + C + D + E + F + G estão operando automaticamente")
    else:
        st.warning("Modo individual: somente a estratégia selecionada será executada")

    max_operacoes = st.number_input(
        "Máximo de operações abertas",
        min_value=1, max_value=50, step=1,
        key="cfg_max_operacoes",
        on_change=salvar_configuracoes,
    )
    tempo_maximo = st.number_input(
        "Tempo máximo por operação (min)",
        min_value=5, max_value=1440, step=5,
        key="cfg_tempo_maximo",
        on_change=salvar_configuracoes,
    )
    banca_inicial = st.number_input(
        "Banca inicial da simulação (R$)",
        min_value=1.0, step=100.0,
        key="cfg_banca_inicial",
        on_change=salvar_configuracoes,
    )
    valor_entrada = st.number_input(
        "Valor por operação (R$)",
        min_value=1.0, max_value=100000.0, step=10.0,
        key="cfg_valor_entrada",
        on_change=salvar_configuracoes,
        help="Valor nominal usado no paper trading e na simulação da banca. Padrão: R$ 100,00.",
    )
    st.divider()
    st.write(f"**Stop:** {ATR_STOP_MULTIPLIER:.1f} × ATR")
    st.write(f"**Alvo:** {ATR_TARGET_MULTIPLIER:.1f} × ATR")
    st.write(f"**Refresh:** {AUTO_REFRESH_SECONDS}s")
    st.write("**Banco:** `btc_trader_v2.db`")
    if estrategia.startswith("B"):
        st.info(
            "B busca reversão à média: COMPRA quando preço <= Bollinger inferior + RSI <= 30 + preço >= 0,30% abaixo do VWAP; "
            "VENDA é o espelho. Todas as 3 condições precisam confirmar."
        )
    if estrategia.startswith("C"):
        st.info(
            "C busca rompimento: COMPRA acima da máxima Donchian dos 20 candles anteriores + EMA20 > EMA50 + Volume Z >= 1 + ATR/Preço >= 0,10%; "
            "VENDA é o espelho."
        )
    if estrategia.startswith("D"):
        st.info(
            "D abre COMPRA + VENDA com 1 ATR de stop e 2 ATR de alvo quando: "
            "GEX proxy < 0, OI ATM está no percentil 75%+ das expirações disponíveis e a próxima expiração está a até 24h."
        )
    if estrategia.startswith("G"):
        st.info(
            "G transforma GEX, Gamma Flip, Put/Call Wall e Volume Z em variáveis relativas ao preço. Não reduz os indicadores por 10x: 10x só é usado para interpretar o alvo de 1,50% como ~15% sobre a margem. Entrada exige GEX de expansão, confirmação do Gamma Flip, Wall dentro de 1,50% e Volume Z >= 1."
        )
    if estrategia.startswith("E"):
        st.info(
            "E usa Brent/WTI como filtro de contexto. O ratio é Brent ÷ WTI; a entrada só é liberada quando |Z-score| >= 2,0 e o sinal técnico do BTC confirma COMPRA ou VENDA. O ratio não determina sozinho a direção."
        )

# ============================================================
# MONITORAMENTO
# ============================================================
@st.fragment(run_every=AUTO_REFRESH_SECONDS)
def monitor():
    try:
        df = calcular_indicadores(buscar_klines())
        preco_atual = buscar_preco()
        orderbook = buscar_orderbook()
        row = df.iloc[-2]
        signal_time = row["close_time"]

        score_compra, score_venda, sinal, regime, fatores_compra, fatores_venda = gerar_score(row, orderbook["imbalance"])
        fechadas = monitorar_operacoes(preco_atual, tempo_maximo)
        if fechadas:
            st.toast(" | ".join(f"#{trade_id}: {resultado}" for trade_id, resultado in fechadas))

        # GEX é consultado sempre para que a tela mostre o estado do mercado.
        gex_data = None
        gex_erro = None
        try:
            opcoes = buscar_opcoes_deribit()
            gex_data = calcular_gex_proxy(opcoes, preco_atual)
        except Exception as exc:
            gex_erro = str(exc)

        ratio_data = None
        ratio_erro = None
        try:
            ratio_data = calcular_ratio_brent_wti()
        except Exception as exc:
            ratio_erro = str(exc)

        if estrategia.startswith("E"):
            sinal_e, motivo_e_status = estrategia_e_signal(sinal, ratio_data)
        else:
            sinal_e, motivo_e_status = sinal, ""

        f_walls = calcular_gex_walls(opcoes, preco_atual) if 'opcoes' in locals() and opcoes else None
        f_estado = f_criar_snapshot(f_walls) if f_walls else f_buscar_estado()
        sinal_f, motivo_f, fatores_f_compra, fatores_f_venda = estrategia_f_signal(row, f_estado)
        sinal_g, motivo_g, fatores_g_compra, fatores_g_venda, score_g_compra, score_g_venda, g_detalhes = estrategia_g_signal(
            row, gex_data, f_walls
        )

        if estrategia.startswith("B"):
            strategy_code = "B"
        elif estrategia.startswith("C"):
            strategy_code = "C"
        elif estrategia.startswith("D"):
            strategy_code = "D"
        elif estrategia.startswith("E"):
            strategy_code = "E"
        elif estrategia.startswith("F"):
            strategy_code = "F"
        elif estrategia.startswith("G"):
            strategy_code = "G"
        else:
            strategy_code = "A"
        ciclo_criado = None
        entradas_realizadas = []

        abertas = buscar_operacoes_abertas()
        quantidade_abertas = len(abertas)

        # ========================================================
        # EXECUÇÃO AUTOMÁTICA DAS ESTRATÉGIAS
        # ========================================================
        # Quando automatizar_todas=True, cada estratégia é avaliada no mesmo
        # candle. O limite de operações abertas continua global e a proteção
        # entrada_ja_registrada impede duplicações a cada refresh.
        estrategias_para_executar = ["A", "B", "C", "D", "E", "F", "G"] if automatizar_todas else [strategy_code]

        sinais_estrategias = {
            "A": sinal,
            "B": estrategia_b_signal(row)[0],
            "C": estrategia_c_signal(row)[0],
            "D": "DUAL" if bool(gex_data and gex_data.get("d_ativa")) else "AGUARDAR",
            "E": sinal_e,
            "F": sinal_f,
            "G": sinal_g,
        }

        # A — score técnico original
        if "A" in estrategias_para_executar:
            motivo = montar_motivo_entrada(sinal, score_compra, score_venda, regime, fatores_compra, fatores_venda)
            if quantidade_abertas < int(max_operacoes) and sinal in ("COMPRA", "VENDA"):
                if not entrada_ja_registrada(signal_time, strategy="A", side=sinal):
                    atr = float(row["ATR"])
                    if np.isfinite(atr) and atr > 0:
                        entrada = float(preco_atual)
                        stop, alvo = calcular_plano(sinal, entrada, atr)
                        score = score_compra if sinal == "COMPRA" else score_venda
                        trade_id = registrar_trade(
                            side=sinal, entry_price=entrada, stop_price=stop, target_price=alvo,
                            score=score, regime=regime, signal_time=signal_time, row=row,
                            imbalance=orderbook["imbalance"], score_compra=score_compra, score_venda=score_venda,
                            signal=sinal, entry_reason=motivo, strategy="A", cycle_id=None,
                            gex_data=gex_data, ratio_data=ratio_data,
                            notes="Estratégia A — sinal confirmado. Snapshot salvo para auditoria.",
                        )
                        entradas_realizadas.append(f"#{trade_id} A {sinal}")
                        quantidade_abertas += 1

        # B — reversão Bollinger + RSI + VWAP
        if "B" in estrategias_para_executar:
            sinal_b, motivo_b, fatores_b_compra, fatores_b_venda = estrategia_b_signal(row)
            if quantidade_abertas < int(max_operacoes) and sinal_b in ("COMPRA", "VENDA"):
                if not entrada_ja_registrada(signal_time, strategy="B", side=sinal_b):
                    atr = float(row["ATR"])
                    if np.isfinite(atr) and atr > 0:
                        entrada = float(preco_atual)
                        stop, alvo = calcular_plano(sinal_b, entrada, atr)
                        fatores_b = fatores_b_compra if sinal_b == "COMPRA" else fatores_b_venda
                        motivo = montar_motivo_entrada_bc("B", sinal_b, row, motivo_b, fatores_b)
                        trade_id = registrar_trade(
                            side=sinal_b, entry_price=entrada, stop_price=stop, target_price=alvo,
                            score=100.0, regime=regime, signal_time=signal_time, row=row,
                            imbalance=orderbook["imbalance"], score_compra=100.0 if sinal_b == "COMPRA" else 0.0,
                            score_venda=100.0 if sinal_b == "VENDA" else 0.0, signal=sinal_b,
                            entry_reason=motivo, strategy="B", cycle_id=None, gex_data=gex_data, ratio_data=ratio_data,
                            notes="Estratégia B — reversão à média com Bollinger + RSI + VWAP.",
                        )
                        entradas_realizadas.append(f"#{trade_id} B {sinal_b}")
                        quantidade_abertas += 1

        # C — rompimento Donchian + tendência + volume + ATR
        if "C" in estrategias_para_executar:
            sinal_c, motivo_c, fatores_c_compra, fatores_c_venda = estrategia_c_signal(row)
            if quantidade_abertas < int(max_operacoes) and sinal_c in ("COMPRA", "VENDA"):
                if not entrada_ja_registrada(signal_time, strategy="C", side=sinal_c):
                    atr = float(row["ATR"])
                    if np.isfinite(atr) and atr > 0:
                        entrada = float(preco_atual)
                        stop, alvo = calcular_plano(sinal_c, entrada, atr)
                        fatores_c = fatores_c_compra if sinal_c == "COMPRA" else fatores_c_venda
                        motivo = montar_motivo_entrada_bc("C", sinal_c, row, motivo_c, fatores_c)
                        trade_id = registrar_trade(
                            side=sinal_c, entry_price=entrada, stop_price=stop, target_price=alvo,
                            score=100.0, regime=regime, signal_time=signal_time, row=row,
                            imbalance=orderbook["imbalance"], score_compra=100.0 if sinal_c == "COMPRA" else 0.0,
                            score_venda=100.0 if sinal_c == "VENDA" else 0.0, signal=sinal_c,
                            entry_reason=motivo, strategy="C", cycle_id=None, gex_data=gex_data, ratio_data=ratio_data,
                            notes="Estratégia C — rompimento Donchian + tendência + volume + ATR.",
                        )
                        entradas_realizadas.append(f"#{trade_id} C {sinal_c}")
                        quantidade_abertas += 1

        # D — GEX + OI ATM + expiração; abre as duas pernas
        if "D" in estrategias_para_executar:
            d_ativa = bool(gex_data and gex_data.get("d_ativa"))
            motivo_dual = montar_motivo_dual_d(score_compra, score_venda, regime, gex_data) if gex_data else "D indisponível: sem dados Deribit"
            if d_ativa and quantidade_abertas + 2 <= int(max_operacoes):
                if not entrada_ja_registrada(signal_time, strategy="D"):
                    ciclo_base = f"D-{pd.Timestamp(signal_time).strftime('%Y%m%d%H%M')}-{uuid.uuid4().hex[:6]}"
                    atr = float(row["ATR"])
                    if np.isfinite(atr) and atr > 0:
                        entrada = float(preco_atual)
                        for lado in ("COMPRA", "VENDA"):
                            stop, alvo = calcular_plano(lado, entrada, atr)
                            trade_id = registrar_trade(
                                side=lado, entry_price=entrada, stop_price=stop, target_price=alvo,
                                score=max(score_compra, score_venda), regime=regime, signal_time=signal_time, row=row,
                                imbalance=orderbook["imbalance"], score_compra=score_compra, score_venda=score_venda,
                                signal="DUAL", entry_reason=motivo_dual, strategy="D", cycle_id=ciclo_base,
                                gex_data=gex_data, ratio_data=ratio_data,
                                notes="Estratégia D — ciclo dual baseado em GEX proxy negativo + OI ATM elevado + expiração <= 24h.",
                            )
                            entradas_realizadas.append(f"#{trade_id} D {lado}")
                            quantidade_abertas += 1
                        ciclo_criado = ciclo_base

        # E — Brent/WTI + confirmação do sinal técnico BTC
        if "E" in estrategias_para_executar:
            if ratio_data and ratio_data.get("ready"):
                if sinal_e in ("COMPRA", "VENDA") and quantidade_abertas < int(max_operacoes):
                    motivo_e = montar_motivo_entrada_e(
                        sinal_e, score_compra, score_venda, regime,
                        fatores_compra, fatores_venda, ratio_data
                    )
                    if not entrada_ja_registrada(signal_time, strategy="E", side=sinal_e):
                        atr = float(row["ATR"])
                        if np.isfinite(atr) and atr > 0:
                            entrada = float(preco_atual)
                            stop, alvo = calcular_plano(sinal_e, entrada, atr)
                            score = score_compra if sinal_e == "COMPRA" else score_venda
                            trade_id = registrar_trade(
                                side=sinal_e, entry_price=entrada, stop_price=stop, target_price=alvo,
                                score=score, regime=regime, signal_time=signal_time, row=row,
                                imbalance=orderbook["imbalance"], score_compra=score_compra, score_venda=score_venda,
                                signal=sinal_e, entry_reason=motivo_e, strategy="E", cycle_id=None,
                                gex_data=gex_data, ratio_data=ratio_data,
                                notes="Estratégia E — extremo do ratio Brent/WTI (|Z| >= 2) + confirmação técnica BTC.",
                            )
                            entradas_realizadas.append(f"#{trade_id} E {sinal_e}")
                            quantidade_abertas += 1
            elif ratio_erro and "E" in estrategias_para_executar:
                st.warning(f"Estratégia E indisponível: {ratio_erro}")

        # F — GEX Walls / primeiro toque do dia
        if "F" in estrategias_para_executar:
            if sinal_f == "BLOQUEAR":
                # Um candle que toca as duas paredes não oferece um primeiro toque inequívoco.
                # Mesmo assim, o dia fica consumido para impedir uma entrada posterior.
                f_marcar_primeiro_toque("AMBIGUO", signal_time)
            elif sinal_f in ("COMPRA", "VENDA"):
                # O primeiro toque é consumido independentemente do limite de
                # operações. Assim, se o robô estiver lotado neste candle, ele
                # não poderá entrar numa segunda visita à mesma wall.
                wall_tocada = "PUT WALL" if sinal_f == "COMPRA" else "CALL WALL"
                toque_consumido = f_marcar_primeiro_toque(wall_tocada, signal_time)

                if toque_consumido and quantidade_abertas < int(max_operacoes):
                    if not entrada_ja_registrada(signal_time, strategy="F", side=sinal_f):
                        atr = float(row["ATR"])
                        if np.isfinite(atr) and atr > 0:
                            entrada = float(preco_atual)
                            stop, alvo = calcular_plano(sinal_f, entrada, atr)
                            motivo_f_entrada = motivo_f + " | " + " | ".join(fatores_f_compra if sinal_f == "COMPRA" else fatores_f_venda)
                            trade_id = registrar_trade(
                                side=sinal_f, entry_price=entrada, stop_price=stop, target_price=alvo,
                                score=100.0, regime=regime, signal_time=signal_time, row=row,
                                imbalance=orderbook["imbalance"], score_compra=100.0 if sinal_f == "COMPRA" else 0.0,
                                score_venda=100.0 if sinal_f == "VENDA" else 0.0, signal=sinal_f,
                                entry_reason=motivo_f_entrada, strategy="F", cycle_id=None,
                                gex_data=gex_data, ratio_data=ratio_data,
                                notes="Estratégia F — Sniper A Seco: entrada exclusivamente no primeiro toque diário de uma GEX Wall. Stop/alvo operacional seguem o plano ATR 1:2 do robô.",
                            )
                            entradas_realizadas.append(f"#{trade_id} F {sinal_f}")
                            quantidade_abertas += 1
                elif toque_consumido and quantidade_abertas >= int(max_operacoes):
                    st.info("F: primeiro toque consumido, mas o limite de operações abertas foi atingido; nenhuma entrada F foi aberta.")
            elif sinal_f == "AGUARDAR" and f_estado and int(f_estado.get("first_touch_done") or 0) == 0:
                # Se a vela tocou uma parede mas não confirmou, a própria função F
                # informa isso; nesse caso o primeiro toque também deve consumir o dia.
                low = float(row["Low"]); high = float(row["High"])
                put_wall = float(f_estado["put_wall"]); call_wall = float(f_estado["call_wall"])
                if low <= put_wall <= high and not (low <= call_wall <= high):
                    f_marcar_primeiro_toque("PUT WALL", signal_time)
                elif low <= call_wall <= high and not (low <= put_wall <= high):
                    f_marcar_primeiro_toque("CALL WALL", signal_time)

        # G — GEX Expansion: alvo fixo de +/-1,50% no BTC.
        if "G" in estrategias_para_executar:
            if quantidade_abertas < int(max_operacoes) and sinal_g in ("COMPRA", "VENDA"):
                if not entrada_ja_registrada(signal_time, strategy="G", side=sinal_g):
                    entrada = float(preco_atual)
                    # Alvo de 1,50% e risco de 0,75%: R/R 1:2.
                    distancia_alvo = entrada * G_TARGET_PCT
                    distancia_stop = distancia_alvo / 2.0
                    if sinal_g == "COMPRA":
                        stop = entrada - distancia_stop
                        alvo = entrada + distancia_alvo
                    else:
                        stop = entrada + distancia_stop
                        alvo = entrada - distancia_alvo
                    motivo_g_entrada = motivo_g + " | " + " | ".join(fatores_g_compra if sinal_g == "COMPRA" else fatores_g_venda)
                    score_g = score_g_compra if sinal_g == "COMPRA" else score_g_venda
                    trade_id = registrar_trade(
                        side=sinal_g, entry_price=entrada, stop_price=stop, target_price=alvo,
                        score=score_g, regime=regime, signal_time=signal_time, row=row,
                        imbalance=orderbook["imbalance"], score_compra=score_g_compra, score_venda=score_g_venda,
                        signal=sinal_g, entry_reason=motivo_g_entrada, strategy="G", cycle_id=None,
                        gex_data=gex_data, ratio_data=ratio_data,
                        notes="Estratégia G — GEX Expansion 1,50%. Indicadores normalizados em relação ao preço; 10x é referência de execução/ROE, não escala dos indicadores. Alvo BTC +/-1,50%; stop 0,75%; R/R 1:2.",
                    )
                    entradas_realizadas.append(f"#{trade_id} G {sinal_g}")
                    quantidade_abertas += 1

        if entradas_realizadas:
            st.success("Nova entrada: " + " | ".join(entradas_realizadas))

        # O painel pode destacar uma estratégia, mas a automação não depende dela.
        if strategy_code == "B":
            sinal_operacional = sinais_estrategias["B"]
        elif strategy_code == "C":
            sinal_operacional = sinais_estrategias["C"]
        elif strategy_code == "D":
            sinal_operacional = sinais_estrategias["D"]
        elif strategy_code == "E":
            sinal_operacional = sinais_estrategias["E"]
        elif strategy_code == "F":
            sinal_operacional = sinais_estrategias["F"]
        elif strategy_code == "G":
            sinal_operacional = sinais_estrategias["G"]
        else:
            sinal_operacional = sinais_estrategias["A"]

        abertas = buscar_operacoes_abertas()
        quantidade_abertas = len(abertas)

        # ========================================================
        # PAINEL
        # ========================================================
        if automatizar_todas:
            df_sinais = pd.DataFrame([
                {"Estratégia": "A — Atual", "Sinal": sinais_estrategias["A"]},
                {"Estratégia": "B — Reversão", "Sinal": sinais_estrategias["B"]},
                {"Estratégia": "C — Rompimento", "Sinal": sinais_estrategias["C"]},
                {"Estratégia": "D — GEX/Dual", "Sinal": sinais_estrategias["D"]},
                {"Estratégia": "E — Brent/WTI", "Sinal": sinais_estrategias["E"]},
                {"Estratégia": "F — GEX Walls", "Sinal": sinais_estrategias["F"]},
                {"Estratégia": "G — GEX Expansion", "Sinal": sinais_estrategias["G"]},
            ])
            st.subheader("🤖 Automação — 7 estratégias")
            st.dataframe(df_sinais, use_container_width=True, hide_index=True)
            st.caption(f"Execução automática ativa. Limite global: {quantidade_abertas}/{int(max_operacoes)} operações abertas.")

        # ========================================================
        # PAINEL
        # ========================================================
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("BTC", f"${preco_atual:,.2f}")
        c2.metric("Sinal operacional", sinal_operacional)
        c3.metric("Score COMPRA", f"{score_compra:.0f}")
        c4.metric("Score VENDA", f"{score_venda:.0f}")
        c5.metric("Regime", regime)
        c6.metric("F", "Primeiro toque" if f_estado and not int(f_estado.get("first_touch_done") or 0) else "Consumido")

        # ========================================================
        # GRÁFICO DE CANDLES
        # ========================================================
        st.subheader("📈 BTC/USDT — Gráfico de 5 minutos")
        try:
            fig_candles = criar_grafico_candles(df, janela=100)

            # Marca a última vela analisada pelo robô.
            fig_candles.add_trace(go.Scatter(
                x=[signal_time],
                y=[float(row["Close"])],
                mode="markers",
                marker=dict(size=10, symbol="circle"),
                name="Candle analisado",
                hovertemplate="Candle analisado<br>%{x}<br>Preço: $%{y:,.2f}<extra></extra>",
            ))

            # Mostra no gráfico as operações atualmente abertas.
            if not abertas.empty:
                for _, trade in abertas.iterrows():
                    entry = float(trade["entry_price"])
                    stop = float(trade["stop_price"])
                    alvo = float(trade["target_price"])
                    lado = str(trade["side"])
                    estrategia_trade = str(trade.get("strategy", "A") or "A")
                    legenda = f"{estrategia_trade} {lado}"
                    fig_candles.add_hline(y=entry, line_dash="solid", annotation_text=f"Entrada {legenda}")
                    fig_candles.add_hline(y=stop, line_dash="dot", annotation_text=f"Stop {legenda}")
                    fig_candles.add_hline(y=alvo, line_dash="dash", annotation_text=f"Alvo {legenda}")

            fig_candles.update_layout(
                height=620,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
            )
            st.plotly_chart(fig_candles, use_container_width=True, config={"displaylogo": False, "scrollZoom": True})
            st.caption("Últimas 100 candles. As linhas de Entrada/Stop/Alvo aparecem quando existem operações abertas.")
        except Exception as exc:
            st.warning(f"Não foi possível montar o gráfico de candles: {exc}")

        st.divider()
        left, right = st.columns([1.15, 1])
        with left:
            st.subheader("Sinal atual")
            if sinal_operacional == "COMPRA":
                st.success("🟢 COMPRA")
            elif sinal_operacional == "VENDA":
                st.error("🔴 VENDA")
            else:
                st.info("🟡 AGUARDAR")
            st.write(f"**Estratégia selecionada:** {estrategia}")
            st.write(f"**Sinal técnico base A:** {sinal}")
            st.write(f"**Candle analisado:** {signal_time}")
            st.write(f"**Preço do candle:** ${float(row['Close']):,.2f}")
            st.write(f"**Preço atual:** ${preco_atual:,.2f}")
            st.write(f"**Order book imbalance:** {orderbook['imbalance']:+.4f}")
            if strategy_code == "B":
                sinal_b_view, motivo_b_view, _, _ = estrategia_b_signal(row)
                st.write(f"**Sinal B:** {sinal_b_view}")
                st.caption(motivo_b_view)
                st.write(f"**BB inferior:** ${float(row['BB_LOWER']):,.2f} | **BB superior:** ${float(row['BB_UPPER']):,.2f}")
            elif strategy_code == "C":
                sinal_c_view, motivo_c_view, _, _ = estrategia_c_signal(row)
                st.write(f"**Sinal C:** {sinal_c_view}")
                st.caption(motivo_c_view)
                st.write(f"**Donchian H:** ${float(row['DONCHIAN_HIGH']):,.2f} | **Donchian L:** ${float(row['DONCHIAN_LOW']):,.2f}")
            elif strategy_code == "F":
                st.write(f"**Sinal F:** {sinal_f}")
                st.caption(motivo_f)

                if f_estado:
                    put_wall_f = float(f_estado["put_wall"])
                    call_wall_f = float(f_estado["call_wall"])
                    operacao_f = buscar_operacao_f_relevante()

                    # A operação registrada no SQLite é a fonte de verdade para
                    # entrada/stop/alvo depois que o primeiro toque foi consumido.
                    entrada_f = None
                    stop_f = None
                    alvo_f = None
                    status_operacao_f = "SEM OPERAÇÃO F REGISTRADA"

                    if operacao_f is not None:
                        try:
                            entrada_f = float(operacao_f["entry_price"])
                            stop_f = float(operacao_f["stop_price"])
                            alvo_f = float(operacao_f["target_price"])
                            if operacao_f.get("exit_time"):
                                status_operacao_f = f"ÚLTIMA F ENCERRADA • #{int(operacao_f['id'])}"
                            else:
                                status_operacao_f = f"F ABERTA • #{int(operacao_f['id'])}"
                        except (TypeError, ValueError, KeyError):
                            entrada_f = stop_f = alvo_f = None

                    # Se ainda não existe uma operação registrada, mas há um novo
                    # sinal F válido nesta execução, mostramos o plano que será
                    # usado pelo executor.
                    if entrada_f is None and sinal_f in ("COMPRA", "VENDA"):
                        try:
                            atr_f = float(row["ATR"])
                            if np.isfinite(atr_f) and atr_f > 0:
                                entrada_f = float(preco_atual)
                                stop_f, alvo_f = calcular_plano(sinal_f, entrada_f, atr_f)
                                status_operacao_f = "PLANO F AGUARDANDO REGISTRO"
                        except (TypeError, ValueError, KeyError):
                            pass

                    st.markdown("### 🎯 F — GEX Walls / First Touch")

                    f1, f2, f3 = st.columns(3)
                    f1.metric("₿ BTC atual", f"US$ {preco_atual:,.2f}")
                    f2.metric(
                        "🎯 Entrada F",
                        f"US$ {entrada_f:,.2f}" if entrada_f is not None else "AGUARDANDO",
                    )
                    f3.metric("📍 Put Wall", f"US$ {put_wall_f:,.2f}")

                    f4, f5, f6 = st.columns(3)
                    f4.metric("📍 Call Wall", f"US$ {call_wall_f:,.2f}")
                    f5.metric(
                        "🛑 Stop F",
                        f"US$ {stop_f:,.2f}" if stop_f is not None else "AGUARDANDO",
                    )
                    f6.metric(
                        "🎯 Alvo F",
                        f"US$ {alvo_f:,.2f}" if alvo_f is not None else "AGUARDANDO",
                    )

                    st.write(
                        f"**Gamma Flip:** US$ {float(f_estado['gamma_flip']):,.2f} | "
                        f"**Pin:** US$ {float(f_estado['pin_candidate']):,.2f}"
                    )
                    st.write(
                        f"**Centróide GEX:** US$ {float(f_estado['gamma_centroid']):,.2f} | "
                        f"**Primeiro toque:** {f_estado.get('first_touch_wall') or 'aguardando'}"
                    )

                    distancia_put = ((preco_atual - put_wall_f) / preco_atual) * 100.0
                    distancia_call = ((call_wall_f - preco_atual) / preco_atual) * 100.0
                    st.caption(
                        f"Distância BTC → Put Wall: {distancia_put:.2f}% | "
                        f"BTC → Call Wall: {distancia_call:.2f}% | {status_operacao_f}"
                    )

                    if entrada_f is not None and stop_f is not None and alvo_f is not None:
                        risco_pct_f = abs((stop_f / entrada_f) - 1.0) * 100.0
                        alvo_pct_f = abs((alvo_f / entrada_f) - 1.0) * 100.0
                        risco_rs_f = float(valor_entrada) * risco_pct_f / 100.0
                        alvo_rs_f = float(valor_entrada) * alvo_pct_f / 100.0
                        st.write(
                            f"**Plano F:** Entrada US$ {entrada_f:,.2f} | "
                            f"Stop US$ {stop_f:,.2f} | Alvo US$ {alvo_f:,.2f} | "
                            f"Risco {risco_pct_f:.3f}% | Alvo {alvo_pct_f:.3f}% | R/R 1:2"
                        )
                        st.caption(
                            f"Valor nominal do paper trade: R$ {valor_entrada:,.2f} | "
                            f"Risco teórico: R$ {risco_rs_f:,.2f} | "
                            f"Alvo teórico: R$ {alvo_rs_f:,.2f}"
                        )
                    elif sinal_f not in ("COMPRA", "VENDA"):
                        st.info("⏳ F aguardando o primeiro toque confirmado. Ainda não existe uma nova entrada operacional.")
            elif strategy_code == "G":
                st.write(f"**Sinal G:** {sinal_g}")
                st.caption(motivo_g)
                if g_detalhes:
                    st.write(
                        f"**GEX norm.:** {g_detalhes.get('gex_norm', 0):+.2f} | "
                        f"**Gamma Flip:** {g_detalhes.get('gamma_flip_dist_pct', 0):+.2f}% | "
                        f"**Put Wall:** {g_detalhes.get('put_wall_dist_pct', 0):+.2f}% | "
                        f"**Call Wall:** {g_detalhes.get('call_wall_dist_pct', 0):+.2f}% | "
                        f"**Volume Z:** {g_detalhes.get('volume_z', 0):+.2f}"
                    )
                    st.write(
                        f"**Score G:** COMPRA {score_g_compra:.0f} | VENDA {score_g_venda:.0f} | "
                        f"**Alvo BTC:** ±{G_TARGET_PCT*100:.2f}% | **ROE teórico em 10x:** ±{G_TARGET_PCT*100*10:.1f}%"
                    )
            sinal_plano = sinal_operacional if sinal_operacional in ("COMPRA", "VENDA") else (sinal if sinal in ("COMPRA", "VENDA") else None)
            if sinal_plano and strategy_code != "F":
                if strategy_code == "G":
                    distancia_alvo_view = preco_atual * G_TARGET_PCT
                    distancia_stop_view = distancia_alvo_view / 2.0
                    if sinal_plano == "COMPRA":
                        stop_view = preco_atual - distancia_stop_view
                        alvo_view = preco_atual + distancia_alvo_view
                    else:
                        stop_view = preco_atual + distancia_stop_view
                        alvo_view = preco_atual - distancia_alvo_view
                else:
                    stop_view, alvo_view = calcular_plano(sinal_plano, preco_atual, float(row["ATR"]))
                risco_pct = abs((stop_view / preco_atual) - 1.0) * 100.0
                alvo_pct = abs((alvo_view / preco_atual) - 1.0) * 100.0
                risco_rs = float(valor_entrada) * risco_pct / 100.0
                alvo_rs = float(valor_entrada) * alvo_pct / 100.0
                st.write(f"**Plano {sinal_plano}:** Entrada ${preco_atual:,.2f} | Stop ${stop_view:,.2f} | Alvo ${alvo_view:,.2f}")
                st.write(f"**Risco:** -{risco_pct:.3f}% ≈ -R$ {risco_rs:,.2f} | **Alvo:** +{alvo_pct:.3f}% ≈ +R$ {alvo_rs:,.2f} | **R/R:** 1:2")
                if strategy_code == "G":
                    st.caption(f"G: alvo fixo de ±{G_TARGET_PCT*100:.2f}% no BTC • referência 10x = ~{G_TARGET_PCT*100*10:.1f}% sobre a margem antes de custos • valor nominal: R$ {valor_entrada:,.2f}")
                else:
                    st.caption(f"Valor nominal do paper trade: R$ {valor_entrada:,.2f}")

        with right:
            st.subheader("GEX / Opções Deribit")
            if gex_data:
                g1, g2, g3 = st.columns(3)
                g1.metric("GEX proxy", f"{gex_data['gex_proxy']:+,.0f}")
                g2.metric("OI ATM", f"{gex_data['atm_oi']:,.1f}")
                g3.metric("DTE", f"{gex_data['dte_hours']:.1f}h")
                st.write(f"**Próxima expiração:** {gex_data['nearest_expiry']}")
                st.write(f"**OI ATM ratio:** {gex_data['atm_oi_ratio']:.2%}")
                if f_estado:
                    st.markdown("**F — GEX Walls / First Touch**")
                    fw1, fw2, fw3 = st.columns(3)
                    fw1.metric("Put Wall", f"${float(f_estado['put_wall']):,.0f}")
                    fw2.metric("Call Wall", f"${float(f_estado['call_wall']):,.0f}")
                    fw3.metric("Gamma Flip", f"${float(f_estado['gamma_flip']):,.0f}")
                    st.caption(f"Pin ${float(f_estado['pin_candidate']):,.0f} • Centr. ${float(f_estado['gamma_centroid']):,.0f} • Primeiro toque: {f_estado.get('first_touch_wall') or 'aguardando'}")
                st.write(f"**Percentil usado para OI elevado:** {D_ATM_ELEVATED_PERCENTILE:.0f}%")
                st.write(f"**Opções utilizadas:** {gex_data.get('options_used', '-')} | **Método:** {gex_data.get('gex_method', '-')}")
                st.write("**Condições:** " + " | ".join(gex_data["condicoes"]))
                if gex_data["d_ativa"]:
                    st.success("🟢 Estratégia D ATIVA")
                else:
                    st.info("⚪ Estratégia D INATIVA")
                st.markdown("**G — GEX Expansion 1,50%**")
                gg1, gg2, gg3 = st.columns(3)
                gg1.metric("Score G COMPRA", f"{score_g_compra:.0f}")
                gg2.metric("Score G VENDA", f"{score_g_venda:.0f}")
                gg3.metric("Alvo BTC", f"±{G_TARGET_PCT*100:.2f}%")
                st.caption(f"ROE teórico em 10x: ±{G_TARGET_PCT*100*10:.1f}% antes de custos • sinal: {sinal_g}")
            else:
                st.warning(f"GEX indisponível: {gex_erro}")

        if estrategia.startswith("E"):
            st.subheader("Brent / WTI — Estratégia E")
            if ratio_data:
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Brent", f"${ratio_data['brent_price']:.2f}")
                r2.metric("WTI", f"${ratio_data['wti_price']:.2f}")
                r3.metric("Ratio", f"{ratio_data['ratio']:.4f}")
                r4.metric("Z-score", "-" if ratio_data.get("z") is None else f"{ratio_data['z']:+.2f}")
                st.write(f"**Condição:** {ratio_data['condition']}")
                st.write(f"**Status E:** {motivo_e_status}")
            else:
                st.warning(f"Brent/WTI indisponível: {ratio_erro}")

        st.divider()
        with st.expander("🔎 Fatores técnicos"):
            fatores = fatores_compra if sinal == "COMPRA" else fatores_venda
            if fatores:
                for fator in fatores:
                    st.write("•", fator)
            else:
                st.write("Nenhum fator direcional relevante.")

        # ========================================================
        # OPERAÇÕES ABERTAS
        # ========================================================
        st.subheader(f"Operações abertas ({quantidade_abertas}/{int(max_operacoes)})")
        if abertas.empty:
            st.info("Nenhuma operação aberta.")
        else:
            exib = abertas.copy()
            exib["Preço entrada"] = exib["entry_price"].map(lambda x: f"${x:,.2f}")
            exib["Stop"] = exib["stop_price"].map(lambda x: f"${x:,.2f}")
            exib["Alvo"] = exib["target_price"].map(lambda x: f"${x:,.2f}")
            pnl_atual = []
            for _, t in exib.iterrows():
                entry = float(t["entry_price"])
                pnl = ((preco_atual / entry) - 1) * 100 if t["side"] == "COMPRA" else ((entry / preco_atual) - 1) * 100
                pnl_atual.append(pnl)
            exib["P&L atual %"] = [f"{x:+.2f}%" for x in pnl_atual]
            exib["Estratégia"] = exib["strategy"].fillna("A")
            exib["Ciclo"] = exib["cycle_id"].fillna("-")
            exib = exib.rename(columns={"id": "#", "side": "Lado", "entry_time": "Entrada", "score": "Score", "regime": "Regime"})
            cols = ["#", "Estratégia", "Ciclo", "Lado", "Entrada", "Preço entrada", "Stop", "Alvo", "P&L atual %", "Score", "Regime"]
            st.dataframe(exib[cols], use_container_width=True, hide_index=True)

        # ========================================================
        # AUDITORIA
        # ========================================================
        with st.expander("🔎 Snapshot técnico / GEX para auditoria"):
            audit = {
                "EMA20": row["EMA20"], "EMA50": row["EMA50"], "EMA200": row["EMA200"],
                "RSI": row["RSI"], "ATR": row["ATR"], "ret_1": row["ret_1"],
                "ret_3": row["ret_3"], "ret_12": row["ret_12"], "ret_48": row["ret_48"],
                "volatility": row["volatility"], "vol_z": row["vol_z"], "volume_ratio": row["volume_ratio"],
                "volume_z": row["volume_z"], "VWAP": row["VWAP"], "cvd_delta": row["cvd_delta"],
                "orderbook_imbalance": orderbook["imbalance"], "score_compra": score_compra,
                "score_venda": score_venda, "regime": regime, "signal": sinal,
                "signal_time": str(signal_time),
            }
            if gex_data:
                audit.update({
                    "gex_proxy": gex_data["gex_proxy"],
                    "gex_calls": gex_data["gex_calls"],
                    "gex_puts": gex_data["gex_puts"],
                    "atm_oi": gex_data["atm_oi"],
                    "atm_oi_total": gex_data["atm_oi_total"],
                    "atm_oi_ratio": gex_data["atm_oi_ratio"],
                    "nearest_expiry": gex_data["nearest_expiry"],
                    "dte_hours": gex_data["dte_hours"],
                    "D_ativa": gex_data["d_ativa"],
                    "gex_options_used": gex_data.get("options_used"),
                    "gex_method": gex_data.get("gex_method"),
                })
            if ratio_data:
                audit.update({
                    "brent_price": ratio_data["brent_price"],
                    "wti_price": ratio_data["wti_price"],
                    "brent_wti_ratio": ratio_data["ratio"],
                    "brent_wti_ratio_mean": ratio_data["ratio_mean"],
                    "brent_wti_ratio_std": ratio_data["ratio_std"],
                    "brent_wti_z": ratio_data["z"],
                    "brent_wti_change": ratio_data["change"],
                    "brent_wti_condition": ratio_data["condition"],
                    "strategy_e_signal": sinal_e,
                })
            if f_estado:
                audit.update({
                    "f_operational_date": f_estado.get("operational_date"),
                    "f_call_wall": f_estado.get("call_wall"),
                    "f_put_wall": f_estado.get("put_wall"),
                    "f_gamma_flip": f_estado.get("gamma_flip"),
                    "f_pin_candidate": f_estado.get("pin_candidate"),
                    "f_gamma_centroid": f_estado.get("gamma_centroid"),
                    "f_first_touch_done": f_estado.get("first_touch_done"),
                    "f_first_touch_wall": f_estado.get("first_touch_wall"),
                    "strategy_f_signal": sinal_f,
                })
            audit.update({
                "g_strategy_signal": sinal_g,
                "g_score_compra": score_g_compra,
                "g_score_venda": score_g_venda,
                "g_gex_norm": g_detalhes.get("gex_norm") if g_detalhes else None,
                "g_gamma_flip_dist_pct": g_detalhes.get("gamma_flip_dist_pct") if g_detalhes else None,
                "g_put_wall_dist_pct": g_detalhes.get("put_wall_dist_pct") if g_detalhes else None,
                "g_call_wall_dist_pct": g_detalhes.get("call_wall_dist_pct") if g_detalhes else None,
                "g_target_pct": G_TARGET_PCT * 100.0,
                "g_roe_target_10x_pct": G_TARGET_PCT * 100.0 * 10.0,
            })
            st.dataframe(pd.DataFrame(list(audit.items()), columns=["Indicador", "Valor"]), use_container_width=True, hide_index=True)

        # ========================================================
        # HISTÓRICO
        # ========================================================
        st.subheader("Histórico")
        historico = buscar_historico(200)
        if historico.empty:
            st.info("Ainda não existem operações no banco.")
        else:
            hist = historico.copy()
            hist["Resultado"] = hist["result"].fillna("ABERTA")
            hist["Saída por"] = hist["exit_reason"].fillna("-")
            hist["Estratégia"] = hist["strategy"].fillna("A")
            hist["Ciclo"] = hist["cycle_id"].fillna("-")
            hist["P&L %"] = hist["pnl_pct"].map(lambda x: "-" if pd.isna(x) else f"{x:+.2f}%")
            for col, label in [("entry_price", "Preço entrada"), ("stop_price", "Stop"), ("target_price", "Alvo"), ("exit_price", "Preço saída")]:
                hist[label] = hist[col].map(lambda x: "-" if pd.isna(x) else f"${float(x):,.2f}")
            hist = hist.rename(columns={"id": "#", "side": "Lado", "entry_time": "Entrada", "exit_time": "Saída", "score": "Score", "regime": "Regime"})
            cols = ["#", "Estratégia", "Ciclo", "Lado", "Entrada", "Preço entrada", "Stop", "Alvo", "Saída", "Preço saída", "P&L %", "Resultado", "Saída por", "Score", "Regime"]
            st.dataframe(hist[cols], use_container_width=True, hide_index=True)

            st.markdown("### 💰 Banca por método — início de R$ 1.000")
            st.caption("Cada método é simulado separadamente usando somente as operações paper fechadas daquela estratégia. Não é backtest histórico.")
            metodos = [
                ("A", "A — Atual"), ("B", "B — Reversão"), ("C", "C — Rompimento"),
                ("D", "D — GEX/Dual"), ("E", "E — Brent/WTI"), ("F", "F — GEX Walls"), ("G", "G — GEX Expansion"),
            ]
            cards = []
            historico_banca = buscar_historico(100000)
            hist_fechado = historico_banca[historico_banca["pnl_pct"].notna()].copy() if not historico_banca.empty else pd.DataFrame()
            for code, nome_metodo in metodos:
                if hist_fechado.empty:
                    sub = hist_fechado
                else:
                    sub = hist_fechado[hist_fechado["strategy"].fillna("A") == code].copy()
                _, final_metodo = simular_banca(sub, 1000.0, valor_entrada)
                operacoes_metodo = len(sub)
                lucro_metodo = final_metodo - 1000.0
                cards.append((nome_metodo, final_metodo, lucro_metodo, operacoes_metodo))
            cols_cards = st.columns(3)
            for idx, (nome_metodo, final_metodo, lucro_metodo, operacoes_metodo) in enumerate(cards):
                with cols_cards[idx % 3]:
                    st.metric(nome_metodo, f"R$ {final_metodo:,.2f}", f"{lucro_metodo:+,.2f} R$")
                    st.caption(f"Início R$ 1.000,00 • {operacoes_metodo} operações fechadas")

            st.markdown("### 💰 Simulação da banca geral")
            sim, banca_final = simular_banca(historico, banca_inicial, valor_entrada)
            st.caption(
                f"Banca inicial: R$ {banca_inicial:,.2f} • Entrada fixa: R$ {valor_entrada:,.2f} por operação • "
                f"Na D, COMPRA e VENDA são duas operações e cada uma usa R$ {valor_entrada:,.2f}."
            )
            if sim.empty:
                st.info("A simulação aparecerá após a primeira operação fechada.")
            else:
                ganhos = int((sim["Resultado"] == "GANHO").sum())
                perdas = int((sim["Resultado"] == "PERDA").sum())
                lucro = banca_final - float(banca_inicial)
                retorno = lucro / float(banca_inicial) * 100
                b1, b2, b3, b4 = st.columns(4)
                b1.metric("Banca atual", f"R$ {banca_final:,.2f}")
                b2.metric("Lucro / prejuízo", f"R$ {lucro:+,.2f}")
                b3.metric("Retorno", f"{retorno:+.2f}%")
                b4.metric("Ganhas / Perdidas", f"{ganhos} / {perdas}")
                sim_exib = sim.copy()
                sim_exib["P&L mercado %"] = sim_exib["P&L mercado %"].map(lambda x: f"{x:+.2f}%")
                for col in ["Banca antes", "Valor entrada", "Resultado R$", "Banca depois"]:
                    sim_exib[col] = sim_exib[col].map(lambda x: f"R$ {x:,.2f}")
                st.dataframe(
                    sim_exib[["#", "Estratégia", "Ciclo", "Lado", "Entrada", "Saída", "P&L mercado %", "Banca antes", "Entrada %", "Valor entrada", "Resultado R$", "Banca depois", "Resultado", "Saída por"]],
                    use_container_width=True,
                    hide_index=True,
                )

            ultimo = historico.iloc[0]
            with st.expander("📋 Motivo da última operação registrada"):
                st.write(f"**Operação #{int(ultimo['id'])} — {ultimo['side']} — Estratégia {ultimo.get('strategy', 'A') or 'A'}**")
                st.write(f"**Resultado:** {ultimo['result'] if pd.notna(ultimo['result']) else 'ABERTA'}")
                st.write(f"**Saída por:** {ultimo['exit_reason'] if pd.notna(ultimo['exit_reason']) else '-'}")
                st.code(str(ultimo.get("entry_reason", "-")), language="text")

        # ========================================================
        # ESTATÍSTICAS
        # ========================================================
        conn = get_conn()
        stats = pd.read_sql_query(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN result = 'GANHO' THEN 1 ELSE 0 END) AS ganhos,
                SUM(CASE WHEN result = 'PERDA' THEN 1 ELSE 0 END) AS perdas,
                COALESCE(SUM(pnl_pct), 0) AS pnl_total
            FROM trades
            WHERE result IS NOT NULL
            """,
            conn,
        )
        conn.close()
        total = int(stats.iloc[0]["total"] or 0)
        ganhos = int(stats.iloc[0]["ganhos"] or 0)
        perdas = int(stats.iloc[0]["perdas"] or 0)
        pnl_total = float(stats.iloc[0]["pnl_total"] or 0)
        wr = ganhos / total * 100 if total else 0
        st.divider()
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Operações fechadas", total)
        s2.metric("Ganhas", ganhos)
        s3.metric("Perdidas", perdas)
        s4.metric("WR", f"{wr:.2f}%")
        st.caption(f"P&L percentual acumulado das operações fechadas: {pnl_total:+.2f}%")

    except requests.RequestException as exc:
        st.error(f"Erro ao consultar API: {exc}")
    except Exception as exc:
        st.error(f"Erro no monitoramento: {exc}")


monitor()
