import sqlite3
import math
from datetime import datetime, timedelta, timezone
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


@st.cache_data(ttl=300, show_spinner=False)
def buscar_klines_historico(limite_total=50000):
    """Busca histórico de 5m em páginas. Usado somente no backtest."""
    cols = ["open_time","Open","High","Low","Close","Volume","close_time","quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"]
    todos = []
    end_time = None
    restante = int(limite_total)
    while restante > 0:
        n = min(1000, restante)
        params = {"symbol": SYMBOL, "interval": INTERVAL, "limit": n}
        if end_time is not None:
            params["endTime"] = end_time
        r = requests.get(f"{BINANCE_API}/api/v3/klines", params=params, timeout=15)
        r.raise_for_status()
        lote = r.json()
        if not lote:
            break
        todos = lote + todos
        restante -= len(lote)
        first_open = int(lote[0][0])
        end_time = first_open - 1
        if len(lote) < n:
            break
    df = pd.DataFrame(todos, columns=cols).drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
    for col in ["Open","High","Low","Close","Volume","quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    return df


def backtest_bc(df, strategy_code, max_trades=1000):
    """Backtest objetivo de B/C com stop 1 ATR e alvo 2 ATR. Sem look-ahead no Donchian."""
    d = calcular_indicadores(df.copy()).reset_index(drop=True)
    resultados = []
    i = 0
    while i < len(d) - 2 and len(resultados) < max_trades:
        row = d.iloc[i]
        if strategy_code == "B":
            sinal_bt, _ = estrategia_b_signal(row)
        else:
            sinal_bt, _ = estrategia_c_signal(row)
        if sinal_bt not in ("COMPRA", "VENDA") or not np.isfinite(row.get("ATR", np.nan)) or row["ATR"] <= 0:
            i += 1; continue
        entrada = float(row["Close"]); atr = float(row["ATR"])
        stop, alvo = calcular_plano(sinal_bt, entrada, atr)
        saida = None; motivo = None; j = i + 1
        while j < len(d):
            h, l = float(d.iloc[j]["High"]), float(d.iloc[j]["Low"])
            if sinal_bt == "COMPRA":
                if l <= stop and h >= alvo:
                    saida, motivo = stop, "STOP (conservador em candle com ambos)"; break
                if l <= stop:
                    saida, motivo = stop, "STOP"; break
                if h >= alvo:
                    saida, motivo = alvo, "ALVO"; break
            else:
                if h >= stop and l <= alvo:
                    saida, motivo = stop, "STOP (conservador em candle com ambos)"; break
                if h >= stop:
                    saida, motivo = stop, "STOP"; break
                if l <= alvo:
                    saida, motivo = alvo, "ALVO"; break
            j += 1
        if saida is not None:
            pnl = ((saida/entrada)-1)*100 if sinal_bt == "COMPRA" else ((entrada/saida)-1)*100
            resultados.append({"Estratégia":strategy_code,"Entrada":d.iloc[i]["close_time"],"Saída":d.iloc[j]["close_time"],"Lado":sinal_bt,"Entrada preço":entrada,"Stop":stop,"Alvo":alvo,"P&L %":pnl,"Saída por":motivo})
            i = j + 1
        else:
            i += 1
    return pd.DataFrame(resultados)


def resumo_backtest(bt, valor_entrada=100.0):
    if bt.empty:
        return {"Operações":0,"WR %":0.0,"Lucro R$":0.0,"ROI %":0.0,"Drawdown máx. %":0.0}
    wins = bt["P&L %"] > 0
    lucro = float((bt["P&L %"] / 100.0 * valor_entrada).sum())
    curva = (bt["P&L %"] / 100.0 * valor_entrada).cumsum()
    pico = curva.cummax()
    dd = (curva - pico)
    return {"Operações":len(bt),"WR %":float(wins.mean()*100),"Lucro R$":lucro,"ROI %":float(lucro/(len(bt)*valor_entrada)*100),"Drawdown máx. R$":float(dd.min())}

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
        ],
        key="cfg_estrategia",
        on_change=salvar_configuracoes,
        help="A = score técnico original; B = reversão à média; C = rompimento/momentum; D = GEX/OI/expiração; E = Brent/WTI como filtro.",
    )
    automatizar_todas = st.checkbox(
        "🤖 Automatizar as 5 estratégias",
        key="cfg_automatizar_todas",
        on_change=salvar_configuracoes,
        help="Quando ativado, A, B, C, D e E são avaliadas a cada candle. A estratégia selecionada acima serve apenas para detalhar o painel.",
    )
    if automatizar_todas:
        st.success("🤖 A + B + C + D + E estão operando automaticamente")
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

        if estrategia.startswith("B"):
            strategy_code = "B"
        elif estrategia.startswith("C"):
            strategy_code = "C"
        elif estrategia.startswith("D"):
            strategy_code = "D"
        elif estrategia.startswith("E"):
            strategy_code = "E"
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
        estrategias_para_executar = ["A", "B", "C", "D", "E"] if automatizar_todas else [strategy_code]

        sinais_estrategias = {
            "A": sinal,
            "B": estrategia_b_signal(row)[0],
            "C": estrategia_c_signal(row)[0],
            "D": "DUAL" if bool(gex_data and gex_data.get("d_ativa")) else "AGUARDAR",
            "E": sinal_e,
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
            ])
            st.subheader("🤖 Automação — 5 estratégias")
            st.dataframe(df_sinais, use_container_width=True, hide_index=True)
            st.caption(f"Execução automática ativa. Limite global: {quantidade_abertas}/{int(max_operacoes)} operações abertas.")

        # ========================================================
        # PAINEL
        # ========================================================
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("BTC", f"${preco_atual:,.2f}")
        c2.metric("Sinal operacional", sinal_operacional)
        c3.metric("Score COMPRA", f"{score_compra:.0f}")
        c4.metric("Score VENDA", f"{score_venda:.0f}")
        c5.metric("Regime", regime)

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
            sinal_plano = sinal_operacional if sinal_operacional in ("COMPRA", "VENDA") else (sinal if sinal in ("COMPRA", "VENDA") else None)
            if sinal_plano:
                stop_view, alvo_view = calcular_plano(sinal_plano, preco_atual, float(row["ATR"]))
                risco_pct = abs((stop_view / preco_atual) - 1.0) * 100.0
                alvo_pct = abs((alvo_view / preco_atual) - 1.0) * 100.0
                risco_rs = float(valor_entrada) * risco_pct / 100.0
                alvo_rs = float(valor_entrada) * alvo_pct / 100.0
                st.write(f"**Plano {sinal_plano}:** Entrada ${preco_atual:,.2f} | Stop ${stop_view:,.2f} | Alvo ${alvo_view:,.2f}")
                st.write(f"**Risco:** -{risco_pct:.3f}% ≈ -R$ {risco_rs:,.2f} | **Alvo:** +{alvo_pct:.3f}% ≈ +R$ {alvo_rs:,.2f} | **R/R:** 1:2")
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
                st.write(f"**Percentil usado para OI elevado:** {D_ATM_ELEVATED_PERCENTILE:.0f}%")
                st.write(f"**Opções utilizadas:** {gex_data.get('options_used', '-')} | **Método:** {gex_data.get('gex_method', '-')}")
                st.write("**Condições:** " + " | ".join(gex_data["condicoes"]))
                if gex_data["d_ativa"]:
                    st.success("🟢 Estratégia D ATIVA")
                else:
                    st.info("⚪ Estratégia D INATIVA")
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

            st.markdown("### 🧪 Backtest — meta de 1.000 operações por estratégia")
            st.caption("O backtest histórico completo é calculado com dados públicos de candles. B e C podem ser testadas diretamente; D/GEX e E/Brent-WTI exigem histórico próprio dos dados externos para uma reprodução 100% fiel.")
            if st.button("🚀 Executar backtest B + C", key="btn_backtest_bc"):
                try:
                    with st.spinner("Baixando histórico e simulando até 1.000 operações por método..."):
                        hist_bt = buscar_klines_historico(50000)
                        bt_b = backtest_bc(hist_bt, "B", 1000)
                        bt_c = backtest_bc(hist_bt, "C", 1000)
                    rows = []
                    for code, bt in [("B", bt_b), ("C", bt_c)]:
                        r = resumo_backtest(bt, valor_entrada)
                        r["Estratégia"] = code
                        rows.append(r)
                        st.session_state[f"bt_{code}"] = bt
                    st.session_state["bt_resumo"] = pd.DataFrame(rows)[["Estratégia","Operações","WR %","Lucro R$","ROI %","Drawdown máx. R$"]]
                except Exception as exc:
                    st.error(f"Backtest não executado: {exc}")
            if "bt_resumo" in st.session_state:
                st.dataframe(st.session_state["bt_resumo"], use_container_width=True, hide_index=True)
                st.info("Para D e E, não vou inventar 1.000 operações: precisamos armazenar o histórico de GEX/OI e Brent/WTI para testar exatamente as regras dessas duas estratégias.")

            st.markdown("### 💰 Simulação da banca")
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
