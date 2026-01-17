#!/usr/bin/env python3
"""
LIVE Polymarket Deep-Bid Bot (single-file) — 20%–30% band + balance-aware risk

What this version does (vs your stink-bid bot):
- Places *deep* BUY limit bids only in the 0.20–0.30 price band (20%–30% implied prob).
- Uses a simple “edge” filter: only bids when mid - bid_price >= MIN_EDGE (and spread gate).
- Balance-aware risk: per-order sizing, max open exposure, and daily loss limit scale with balance.
- Enforces minimums: MIN_SHARES=5 and MIN_NOTIONAL_USD=1.00 (price*shares >= 1)
- Cancels stale deep bids, places take-profit SELLs on fills, tracks daily PnL + cooldown.

⚠️ WARNING
This places REAL orders. Start small. Watch logs/CSV. You are responsible for results.
No strategy can guarantee “good EV/alpha”; this uses heuristics + risk controls.

Required env vars:
  PRIVATE_KEY
  POLY_API_KEY
  POLY_API_SECRET
  POLY_API_PASSPHRASE
  SIGNATURE_TYPE        # 0=EOA, 1=POLY_PROXY, 2=GNOSIS_SAFE
  FUNDER_ADDRESS

Optional env vars:
  LOG_LEVEL             # DEBUG/INFO/WARNING/ERROR (default INFO)
"""

import os
import sys
import json
import csv
import math
import time
import asyncio
import logging
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Set

import requests

# ---- py-clob-client ----
try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import ApiCreds, OrderArgs, OrderType
    from py_clob_client.order_builder.constants import BUY, SELL
except Exception:
    print("ERROR: py-clob-client import failed. Install with: pip install py-clob-client")
    raise


# ============================================================
# CONFIG
# ============================================================

class Config:
    # -------- endpoints --------
    GAMMA_API_BASE = "https://gamma-api.polymarket.com"
    CLOB_API_BASE = "https://clob.polymarket.com"

    # -------- discovery --------
    GAMMA_LIMIT = 300
    REFRESH_MARKETS_INTERVAL_SEC = 120

    # Loose matching for "15-minute BTC" (change if you want other markets)
    MUST_HAVE_ANY = ["btc", "bitcoin"]
    MUST_HAVE_TIME_ANY = ["15m", "15 min", "15-min", "15 minute", "15-minute", "15 minutes"]

    # Outcome choice keywords (first match wins)
    TARGET_OUTCOME_KEYWORDS = ["up", "yes"]

    # -------- deep-bid strategy (core) --------
    DEEP_BID_MIN = 0.20     # 20%
    DEEP_BID_MAX = 0.30     # 30%
    # Only place bid if "edge" vs mid is at least this:
    MIN_EDGE = 0.03         # require mid - bid_px >= 3c
    # Spread gate (avoid tight books)
    MIN_SPREAD = 0.05       # ask-bid must be >= 5c
    # How far below mid we try to bid (then clamped into 20–30 band)
    TARGET_DISCOUNT_FROM_MID = 0.06

    # Don’t open near endDate
    FINAL_SECONDS_BEFORE_END = 45

    # -------- take-profit --------
    BASE_PROFIT_TARGET = 0.03           # at least +3c
    PROFIT_FRACTION_OF_EDGE = 0.50      # also take some of the edge
    PROFIT_FRACTION_OF_SPREAD = 0.25    # and some of spread
    MAX_TP_ADD = 0.20                   # cap TP distance

    # -------- order mgmt --------
    MAX_STALE_BID_SECONDS = 240
    LOOP_SLEEP_SECONDS = 5
    PLACE_AT_MOST_ONE_NEW_BID_PER_LOOP = True

    # -------- minimums --------
    MIN_SHARES_LIMIT = 5
    MIN_NOTIONAL_USD = 1.00

    # -------- balance-aware sizing & risk --------
    USE_BALANCE_SIZING = True

    # Per-order budget is a fraction of available USDC (tiered; see function below)
    # Absolute hard cap per order:
    MAX_ORDER_NOTIONAL_USD = 15.0

    # Exposure cap and daily loss cap scale with balance (tiered; see functions below)
    HARD_MAX_OPEN_EXPOSURE_USD = 60.0   # absolute ceiling regardless of balance
    HARD_MAX_DAILY_LOSS_USD = 30.0      # absolute ceiling regardless of balance

    MAX_OPEN_ORDERS_TOTAL = 12
    MAX_OPEN_ORDERS_PER_MARKET = 1

    CONSECUTIVE_LOSS_LIMIT = 3
    COOLDOWN_SECONDS = 300

    # -------- reconciliation --------
    TRADE_LOOKBACK_MINUTES = 240

    # -------- misc --------
    REQUEST_TIMEOUT = 15

    # -------- logging --------
    CSV_LOG_FILE = "bot_trades.csv"
    LOG_FILE = "deepbidbot.log"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


# ============================================================
# LOGGING
# ============================================================

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("DeepBidBot")
    level = getattr(logging, Config.LOG_LEVEL, logging.INFO)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(Config.LOG_FILE)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger

logger = setup_logging()


# ============================================================
# DATA STRUCTURES
# ============================================================

class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(str, Enum):
    OPEN = "OPEN"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"

@dataclass
class Market:
    market_id: str
    question: str
    description: str
    tokens: Dict[str, str]                 # outcome_title -> token_id
    end_time: Optional[datetime] = None
    active: bool = True

    volume24hr: float = 0.0
    liquidity: float = 0.0

@dataclass
class OrderRecord:
    order_id: str
    market_id: str
    token_id: str
    outcome_title: str
    side: Side
    price: float
    size: float
    created_at: datetime
    status: OrderStatus = OrderStatus.OPEN

    filled_size: float = 0.0
    avg_fill_price: float = 0.0
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    parent_buy_order_id: Optional[str] = None

@dataclass
class BotState:
    markets: Dict[str, Market] = field(default_factory=dict)
    last_market_refresh: Optional[datetime] = None

    open_orders: Dict[str, OrderRecord] = field(default_factory=dict)
    tp_placed_for_buy: Set[str] = field(default_factory=set)

    daily_pnl: float = 0.0
    consecutive_losses: int = 0
    cooldown_until: Optional[datetime] = None

    last_trade_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc) - timedelta(minutes=10))

    tick_cache: Dict[str, Tuple[float, bool, datetime]] = field(default_factory=dict)
    balance_cache: Tuple[Optional[float], datetime] = (None, datetime.now(timezone.utc) - timedelta(minutes=10))


# ============================================================
# CSV EVENT LOG
# ============================================================

def log_event(event: str, data: Dict[str, Any]) -> None:
    file_exists = Path(Config.CSV_LOG_FILE).exists()
    try:
        with open(Config.CSV_LOG_FILE, "a", newline="") as f:
            w = csv.writer(f)
            if not file_exists:
                w.writerow([
                    "ts", "event", "order_id", "market_id", "token_id", "outcome",
                    "side", "price", "size", "filled_size", "avg_fill_price", "details"
                ])
            w.writerow([
                datetime.now(timezone.utc).isoformat(),
                event,
                data.get("order_id", ""),
                data.get("market_id", ""),
                data.get("token_id", ""),
                data.get("outcome_title", ""),
                data.get("side", ""),
                data.get("price", ""),
                data.get("size", ""),
                data.get("filled_size", ""),
                data.get("avg_fill_price", ""),
                json.dumps(data, default=str),
            ])
    except Exception as e:
        logger.error(f"CSV log failed: {e}")


# ============================================================
# UTILS
# ============================================================

def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def must_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v

def parse_iso_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        if isinstance(s, (int, float)):
            return datetime.fromtimestamp(float(s), tz=timezone.utc)
        s = str(s)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None

def norm(s: str) -> str:
    return (s or "").strip().lower()


# ============================================================
# AUTH / CLIENT
# ============================================================

def load_auth() -> Dict[str, str]:
    return {
        "PRIVATE_KEY": must_env("PRIVATE_KEY"),
        "POLY_API_KEY": must_env("POLY_API_KEY"),
        "POLY_API_SECRET": must_env("POLY_API_SECRET"),
        "POLY_API_PASSPHRASE": must_env("POLY_API_PASSPHRASE"),
        "SIGNATURE_TYPE": must_env("SIGNATURE_TYPE"),
        "FUNDER_ADDRESS": must_env("FUNDER_ADDRESS"),
    }

def init_client(auth: Dict[str, str]) -> ClobClient:
    creds = ApiCreds(
        api_key=auth["POLY_API_KEY"],
        api_secret=auth["POLY_API_SECRET"],
        api_passphrase=auth["POLY_API_PASSPHRASE"],
    )
    client = ClobClient(
        Config.CLOB_API_BASE,
        key=auth["PRIVATE_KEY"],
        chain_id=137,
        creds=creds,
        signature_type=int(auth["SIGNATURE_TYPE"]),
        funder=auth["FUNDER_ADDRESS"],
    )
    logger.info(f"✓ ClobClient initialized (signature_type={auth['SIGNATURE_TYPE']}, funder={auth['FUNDER_ADDRESS'][:10]}...)")
    return client


# ============================================================
# GAMMA DISCOVERY
# ============================================================

def gamma_get_markets() -> List[Dict[str, Any]]:
    url = f"{Config.GAMMA_API_BASE}/markets"
    params = {
        "limit": Config.GAMMA_LIMIT,
        "active": True,
        "closed": False,
        "archived": False,
        "order": "volume24hr",
        "ascending": False,
    }
    r = requests.get(url, params=params, timeout=Config.REQUEST_TIMEOUT)
    r.raise_for_status()
    payload = r.json()
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            return payload["data"]
        if isinstance(payload.get("markets"), list):
            return payload["markets"]
    return []

def extract_tokens(item: Dict[str, Any]) -> Dict[str, str]:
    tokens: Dict[str, str] = {}

    toks = item.get("tokens")
    if isinstance(toks, list):
        for t in toks:
            if not isinstance(t, dict):
                continue
            token_id = t.get("token_id") or t.get("tokenId") or t.get("tokenID") or t.get("id")
            title = (t.get("title") or t.get("outcome") or t.get("name") or t.get("label") or "").strip()
            if token_id and title:
                tokens[title] = str(token_id)

    outcomes = item.get("outcomes")
    if isinstance(outcomes, list):
        for o in outcomes:
            if not isinstance(o, dict):
                continue
            title = (o.get("title") or o.get("name") or o.get("outcome") or "").strip()
            tok = o.get("token")
            token_id = None
            if isinstance(tok, dict):
                token_id = tok.get("token_id") or tok.get("tokenId") or tok.get("tokenID") or tok.get("id")
            token_id = token_id or o.get("token_id") or o.get("tokenId") or o.get("tokenID") or o.get("id")
            if token_id and title and title not in tokens:
                tokens[title] = str(token_id)

    return tokens

def is_target_market(item: Dict[str, Any]) -> bool:
    q = norm(item.get("question") or item.get("title") or "")
    d = norm(item.get("description") or "")
    text = f"{q} {d}"

    if not any(k in text for k in Config.MUST_HAVE_ANY):
        return False
    if not any(k in text for k in Config.MUST_HAVE_TIME_ANY):
        return False
    if item.get("active") is False:
        return False
    return True

def discover_markets() -> List[Market]:
    found: List[Market] = []
    try:
        raw = gamma_get_markets()
        for item in raw:
            if not isinstance(item, dict):
                continue
            if not is_target_market(item):
                continue

            market_id = str(item.get("id") or item.get("marketId") or "")
            if not market_id:
                continue

            tokens = extract_tokens(item)
            if len(tokens) < 2:
                continue

            q = item.get("question") or item.get("title") or ""
            d = item.get("description") or ""
            end_time = parse_iso_dt(item.get("endDate") or item.get("end_date") or item.get("closeTime") or item.get("resolution_date"))

            v24 = 0.0
            liq = 0.0
            for k in ("volume24hr", "volume_24hr", "volume"):
                if item.get(k) is not None:
                    try: v24 = float(item.get(k))
                    except Exception: pass
                    break
            for k in ("liquidity", "liquidityNum", "liquidity_num"):
                if item.get(k) is not None:
                    try: liq = float(item.get(k))
                    except Exception: pass
                    break

            found.append(Market(
                market_id=market_id,
                question=q,
                description=d,
                tokens=tokens,
                end_time=end_time,
                active=True,
                volume24hr=v24,
                liquidity=liq,
            ))

        return found
    except Exception as e:
        logger.warning(f"Market discovery failed: {e}")
        return []


# ============================================================
# PRICE / TICK HELPERS
# ============================================================

def get_best_bid_ask(client: ClobClient, token_id: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Uses get_price(token_id, BUY/SELL):
      BUY  => best bid
      SELL => best ask
    """
    bid = ask = None
    try:
        b = client.get_price(token_id, BUY)
        if isinstance(b, dict) and b.get("price") is not None:
            bid = float(b["price"])
    except Exception:
        pass
    try:
        a = client.get_price(token_id, SELL)
        if isinstance(a, dict) and a.get("price") is not None:
            ask = float(a["price"])
    except Exception:
        pass
    return bid, ask

def quantize_down(px: float, tick: float) -> float:
    if tick <= 0:
        return round(px, 2)
    return round(math.floor(px / tick + 1e-12) * tick, 6)

def quantize_up(px: float, tick: float) -> float:
    if tick <= 0:
        return round(px, 2)
    return round(math.ceil(px / tick - 1e-12) * tick, 6)

def clamp_price(px: float, tick: float) -> float:
    tick = float(tick) if tick else 0.01
    px = float(px)
    return max(tick, min(0.99, px))

def choose_outcome(market: Market) -> Optional[Tuple[str, str]]:
    keys = [k.lower() for k in Config.TARGET_OUTCOME_KEYWORDS]
    for title, tok in market.tokens.items():
        t = title.strip().lower()
        if any(k in t for k in keys):
            return title, tok
    return None


# ============================================================
# BALANCE (best-effort probing)
# ============================================================

def _try_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def get_available_balance_usd_best_effort(client: ClobClient) -> Optional[float]:
    """
    py-clob-client versions differ. We probe common method names and response shapes.
    Returns available USD/USDC balance if found, else None.
    """
    candidates = [
        "get_balance",
        "get_balances",
        "get_user_balance",
        "get_collateral_balance",
        "get_account",
        "get_portfolio",
    ]
    for name in candidates:
        fn = getattr(client, name, None)
        if not callable(fn):
            continue
        try:
            resp = fn()
        except Exception:
            continue

        if isinstance(resp, (int, float, str)):
            v = _try_float(resp)
            if v is not None:
                return v

        if isinstance(resp, dict):
            for k in ("available", "availableBalance", "available_balance", "usdc", "USDC", "balance", "collateral"):
                v = _try_float(resp.get(k))
                if v is not None:
                    return v

            for k in ("balances", "assets", "holdings"):
                arr = resp.get(k)
                if isinstance(arr, list):
                    for it in arr:
                        if not isinstance(it, dict):
                            continue
                        sym = (it.get("symbol") or it.get("asset") or it.get("currency") or "").upper()
                        if sym in ("USDC", "USD"):
                            v = _try_float(it.get("available") or it.get("free") or it.get("balance") or it.get("amount"))
                            if v is not None:
                                return v

    return None

def get_balance_cached(client: ClobClient, state: BotState, cache_seconds: int = 10) -> Optional[float]:
    bal, ts = state.balance_cache
    if (utcnow() - ts).total_seconds() < cache_seconds:
        return bal
    bal2 = get_available_balance_usd_best_effort(client)
    state.balance_cache = (bal2, utcnow())
    return bal2


# ============================================================
# BALANCE-AWARE RISK CONTROLS
# ============================================================

def per_order_fraction_for_balance(bal: float) -> float:
    """
    Conservative tiering:
    - small balances: higher fraction so you still trade, but capped by MAX_ORDER_NOTIONAL_USD
    - larger balances: lower fraction to reduce risk concentration
    """
    if bal <= 10:
        return 0.20   # up to 20% of balance (but max notional cap applies)
    if bal <= 50:
        return 0.10
    if bal <= 200:
        return 0.05
    return 0.03

def exposure_fraction_for_balance(bal: float) -> float:
    if bal <= 10:
        return 0.50
    if bal <= 50:
        return 0.35
    if bal <= 200:
        return 0.25
    return 0.20

def daily_loss_fraction_for_balance(bal: float) -> float:
    if bal <= 10:
        return 0.25
    if bal <= 50:
        return 0.18
    if bal <= 200:
        return 0.12
    return 0.08

def dynamic_exposure_cap(client: ClobClient, state: BotState) -> float:
    cap = float(Config.HARD_MAX_OPEN_EXPOSURE_USD)
    if not Config.USE_BALANCE_SIZING:
        return cap
    bal = get_balance_cached(client, state)
    if bal is None:
        return min(cap, 10.0)
    frac_cap = exposure_fraction_for_balance(float(bal)) * float(bal)
    return min(cap, max(1.0, frac_cap))

def dynamic_daily_loss_limit(client: ClobClient, state: BotState) -> float:
    lim = float(Config.HARD_MAX_DAILY_LOSS_USD)
    if not Config.USE_BALANCE_SIZING:
        return lim
    bal = get_balance_cached(client, state)
    if bal is None:
        return min(lim, 5.0)
    frac_lim = daily_loss_fraction_for_balance(float(bal)) * float(bal)
    return min(lim, max(1.0, frac_lim))


# ============================================================
# ELIGIBILITY / STATE
# ============================================================

def in_cooldown(state: BotState) -> bool:
    return state.cooldown_until is not None and utcnow() < state.cooldown_until

def open_exposure(state: BotState) -> float:
    return sum(o.price * o.size for o in state.open_orders.values()
               if o.status == OrderStatus.OPEN and o.side == Side.BUY)

def open_orders_total(state: BotState) -> int:
    return sum(1 for o in state.open_orders.values() if o.status == OrderStatus.OPEN)

def open_orders_on_market(state: BotState, market_id: str) -> int:
    return sum(1 for o in state.open_orders.values()
               if o.status == OrderStatus.OPEN and o.market_id == market_id)

def can_place_bid(client: ClobClient, state: BotState, market: Market, spread: Optional[float]) -> Tuple[bool, str]:
    if in_cooldown(state):
        return False, "cooldown"

    daily_loss_limit = dynamic_daily_loss_limit(client, state)
    if state.daily_pnl <= -daily_loss_limit:
        return False, f"daily loss limit ({daily_loss_limit:.2f})"

    if open_orders_total(state) >= Config.MAX_OPEN_ORDERS_TOTAL:
        return False, "max open orders"
    if open_orders_on_market(state, market.market_id) >= Config.MAX_OPEN_ORDERS_PER_MARKET:
        return False, "max per market"
    if spread is None or spread < Config.MIN_SPREAD:
        return False, f"spread {spread}"
    if market.end_time:
        secs = (market.end_time - utcnow()).total_seconds()
        if secs < Config.FINAL_SECONDS_BEFORE_END:
            return False, "too close to end"

    cap = dynamic_exposure_cap(client, state)
    if open_exposure(state) >= cap:
        return False, f"max exposure cap={cap:.2f}"

    return True, "ok"


# ============================================================
# TICK / MARKET META (cached)
# ============================================================

def get_tick_and_negrisk_cached(client: ClobClient, state: BotState, token_id: str, ttl_sec: int = 300) -> Tuple[float, bool]:
    hit = state.tick_cache.get(token_id)
    if hit:
        tick, neg, ts = hit
        if (utcnow() - ts).total_seconds() < ttl_sec:
            return tick, neg

    m = client.get_market(token_id)
    tick = float(m.get("tickSize") or m.get("tick_size") or 0.01)
    neg = bool(m.get("negRisk") or m.get("neg_risk") or False)
    state.tick_cache[token_id] = (tick, neg, utcnow())
    return tick, neg


# ============================================================
# ORDER SIZING (min shares + min $1 + balance-aware)
# ============================================================

def min_size_for_notional(price: float) -> int:
    if price <= 0:
        return Config.MIN_SHARES_LIMIT
    return max(Config.MIN_SHARES_LIMIT, int(math.ceil(Config.MIN_NOTIONAL_USD / float(price))))

def size_from_balance(price: float, available: float) -> int:
    """
    per-order budget = min(MAX_ORDER_NOTIONAL_USD, available * tier_fraction)
    size = floor(budget / price), respecting min shares & min notional.
    """
    price = float(price)
    if price <= 0:
        return Config.MIN_SHARES_LIMIT

    frac = per_order_fraction_for_balance(float(available))
    budget = min(float(Config.MAX_ORDER_NOTIONAL_USD), float(available) * float(frac))

    raw = int(math.floor(budget / price))
    raw = max(raw, min_size_for_notional(price))
    return raw


# ============================================================
# DEEP BID + TP PRICING
# ============================================================

def compute_deep_bid_price(bid: float, ask: float, tick: float) -> Tuple[Optional[float], float, float, float]:
    """
    Returns (deep_px, mid, spread, edge).
    - deep_px is clamped to [0.20, 0.30]
    - also forced to be <= best_bid - 1 tick (so it's truly 'deep' and not top-of-book)
    """
    if bid is None or ask is None:
        return None, 0.0, 0.0, 0.0

    tick = float(tick) if tick else 0.01
    bid = float(bid)
    ask = float(ask)
    spread = ask - bid
    mid = 0.5 * (bid + ask)

    # Start from "mid - discount"
    target = mid - float(Config.TARGET_DISCOUNT_FROM_MID)
    # Clamp to deep band
    target = max(float(Config.DEEP_BID_MIN), min(float(Config.DEEP_BID_MAX), target))

    # Ensure truly deep vs current best bid
    max_allowed = bid - tick
    if max_allowed < float(Config.DEEP_BID_MIN):
        # If best bid is already below 0.20, we skip this market for "20-30%" bot
        return None, mid, spread, 0.0

    target = min(target, max_allowed)

    deep_px = quantize_down(target, tick)
    deep_px = clamp_price(deep_px, tick)

    edge = mid - deep_px
    return deep_px, mid, spread, edge

def compute_tp_price(entry: float, edge: float, spread: float, tick: float) -> float:
    """
    TP = entry + max(BASE_PROFIT_TARGET, edge*F_edge, spread*F_spread), capped.
    """
    add = max(
        float(Config.BASE_PROFIT_TARGET),
        float(edge) * float(Config.PROFIT_FRACTION_OF_EDGE),
        float(spread) * float(Config.PROFIT_FRACTION_OF_SPREAD),
    )
    add = min(add, float(Config.MAX_TP_ADD))
    tp = entry + add
    tp = quantize_up(tp, float(tick))
    return clamp_price(tp, float(tick))


# ============================================================
# ORDER PLACEMENT
# ============================================================

def place_limit_order(
    client: ClobClient,
    state: BotState,
    market_id: str,
    outcome_title: str,
    token_id: str,
    side: Side,
    price: float,
    size: float,
    parent_buy_order_id: Optional[str] = None,
) -> str:
    tick, neg = get_tick_and_negrisk_cached(client, state, token_id)

    if side == Side.BUY:
        price = quantize_down(price, tick)
    else:
        price = quantize_up(price, tick)

    price = clamp_price(price, tick)

    # enforce minimums
    size_int = int(math.floor(float(size)))
    size_int = max(size_int, min_size_for_notional(price))

    resp = client.create_and_post_order(
        OrderArgs(
            token_id=token_id,
            price=float(price),
            size=float(size_int),
            side=BUY if side == Side.BUY else SELL,
        ),
        options={"tick_size": tick, "neg_risk": neg},
        order_type=OrderType.GTC,
    )

    order_id = resp.get("orderID") or resp.get("orderId") or resp.get("id")
    if not order_id:
        raise RuntimeError(f"Unexpected order response: {resp}")

    rec = OrderRecord(
        order_id=str(order_id),
        market_id=market_id,
        token_id=token_id,
        outcome_title=outcome_title,
        side=side,
        price=float(price),
        size=float(size_int),
        created_at=utcnow(),
        status=OrderStatus.OPEN,
        parent_buy_order_id=parent_buy_order_id,
    )
    state.open_orders[rec.order_id] = rec

    logger.info(f"[ORDER] {side.value} px={rec.price:.4f} sz={rec.size:.0f} notional=${rec.price*rec.size:.2f} outcome={outcome_title} id={rec.order_id}")
    log_event("ORDER_PLACED", {
        "order_id": rec.order_id,
        "market_id": rec.market_id,
        "token_id": rec.token_id,
        "outcome_title": rec.outcome_title,
        "side": rec.side.value,
        "price": rec.price,
        "size": rec.size,
        "notional": rec.price * rec.size,
    })
    return rec.order_id


# ============================================================
# RECONCILIATION: TRADES -> FILLS
# ============================================================

def get_trades_best_effort(client: ClobClient) -> List[Dict[str, Any]]:
    trades = client.get_trades()
    if isinstance(trades, list):
        return [t for t in trades if isinstance(t, dict)]
    return []

def trade_order_id(t: Dict[str, Any]) -> Optional[str]:
    oid = t.get("orderID") or t.get("orderId") or t.get("order_id") or t.get("id")
    return str(oid) if oid else None

def trade_price(t: Dict[str, Any]) -> Optional[float]:
    p = t.get("price")
    try:
        return float(p) if p is not None else None
    except Exception:
        return None

def trade_size(t: Dict[str, Any]) -> Optional[float]:
    s = t.get("size") or t.get("amount") or t.get("filledSize") or t.get("quantity")
    try:
        return float(s) if s is not None else None
    except Exception:
        return None

def trade_ts(t: Dict[str, Any]) -> Optional[datetime]:
    ts = t.get("timestamp") or t.get("createdAt") or t.get("created_at") or t.get("time")
    return parse_iso_dt(ts)

def aggregate_fills(trades: List[Dict[str, Any]], since: datetime) -> Dict[str, Tuple[float, float]]:
    """
    Returns order_id -> (filled_size, vwap)
    """
    acc: Dict[str, Tuple[float, float]] = {}
    for tr in trades:
        dt = trade_ts(tr)
        if dt and dt < since:
            continue
        oid = trade_order_id(tr)
        px = trade_price(tr)
        sz = trade_size(tr)
        if not oid or px is None or sz is None:
            continue
        sum_sz, sum_px_sz = acc.get(oid, (0.0, 0.0))
        sum_sz += sz
        sum_px_sz += sz * px
        acc[oid] = (sum_sz, sum_px_sz)

    out: Dict[str, Tuple[float, float]] = {}
    for oid, (sum_sz, sum_px_sz) in acc.items():
        if sum_sz > 0:
            out[oid] = (sum_sz, sum_px_sz / sum_sz)
    return out

def cancel_stale_bids(client: ClobClient, state: BotState) -> None:
    now = utcnow()
    for oid, rec in list(state.open_orders.items()):
        if rec.status != OrderStatus.OPEN:
            continue
        if rec.side != Side.BUY:
            continue
        age = (now - rec.created_at).total_seconds()
        if age > Config.MAX_STALE_BID_SECONDS:
            try:
                client.cancel_order(rec.order_id)
                rec.status = OrderStatus.CANCELLED
                rec.last_update = now
                logger.info(f"[CANCEL] stale BUY {rec.order_id} age={int(age)}s")
                log_event("ORDER_CANCELLED_STALE", {
                    "order_id": rec.order_id,
                    "market_id": rec.market_id,
                    "token_id": rec.token_id,
                    "outcome_title": rec.outcome_title,
                    "side": rec.side.value,
                })
            except Exception as e:
                logger.warning(f"Cancel failed {rec.order_id}: {e}")
            finally:
                state.open_orders.pop(oid, None)

def reconcile_fills_and_manage_tp(client: ClobClient, state: BotState) -> None:
    cancel_stale_bids(client, state)

    now = utcnow()
    since = now - timedelta(minutes=Config.TRADE_LOOKBACK_MINUTES)
    since = min(state.last_trade_check, since)

    try:
        trades = get_trades_best_effort(client)
    except Exception as e:
        logger.warning(f"get_trades failed: {e}")
        return

    fills = aggregate_fills(trades, since=since)
    state.last_trade_check = now

    # apply fill updates
    for oid, (f_sz, vwap) in fills.items():
        rec = state.open_orders.get(oid)
        if not rec:
            continue

        rec.filled_size = max(rec.filled_size, min(rec.size, f_sz))
        rec.avg_fill_price = vwap
        rec.last_update = now
        if rec.filled_size >= rec.size - 1e-9:
            rec.status = OrderStatus.FILLED
        elif rec.filled_size > 0:
            rec.status = OrderStatus.PARTIAL

        log_event("ORDER_FILL_UPDATE", {
            "order_id": rec.order_id,
            "market_id": rec.market_id,
            "token_id": rec.token_id,
            "outcome_title": rec.outcome_title,
            "side": rec.side.value,
            "price": rec.price,
            "size": rec.size,
            "filled_size": rec.filled_size,
            "avg_fill_price": rec.avg_fill_price,
        })

    # place TP for BUY fills
    for oid, rec in list(state.open_orders.items()):
        if rec.side != Side.BUY or rec.status != OrderStatus.FILLED:
            continue
        if rec.order_id in state.tp_placed_for_buy:
            continue

        entry = float(rec.avg_fill_price or rec.price)
        bid, ask = get_best_bid_ask(client, rec.token_id)
        spread = (ask - bid) if (bid is not None and ask is not None) else Config.MIN_SPREAD
        mid = 0.5 * (bid + ask) if (bid is not None and ask is not None) else entry
        edge = mid - entry

        try:
            tick, _neg = get_tick_and_negrisk_cached(client, state, rec.token_id)
            tp_price = compute_tp_price(entry=entry, edge=float(edge), spread=float(spread), tick=tick)
        except Exception:
            tp_price = clamp_price(entry + Config.BASE_PROFIT_TARGET, 0.01)

        try:
            sell_oid = place_limit_order(
                client=client,
                state=state,
                market_id=rec.market_id,
                outcome_title=rec.outcome_title,
                token_id=rec.token_id,
                side=Side.SELL,
                price=tp_price,
                size=rec.filled_size if rec.filled_size > 0 else rec.size,
                parent_buy_order_id=rec.order_id,
            )
            state.tp_placed_for_buy.add(rec.order_id)
            logger.info(f"[TP] placed SELL {sell_oid} at {tp_price:.4f} for BUY {rec.order_id}")
            log_event("TP_PLACED", {
                "order_id": sell_oid,
                "market_id": rec.market_id,
                "token_id": rec.token_id,
                "outcome_title": rec.outcome_title,
                "side": "SELL",
                "price": tp_price,
                "size": rec.filled_size if rec.filled_size > 0 else rec.size,
                "parent_buy_order_id": rec.order_id,
            })
        except Exception as e:
            logger.error(f"TP placement failed for buy {rec.order_id}: {e}")

    # compute PnL on SELL fills and cleanup
    for oid, rec in list(state.open_orders.items()):
        if rec.side != Side.SELL or rec.status != OrderStatus.FILLED:
            continue

        parent = rec.parent_buy_order_id
        if parent and parent in state.open_orders:
            buy = state.open_orders[parent]
            qty = min(rec.filled_size or rec.size, buy.filled_size or buy.size)
            buy_px = float(buy.avg_fill_price or buy.price)
            sell_px = float(rec.avg_fill_price or rec.price)
            pnl = qty * (sell_px - buy_px)

            state.daily_pnl += pnl
            if pnl < 0:
                state.consecutive_losses += 1
            else:
                state.consecutive_losses = 0

            logger.info(f"[PNL] qty={qty:.0f} buy={buy_px:.4f} sell={sell_px:.4f} pnl={pnl:+.4f} daily={state.daily_pnl:+.4f}")
            log_event("ROUND_TRIP_PNL", {
                "buy_order_id": buy.order_id,
                "sell_order_id": rec.order_id,
                "market_id": rec.market_id,
                "token_id": rec.token_id,
                "outcome_title": rec.outcome_title,
                "qty": qty,
                "buy_px": buy_px,
                "sell_px": sell_px,
                "pnl": pnl,
                "daily_pnl": state.daily_pnl,
            })

            state.open_orders.pop(oid, None)
            state.open_orders.pop(parent, None)
        else:
            state.open_orders.pop(oid, None)

        if state.consecutive_losses >= Config.CONSECUTIVE_LOSS_LIMIT:
            state.cooldown_until = utcnow() + timedelta(seconds=Config.COOLDOWN_SECONDS)
            logger.warning(f"[COOLDOWN] until {state.cooldown_until} after losses={Config.CONSECUTIVE_LOSS_LIMIT}")
            log_event("COOLDOWN_TRIGGERED", {
                "cooldown_until": state.cooldown_until.isoformat(),
                "daily_pnl": state.daily_pnl,
            })
            state.consecutive_losses = 0


# ============================================================
# MARKET RANKING / SELECTION
# ============================================================

def market_is_tradeable_now(m: Market) -> bool:
    if not m.active:
        return False
    if m.end_time:
        secs = (m.end_time - utcnow()).total_seconds()
        if secs < Config.FINAL_SECONDS_BEFORE_END:
            return False
    return True

def rank_markets(
    client: ClobClient,
    state: BotState,
    markets: List[Market],
) -> List[Tuple[float, Market, str, str, float, float, float, float, float]]:
    """
    Returns list of:
      (score, market, outcome_title, token_id, bid, ask, spread, deep_px, edge)

    Score tries to prefer:
    - bigger edge (mid - deep_px)
    - bigger spread (more room)
    - more liquidity
    - more time remaining (small bonus)
    """
    ranked = []
    for m in markets:
        if not market_is_tradeable_now(m):
            continue
        choice = choose_outcome(m)
        if not choice:
            continue
        outcome_title, token_id = choice

        bid, ask = get_best_bid_ask(client, token_id)
        if bid is None or ask is None:
            continue
        spread = ask - bid
        if spread < Config.MIN_SPREAD:
            continue

        try:
            tick, _neg = get_tick_and_negrisk_cached(client, state, token_id)
        except Exception:
            tick = 0.01

        deep_px, mid, spread2, edge = compute_deep_bid_price(bid=bid, ask=ask, tick=tick)
        if deep_px is None:
            continue
        if edge < Config.MIN_EDGE:
            continue

        time_score = 0.0
        if m.end_time:
            mins = max(0.0, (m.end_time - utcnow()).total_seconds() / 60.0)
            time_score = min(1.0, mins / 60.0)

        liq_score = 0.0
        if m.volume24hr > 0:
            liq_score += min(1.0, math.log10(m.volume24hr + 1) / 5.0)
        if m.liquidity > 0:
            liq_score += min(1.0, math.log10(m.liquidity + 1) / 5.0)

        # edge dominates, then spread, then light bonuses
        score = (edge * 12.0) + (spread * 3.0) + (liq_score * 0.4) + (time_score * 0.2)
        ranked.append((score, m, outcome_title, token_id, bid, ask, spread, deep_px, edge))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked


# ============================================================
# MAIN LOOP
# ============================================================

async def main_loop(client: ClobClient) -> None:
    state = BotState()
    logger.info("=" * 70)
    logger.info("POLYMARKET DEEP-BID BOT (20%–30% band) — BALANCE-AWARE RISK")
    logger.info("⚠️  Real orders. Start small. Watch logs + CSV.")
    logger.info("=" * 70)

    while True:
        now = utcnow()

        # refresh markets
        if (state.last_market_refresh is None) or ((now - state.last_market_refresh).total_seconds() > Config.REFRESH_MARKETS_INTERVAL_SEC):
            markets = discover_markets()
            state.markets = {m.market_id: m for m in markets}
            state.last_market_refresh = now
            logger.info(f"[MARKETS] refreshed candidates={len(state.markets)}")

        # reconcile fills + TP + pnl + cancel stales
        try:
            reconcile_fills_and_manage_tp(client, state)
        except Exception as e:
            logger.warning(f"reconcile error: {e}")

        # risk stop (dynamic daily loss)
        daily_loss_limit = dynamic_daily_loss_limit(client, state)
        if state.daily_pnl <= -daily_loss_limit:
            logger.warning(f"[RISK] daily loss limit hit ({state.daily_pnl:+.4f} <= -{daily_loss_limit:.2f}). Sleeping.")
            await asyncio.sleep(Config.LOOP_SLEEP_SECONDS)
            continue

        if in_cooldown(state):
            logger.info(f"[COOLDOWN] active until {state.cooldown_until}")
            await asyncio.sleep(Config.LOOP_SLEEP_SECONDS)
            continue

        # choose best market(s)
        market_list = list(state.markets.values())
        ranked = rank_markets(client, state, market_list)

        placed = False
        for _score, m, outcome_title, token_id, bid, ask, spread, deep_px, edge in ranked[:25]:
            ok, _reason = can_place_bid(client, state, m, spread)
            if not ok:
                continue

            # size based on balance + minimums
            size = Config.MIN_SHARES_LIMIT
            if Config.USE_BALANCE_SIZING:
                bal = get_balance_cached(client, state)
                if bal is not None and bal > 0:
                    size = size_from_balance(deep_px, bal)
                else:
                    size = min_size_for_notional(deep_px)
            else:
                size = min_size_for_notional(deep_px)

            # cap per-order notional
            notional = deep_px * float(size)
            if notional > Config.MAX_ORDER_NOTIONAL_USD:
                size = max(Config.MIN_SHARES_LIMIT, int(math.floor(Config.MAX_ORDER_NOTIONAL_USD / deep_px)))
                size = max(size, min_size_for_notional(deep_px))
                notional = deep_px * float(size)

            # post-check exposure
            cap = dynamic_exposure_cap(client, state)
            projected = open_exposure(state) + notional
            if projected > cap:
                continue

            try:
                place_limit_order(
                    client=client,
                    state=state,
                    market_id=m.market_id,
                    outcome_title=outcome_title,
                    token_id=token_id,
                    side=Side.BUY,
                    price=deep_px,
                    size=float(size),
                )
                placed = True
                logger.info(f"[DEEP] bid_px={deep_px:.4f} mid={(0.5*(bid+ask)):.4f} edge={edge:.4f} spread={spread:.4f}")
                log_event("DEEP_BID_SIGNAL", {
                    "market_id": m.market_id,
                    "token_id": token_id,
                    "outcome_title": outcome_title,
                    "best_bid": bid,
                    "best_ask": ask,
                    "deep_px": deep_px,
                    "edge": edge,
                    "spread": spread,
                    "size": size,
                    "notional": notional,
                })
            except Exception as e:
                logger.error(f"place order failed: {e}")
                log_event("ORDER_PLACE_FAILED", {
                    "market_id": m.market_id,
                    "token_id": token_id,
                    "outcome_title": outcome_title,
                    "error": str(e),
                })

            if Config.PLACE_AT_MOST_ONE_NEW_BID_PER_LOOP:
                break

        bal = get_balance_cached(client, state)
        cap = dynamic_exposure_cap(client, state)
        logger.info(
            f"[STATUS] markets={len(state.markets)} ranked={len(ranked)} open={len(state.open_orders)} "
            f"exposure=${open_exposure(state):.2f}/${cap:.2f} bal={bal if bal is not None else 'NA'} "
            f"daily_pnl={state.daily_pnl:+.4f} daily_loss_limit={daily_loss_limit:.2f} placed={placed}"
        )

        await asyncio.sleep(Config.LOOP_SLEEP_SECONDS)


def run() -> None:
    auth = load_auth()
    client = init_client(auth)
    try:
        asyncio.run(main_loop(client))
    except KeyboardInterrupt:
        logger.info("Shutdown requested.")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run()
