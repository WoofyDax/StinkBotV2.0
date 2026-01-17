#!/usr/bin/env python3
"""
POLYMARKET STINK BID BOT (UPGRADED)

Implements the "gotchas":
1) Trades BOTH sides (YES/UP and NO/DOWN tokens).
2) Avoids duplicates: one open order per (token_id, side, price).
3) Detects fills (partial/full) and auto-posts take-profit SELL orders.
4) Reconciles local state with exchange state periodically.

NO telegram
NO prediction
Pure orderbook convexity

RISK IS CAPPED BY BALANCE
"""

import os
import sys
import math
import time
import logging
import requests
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY, SELL

# ================= CONFIG =================

class Config:
    GAMMA_API = "https://gamma-api.polymarket.com"
    CLOB_API  = "https://clob.polymarket.com"

    # stink ladder
    STINK_PRICES = [0.01, 0.02, 0.03]

    # selection filters
    MIN_24H_VOLUME = 15_000
    MAX_BID_DEPTH_ABOVE = 3_000     # USD depth above stink price
    MIN_SPREAD = 0.05
    MIN_TIME_TO_END_MIN = 120       # avoid imminent resolution

    # risk
    MAX_OPEN_EXPOSURE_FRAC = 0.20   # max 20% of balance deployed
    MAX_PER_MARKET_FRAC    = 0.05
    HARD_MAX_EXPOSURE      = 200.0

    # order rules
    MIN_NOTIONAL = 1.00
    MIN_SHARES   = 5
    STALE_SECONDS = 180

    # exits
    # After a stink fill, place a take-profit sell around "fair-ish".
    # This is intentionally simple & conservative.
    TAKE_PROFIT_MIN_EDGE = 0.08     # at least +8c over fill price when possible
    TAKE_PROFIT_UNDER_ASK = 0.01    # sell at (best_ask - 1c) when possible
    PRICE_TICK = 0.01              # Polymarket is typically 1-cent ticks

    LOOP_SLEEP = 6
    REFRESH_MARKETS = 120
    RECONCILE_OPEN_ORDERS = 60      # pull open orders from exchange periodically

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# ================= LOGGING =================

logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("STINK+")

# ================= UTIL =================

def utcnow():
    return datetime.now(timezone.utc)

def must_env(k: str) -> str:
    v = os.getenv(k)
    if not v:
        raise RuntimeError(f"Missing env var {k}")
    return v

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def round_tick(px: float) -> float:
    # round to nearest tick
    t = Config.PRICE_TICK
    return round(px / t) * t

# ================= CLIENT =================

def init_client():
    creds = ApiCreds(
        api_key=must_env("POLY_API_KEY"),
        api_secret=must_env("POLY_API_SECRET"),
        api_passphrase=must_env("POLY_API_PASSPHRASE"),
    )
    return ClobClient(
        Config.CLOB_API,
        key=must_env("PRIVATE_KEY"),
        chain_id=137,
        creds=creds,
        signature_type=int(must_env("SIGNATURE_TYPE")),
        funder=must_env("FUNDER_ADDRESS"),
    )

# ================= MARKET DISCOVERY =================

def get_markets():
    r = requests.get(
        f"{Config.GAMMA_API}/markets",
        params=dict(active=True, closed=False, limit=400),
        timeout=15
    )
    r.raise_for_status()
    return r.json()["markets"]

def extract_tokens_both_sides(mkt) -> List[str]:
    """
    Return both outcome token_ids (YES/UP and NO/DOWN) when we can identify them.
    We key off titles containing yes/up and no/down.
    """
    yes = None
    no = None
    for o in mkt.get("outcomes", []):
        title = (o.get("title") or "").lower()
        tok = (o.get("token") or {}).get("token_id")
        if not tok:
            continue
        if ("yes" in title) or ("up" in title):
            yes = tok
        if ("no" in title) or ("down" in title):
            no = tok

    out = []
    if yes: out.append(yes)
    if no:  out.append(no)
    return out

def market_ok(mkt):
    if mkt.get("volume24hr", 0) < Config.MIN_24H_VOLUME:
        return False
    end = mkt.get("endDate")
    if end:
        dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
        if (dt - utcnow()).total_seconds() < Config.MIN_TIME_TO_END_MIN * 60:
            return False
    return True

# ================= ORDERBOOK =================

def get_book(token_id):
    r = requests.get(f"{Config.CLOB_API}/book/{token_id}", timeout=10)
    r.raise_for_status()
    return r.json()

def best_bid_ask(book) -> Tuple[Optional[float], Optional[float]]:
    bid = float(book["bids"][0][0]) if book.get("bids") else None
    ask = float(book["asks"][0][0]) if book.get("asks") else None
    return bid, ask

def bid_depth_above(book, px):
    depth = 0.0
    for p, s in book.get("bids", []):
        if float(p) >= px:
            depth += float(p) * float(s)
    return depth

# ================= BALANCE =================

def get_balance(client) -> Optional[float]:
    try:
        return float(client.get_balance())
    except Exception:
        return None

# ================= BOT STATE =================

@dataclass
class TrackedOrder:
    order_id: str
    token_id: str
    side: str         # BUY or SELL
    price: float
    size: float
    created: datetime
    filled_size: float = 0.0  # last-known filled amount (for delta detection)

@dataclass
class PendingExit:
    token_id: str
    size: float
    target_price: float

class StinkBotPlus:
    def __init__(self, client: ClobClient):
        self.client = client

        # Keyed by (token_id, side, price) so we don't duplicate
        self.open_orders: Dict[Tuple[str, str, float], TrackedOrder] = {}

        self.last_refresh = utcnow()
        self.last_reconcile = utcnow()
        self.markets = []

        # Exits to place (generated from fills)
        self.exit_queue: List[PendingExit] = []

    # -------- exchange helpers (robust to minor API diffs) --------

    def _cancel(self, order_id: str):
        # py_clob_client usually has cancel_order
        return self.client.cancel_order(order_id)

    def _place_order(self, token_id: str, side: str, price: float, size: float) -> Optional[str]:
        resp = self.client.create_and_post_order(
            OrderArgs(token_id=token_id, price=price, size=size, side=side),
            order_type=OrderType.GTC
        )
        oid = resp.get("orderID") or resp.get("id") or resp.get("order_id")
        return oid

    def _get_open_orders_exchange(self) -> List[dict]:
        """
        Try to pull open orders from exchange.
        Method names can vary; we attempt common ones.
        """
        for name in ("get_open_orders", "get_orders", "list_orders"):
            fn = getattr(self.client, name, None)
            if callable(fn):
                try:
                    res = fn()
                    # Some versions wrap results in dicts
                    if isinstance(res, dict):
                        for k in ("orders", "data", "results"):
                            if k in res and isinstance(res[k], list):
                                return res[k]
                        # if dict but not recognized, fall through
                    if isinstance(res, list):
                        return res
                except Exception:
                    pass
        return []

    def _get_order_status(self, order_id: str) -> Optional[dict]:
        """
        Pull order info for fill detection. Try common names.
        """
        for name in ("get_order", "get_order_by_id", "read_order"):
            fn = getattr(self.client, name, None)
            if callable(fn):
                try:
                    return fn(order_id)
                except Exception:
                    pass
        return None

    # -------- core logic --------

    def cleanup_stale(self):
        now = utcnow()
        for key, o in list(self.open_orders.items()):
            age = (now - o.created).total_seconds()
            if age > Config.STALE_SECONDS and o.side == BUY:
                # We only auto-cancel stale BUY stink orders.
                # If a SELL TP is stale, you might still want it working.
                try:
                    self._cancel(o.order_id)
                    log.info(f"Canceled stale BUY {o.order_id} ({o.token_id} @ {o.price:.2f})")
                except Exception:
                    pass
                self.open_orders.pop(key, None)

    def reconcile_open_orders(self):
        """
        Sync local open_orders map with exchange to:
        - remove canceled/filled orders
        - ensure restarts don't duplicate
        """
        exch = self._get_open_orders_exchange()
        if not exch:
            return

        live_ids = set()
        # Build a set of live keys we see
        live_keys = set()

        for od in exch:
            oid = od.get("orderID") or od.get("id") or od.get("order_id")
            token = od.get("token_id") or od.get("tokenId")
            side  = od.get("side")
            price = od.get("price")
            size  = od.get("size")

            if not (oid and token and side and price and size):
                continue

            price = float(price)
            size  = float(size)
            live_ids.add(oid)
            k = (token, side, price)
            live_keys.add(k)

            if k not in self.open_orders:
                # Add it with "now" as created if we didn't know about it
                self.open_orders[k] = TrackedOrder(
                    order_id=oid,
                    token_id=token,
                    side=side,
                    price=price,
                    size=size,
                    created=utcnow(),
                    filled_size=float(od.get("filled_size") or od.get("filledSize") or 0.0),
                )

        # Remove local orders that are no longer live on exchange
        for k, o in list(self.open_orders.items()):
            if o.order_id not in live_ids:
                self.open_orders.pop(k, None)

        log.info(f"Reconciled open orders: {len(self.open_orders)}")

    def compute_take_profit(self, fill_px: float, book) -> float:
        bid, ask = best_bid_ask(book)
        # Aim for (ask - 1c), but ensure we capture at least +edge over fill
        target = fill_px + Config.TAKE_PROFIT_MIN_EDGE

        if ask is not None:
            target = max(target, ask - Config.TAKE_PROFIT_UNDER_ASK)

        # Clamp to valid [0.01, 0.99] range and tick-round
        target = clamp(target, 0.01, 0.99)
        target = round_tick(target)
        return target

    def detect_fills_and_queue_exits(self):
        """
        For every tracked BUY order, pull status and see if filled increased.
        If so, enqueue a SELL take-profit for the delta filled amount.
        """
        for k, o in list(self.open_orders.items()):
            if o.side != BUY:
                continue

            st = self._get_order_status(o.order_id)
            if not st:
                continue

            # Typical fields: filled_size / filledSize / executed / etc.
            filled = (
                st.get("filled_size")
                or st.get("filledSize")
                or st.get("filled")
                or 0.0
            )
            try:
                filled = float(filled)
            except Exception:
                filled = o.filled_size

            delta = filled - o.filled_size
            if delta > 0:
                # We got new fills. Queue a TP sell for just the delta.
                try:
                    book = get_book(o.token_id)
                    tp = self.compute_take_profit(o.price, book)
                except Exception:
                    tp = round_tick(clamp(o.price + Config.TAKE_PROFIT_MIN_EDGE, 0.01, 0.99))

                self.exit_queue.append(PendingExit(token_id=o.token_id, size=delta, target_price=tp))
                o.filled_size = filled
                self.open_orders[k] = o

                log.info(f"FILL detected: {o.token_id} BUY @ {o.price:.2f} +{delta:.0f}sh (total filled={filled:.0f}). Queue TP SELL @ {tp:.2f}")

            # If the order is fully filled or not open anymore, exchange reconcile will remove it later.
            # But we can also drop it if status says closed/filled.
            status = (st.get("status") or "").lower()
            if status in ("filled", "canceled", "cancelled", "rejected", "expired"):
                # remove from local (exits already queued via delta fill)
                self.open_orders.pop(k, None)

    def place_exits(self, bal: float):
        """
        Place queued TP SELL orders, avoiding duplicates similarly.
        """
        new_queue = []
        for ex in self.exit_queue:
            # ensure minimums
            px = ex.target_price
            size = float(ex.size)
            notional = px * size

            if size < Config.MIN_SHARES:
                # If delta fill smaller than min shares, accumulate it by re-queuing.
                new_queue.append(ex)
                continue
            if notional < Config.MIN_NOTIONAL:
                new_queue.append(ex)
                continue

            key = (ex.token_id, SELL, px)
            if key in self.open_orders:
                # already have a TP at this price, don't duplicate
                continue

            try:
                oid = self._place_order(ex.token_id, SELL, px, size)
                if oid:
                    self.open_orders[key] = TrackedOrder(
                        order_id=oid,
                        token_id=ex.token_id,
                        side=SELL,
                        price=px,
                        size=size,
                        created=utcnow(),
                        filled_size=0.0
                    )
                    log.info(f"TP SELL placed: {ex.token_id} @ {px:.2f} size={size:.0f}")
                else:
                    new_queue.append(ex)
            except Exception as e:
                log.warning(f"Failed placing TP SELL: {e}")
                new_queue.append(ex)

        self.exit_queue = new_queue

    def deployed_notional(self) -> float:
        # Counts open BUY notional as "deployed"; SELLs are exits (not extra exposure)
        dep = 0.0
        for o in self.open_orders.values():
            if o.side == BUY:
                dep += o.price * o.size
        return dep

    def maybe_place_stink_buys(self, bal: float):
        max_exposure = min(Config.HARD_MAX_EXPOSURE, bal * Config.MAX_OPEN_EXPOSURE_FRAC)
        deployed = self.deployed_notional()
        if deployed >= max_exposure:
            return

        for mkt in self.markets:
            if not market_ok(mkt):
                continue

            tokens = extract_tokens_both_sides(mkt)
            if not tokens:
                continue

            # Per-market cap
            per_mkt_cap = bal * Config.MAX_PER_MARKET_FRAC

            for token in tokens:
                try:
                    book = get_book(token)
                except Exception:
                    continue

                bid, ask = best_bid_ask(book)
                if not bid or not ask or (ask - bid) < Config.MIN_SPREAD:
                    continue

                for px in Config.STINK_PRICES:
                    # Skip if we already have this stink order live
                    key = (token, BUY, float(px))
                    if key in self.open_orders:
                        continue

                    depth = bid_depth_above(book, px)
                    if depth > Config.MAX_BID_DEPTH_ABOVE:
                        continue

                    budget = min(per_mkt_cap, max_exposure - deployed)
                    if budget < Config.MIN_NOTIONAL:
                        continue

                    size = max(Config.MIN_SHARES, int(math.ceil(Config.MIN_NOTIONAL / px)))
                    if px * size > budget:
                        continue

                    try:
                        oid = self._place_order(token, BUY, px, size)
                        if oid:
                            self.open_orders[key] = TrackedOrder(
                                order_id=oid,
                                token_id=token,
                                side=BUY,
                                price=px,
                                size=size,
                                created=utcnow(),
                                filled_size=0.0
                            )
                            deployed += px * size
                            log.info(
                                f"STINK BUY {px:.2f} size={size} depth_above=${depth:.0f} "
                                f"vol24h=${mkt.get('volume24hr', 0):.0f} token={token[:8]}..."
                            )
                    except Exception as e:
                        log.warning(f"Order post failed: {e}")

                    if deployed >= max_exposure:
                        return

    def run(self):
        self.markets = get_markets()
        self.last_refresh = utcnow()
        log.info(f"Initial markets: {len(self.markets)}")

        while True:
            try:
                self.cleanup_stale()

                if (utcnow() - self.last_refresh).total_seconds() > Config.REFRESH_MARKETS:
                    self.markets = get_markets()
                    self.last_refresh = utcnow()
                    log.info(f"Markets refreshed: {len(self.markets)}")

                if (utcnow() - self.last_reconcile).total_seconds() > Config.RECONCILE_OPEN_ORDERS:
                    self.reconcile_open_orders()
                    self.last_reconcile = utcnow()

                bal = get_balance(self.client)
                if not bal:
                    time.sleep(5)
                    continue

                # 1) detect fills and queue exits
                self.detect_fills_and_queue_exits()

                # 2) place queued TP exits
                self.place_exits(bal)

                # 3) place new stink buys if within risk
                self.maybe_place_stink_buys(bal)

                time.sleep(Config.LOOP_SLEEP)

            except Exception as e:
                log.error(f"Loop error: {e}")
                time.sleep(5)

# ================= ENTRY =================

if __name__ == "__main__":
    client = init_client()
    bot = StinkBotPlus(client)
    bot.run()
