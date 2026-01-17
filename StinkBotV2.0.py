#!/usr/bin/env python3
"""
POLYMARKET STINK BID BOT (HARDENED / "ALL GOTCHAS")

Core idea:
- Post low "stink" BUY bids (1c/2c/3c) on both outcomes when book convexity looks favorable.
- If filled, automatically place a take-profit SELL near (ask - 1c) or at least +edge.
- Strict risk caps based on balance, plus per-market caps.
- Avoid duplicates robustly (canonical tick prices + exchange reconciliation).
- Persist state to disk for restart safety.
- Retry/backoff for flaky APIs.

NO telegram
NO prediction
Pure orderbook convexity

RISK IS CAPPED BY BALANCE
"""

import os
import sys
import math
import time
import json
import random
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from decimal import Decimal, getcontext, ROUND_FLOOR, ROUND_CEILING, ROUND_HALF_UP

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY, SELL


# ================= CONFIG =================

class Config:
    GAMMA_API = "https://gamma-api.polymarket.com"
    CLOB_API  = "https://clob.polymarket.com"

    # stink ladder (as strings so Decimal is exact)
    STINK_PRICES = ["0.01", "0.02", "0.03"]

    # selection filters
    MIN_24H_VOLUME = 15_000
    MAX_BID_DEPTH_ABOVE = 3_000     # USD depth above stink price
    MIN_SPREAD = Decimal("0.05")
    MIN_TIME_TO_END_MIN = 120       # avoid imminent resolution

    # risk
    MAX_OPEN_EXPOSURE_FRAC = Decimal("0.20")   # max 20% of balance deployed
    MAX_PER_MARKET_FRAC    = Decimal("0.05")
    HARD_MAX_EXPOSURE      = Decimal("200.0")

    # order rules
    MIN_NOTIONAL = Decimal("1.00")
    MIN_SHARES   = Decimal("5")
    STALE_SECONDS = 180

    # exits
    TAKE_PROFIT_MIN_EDGE = Decimal("0.08")     # at least +8c over fill price when possible
    TAKE_PROFIT_UNDER_ASK = Decimal("0.01")    # sell at (best_ask - 1c) when possible
    PRICE_TICK = Decimal("0.01")               # 1-cent ticks
    MIN_PRICE = Decimal("0.01")
    MAX_PRICE = Decimal("0.99")

    LOOP_SLEEP = 6
    REFRESH_MARKETS = 120
    RECONCILE_OPEN_ORDERS = 60
    CHECK_FILLS_EVERY = 1  # check fills each loop (can raise if rate-limited)

    # networking
    HTTP_TIMEOUT = 12
    HTTP_RETRIES = 3
    HTTP_BACKOFF = 0.4

    # state persistence
    STATE_FILE = os.getenv("STINK_STATE_FILE", "stink_state.json")

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


# Decimal precision
getcontext().prec = 28


# ================= LOGGING =================

logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("STINK-HARDENED")


# ================= UTIL =================

def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def must_env(k: str) -> str:
    v = os.getenv(k)
    if not v:
        raise RuntimeError(f"Missing env var {k}")
    return v

def env_int(k: str, default: int) -> int:
    v = os.getenv(k)
    if not v:
        return default
    return int(v)

def clamp_dec(x: Decimal, lo: Decimal, hi: Decimal) -> Decimal:
    return max(lo, min(hi, x))

def tick_quantize(px: Decimal) -> Decimal:
    """
    Canonicalize price to the exchange tick (1c).
    ROUND_HALF_UP to nearest tick.
    """
    t = Config.PRICE_TICK
    q = (px / t).to_integral_value(rounding=ROUND_HALF_UP) * t
    q = clamp_dec(q, Config.MIN_PRICE, Config.MAX_PRICE)
    # normalize exponent
    return q.quantize(t)

def floor_to_tick(px: Decimal) -> Decimal:
    t = Config.PRICE_TICK
    q = (px / t).to_integral_value(rounding=ROUND_FLOOR) * t
    q = clamp_dec(q, Config.MIN_PRICE, Config.MAX_PRICE)
    return q.quantize(t)

def ceil_to_tick(px: Decimal) -> Decimal:
    t = Config.PRICE_TICK
    q = (px / t).to_integral_value(rounding=ROUND_CEILING) * t
    q = clamp_dec(q, Config.MIN_PRICE, Config.MAX_PRICE)
    return q.quantize(t)

def safe_iso(dt_str: str) -> Optional[datetime]:
    if not dt_str:
        return None
    try:
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except Exception:
        return None


# ================= HTTP SESSION (retries) =================

def make_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=Config.HTTP_RETRIES,
        backoff_factor=Config.HTTP_BACKOFF,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    return s


# ================= CLIENT =================

def init_client() -> ClobClient:
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

def get_markets(sess: requests.Session) -> List[dict]:
    r = sess.get(
        f"{Config.GAMMA_API}/markets",
        params=dict(active=True, closed=False, limit=400),
        timeout=Config.HTTP_TIMEOUT
    )
    r.raise_for_status()
    j = r.json()
    return j.get("markets", [])


def market_ok(mkt: dict) -> bool:
    if float(mkt.get("volume24hr", 0) or 0) < Config.MIN_24H_VOLUME:
        return False
    end = mkt.get("endDate")
    dt = safe_iso(end) if end else None
    if dt:
        if (dt - utcnow()).total_seconds() < Config.MIN_TIME_TO_END_MIN * 60:
            return False
    return True


def extract_tokens_both_sides(mkt: dict) -> List[str]:
    """
    More robust token discovery:
    - First try YES/NO or UP/DOWN title matching
    - If that fails but there are exactly 2 outcomes w/ token_ids, return both
    """
    outs = mkt.get("outcomes", []) or []
    found = []
    yes = None
    no = None

    def tokid(o: dict) -> Optional[str]:
        t = o.get("token") or {}
        return t.get("token_id") or t.get("tokenId") or o.get("token_id") or o.get("tokenId")

    for o in outs:
        title = (o.get("title") or "").strip().lower()
        tid = tokid(o)
        if not tid:
            continue
        if ("yes" in title) or ("up" in title) or (title == "y"):
            yes = tid
        elif ("no" in title) or ("down" in title) or (title == "n"):
            no = tid

    if yes:
        found.append(yes)
    if no and no != yes:
        found.append(no)

    # Fallback: exactly 2 outcomes
    if len(found) == 0:
        tids = []
        for o in outs:
            tid = tokid(o)
            if tid:
                tids.append(tid)
        tids = list(dict.fromkeys(tids))
        if len(tids) == 2:
            return tids

    return found


# ================= ORDERBOOK =================

def get_book(sess: requests.Session, token_id: str) -> dict:
    r = sess.get(f"{Config.CLOB_API}/book/{token_id}", timeout=Config.HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()

def best_bid_ask(book: dict) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    bid = None
    ask = None
    try:
        if book.get("bids"):
            bid = Decimal(str(book["bids"][0][0]))
        if book.get("asks"):
            ask = Decimal(str(book["asks"][0][0]))
    except Exception:
        pass
    return bid, ask

def bid_depth_above(book: dict, px: Decimal) -> Decimal:
    depth = Decimal("0")
    for p, s in book.get("bids", []) or []:
        try:
            pd = Decimal(str(p))
            sd = Decimal(str(s))
        except Exception:
            continue
        if pd >= px:
            depth += pd * sd
    return depth


# ================= BOT STATE =================

@dataclass
class TrackedOrder:
    order_id: str
    token_id: str
    side: str         # BUY or SELL
    price: str        # store as string for exact persistence
    size: str         # store as string for exact persistence
    created: str      # iso
    filled_size: str = "0"

    @staticmethod
    def from_live(order_id: str, token_id: str, side: str, price: Decimal, size: Decimal, filled: Decimal) -> "TrackedOrder":
        return TrackedOrder(
            order_id=order_id,
            token_id=token_id,
            side=side,
            price=str(tick_quantize(price)),
            size=str(size),
            created=utcnow().isoformat(),
            filled_size=str(filled)
        )

    def price_dec(self) -> Decimal:
        return Decimal(self.price)

    def size_dec(self) -> Decimal:
        return Decimal(self.size)

    def filled_dec(self) -> Decimal:
        return Decimal(self.filled_size)

    def created_dt(self) -> datetime:
        return safe_iso(self.created) or utcnow()


@dataclass
class PendingExit:
    token_id: str
    size: str
    target_price: str

    def size_dec(self) -> Decimal:
        return Decimal(self.size)

    def price_dec(self) -> Decimal:
        return Decimal(self.target_price)


class StinkBotHardened:
    def __init__(self, client: ClobClient):
        self.client = client
        self.sess = make_session()

        # Keyed by (token_id, side, canonical_price_string)
        self.open_orders: Dict[Tuple[str, str, str], TrackedOrder] = {}

        # exits aggregated per (token_id, target_price)
        self.exit_queue: Dict[Tuple[str, str], Decimal] = {}

        self.last_refresh = utcnow()
        self.last_reconcile = utcnow()
        self.markets: List[dict] = []

        self._load_state()

    # -------- persistence --------

    def _load_state(self):
        fn = Config.STATE_FILE
        if not os.path.exists(fn):
            return
        try:
            with open(fn, "r", encoding="utf-8") as f:
                st = json.load(f)
            oo = st.get("open_orders", {})
            self.open_orders = {}
            for k, v in oo.items():
                # key is "token|side|price"
                parts = k.split("|")
                if len(parts) != 3:
                    continue
                token, side, price = parts
                self.open_orders[(token, side, price)] = TrackedOrder(**v)

            eq = st.get("exit_queue", {})
            self.exit_queue = {}
            for k, v in eq.items():
                # key "token|price"
                parts = k.split("|")
                if len(parts) != 2:
                    continue
                token, price = parts
                self.exit_queue[(token, price)] = Decimal(str(v))

            log.info(f"Loaded state: {len(self.open_orders)} open orders, {len(self.exit_queue)} queued exits")
        except Exception as e:
            log.warning(f"Failed to load state ({fn}): {e}")

    def _save_state(self):
        fn = Config.STATE_FILE
        try:
            oo = {}
            for (token, side, price), o in self.open_orders.items():
                oo[f"{token}|{side}|{price}"] = asdict(o)
            eq = {}
            for (token, price), sz in self.exit_queue.items():
                eq[f"{token}|{price}"] = str(sz)

            tmp = {
                "saved_at": utcnow().isoformat(),
                "open_orders": oo,
                "exit_queue": eq,
            }
            with open(fn, "w", encoding="utf-8") as f:
                json.dump(tmp, f, indent=2)
        except Exception as e:
            log.warning(f"Failed to save state ({fn}): {e}")

    # -------- exchange helpers --------

    def _cancel(self, order_id: str):
        return self.client.cancel_order(order_id)

    def _place_order(self, token_id: str, side: str, price: Decimal, size: Decimal) -> Optional[str]:
        price = tick_quantize(price)
        resp = self.client.create_and_post_order(
            OrderArgs(token_id=token_id, price=float(price), size=float(size), side=side),
            order_type=OrderType.GTC
        )
        oid = resp.get("orderID") or resp.get("id") or resp.get("order_id")
        return oid

    def _get_open_orders_exchange(self) -> List[dict]:
        for name in ("get_open_orders", "get_orders", "list_orders"):
            fn = getattr(self.client, name, None)
            if callable(fn):
                try:
                    res = fn()
                    if isinstance(res, dict):
                        for k in ("orders", "data", "results"):
                            if k in res and isinstance(res[k], list):
                                return res[k]
                    if isinstance(res, list):
                        return res
                except Exception:
                    pass
        return []

    def _get_order_status(self, order_id: str) -> Optional[dict]:
        for name in ("get_order", "get_order_by_id", "read_order"):
            fn = getattr(self.client, name, None)
            if callable(fn):
                try:
                    return fn(order_id)
                except Exception:
                    pass
        return None

    def _get_positions(self) -> Dict[str, Decimal]:
        """
        Try to get current position sizes by token_id.
        If unavailable, return {} and we fall back to conservative local accounting.
        """
        for name in ("get_positions", "get_holdings", "list_holdings", "get_token_balances"):
            fn = getattr(self.client, name, None)
            if callable(fn):
                try:
                    res = fn()
                    pos: Dict[str, Decimal] = {}
                    # Try a few plausible formats
                    if isinstance(res, dict):
                        items = res.get("positions") or res.get("holdings") or res.get("data") or res.get("results") or []
                    else:
                        items = res if isinstance(res, list) else []
                    for it in items:
                        tid = it.get("token_id") or it.get("tokenId") or it.get("token")
                        sz = it.get("size") or it.get("balance") or it.get("quantity") or 0
                        if tid:
                            try:
                                pos[tid] = Decimal(str(sz))
                            except Exception:
                                continue
                    return pos
                except Exception:
                    pass
        return {}

    def get_balance(self) -> Optional[Decimal]:
        try:
            return Decimal(str(self.client.get_balance()))
        except Exception:
            return None

    # -------- risk/exposure --------

    def deployed_notional_local(self) -> Decimal:
        dep = Decimal("0")
        for o in self.open_orders.values():
            if o.side == BUY:
                dep += o.price_dec() * o.size_dec()
        return dep

    def deployed_notional_exchange_if_possible(self) -> Optional[Decimal]:
        exch = self._get_open_orders_exchange()
        if not exch:
            return None
        dep = Decimal("0")
        for od in exch:
            side = od.get("side")
            if side != BUY:
                continue
            try:
                px = Decimal(str(od.get("price")))
                sz = Decimal(str(od.get("size")))
                dep += tick_quantize(px) * sz
            except Exception:
                continue
        return dep

    # -------- gotcha-proof maintenance --------

    def cleanup_stale_buys(self):
        now = utcnow()
        for key, o in list(self.open_orders.items()):
            if o.side != BUY:
                continue
            age = (now - o.created_dt()).total_seconds()
            if age > Config.STALE_SECONDS:
                try:
                    self._cancel(o.order_id)
                    log.info(f"Canceled stale BUY {o.order_id} ({o.token_id} @ {o.price})")
                except Exception:
                    pass
                self.open_orders.pop(key, None)

    def reconcile_open_orders(self):
        """
        Sync local open_orders with exchange, using canonical tick prices.
        Removes orders no longer live; adds unknown live orders.
        """
        exch = self._get_open_orders_exchange()
        if exch is None:
            return

        live_ids = set()
        live_keys = set()

        for od in exch:
            oid = od.get("orderID") or od.get("id") or od.get("order_id")
            token = od.get("token_id") or od.get("tokenId")
            side  = od.get("side")
            price = od.get("price")
            size  = od.get("size")
            if not (oid and token and side and price and size):
                continue

            try:
                px = tick_quantize(Decimal(str(price)))
                sz = Decimal(str(size))
                filled = Decimal(str(od.get("filled_size") or od.get("filledSize") or 0))
            except Exception:
                continue

            live_ids.add(oid)
            k = (token, side, str(px))
            live_keys.add(k)

            if k not in self.open_orders:
                self.open_orders[k] = TrackedOrder.from_live(
                    order_id=oid, token_id=token, side=side, price=px, size=sz, filled=filled
                )

        # drop locals not live
        for k, o in list(self.open_orders.items()):
            if o.order_id not in live_ids:
                self.open_orders.pop(k, None)

        log.info(f"Reconciled open orders: {len(self.open_orders)}")
        self._save_state()

    # -------- fills + exits --------

    def compute_take_profit(self, fill_px: Decimal, book: dict) -> Decimal:
        bid, ask = best_bid_ask(book)
        target = fill_px + Config.TAKE_PROFIT_MIN_EDGE
        if ask is not None:
            target = max(target, ask - Config.TAKE_PROFIT_UNDER_ASK)
        target = clamp_dec(target, Config.MIN_PRICE, Config.MAX_PRICE)
        return tick_quantize(target)

    def queue_exit(self, token_id: str, target_px: Decimal, delta_size: Decimal):
        if delta_size <= 0:
            return
        px = str(tick_quantize(target_px))
        k = (token_id, px)
        self.exit_queue[k] = self.exit_queue.get(k, Decimal("0")) + delta_size

    def detect_fills_and_queue_exits(self):
        """
        For every tracked BUY order, pull status and detect fill deltas.
        Queue exit sells aggregated by (token, price).
        """
        for k, o in list(self.open_orders.items()):
            if o.side != BUY:
                continue

            st = self._get_order_status(o.order_id)
            if not st:
                continue

            filled_raw = st.get("filled_size") or st.get("filledSize") or st.get("filled") or 0
            try:
                filled = Decimal(str(filled_raw))
            except Exception:
                filled = o.filled_dec()

            prev = o.filled_dec()
            delta = filled - prev

            if delta > 0:
                # compute TP
                try:
                    book = get_book(self.sess, o.token_id)
                    tp = self.compute_take_profit(o.price_dec(), book)
                except Exception:
                    tp = tick_quantize(clamp_dec(o.price_dec() + Config.TAKE_PROFIT_MIN_EDGE, Config.MIN_PRICE, Config.MAX_PRICE))

                self.queue_exit(o.token_id, tp, delta)

                o.filled_size = str(filled)
                self.open_orders[k] = o

                log.info(f"FILL: {o.token_id} BUY @ {o.price} +{delta}sh (total={filled}). Queue TP @ {tp}")

            status = (st.get("status") or "").lower()
            if status in ("filled", "canceled", "cancelled", "rejected", "expired"):
                self.open_orders.pop(k, None)

        self._save_state()

    def place_exits(self):
        """
        Place TP SELLs from the aggregated exit queue.
        - Enforces min shares / min notional.
        - Avoids duplicates (token, SELL, price).
        - Attempts to avoid overselling by checking positions if possible.
        """
        if not self.exit_queue:
            return

        # If we can, use real positions to cap sells
        pos = self._get_positions()  # token -> size
        have_positions = bool(pos)

        new_queue: Dict[Tuple[str, str], Decimal] = {}

        for (token, px_str), size in list(self.exit_queue.items()):
            px = Decimal(px_str)
            size = Decimal(size)

            if size <= 0:
                continue

            # enforce minimums
            notional = px * size
            if size < Config.MIN_SHARES or notional < Config.MIN_NOTIONAL:
                new_queue[(token, px_str)] = new_queue.get((token, px_str), Decimal("0")) + size
                continue

            # oversell guard: cap to available position if we can read it
            if have_positions:
                available = pos.get(token, Decimal("0"))
                if available <= 0:
                    # no inventory (or cannot see it), keep queued but don't place
                    new_queue[(token, px_str)] = new_queue.get((token, px_str), Decimal("0")) + size
                    continue
                if size > available:
                    # place only what we have, keep remainder queued
                    place_size = available
                    rem = size - available
                else:
                    place_size = size
                    rem = Decimal("0")
            else:
                place_size = size
                rem = Decimal("0")

            # avoid duplicates
            key = (token, SELL, px_str)
            if key in self.open_orders:
                # already have a TP at this price; keep remainder queued (if any)
                if rem > 0:
                    new_queue[(token, px_str)] = new_queue.get((token, px_str), Decimal("0")) + rem
                continue

            try:
                oid = self._place_order(token, SELL, px, place_size)
                if oid:
                    self.open_orders[key] = TrackedOrder(
                        order_id=oid,
                        token_id=token,
                        side=SELL,
                        price=px_str,
                        size=str(place_size),
                        created=utcnow().isoformat(),
                        filled_size="0"
                    )
                    log.info(f"TP SELL placed: {token} @ {px_str} size={place_size}")
                else:
                    new_queue[(token, px_str)] = new_queue.get((token, px_str), Decimal("0")) + size
                    continue
            except Exception as e:
                log.warning(f"Failed placing TP SELL ({token} @ {px_str}): {e}")
                new_queue[(token, px_str)] = new_queue.get((token, px_str), Decimal("0")) + size
                continue

            if rem > 0:
                new_queue[(token, px_str)] = new_queue.get((token, px_str), Decimal("0")) + rem

        self.exit_queue = new_queue
        self._save_state()

    # -------- placing stink buys --------

    def maybe_place_stink_buys(self, bal: Decimal):
        # risk caps
        max_exposure = min(Config.HARD_MAX_EXPOSURE, bal * Config.MAX_OPEN_EXPOSURE_FRAC)

        dep_exch = self.deployed_notional_exchange_if_possible()
        deployed = dep_exch if dep_exch is not None else self.deployed_notional_local()

        if deployed >= max_exposure:
            return

        for mkt in self.markets:
            if not market_ok(mkt):
                continue

            tokens = extract_tokens_both_sides(mkt)
            if not tokens:
                continue

            per_mkt_cap = bal * Config.MAX_PER_MARKET_FRAC

            for token in tokens:
                # pull book once per token per loop
                try:
                    book = get_book(self.sess, token)
                except Exception:
                    continue

                bid, ask = best_bid_ask(book)
                if bid is None or ask is None:
                    continue
                spread = ask - bid
                if spread < Config.MIN_SPREAD:
                    continue

                for px_s in Config.STINK_PRICES:
                    px = Decimal(px_s)
                    px = tick_quantize(px)

                    # duplicate prevention (canonical tick price string)
                    key = (token, BUY, str(px))
                    if key in self.open_orders:
                        continue

                    depth = bid_depth_above(book, px)
                    if depth > Decimal(str(Config.MAX_BID_DEPTH_ABOVE)):
                        continue

                    budget = min(per_mkt_cap, max_exposure - deployed)
                    if budget < Config.MIN_NOTIONAL:
                        continue

                    # size is max(MIN_SHARES, ceil(MIN_NOTIONAL/px))
                    need = (Config.MIN_NOTIONAL / px).to_integral_value(rounding=ROUND_CEILING)
                    size = max(Config.MIN_SHARES, need)

                    if px * size > budget:
                        continue

                    try:
                        oid = self._place_order(token, BUY, px, size)
                        if oid:
                            self.open_orders[key] = TrackedOrder(
                                order_id=oid,
                                token_id=token,
                                side=BUY,
                                price=str(px),
                                size=str(size),
                                created=utcnow().isoformat(),
                                filled_size="0"
                            )
                            deployed += px * size
                            log.info(
                                f"STINK BUY {px} size={size} depth_above=${depth} "
                                f"vol24h=${mkt.get('volume24hr', 0)} token={token[:8]}..."
                            )
                            self._save_state()
                    except Exception as e:
                        log.warning(f"Order post failed ({token} BUY {px}): {e}")

                    if deployed >= max_exposure:
                        return

    # -------- main loop --------

    def run(self):
        self.markets = get_markets(self.sess)
        self.last_refresh = utcnow()
        log.info(f"Initial markets: {len(self.markets)}")

        # important: reconcile on boot so we don't duplicate after restart
        try:
            self.reconcile_open_orders()
        except Exception:
            pass

        loops = 0
        while True:
            try:
                loops += 1

                # cancel stale stink BUYs
                self.cleanup_stale_buys()

                # refresh markets
                if (utcnow() - self.last_refresh).total_seconds() > Config.REFRESH_MARKETS:
                    self.markets = get_markets(self.sess)
                    self.last_refresh = utcnow()
                    log.info(f"Markets refreshed: {len(self.markets)}")

                # reconcile exchange state
                if (utcnow() - self.last_reconcile).total_seconds() > Config.RECONCILE_OPEN_ORDERS:
                    self.reconcile_open_orders()
                    self.last_reconcile = utcnow()

                bal = self.get_balance()
                if bal is None or bal <= 0:
                    time.sleep(5)
                    continue

                # detect fills + queue exits (can be rate-limited; still best to do frequently)
                if Config.CHECK_FILLS_EVERY <= 1 or (loops % Config.CHECK_FILLS_EVERY == 0):
                    self.detect_fills_and_queue_exits()

                # place exits
                self.place_exits()

                # place new stink buys
                self.maybe_place_stink_buys(bal)

                # jitter to avoid synchronized bursts
                time.sleep(Config.LOOP_SLEEP + random.uniform(0, 0.5))

            except Exception as e:
                log.error(f"Loop error: {e}")
                time.sleep(5)


# ================= ENTRY =================

if __name__ == "__main__":
    client = init_client()
    bot = StinkBotHardened(client)
    bot.run()
