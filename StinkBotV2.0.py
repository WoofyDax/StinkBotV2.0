#!/usr/bin/env python3
"""
POLYMARKET STINK BID BOT
Ultra-low bids (1c / 2c / 3c) hunting fat-finger & panic sells.

NO telegram
NO prediction
Pure orderbook convexity

RISK IS CAPPED BY BALANCE
"""

import os
import sys
import math
import time
import asyncio
import logging
import requests
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY, SELL

# ================= CONFIG =================

class Config:
    GAMMA_API = "https://gamma-api.polymarket.com"
    CLOB_API = "https://clob.polymarket.com"

    # stink ladder
    STINK_PRICES = [0.01, 0.02, 0.03]

    # selection filters
    MIN_24H_VOLUME = 15_000
    MAX_BID_DEPTH_ABOVE = 3_000     # USD depth above stink price
    MIN_SPREAD = 0.05
    MIN_TIME_TO_END_MIN = 120       # avoid imminent resolution

    # risk
    MAX_OPEN_EXPOSURE_FRAC = 0.20   # max 20% of balance deployed
    MAX_PER_MARKET_FRAC = 0.05
    HARD_MAX_EXPOSURE = 200.0

    # order rules
    MIN_NOTIONAL = 1.00
    MIN_SHARES = 5
    STALE_SECONDS = 180

    LOOP_SLEEP = 6
    REFRESH_MARKETS = 120

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# ================= LOGGING =================

logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("STINK")

# ================= UTIL =================

def utcnow():
    return datetime.now(timezone.utc)

def must_env(k):
    v = os.getenv(k)
    if not v:
        raise RuntimeError(f"Missing env var {k}")
    return v

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

def extract_token(mkt):
    for o in mkt.get("outcomes", []):
        title = (o.get("title") or "").lower()
        if "yes" in title or "up" in title:
            return o["token"]["token_id"]
    return None

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

def bid_depth_above(book, px):
    depth = 0.0
    for p, s in book.get("bids", []):
        if float(p) >= px:
            depth += float(p) * float(s)
    return depth

# ================= BALANCE =================

def get_balance(client):
    try:
        return float(client.get_balance())
    except:
        return None

# ================= BOT =================

@dataclass
class OpenOrder:
    order_id: str
    price: float
    size: float
    created: datetime

class StinkBot:
    def __init__(self, client):
        self.client = client
        self.open_orders: Dict[str, OpenOrder] = {}
        self.last_refresh = utcnow()

    def cleanup(self):
        now = utcnow()
        for oid, o in list(self.open_orders.items()):
            if (now - o.created).total_seconds() > Config.STALE_SECONDS:
                try:
                    self.client.cancel_order(oid)
                    log.info(f"Canceled stale {oid}")
                except:
                    pass
                self.open_orders.pop(oid, None)

    def run(self):
        while True:
            try:
                self.cleanup()

                if (utcnow() - self.last_refresh).total_seconds() > Config.REFRESH_MARKETS:
                    self.markets = get_markets()
                    self.last_refresh = utcnow()
                    log.info(f"Markets refreshed: {len(self.markets)}")

                bal = get_balance(self.client)
                if not bal:
                    time.sleep(5)
                    continue

                max_exposure = min(
                    Config.HARD_MAX_EXPOSURE,
                    bal * Config.MAX_OPEN_EXPOSURE_FRAC
                )

                deployed = sum(o.price * o.size for o in self.open_orders.values())
                if deployed >= max_exposure:
                    time.sleep(Config.LOOP_SLEEP)
                    continue

                for mkt in self.markets:
                    if not market_ok(mkt):
                        continue

                    token = extract_token(mkt)
                    if not token:
                        continue

                    book = get_book(token)
                    bid = float(book["bids"][0][0]) if book["bids"] else None
                    ask = float(book["asks"][0][0]) if book["asks"] else None
                    if not bid or not ask or ask - bid < Config.MIN_SPREAD:
                        continue

                    for px in Config.STINK_PRICES:
                        depth = bid_depth_above(book, px)
                        if depth > Config.MAX_BID_DEPTH_ABOVE:
                            continue

                        per_mkt_cap = bal * Config.MAX_PER_MARKET_FRAC
                        budget = min(per_mkt_cap, max_exposure - deployed)
                        if budget < Config.MIN_NOTIONAL:
                            continue

                        size = max(
                            Config.MIN_SHARES,
                            int(math.ceil(Config.MIN_NOTIONAL / px))
                        )
                        if px * size > budget:
                            continue

                        resp = self.client.create_and_post_order(
                            OrderArgs(
                                token_id=token,
                                price=px,
                                size=size,
                                side=BUY
                            ),
                            order_type=OrderType.GTC
                        )
                        oid = resp.get("orderID") or resp.get("id")
                        if oid:
                            self.open_orders[oid] = OpenOrder(
                                oid, px, size, utcnow()
                            )
                            deployed += px * size
                            log.info(
                                f"STINK BID {px:.2f} size={size} "
                                f"depth_above=${depth:.0f} vol24h=${mkt['volume24hr']}"
                            )

                        if deployed >= max_exposure:
                            break

                time.sleep(Config.LOOP_SLEEP)

            except Exception as e:
                log.error(f"Loop error: {e}")
                time.sleep(5)

# ================= ENTRY =================

if __name__ == "__main__":
    client = init_client()
    bot = StinkBot(client)
    bot.run()
