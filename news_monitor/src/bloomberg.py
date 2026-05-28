"""
Bloomberg Market Confirmation — connects to Bloomberg DAPI via blpapi
to stream real-time market data and confirm/deny news-driven factor moves.

If Bloomberg is not available, operates in "offline" mode with no market data.
"""

import logging
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Callable

from .taxonomy import get_taxonomy

logger = logging.getLogger(__name__)

# Try to import blpapi — gracefully degrade if unavailable
try:
    import blpapi

    BLPAPI_AVAILABLE = True
except ImportError:
    BLPAPI_AVAILABLE = False
    logger.info("blpapi not available — Bloomberg market confirmation disabled")


class MarketDataCache:
    """Thread-safe cache of recent market data snapshots."""

    def __init__(self):
        self._lock = threading.Lock()
        # ticker -> list of {timestamp, field, value}
        self._data: dict[str, list[dict]] = defaultdict(list)
        self._max_age_hours = 24

    def update(self, ticker: str, field: str, value: float):
        with self._lock:
            self._data[ticker].append({
                "timestamp": datetime.utcnow().isoformat(),
                "field": field,
                "value": value,
            })
            # Prune old entries
            cutoff = (datetime.utcnow() - timedelta(hours=self._max_age_hours)).isoformat()
            self._data[ticker] = [
                d for d in self._data[ticker] if d["timestamp"] > cutoff
            ]

    def get_latest(self, ticker: str) -> dict | None:
        with self._lock:
            entries = self._data.get(ticker, [])
            return entries[-1] if entries else None

    def get_change(self, ticker: str, hours: int = 1) -> float | None:
        """Get price/yield change over last N hours."""
        with self._lock:
            entries = self._data.get(ticker, [])
            if len(entries) < 2:
                return None
            cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
            old_entries = [e for e in entries if e["timestamp"] <= cutoff]
            if not old_entries:
                old_val = entries[0]["value"]
            else:
                old_val = old_entries[-1]["value"]
            new_val = entries[-1]["value"]
            if old_val == 0:
                return None
            return (new_val - old_val) / abs(old_val) * 100  # percent change

    def get_all_latest(self) -> dict[str, dict]:
        with self._lock:
            result = {}
            for ticker, entries in self._data.items():
                if entries:
                    result[ticker] = entries[-1]
            return result


# Module-level cache
_market_cache = MarketDataCache()


class BloombergStreamer:
    """
    Manages Bloomberg DAPI subscription for real-time data.
    Subscribes to tickers defined in taxonomy.yaml.
    """

    def __init__(self, host: str = "localhost", port: int = 8194):
        if not BLPAPI_AVAILABLE:
            raise RuntimeError("blpapi is not installed. Install Bloomberg API SDK.")
        self._host = host
        self._port = port
        self._session: Any = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._callbacks: list[Callable] = []

    def start(self):
        """Start the Bloomberg session and subscribe to tickers."""
        session_opts = blpapi.SessionOptions()
        session_opts.setServerHost(self._host)
        session_opts.setServerPort(self._port)

        self._session = blpapi.Session(session_opts)
        if not self._session.start():
            raise RuntimeError(f"Failed to connect to Bloomberg at {self._host}:{self._port}")
        if not self._session.openService("//blp/mktdata"):
            raise RuntimeError("Failed to open //blp/mktdata service")

        # Subscribe to all tickers from taxonomy
        taxonomy = get_taxonomy()
        all_tickers = taxonomy.get_all_tickers_flat()
        subscriptions = blpapi.SubscriptionList()
        for t in all_tickers:
            topic = f"//blp/mktdata/{t['ticker']}"
            subscriptions.add(topic, "LAST_PRICE,NET_CHANGE,PCT_CHANGE", "", blpapi.CorrelationId(t["ticker"]))

        self._session.subscribe(subscriptions)
        self._running = True
        self._thread = threading.Thread(target=self._event_loop, daemon=True)
        self._thread.start()
        logger.info(f"Bloomberg streamer started — subscribed to {len(all_tickers)} tickers")

    def stop(self):
        self._running = False
        if self._session:
            self._session.stop()
        if self._thread:
            self._thread.join(timeout=5)

    def add_callback(self, fn: Callable):
        """Register a callback for market data updates: fn(ticker, field, value)"""
        self._callbacks.append(fn)

    def _event_loop(self):
        while self._running:
            event = self._session.nextEvent(500)
            for msg in event:
                if event.eventType() in (
                    blpapi.Event.SUBSCRIPTION_DATA,
                    blpapi.Event.PARTIAL_RESPONSE,
                ):
                    ticker = str(msg.correlationIds()[0].value())
                    for field in ["LAST_PRICE", "NET_CHANGE", "PCT_CHANGE"]:
                        if msg.hasElement(field):
                            value = msg.getElementAsFloat(field)
                            _market_cache.update(ticker, field, value)
                            for cb in self._callbacks:
                                try:
                                    cb(ticker, field, value)
                                except Exception as e:
                                    logger.error(f"Callback error: {e}")


def get_market_cache() -> MarketDataCache:
    """Get the module-level market data cache."""
    return _market_cache


def compute_market_confirmation(
    tags: list[dict[str, Any]],
    lookback_hours: int = 1,
) -> list[dict[str, Any]]:
    """
    For each article tag, check if the relevant Bloomberg tickers have moved
    in the expected direction. Returns enhanced tags with market_confirmation field.

    market_confirmation: 'confirmed', 'contradicted', or 'no_data'
    """
    cache = get_market_cache()
    enhanced = []

    for tag in tags:
        tickers = tag.get("bloomberg_tickers", [])
        confirmations = 0
        contradictions = 0
        no_data = 0

        ticker_moves = []
        for t in tickers:
            ticker_id = t["ticker"]
            change = cache.get_change(ticker_id, hours=lookback_hours)
            if change is None:
                no_data += 1
                ticker_moves.append({"ticker": ticker_id, "change_pct": None, "status": "no_data"})
                continue

            expected = t.get("expected_direction", "unknown")
            actual_dir = "up" if change > 0.1 else ("down" if change < -0.1 else "flat")

            if expected == actual_dir:
                confirmations += 1
                status = "confirmed"
            elif expected != "unknown" and expected != "neutral" and actual_dir != "flat":
                contradictions += 1
                status = "contradicted"
            else:
                status = "neutral"

            ticker_moves.append({
                "ticker": ticker_id,
                "change_pct": round(change, 3),
                "expected": expected,
                "actual": actual_dir,
                "status": status,
            })

        # Overall confirmation status
        if confirmations > contradictions and confirmations > 0:
            overall = "confirmed"
            conf_boost = 0.1
        elif contradictions > confirmations and contradictions > 0:
            overall = "contradicted"
            conf_boost = -0.15
        else:
            overall = "no_data"
            conf_boost = 0.0

        enhanced_tag = dict(tag)
        enhanced_tag["market_confirmation"] = overall
        enhanced_tag["ticker_moves"] = ticker_moves
        # Adjust confidence
        new_conf = tag.get("confidence", 0.5) + conf_boost
        enhanced_tag["confidence"] = round(min(max(new_conf, 0.05), 1.0), 3)
        enhanced.append(enhanced_tag)

    return enhanced


def get_key_market_movers(hours: int = 1, threshold_pct: float = 0.5) -> list[dict]:
    """
    Get tickers that have moved more than threshold in the last N hours.
    Returns list of {ticker, name, change_pct, category}.
    """
    taxonomy = get_taxonomy()
    cache = get_market_cache()
    all_tickers = taxonomy.get_all_tickers_flat()
    movers = []

    for t in all_tickers:
        change = cache.get_change(t["ticker"], hours=hours)
        if change is not None and abs(change) >= threshold_pct:
            movers.append({
                "ticker": t["ticker"],
                "name": t["name"],
                "change_pct": round(change, 3),
                "category": t.get("category", "other"),
                "factor": t.get("factor", ""),
                "themes": t.get("themes", []),
            })

    movers.sort(key=lambda x: abs(x["change_pct"]), reverse=True)
    return movers
