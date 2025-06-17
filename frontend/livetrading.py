from nicegui import ui
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

async def render_live_trading_page(fetch_api, user_storage, broker):
    ui.label(f"Live Trading Activity Monitor - {broker}").classes("text-2xl font-semibold p-4")
    live_activity_container = ui.column().classes("w-full p-4 gap-4")

    async def refresh_live_activity():
        live_activity_container.clear()
        with live_activity_container:
            ui.label("Recent Trades").classes("text-xl font-semibold mb-2")
            recent_trades_area = ui.column().classes("w-full")
            trades = await fetch_api(f"/trade-history/{broker}", params={"limit": 10})
            if trades and isinstance(trades, list):
                if not trades:
                    with recent_trades_area:
                        ui.label("No recent trades.").classes("text-gray-500")
                else:
                    with recent_trades_area:
                        for trade in trades:
                            trade_time_str = trade.get('timestamp', datetime.now().isoformat())
                            try:
                                trade_time = datetime.fromisoformat(trade_time_str.replace("Z", "+00:00")).strftime(
                                    '%H:%M:%S')
                            except ValueError:
                                trade_time = trade_time_str
                            pnl_val = trade.get('pnl', 0)
                            pnl_color = 'text-green-500' if pnl_val >= 0 else 'text-red-500'
                            symbol_display = trade.get('tradingsymbol', trade.get('symbol', 'Unknown'))
                            with ui.card().classes("p-3 w-full shadow-sm"):
                                ui.label(
                                    f"[{trade_time}] {trade.get('type', '')} {symbol_display} @ {trade.get('price', 'N/A'):.2f} | Qty: {trade.get('qty', 'N/A')} | P&L: <span class='{pnl_color} font-semibold'>{pnl_val:,.2f}</span>").classes(
                                    "text-sm").props("html")
            else:
                with recent_trades_area:
                    ui.label("Could not fetch recent trades.").classes("text-orange-600")
            ui.label("Running Automated Strategies").classes("text-xl font-semibold mt-6 mb-2")
            running_strats_area = ui.column().classes("w-full")
            with running_strats_area:
                ui.label("Automated strategy status monitoring coming soon.").classes("text-gray-500")

    await refresh_live_activity()
    ui.timer(10, refresh_live_activity)