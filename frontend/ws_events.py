from typing import Callable, Awaitable, Any, Set
import asyncio

_order_callbacks: Set[Callable[[dict], Any]] = set()

def register_order_ws_callback(cb: Callable[[dict], Any]) -> None:
    _order_callbacks.add(cb)

def unregister_order_ws_callback(cb: Callable[[dict], Any]) -> None:
    _order_callbacks.discard(cb)

async def emit_order_event(data: dict) -> None:
    for cb in list(_order_callbacks):
        try:
            res = cb(data)
            if asyncio.iscoroutine(res):
                await res
        except Exception:
            # Keep going on errors
            pass

