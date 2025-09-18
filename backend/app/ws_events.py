from typing import Dict, Set, Any, Callable, Awaitable
import asyncio
import logging

logger = logging.getLogger(__name__)

# Registry of user_id -> set of websockets
_order_ws_clients: Dict[str, Set[Any]] = {}
_lock = asyncio.Lock()

async def register_client(user_id: str, websocket: Any) -> None:
    async with _lock:
        _order_ws_clients.setdefault(user_id, set()).add(websocket)
        logger.info(f"Registered orders WebSocket for user {user_id}. Total: {len(_order_ws_clients[user_id])}")

async def unregister_client(user_id: str, websocket: Any) -> None:
    async with _lock:
        clients = _order_ws_clients.get(user_id)
        if clients and websocket in clients:
            clients.remove(websocket)
            logger.info(f"Unregistered orders WebSocket for user {user_id}. Remaining: {len(clients)}")
        if clients is not None and len(clients) == 0:
            _order_ws_clients.pop(user_id, None)

async def broadcast_event(user_id: str, event: str, payload: Dict[str, Any]) -> None:
    """Broadcast an order-related event to all WebSocket clients for this user."""
    message = {"event": event, **payload}
    async with _lock:
        clients = list(_order_ws_clients.get(user_id, set()))
    if not clients:
        return
    to_remove = []
    for ws in clients:
        try:
            # FastAPI WebSocket has send_json; aiohttp WS has send_json/can send str
            if hasattr(ws, 'send_json'):
                await ws.send_json(message)
            elif hasattr(ws, 'send_str'):
                import json
                await ws.send_str(json.dumps(message))
            else:
                logger.debug("Unknown websocket type; skipping send")
        except Exception as e:
            logger.warning(f"Error sending WS event to user {user_id}: {e}")
            to_remove.append(ws)
    # Cleanup failed clients
    if to_remove:
        async with _lock:
            s = _order_ws_clients.get(user_id)
            if s:
                for ws in to_remove:
                    s.discard(ws)
                if not s:
                    _order_ws_clients.pop(user_id, None)

