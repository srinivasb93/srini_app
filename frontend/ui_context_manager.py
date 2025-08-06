"""
Context-aware UI utilities for NiceGUI application.
Handles UI operations from async tasks safely.
"""
import logging
import asyncio
from typing import Optional, Any, Callable
from nicegui import ui, context
from functools import wraps

logger = logging.getLogger(__name__)


class UIContextManager:
    """Manages UI context for async operations"""

    def __init__(self):
        self._notification_queue = asyncio.Queue()
        self._processing_notifications = False

    @staticmethod
    def safe_notify(message: str, notify_type: str = "info", **kwargs) -> bool:
        """
        Safely show a notification, handling context issues.
        Returns True if notification was shown, False if queued for later.
        """
        try:
            # Try to get current context
            current_client = context.client
            if current_client:
                ui.notify(message, type=notify_type, **kwargs)
                return True
        except RuntimeError as e:
            if "slot stack" in str(e).lower():
                logger.warning(f"No UI context available for notification: {message}")
                # Could implement a queue here if needed
                return False
            else:
                raise
        except Exception as e:
            logger.error(f"Unexpected error in safe_notify: {e}")
            return False

        return False

    @staticmethod
    def safe_ui_update(update_func: Callable, *args, **kwargs) -> bool:
        """
        Safely execute a UI update function with proper context handling.
        Returns True if update was executed, False otherwise.
        """
        try:
            current_client = context.client
            if current_client:
                update_func(*args, **kwargs)
                return True
        except RuntimeError as e:
            if "slot stack" in str(e).lower():
                logger.warning(f"No UI context available for update: {update_func.__name__}")
                return False
            else:
                raise
        except Exception as e:
            logger.error(f"Error in safe_ui_update: {e}")
            return False

        return False

    @staticmethod
    def with_ui_context(container_element=None):
        """
        Decorator to ensure UI operations have proper context.
        Usage: @UIContextManager.with_ui_context()
        """
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    if container_element:
                        with container_element:
                            return await func(*args, **kwargs)
                    else:
                        return await func(*args, **kwargs)
                except RuntimeError as e:
                    if "slot stack" in str(e).lower():
                        logger.warning(f"UI context error in {func.__name__}: {e}")
                        # Return None or appropriate default
                        return None
                    else:
                        raise
                except Exception as e:
                    logger.error(f"Error in {func.__name__}: {e}")
                    raise

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    if container_element:
                        with container_element:
                            return func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                except RuntimeError as e:
                    if "slot stack" in str(e).lower():
                        logger.warning(f"UI context error in {func.__name__}: {e}")
                        return None
                    else:
                        raise
                except Exception as e:
                    logger.error(f"Error in {func.__name__}: {e}")
                    raise

            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator


# Global instance
ui_context_manager = UIContextManager()


def safe_notify(message: str, notify_type: str = "info", **kwargs) -> bool:
    """Global function for safe notifications"""
    return ui_context_manager.safe_notify(message, notify_type, **kwargs)


def safe_ui_update(update_func: Callable, *args, **kwargs) -> bool:
    """Global function for safe UI updates"""
    return ui_context_manager.safe_ui_update(update_func, *args, **kwargs)


class ContextAwareAsyncTask:
    """Helper class for creating context-aware async tasks"""

    @staticmethod
    def create_task(coro, *, name=None, context_container=None):
        """
        Create an async task with proper error handling for UI context issues.
        """
        async def wrapped_coro():
            try:
                if context_container:
                    with context_container:
                        return await coro
                else:
                    return await coro
            except RuntimeError as e:
                if "slot stack" in str(e).lower():
                    logger.warning(f"UI context error in task {name or 'unnamed'}: {e}")
                    return None
                else:
                    logger.error(f"Runtime error in task {name or 'unnamed'}: {e}")
                    raise
            except Exception as e:
                logger.error(f"Error in task {name or 'unnamed'}: {e}")
                # Don't re-raise to prevent "Task exception was never retrieved"
                return None

        return asyncio.create_task(wrapped_coro(), name=name)


def create_safe_task(coro, *, name=None, context_container=None):
    """Global function to create safe async tasks"""
    return ContextAwareAsyncTask.create_task(coro, name=name, context_container=context_container)


# Context manager for UI operations
class UIContext:
    """Context manager for ensuring UI operations have proper context"""

    def __init__(self, container=None):
        self.container = container
        self.original_context = None

    def __enter__(self):
        try:
            if self.container:
                return self.container.__enter__()
            return self
        except Exception as e:
            logger.error(f"Error entering UI context: {e}")
            return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.container:
                return self.container.__exit__(exc_type, exc_val, exc_tb)
        except Exception as e:
            logger.error(f"Error exiting UI context: {e}")
        return False


def with_safe_ui_context(container=None):
    """Context manager for safe UI operations"""
    return UIContext(container)
