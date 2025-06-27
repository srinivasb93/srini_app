# frontend/optimization_ui.py
"""
Simple optimization UI components
"""

from nicegui import ui
from typing import Dict, Optional


class OptimizationUI:
    """Simple optimization UI for backtesting.py integration"""

    def __init__(self):
        self.optimization_enabled = False

    def render_optimization_controls(self):
        """Render optimization controls in the given container"""

        with ui.expansion("🚀 Parameter Optimization", icon="tune").classes("w-full"):
                # Main toggle
                self.opt_enabled = ui.switch(
                    "Enable Parameter Optimization",
                    value=False,
                    on_change=self._toggle_optimization
                ).classes("mb-4")

                # Optimization container
                with ui.column().classes("w-full") as self.opt_container:
                    self.opt_container.set_visibility(False)

                    # Strategy selection
                    with ui.card().classes("w-full mb-4"):
                        ui.label("📊 Strategy Parameters").classes("text-h6 mb-2")

                        # RSI Parameters
                        with ui.expansion("RSI Parameters", icon="trending_up"):
                            self.rsi_enabled = ui.switch("Optimize RSI", value=False).classes("mb-2")

                            with ui.row().classes("w-full gap-4"):
                                self.rsi_period_min = ui.number("Period Min", value=10, min=5, max=30).props("dense")
                                self.rsi_period_max = ui.number("Period Max", value=20, min=5, max=30).props("dense")

                            with ui.row().classes("w-full gap-4"):
                                self.rsi_oversold_min = ui.number("Oversold Min", value=25, min=10, max=40).props(
                                    "dense")
                                self.rsi_oversold_max = ui.number("Oversold Max", value=35, min=10, max=40).props(
                                    "dense")

                            with ui.row().classes("w-full gap-4"):
                                self.rsi_overbought_min = ui.number("Overbought Min", value=65, min=60, max=85).props(
                                    "dense")
                                self.rsi_overbought_max = ui.number("Overbought Max", value=75, min=60, max=85).props(
                                    "dense")

                        # MACD Parameters
                        with ui.expansion("MACD Parameters", icon="show_chart"):
                            self.macd_enabled = ui.switch("Optimize MACD", value=False).classes("mb-2")

                            with ui.row().classes("w-full gap-4"):
                                self.macd_fast_min = ui.number("Fast Min", value=8, min=5, max=20).props("dense")
                                self.macd_fast_max = ui.number("Fast Max", value=15, min=5, max=20).props("dense")

                            with ui.row().classes("w-full gap-4"):
                                self.macd_slow_min = ui.number("Slow Min", value=21, min=15, max=35).props("dense")
                                self.macd_slow_max = ui.number("Slow Max", value=30, min=15, max=35).props("dense")

                        # Bollinger Bands Parameters
                        with ui.expansion("Bollinger Bands Parameters", icon="waterfall_chart"):
                            self.bb_enabled = ui.switch("Optimize Bollinger Bands", value=False).classes("mb-2")

                            with ui.row().classes("w-full gap-4"):
                                self.bb_period_min = ui.number("Period Min", value=15, min=10, max=30).props("dense")
                                self.bb_period_max = ui.number("Period Max", value=25, min=10, max=30).props("dense")

                            with ui.row().classes("w-full gap-4"):
                                self.bb_std_min = ui.number("Std Dev Min", value=1.5, min=1.0, max=3.0, step=0.1).props(
                                    "dense")
                                self.bb_std_max = ui.number("Std Dev Max", value=2.5, min=1.0, max=3.0, step=0.1).props(
                                    "dense")

                    # Optimization Settings
                    with ui.card().classes("w-full"):
                        ui.label("⚙️ Optimization Settings").classes("text-h6 mb-2")

                        with ui.row().classes("w-full gap-4"):
                            self.max_tries = ui.number("Max Combinations", value=50, min=10, max=500).props("dense")
                            self.optimize_for = ui.select(
                                options={
                                    'Return [%]': 'Total Return',
                                    'Sharpe Ratio': 'Sharpe Ratio',
                                    'Calmar Ratio': 'Calmar Ratio'
                                },
                                value='Return [%]',
                                label="Optimize For"
                            ).props("dense")

    def _toggle_optimization(self, e):
        """Toggle optimization UI visibility"""
        self.optimization_enabled = e.value
        self.opt_container.set_visibility(e.value)

    def get_optimization_config(self) -> Optional[Dict]:
        """Get optimization configuration"""
        if not self.optimization_enabled:
            return None

        config = {
            'enabled': True,
            'max_tries': int(self.max_tries.value),
            'optimize_for': self.optimize_for.value,
            'parameters': {}
        }

        # RSI parameters
        if hasattr(self, 'rsi_enabled') and self.rsi_enabled.value:
            config['parameters'].update({
                'rsi_period': list(range(int(self.rsi_period_min.value), int(self.rsi_period_max.value) + 1)),
                'oversold': list(range(int(self.rsi_oversold_min.value), int(self.rsi_oversold_max.value) + 1)),
                'overbought': list(range(int(self.rsi_overbought_min.value), int(self.rsi_overbought_max.value) + 1))
            })

        # MACD parameters
        if hasattr(self, 'macd_enabled') and self.macd_enabled.value:
            config['parameters'].update({
                'fast_period': list(range(int(self.macd_fast_min.value), int(self.macd_fast_max.value) + 1)),
                'slow_period': list(range(int(self.macd_slow_min.value), int(self.macd_slow_max.value) + 1))
            })

        # Bollinger Bands parameters
        if hasattr(self, 'bb_enabled') and self.bb_enabled.value:
            import numpy as np
            config['parameters'].update({
                'bb_period': list(range(int(self.bb_period_min.value), int(self.bb_period_max.value) + 1)),
                'bb_std': list(np.arange(self.bb_std_min.value, self.bb_std_max.value + 0.1, 0.1).round(1))
            })

        return config if config['parameters'] else None