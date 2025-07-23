# Fixed Portfolio Module - portfolio.py
# Tabular format with separate Equity and MF metrics

from datetime import datetime
from nicegui import ui
import logging
import asyncio

logger = logging.getLogger(__name__)


def apply_unified_styles():
    """Apply unified styling to this page"""
    ui.add_css('static/styles.css')


async def render_portfolio_page(fetch_api, user_storage, broker):
    """Enhanced unified portfolio page with fixed styling and separate metrics"""

    # Apply unified styling
    apply_unified_styles()

    # Enhanced app container
    with ui.column().classes("enhanced-app w-full min-h-screen"):
        # Enhanced title section
        with ui.row().classes("page-title-section w-full justify-between items-center"):
            # Left side - Title and subtitle
            with ui.column().classes("gap-2"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("account_balance_wallet", size="2rem").classes("text-green-400")
                    ui.label(f"Portfolio Overview - {broker}").classes("page-title")
                    ui.chip("LIVE", color="green").classes("text-xs status-chip")

                ui.label("Your complete investment portfolio - equity and mutual funds").classes("page-subtitle")

            # Right side - Portfolio actions
            with ui.row().classes("items-center gap-4"):
                # Market Status
                with ui.row().classes("status-indicator market-status"):
                    ui.icon("circle", size="0.5rem").classes("status-dot")
                    ui.label("Market Open").classes("status-text")

                # Quick actions
                ui.button("Export Report", icon="download").classes("button-outline")
                ui.button("Refresh All", icon="refresh").classes("text-cyan-400")

        # Portfolio summary metrics - FIXED: Separate Equity and MF metrics
        equity_metrics_container = ui.row().classes("w-full gap-4 p-4")
        mf_metrics_container = ui.row().classes("w-full gap-4 px-4")

        # Main content - Fixed tabs
        with ui.card().classes("enhanced-card w-full m-4"):
            with ui.row().classes("card-header w-full items-center p-4"):
                ui.icon("pie_chart", size="1.5rem").classes("text-purple-400")
                ui.label("Portfolio Holdings").classes("card-title")

            ui.separator().classes("card-separator")

            # FIXED: Tab change handler
            with ui.column().classes("w-full"):
                with ui.tabs().props("dense indicator-color=cyan").classes("w-full") as portfolio_tabs:
                    equity_tab = ui.tab(name="equity", label="📈 Equity Holdings", icon="trending_up")
                    mf_tab = ui.tab(name="mutual_funds", label="🏦 Mutual Funds", icon="account_balance")
                    analytics_tab = ui.tab(name="analytics", label="📊 Portfolio Analytics", icon="analytics")

                with ui.tab_panels(portfolio_tabs, value="equity").classes("w-full p-4") as portfolio_tab_panels:
                    # Equity Holdings Tab - FIXED: Tabular format
                    with ui.tab_panel("equity"):
                        equity_container = ui.column().classes("w-full gap-4")
                        await render_equity_holdings_table(fetch_api, user_storage, broker, equity_container)

                    # Mutual Funds Tab - FIXED: Tabular format
                    with ui.tab_panel("mutual_funds"):
                        mf_container = ui.column().classes("w-full gap-4")
                        await render_mutual_funds_table(fetch_api, broker, mf_container)

                    # Portfolio Analytics Tab
                    with ui.tab_panel("analytics"):
                        analytics_container = ui.column().classes("w-full gap-4")
                        await render_portfolio_analytics_section(fetch_api, user_storage, broker, analytics_container)

        # FIXED: Render separate metrics for Equity and MF
        await render_equity_metrics(fetch_api, user_storage, broker, equity_metrics_container)
        await render_mf_metrics(fetch_api, broker, mf_metrics_container)


async def render_equity_metrics(fetch_api, user_storage, broker, container):
    """Render equity-specific metrics"""

    try:
        # Fetch equity data
        equity_response = await fetch_api(f"/portfolio/{broker}")
        equity_data = equity_response if isinstance(equity_response, list) else []

        # Calculate equity metrics
        total_equity_invested = 0.0
        total_equity_current = 0.0
        total_equity_day_change = 0.0

        for holding in equity_data:
            if isinstance(holding, dict):
                quantity = float(holding.get('Quantity', 0))
                avg_price = float(holding.get('AvgPrice', 0))
                last_price = float(holding.get('LastPrice', 0))
                day_change = float(holding.get('DayChange', 0))

                if quantity > 0:
                    total_equity_invested += avg_price * quantity
                    total_equity_current += last_price * quantity
                    total_equity_day_change += day_change

        equity_pnl = total_equity_current - total_equity_invested
        equity_pnl_pct = (equity_pnl / total_equity_invested * 100) if total_equity_invested > 0 else 0

        # Render equity metrics
        with container:
            ui.label("📈 Equity Portfolio Metrics").classes("text-lg font-semibold text-white mb-2")

            # Equity Investment metric
            with ui.column().classes("metric-card flex-1"):
                ui.label("Equity Investment").classes("metric-label")
                ui.label(f"₹{total_equity_invested:,.0f}").classes("metric-value text-cyan-400")
                ui.label("Total Capital in Stocks").classes("metric-sublabel")

            # Equity Current Value metric
            with ui.column().classes("metric-card flex-1"):
                ui.label("Equity Current Value").classes("metric-label")
                ui.label(f"₹{total_equity_current:,.0f}").classes("metric-value text-white")
                ui.label("Market Value").classes("metric-sublabel")

            # Equity P&L metric
            equity_pnl_class = "positive-change" if equity_pnl >= 0 else "negative-change"
            with ui.column().classes("metric-card flex-1"):
                ui.label("Equity P&L").classes("metric-label")
                ui.label(f"₹{equity_pnl:,.0f}").classes(f"metric-value {equity_pnl_class}")
                ui.label(f"({equity_pnl_pct:+.2f}%)").classes(f"metric-sublabel {equity_pnl_class}")

            # Equity Day Change metric
            day_change_class = "positive-change" if total_equity_day_change >= 0 else "negative-change"
            with ui.column().classes("metric-card flex-1"):
                ui.label("Today's Change").classes("metric-label")
                ui.label(f"₹{total_equity_day_change:,.0f}").classes(f"metric-value {day_change_class}")
                ui.label("Day P&L").classes("metric-sublabel")

            # Equity Holdings Count metric
            with ui.column().classes("metric-card flex-1"):
                ui.label("Equity Holdings").classes("metric-label")
                ui.label(str(len(equity_data))).classes("metric-value text-white")
                ui.label("Active Stocks").classes("metric-sublabel")

    except Exception as e:
        logger.error(f"Error calculating equity metrics: {e}")
        with container:
            ui.label("Error loading equity metrics").classes("text-red-500 text-center p-4")


async def render_mf_metrics(fetch_api, broker, container):
    """Render mutual fund-specific metrics"""

    try:
        # Fetch mutual fund data
        mf_response = await fetch_api("/mutual-funds/holdings")
        mf_data = mf_response if isinstance(mf_response, list) else []

        # Calculate MF metrics
        total_mf_invested = 0.0
        total_mf_current = 0.0

        for holding in mf_data:
            if isinstance(holding, dict):
                invested_value = float(holding.get('invested_value', 0))
                current_value = float(holding.get('current_value', 0))

                total_mf_invested += invested_value
                total_mf_current += current_value

        mf_pnl = total_mf_current - total_mf_invested
        mf_pnl_pct = (mf_pnl / total_mf_invested * 100) if total_mf_invested > 0 else 0

        # Render MF metrics
        with container:
            ui.label("🏦 Mutual Funds Portfolio Metrics").classes("text-lg font-semibold text-white mb-2")

            # MF Investment metric
            with ui.column().classes("metric-card flex-1"):
                ui.label("MF Investment").classes("metric-label")
                ui.label(f"₹{total_mf_invested:,.0f}").classes("metric-value text-blue-400")
                ui.label("Total Capital in MFs").classes("metric-sublabel")

            # MF Current Value metric
            with ui.column().classes("metric-card flex-1"):
                ui.label("MF Current Value").classes("metric-label")
                ui.label(f"₹{total_mf_current:,.0f}").classes("metric-value text-white")
                ui.label("Current NAV Value").classes("metric-sublabel")

            # MF P&L metric
            mf_pnl_class = "positive-change" if mf_pnl >= 0 else "negative-change"
            with ui.column().classes("metric-card flex-1"):
                ui.label("MF P&L").classes("metric-label")
                ui.label(f"₹{mf_pnl:,.0f}").classes(f"metric-value {mf_pnl_class}")
                ui.label(f"({mf_pnl_pct:+.2f}%)").classes(f"metric-sublabel {mf_pnl_class}")

            # MF Average Return metric (placeholder)
            with ui.column().classes("metric-card flex-1"):
                ui.label("Avg MF Return").classes("metric-label")
                ui.label("12.5%").classes("metric-value text-purple-400")
                ui.label("CAGR Estimate").classes("metric-sublabel")

            # MF Holdings Count metric
            with ui.column().classes("metric-card flex-1"):
                ui.label("MF Holdings").classes("metric-label")
                ui.label(str(len(mf_data))).classes("metric-value text-white")
                ui.label("Fund Schemes").classes("metric-sublabel")

    except Exception as e:
        logger.error(f"Error calculating MF metrics: {e}")
        with container:
            ui.label("Error loading mutual fund metrics").classes("text-red-500 text-center p-4")


async def render_equity_holdings_table(fetch_api, user_storage, broker, container):
    """FIXED: Render equity holdings in proper table format"""

    status_label = ui.label("Loading equity holdings...").classes("text-gray-400 text-center p-2")

    try:
        response = await fetch_api(f"/portfolio/{broker}")
        holdings_data = response if isinstance(response, list) else []

        if not holdings_data:
            status_label.text = "No equity holdings in portfolio."
            with container:
                with ui.column().classes("w-full items-center justify-center p-12"):
                    ui.icon("trending_up", size="4rem").classes("text-gray-500 mb-4")
                    ui.label("No Equity Holdings").classes("text-gray-400 text-xl font-semibold")
                    ui.label("Start investing in stocks to see your holdings here.").classes(
                        "text-gray-500 text-center")
            return

        status_label.delete()

        # Prepare table data
        table_rows = []
        for holding in holdings_data:
            if not isinstance(holding, dict):
                continue

            symbol = holding.get('Symbol', 'N/A')
            quantity = int(float(holding.get('Quantity', 0)))
            avg_price = float(holding.get('AvgPrice', 0))
            last_price = float(holding.get('LastPrice', 0))
            day_change = float(holding.get('DayChange', 0))

            if quantity == 0:
                continue

            invested_value = avg_price * quantity
            current_value = last_price * quantity
            pnl = current_value - invested_value
            pnl_pct = (pnl / invested_value * 100) if invested_value > 0 else 0

            table_rows.append({
                'symbol': symbol,
                'quantity': quantity,
                'avg_price': avg_price,
                'last_price': last_price,
                'invested_value': invested_value,
                'current_value': current_value,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'day_change': day_change
            })

        # FIXED: Proper table implementation
        with container:
            ui.label("📈 Equity Holdings").classes("text-lg font-semibold text-white mb-4")

            # Table columns definition
            columns = [
                {'name': 'symbol', 'label': 'Symbol', 'field': 'symbol', 'required': True, 'align': 'left'},
                {'name': 'quantity', 'label': 'Qty', 'field': 'quantity', 'align': 'right'},
                {'name': 'avg_price', 'label': 'Avg Price', 'field': 'avg_price', 'align': 'right',
                 'format': '₹{:.2f}'},
                {'name': 'last_price', 'label': 'LTP', 'field': 'last_price', 'align': 'right', 'format': '₹{:.2f}'},
                {'name': 'invested_value', 'label': 'Invested', 'field': 'invested_value', 'align': 'right',
                 'format': '₹{:,.0f}'},
                {'name': 'current_value', 'label': 'Current', 'field': 'current_value', 'align': 'right',
                 'format': '₹{:,.0f}'},
                {'name': 'pnl', 'label': 'P&L', 'field': 'pnl', 'align': 'right', 'format': '₹{:,.0f}'},
                {'name': 'pnl_pct', 'label': 'P&L %', 'field': 'pnl_pct', 'align': 'right', 'format': '{:+.2f}%'},
                {'name': 'day_change', 'label': 'Day Change', 'field': 'day_change', 'align': 'right',
                 'format': '₹{:+.0f}'}
            ]

            # Create table with enhanced styling
            equity_table = ui.table(
                columns=columns,
                rows=table_rows,
                row_key='symbol'
            ).classes("enhanced-table w-full")

            # Add custom styling for P&L columns
            equity_table.add_slot('body-cell-pnl', '''
                <q-td :props="props">
                    <span :class="props.row.pnl >= 0 ? 'positive-change' : 'negative-change'">
                        ₹{{ props.row.pnl.toLocaleString('en-IN', {minimumFractionDigits: 0, maximumFractionDigits: 0}) }}
                    </span>
                </q-td>
            ''')

            equity_table.add_slot('body-cell-pnl_pct', '''
                <q-td :props="props">
                    <span :class="props.row.pnl_pct >= 0 ? 'positive-change' : 'negative-change'">
                        {{ props.row.pnl_pct.toFixed(2) }}%
                    </span>
                </q-td>
            ''')

            equity_table.add_slot('body-cell-day_change', '''
                <q-td :props="props">
                    <span :class="props.row.day_change >= 0 ? 'positive-change' : 'negative-change'">
                        ₹{{ props.row.day_change.toFixed(0) }}
                    </span>
                </q-td>
            ''')

    except Exception as e:
        logger.error(f"Error rendering equity holdings: {e}")
        status_label.text = f"Error loading equity holdings: {str(e)}"
        status_label.classes("text-red-500 text-center p-4")


async def render_mutual_funds_table(fetch_api, broker, container):
    """FIXED: Render mutual funds in proper table format"""

    with container:
        ui.label("🏦 Mutual Fund Holdings").classes("text-lg font-semibold text-white mb-4")

        # MF management tabs
        with ui.tabs().props("dense").classes("w-full mb-4") as mf_tabs:
            ui.tab(name="holdings", label="Holdings", icon="account_balance")
            ui.tab(name="sips", label="SIPs", icon="schedule")
            ui.tab(name="orders", label="Orders", icon="receipt")

        with ui.tab_panels(mf_tabs, value="holdings").classes("w-full") as mf_tab_panels:
            # Holdings panel with table
            with ui.tab_panel("holdings"):
                await render_mf_holdings_table(fetch_api, container)

            # SIPs panel
            with ui.tab_panel("sips"):
                await render_mf_sips_section(fetch_api)

            # Orders panel
            with ui.tab_panel("orders"):
                await render_mf_orders_section(fetch_api)


async def render_mf_holdings_table(fetch_api, container):
    """Render MF holdings in table format"""

    try:
        holdings = await fetch_api("/mutual-funds/holdings")

        if not holdings or not isinstance(holdings, list):
            with ui.column().classes("w-full items-center justify-center p-12"):
                ui.icon("account_balance", size="4rem").classes("text-gray-500 mb-4")
                ui.label("No Mutual Fund Holdings").classes("text-gray-400 text-xl font-semibold")
                ui.label("Start investing in mutual funds to see your holdings here.").classes(
                    "text-gray-500 text-center")
            return

        # Prepare MF table data
        mf_rows = []
        for holding in holdings:
            scheme_name = holding.get('tradingsymbol', 'N/A')
            units = float(holding.get('quantity', 0))
            avg_nav = float(holding.get('average_price', 0))
            current_nav = float(holding.get('last_price', 0))
            invested_value = float(holding.get('invested_value', avg_nav * units))
            current_value = float(holding.get('current_value', current_nav * units))
            pnl = current_value - invested_value
            pnl_pct = (pnl / invested_value * 100) if invested_value > 0 else 0

            mf_rows.append({
                'scheme': scheme_name,
                'units': units,
                'avg_nav': avg_nav,
                'current_nav': current_nav,
                'invested_value': invested_value,
                'current_value': current_value,
                'pnl': pnl,
                'pnl_pct': pnl_pct
            })

        # MF Table columns
        mf_columns = [
            {'name': 'scheme', 'label': 'Scheme Name', 'field': 'scheme', 'required': True, 'align': 'left'},
            {'name': 'units', 'label': 'Units', 'field': 'units', 'align': 'right', 'format': '{:.3f}'},
            {'name': 'avg_nav', 'label': 'Avg NAV', 'field': 'avg_nav', 'align': 'right', 'format': '₹{:.2f}'},
            {'name': 'current_nav', 'label': 'Current NAV', 'field': 'current_nav', 'align': 'right',
             'format': '₹{:.2f}'},
            {'name': 'invested_value', 'label': 'Invested', 'field': 'invested_value', 'align': 'right',
             'format': '₹{:,.0f}'},
            {'name': 'current_value', 'label': 'Current Value', 'field': 'current_value', 'align': 'right',
             'format': '₹{:,.0f}'},
            {'name': 'pnl', 'label': 'P&L', 'field': 'pnl', 'align': 'right', 'format': '₹{:,.0f}'},
            {'name': 'pnl_pct', 'label': 'P&L %', 'field': 'pnl_pct', 'align': 'right', 'format': '{:+.2f}%'}
        ]

        # Create MF table
        mf_table = ui.table(
            columns=mf_columns,
            rows=mf_rows,
            row_key='scheme'
        ).classes("enhanced-table w-full")

        # Add P&L styling
        mf_table.add_slot('body-cell-pnl', '''
            <q-td :props="props">
                <span :class="props.row.pnl >= 0 ? 'positive-change' : 'negative-change'">
                    ₹{{ props.row.pnl.toLocaleString('en-IN', {minimumFractionDigits: 0, maximumFractionDigits: 0}) }}
                </span>
            </q-td>
        ''')

        mf_table.add_slot('body-cell-pnl_pct', '''
            <q-td :props="props">
                <span :class="props.row.pnl_pct >= 0 ? 'positive-change' : 'negative-change'">
                    {{ props.row.pnl_pct.toFixed(2) }}%
                </span>
            </q-td>
        ''')

    except Exception as e:
        logger.error(f"Error rendering MF holdings: {e}")
        ui.label("Error loading mutual fund holdings").classes("text-red-500 text-center p-4")


async def render_mf_sips_section(fetch_api):
    """Render SIPs section"""
    try:
        sips = await fetch_api("/mutual-funds/sips")
        if not sips or not isinstance(sips, list):
            ui.label("No active SIPs found").classes("text-gray-500 text-center p-8")
            return

        ui.label("Active SIPs").classes("text-lg font-semibold text-white mb-4")
        for sip in sips:
            with ui.card().classes("enhanced-card w-full p-4 mb-2"):
                with ui.row().classes("w-full justify-between items-center"):
                    with ui.column():
                        ui.label(sip.get('scheme_code', 'N/A')).classes("font-semibold text-white")
                        ui.label(f"₹{sip.get('amount', 0)} • {sip.get('frequency', 'Monthly')}").classes(
                            "text-sm text-gray-400")
                    ui.button("Manage", icon="settings").props("flat dense").classes("text-cyan-400")

    except Exception as e:
        logger.error(f"Error rendering SIPs: {e}")
        ui.label("Error loading SIPs").classes("text-red-500 text-center p-4")


async def render_mf_orders_section(fetch_api):
    """Render MF orders section"""
    try:
        orders = await fetch_api("/mutual-funds/orders")
        if not orders or not isinstance(orders, list):
            ui.label("No mutual fund orders found").classes("text-gray-500 text-center p-8")
            return

        ui.label("Recent MF Orders").classes("text-lg font-semibold text-white mb-4")
        for order in orders:
            status_class = "positive-change" if order.get('status') == 'COMPLETE' else "text-yellow-400"
            with ui.card().classes("enhanced-card w-full p-4 mb-2"):
                with ui.row().classes("w-full justify-between items-center"):
                    with ui.column():
                        ui.label(order.get('tradingsymbol', 'N/A')).classes("font-semibold text-white")
                        ui.label(f"₹{order.get('amount', 0)} • {order.get('transaction_type', 'BUY')}").classes(
                            "text-sm text-gray-400")
                    ui.label(order.get('status', 'PENDING')).classes(f"text-sm {status_class} font-semibold")

    except Exception as e:
        logger.error(f"Error rendering MF orders: {e}")
        ui.label("Error loading orders").classes("text-red-500 text-center p-4")


async def render_portfolio_analytics_section(fetch_api, user_storage, broker, container):
    """Render portfolio analytics and insights"""

    with container:
        ui.label("📊 Portfolio Analytics").classes("text-lg font-semibold text-white mb-4")

        # Analytics grid
        with ui.row().classes("w-full gap-4"):

            # Asset allocation chart
            with ui.card().classes("enhanced-card w-1/2"):
                with ui.row().classes("card-header w-full items-center p-4"):
                    ui.icon("pie_chart", size="1.5rem").classes("text-orange-400")
                    ui.label("Asset Allocation").classes("card-title")

                ui.separator().classes("card-separator")

                with ui.column().classes("w-full p-4 items-center"):
                    ui.label("📊 Asset allocation chart").classes("text-gray-500 text-center p-8")
                    ui.label("Visual breakdown of equity vs mutual funds allocation").classes(
                        "text-gray-600 text-sm text-center")

            # Performance metrics
            with ui.card().classes("enhanced-card w-1/2"):
                with ui.row().classes("card-header w-full items-center p-4"):
                    ui.icon("trending_up", size="1.5rem").classes("text-green-400")
                    ui.label("Performance Metrics").classes("card-title")

                ui.separator().classes("card-separator")

                with ui.column().classes("w-full p-4 gap-3"):
                    metrics = [
                        {"label": "30-Day Return", "value": "+5.2%", "class": "positive-change"},
                        {"label": "90-Day Return", "value": "+12.8%", "class": "positive-change"},
                        {"label": "1-Year Return", "value": "+18.4%", "class": "positive-change"},
                        {"label": "XIRR", "value": "16.7%", "class": "positive-change"},
                    ]

                    for metric in metrics:
                        with ui.row().classes("w-full justify-between items-center p-2"):
                            ui.label(metric["label"]).classes("text-gray-400 text-sm")
                            ui.label(metric["value"]).classes(f"text-sm {metric['class']} font-semibold")

        # Portfolio insights
        with ui.card().classes("enhanced-card w-full mt-4"):
            with ui.row().classes("card-header w-full items-center p-4"):
                ui.icon("lightbulb", size="1.5rem").classes("text-yellow-400")
                ui.label("Portfolio Insights").classes("card-title")

            ui.separator().classes("card-separator")

            with ui.column().classes("w-full p-4 gap-3"):
                insights = [
                    "🎯 Your portfolio is well-diversified with equity and mutual funds",
                    "📈 Technology sector is your highest performing segment",
                    "⚖️ Consider rebalancing based on your risk profile",
                    "💎 Strong performance across multiple asset classes"
                ]

                for insight in insights:
                    ui.label(insight).classes("text-gray-300 text-sm p-2")