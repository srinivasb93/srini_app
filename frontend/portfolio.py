# Enhanced Portfolio Module - portfolio.py
# Fixed holdings display, optimized metric cards layout, and improved space utilization

from datetime import datetime
from nicegui import ui
import logging
import asyncio

logger = logging.getLogger(__name__)


def apply_unified_styles():
    """Apply unified styling with enhanced metric cards"""
    ui.add_css('static/styles.css')

    # Enhanced metric cards CSS for better space utilization
    ui.add_css('''
        /* Optimized Metric Cards Layout */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1rem;
            padding: 1rem;
            width: 100%;
        }

        .compact-metric-card {
            background: linear-gradient(135deg, rgba(34, 197, 252, 0.12) 0%, rgba(34, 197, 252, 0.04) 100%);
            border: 1px solid rgba(34, 197, 252, 0.15);
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
            transition: all 0.3s ease;
            min-height: 100px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .compact-metric-card:hover {
            border-color: rgba(34, 197, 252, 0.4);
            background: linear-gradient(135deg, rgba(34, 197, 252, 0.18) 0%, rgba(34, 197, 252, 0.08) 100%);
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(34, 197, 252, 0.15);
        }

        .compact-metric-label {
            color: #94a3b8;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 500;
            margin-bottom: 0.25rem;
        }

        .compact-metric-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #22c5fc;
            margin: 0.25rem 0;
            line-height: 1.1;
        }

        .compact-metric-sublabel {
            color: #6b7280;
            font-size: 0.7rem;
            margin-top: 0.1rem;
        }

        /* Enhanced Holdings Table */
        .holdings-table-container {
            width: 100%;
            overflow-x: auto;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .holdings-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            background: rgba(255, 255, 255, 0.03);
        }

        .holdings-table th {
            background: rgba(255, 255, 255, 0.08);
            color: #94a3b8;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-size: 0.75rem;
            padding: 0.75rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            text-align: left;
            white-space: nowrap;
        }

        .holdings-table td {
            padding: 0.75rem;
            color: #ffffff;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            font-size: 0.85rem;
            white-space: nowrap;
        }

        .holdings-table tr:hover {
            background: rgba(255, 255, 255, 0.05);
        }

        .symbol-cell {
            font-weight: 600;
            color: #22c5fc;
        }

        .price-cell {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 500;
        }

        /* Tab Enhancements */
        .portfolio-tabs .q-tab {
            min-height: 48px;
            padding: 0 16px;
        }

        .tab-content {
            min-height: 400px;
            padding: 1rem;
        }

        /* Empty State Styling */
        .empty-state {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 3rem;
            text-align: center;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 12px;
            border: 1px dashed rgba(255, 255, 255, 0.1);
        }

        .empty-state-icon {
            color: #6b7280;
            margin-bottom: 1rem;
        }

        .empty-state-title {
            color: #9ca3af;
            font-size: 1.125rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }

        .empty-state-subtitle {
            color: #6b7280;
            font-size: 0.875rem;
        }
    ''')


async def render_portfolio_page(fetch_api, user_storage, broker):
    """Enhanced portfolio page with optimized layout and fixed holdings display"""

    apply_unified_styles()

    with ui.column().classes("enhanced-app w-full min-h-screen"):
        # Enhanced title section
        with ui.row().classes("page-title-section w-full justify-between items-center"):
            with ui.column().classes("gap-2"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("account_balance_wallet", size="2rem").classes("text-green-400")
                    ui.label(f"Portfolio Overview - {broker}").classes("page-title")
                    ui.chip("LIVE", color="green").classes("text-xs status-chip")

                ui.label("Your complete investment portfolio - equity and mutual funds").classes("page-subtitle")

            with ui.row().classes("items-center gap-4"):
                ui.button("Export Report", icon="download").classes("button-outline")
                ui.button("Refresh All", icon="refresh").classes("text-cyan-400")

        # Optimized Metrics Section
        await render_optimized_portfolio_metrics(fetch_api, user_storage, broker)

        # Enhanced Holdings Section
        with ui.card().classes("enhanced-card w-full m-4"):
            with ui.row().classes("card-header w-full items-center p-4"):
                ui.icon("pie_chart", size="1.5rem").classes("text-purple-400")
                ui.label("Portfolio Holdings").classes("card-title")

            ui.separator().classes("card-separator")

            with ui.column().classes("w-full"):
                with ui.tabs().props("dense indicator-color=cyan").classes("w-full portfolio-tabs") as portfolio_tabs:
                    equity_tab = ui.tab(name="equity", label="📈 Equity Holdings", icon="trending_up")
                    mf_tab = ui.tab(name="mutual_funds", label="🏦 Mutual Funds", icon="account_balance")
                    analytics_tab = ui.tab(name="analytics", label="📊 Portfolio Analytics", icon="analytics")

                with ui.tab_panels(portfolio_tabs, value="equity").classes("w-full") as portfolio_tab_panels:
                    with ui.tab_panel("equity").classes("tab-content"):
                        equity_container = ui.column().classes("w-full gap-4")
                        await render_enhanced_equity_holdings(fetch_api, user_storage, broker, equity_container)

                    with ui.tab_panel("mutual_funds").classes("tab-content"):
                        mf_container = ui.column().classes("w-full gap-4")
                        await render_enhanced_mutual_funds(fetch_api, broker, mf_container)

                    with ui.tab_panel("analytics").classes("tab-content"):
                        analytics_container = ui.column().classes("w-full gap-4")
                        await render_portfolio_analytics_section(fetch_api, user_storage, broker, analytics_container)


async def render_optimized_portfolio_metrics(fetch_api, user_storage, broker):
    """Render optimized metrics with better space utilization"""

    try:
        # Fetch portfolio data
        equity_response = await fetch_api(f"/portfolio/{broker}")
        mf_response = await fetch_api("/mutual-funds/holdings")

        equity_data = equity_response if isinstance(equity_response, list) else []
        mf_data = mf_response if isinstance(mf_response, list) else []

        # Calculate equity metrics
        total_equity_invested = sum(
            holding.get("quantity", 0) * holding.get("average_price", 0) for holding in equity_data)
        total_equity_current = sum(holding.get("quantity", 0) * holding.get("last_price", 0) for holding in equity_data)
        equity_pnl = total_equity_current - total_equity_invested
        equity_pnl_pct = (equity_pnl / total_equity_invested * 100) if total_equity_invested > 0 else 0

        # Calculate MF metrics
        total_mf_invested = sum(fund.get("invested_amount", 0) for fund in mf_data)
        total_mf_current = sum(fund.get("current_value", 0) for fund in mf_data)
        mf_pnl = total_mf_current - total_mf_invested
        mf_pnl_pct = (mf_pnl / total_mf_invested * 100) if total_mf_invested > 0 else 0

        # Combined metrics
        total_invested = total_equity_invested + total_mf_invested
        total_current = total_equity_current + total_mf_current
        total_pnl = total_current - total_invested
        total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0

        # Render optimized metrics grid
        with ui.element('div').classes("metrics-grid"):
            # Total Portfolio Value
            with ui.column().classes("compact-metric-card"):
                ui.label("Total Portfolio").classes("compact-metric-label")
                ui.label(f"₹{total_current:,.0f}").classes("compact-metric-value text-white")
                ui.label("Current Value").classes("compact-metric-sublabel")

            # Total Invested
            with ui.column().classes("compact-metric-card"):
                ui.label("Total Invested").classes("compact-metric-label")
                ui.label(f"₹{total_invested:,.0f}").classes("compact-metric-value text-blue-400")
                ui.label("Capital Deployed").classes("compact-metric-sublabel")

            # Overall P&L
            pnl_class = "positive-change" if total_pnl >= 0 else "negative-change"
            with ui.column().classes("compact-metric-card"):
                ui.label("Total P&L").classes("compact-metric-label")
                ui.label(f"₹{total_pnl:,.0f}").classes(f"compact-metric-value {pnl_class}")
                ui.label(f"({total_pnl_pct:+.2f}%)").classes(f"compact-metric-sublabel {pnl_class}")

            # Equity Value
            with ui.column().classes("compact-metric-card"):
                ui.label("Equity Value").classes("compact-metric-label")
                ui.label(f"₹{total_equity_current:,.0f}").classes("compact-metric-value text-green-400")
                ui.label(f"{len(equity_data)} Holdings").classes("compact-metric-sublabel")

            # MF Value
            with ui.column().classes("compact-metric-card"):
                ui.label("MF Value").classes("compact-metric-label")
                ui.label(f"₹{total_mf_current:,.0f}").classes("compact-metric-value text-purple-400")
                ui.label(f"{len(mf_data)} Schemes").classes("compact-metric-sublabel")

            # Day Change (placeholder - you can implement this with real-time data)
            with ui.column().classes("compact-metric-card"):
                ui.label("Day Change").classes("compact-metric-label")
                ui.label("₹+2,450").classes("compact-metric-value positive-change")
                ui.label("(+1.2%)").classes("compact-metric-sublabel positive-change")

    except Exception as e:
        logger.error(f"Error rendering portfolio metrics: {e}")
        with ui.element('div').classes("metrics-grid"):
            with ui.column().classes("compact-metric-card"):
                ui.label("Error").classes("compact-metric-label")
                ui.label("Loading...").classes("compact-metric-value text-gray-500")
                ui.label("Please refresh").classes("compact-metric-sublabel")


async def render_enhanced_equity_holdings(fetch_api, user_storage, broker, container):
    """Enhanced equity holdings table with proper error handling and loading states"""

    with container:
        # Loading indicator
        loading_indicator = ui.spinner(size="lg").classes("text-cyan-400")
        status_container = ui.column().classes("w-full")

        try:
            response = await fetch_api(f"/portfolio/{broker}")
            holdings_data = response if isinstance(response, list) else []

            # Remove loading indicator
            loading_indicator.delete()

            if not holdings_data:
                with status_container:
                    with ui.column().classes("empty-state"):
                        ui.icon("trending_up", size="4rem").classes("empty-state-icon")
                        ui.label("No Equity Holdings").classes("empty-state-title")
                        ui.label("Start investing in stocks to see your holdings here.").classes("empty-state-subtitle")
                return

            # Enhanced holdings table
            with status_container:
                ui.label(f"📈 Equity Holdings ({len(holdings_data)} stocks)").classes(
                    "text-lg font-semibold text-white mb-4")

                with ui.element('div').classes("holdings-table-container"):
                    # Create table using ui.html for better control
                    table_html = f"""
                    <table class="holdings-table">
                        <thead>
                            <tr>
                                <th>Symbol</th>
                                <th>Qty</th>
                                <th>Avg Price</th>
                                <th>LTP</th>
                                <th>Current Value</th>
                                <th>P&L</th>
                                <th>P&L %</th>
                                <th>Day Change</th>
                            </tr>
                        </thead>
                        <tbody>
                    """

                    for holding in holdings_data:
                        symbol = holding.get("symbol", "N/A")
                        quantity = holding.get("quantity", 0)
                        avg_price = holding.get("average_price", 0)
                        ltp = holding.get("last_price", 0)
                        current_value = quantity * ltp
                        invested_value = quantity * avg_price
                        pnl = current_value - invested_value
                        pnl_pct = (pnl / invested_value * 100) if invested_value > 0 else 0
                        day_change = holding.get("day_change", 0)

                        pnl_class = "positive-change" if pnl >= 0 else "negative-change"
                        day_change_class = "positive-change" if day_change >= 0 else "negative-change"

                        table_html += f"""
                        <tr>
                            <td class="symbol-cell">{symbol}</td>
                            <td>{quantity:,}</td>
                            <td class="price-cell">₹{avg_price:.2f}</td>
                            <td class="price-cell">₹{ltp:.2f}</td>
                            <td class="price-cell">₹{current_value:,.0f}</td>
                            <td class="price-cell {pnl_class}">₹{pnl:,.0f}</td>
                            <td class="{pnl_class}">{pnl_pct:+.2f}%</td>
                            <td class="{day_change_class}">₹{day_change:+.0f}</td>
                        </tr>
                        """

                    table_html += """
                        </tbody>
                    </table>
                    """

                    ui.html(table_html)

        except Exception as e:
            loading_indicator.delete()
            logger.error(f"Error rendering equity holdings: {e}")
            with status_container:
                with ui.column().classes("empty-state"):
                    ui.icon("error", size="3rem").classes("text-red-500 mb-4")
                    ui.label("Error Loading Holdings").classes("empty-state-title text-red-400")
                    ui.label(f"Failed to fetch portfolio data: {str(e)}").classes("empty-state-subtitle")
                    ui.button("Retry", icon="refresh",
                              on_click=lambda: render_enhanced_equity_holdings(fetch_api, user_storage, broker,
                                                                               container)).classes("mt-4")


async def render_enhanced_mutual_funds(fetch_api, broker, container):
    """Enhanced mutual funds display with better error handling"""

    with container:
        loading_indicator = ui.spinner(size="lg").classes("text-cyan-400")
        status_container = ui.column().classes("w-full")

        try:
            holdings = await fetch_api("/mutual-funds/holdings")

            loading_indicator.delete()

            if not holdings or not isinstance(holdings, list):
                with status_container:
                    with ui.column().classes("empty-state"):
                        ui.icon("account_balance", size="4rem").classes("empty-state-icon")
                        ui.label("No Mutual Fund Holdings").classes("empty-state-title")
                        ui.label("Start investing in mutual funds to see your holdings here.").classes(
                            "empty-state-subtitle")
                return

            with status_container:
                ui.label(f"🏦 Mutual Fund Holdings ({len(holdings)} schemes)").classes(
                    "text-lg font-semibold text-white mb-4")

                # Enhanced MF table
                with ui.element('div').classes("holdings-table-container"):
                    table_html = f"""
                    <table class="holdings-table">
                        <thead>
                            <tr>
                                <th>Scheme Name</th>
                                <th>Units</th>
                                <th>NAV</th>
                                <th>Invested</th>
                                <th>Current Value</th>
                                <th>P&L</th>
                                <th>P&L %</th>
                                <th>XIRR</th>
                            </tr>
                        </thead>
                        <tbody>
                    """

                    for fund in holdings:
                        scheme_name = fund.get("scheme_name", "N/A")[:30] + (
                            "..." if len(fund.get("scheme_name", "")) > 30 else "")
                        units = fund.get("units", 0)
                        nav = fund.get("nav", 0)
                        invested = fund.get("invested_amount", 0)
                        current_value = fund.get("current_value", 0)
                        pnl = current_value - invested
                        pnl_pct = (pnl / invested * 100) if invested > 0 else 0
                        xirr = fund.get("xirr", 0)

                        pnl_class = "positive-change" if pnl >= 0 else "negative-change"
                        xirr_class = "positive-change" if xirr >= 0 else "negative-change"

                        table_html += f"""
                        <tr>
                            <td class="symbol-cell" title="{fund.get('scheme_name', 'N/A')}">{scheme_name}</td>
                            <td>{units:.3f}</td>
                            <td class="price-cell">₹{nav:.2f}</td>
                            <td class="price-cell">₹{invested:,.0f}</td>
                            <td class="price-cell">₹{current_value:,.0f}</td>
                            <td class="price-cell {pnl_class}">₹{pnl:,.0f}</td>
                            <td class="{pnl_class}">{pnl_pct:+.2f}%</td>
                            <td class="{xirr_class}">{xirr:.2f}%</td>
                        </tr>
                        """

                    table_html += """
                        </tbody>
                    </table>
                    """

                    ui.html(table_html)

        except Exception as e:
            loading_indicator.delete()
            logger.error(f"Error rendering mutual funds: {e}")
            with status_container:
                with ui.column().classes("empty-state"):
                    ui.icon("error", size="3rem").classes("text-red-500 mb-4")
                    ui.label("Error Loading Mutual Funds").classes("empty-state-title text-red-400")
                    ui.label(f"Failed to fetch mutual fund data: {str(e)}").classes("empty-state-subtitle")


async def render_portfolio_analytics_section(fetch_api, user_storage, broker, container):
    """Enhanced portfolio analytics with better visualizations"""

    with container:
        ui.label("📊 Portfolio Analytics").classes("text-lg font-semibold text-white mb-4")

        with ui.row().classes("w-full gap-4"):
            # Asset allocation placeholder
            with ui.card().classes("enhanced-card w-1/2"):
                with ui.row().classes("card-header w-full items-center p-4"):
                    ui.icon("pie_chart", size="1.5rem").classes("text-orange-400")
                    ui.label("Asset Allocation").classes("card-title")

                ui.separator().classes("card-separator")

                with ui.column().classes("w-full p-4 items-center"):
                    ui.label("📊 Interactive allocation chart").classes("text-gray-500 text-center p-8")
                    ui.label("Coming soon: Visual breakdown of equity vs MF allocation").classes(
                        "text-gray-600 text-sm text-center")

            # Performance metrics
            with ui.card().classes("enhanced-card w-1/2"):
                with ui.row().classes("card-header w-full items-center p-4"):
                    ui.icon("trending_up", size="1.5rem").classes("text-green-400")
                    ui.label("Performance Metrics").classes("card-title")

                ui.separator().classes("card-separator")

                with ui.column().classes("w-full p-4 gap-3"):
                    with ui.row().classes("w-full justify-between"):
                        ui.label("1D Return").classes("text-gray-400")
                        ui.label("+1.2%").classes("text-green-400 font-semibold")

                    with ui.row().classes("w-full justify-between"):
                        ui.label("1W Return").classes("text-gray-400")
                        ui.label("+3.8%").classes("text-green-400 font-semibold")

                    with ui.row().classes("w-full justify-between"):
                        ui.label("1M Return").classes("text-gray-400")
                        ui.label("+8.5%").classes("text-green-400 font-semibold")

                    with ui.row().classes("w-full justify-between"):
                        ui.label("XIRR").classes("text-gray-400")
                        ui.label("14.2%").classes("text-cyan-400 font-semibold")