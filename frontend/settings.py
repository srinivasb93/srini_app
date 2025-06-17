from nicegui import ui
import logging
import asyncio

logger = logging.getLogger(__name__)

async def render_settings_page(fetch_api, user_storage, apply_theme_from_storage):
    broker = user_storage.get('default_broker', "Zerodha")
    ui.label("Application Settings").classes("text-h4")

    with ui.card().classes("card"):
        ui.label("Preferences").classes("text-h6 border-b pb-2 mb-4")
        with ui.column().classes("space-y-4"):
            current_theme = user_storage.get('app_theme', "Dark")
            ui.label("Select Theme").classes("text-subtitle1")
            ui.select(["Dark", "Light"], value=current_theme,
                      on_change=lambda e: (
                          user_storage.update({'app_theme': e.value}),
                          apply_theme_from_storage()
                      )).classes("input w-full md:w-1/2")
            current_broker = user_storage.get('default_broker', "Zerodha")
            ui.label("Default Trading Broker").classes("text-subtitle1")
            ui.select(["Upstox", "Zerodha"], value=current_broker,
                      on_change=lambda e: user_storage.update({'default_broker': e.value})
                      ).classes("input w-full md:w-1/2")

    with ui.card().classes("card"):
        ui.label("Connect Broker").classes("text-h6 border-b pb-2 mb-4")
        ui.label("Log in with your Zerodha or Upstox credentials.").classes("text-subtitle1 text-gray-400 mb-4")
        status_area = ui.column().classes("space-y-4")

        async def check_and_display_status(broker_name, display_container):
            profile = await fetch_api(f"/profile/{broker_name}")
            display_container.clear()
            with display_container:
                if profile and profile.get("name"):
                    ui.html(
                        f"<p class='text-lg'>{broker_name} Status: <span class='text-green font-semibold'>Connected as {profile['name']}</span></p>")
                else:
                    ui.html(
                        f"<p class='text-lg'>{broker_name} Status: <span class='text-red font-semibold'>Not Connected</span></p>")
                    if broker_name == "Upstox":
                        with ui.column().classes("space-y-4"):
                            ui.label("Broker").classes("text-subtitle1")
                            broker_select = ui.select(options=["Upstox"], value="Upstox").classes("input w-full")
                            ui.label("Client ID").classes("text-subtitle1")
                            auth_code_input = ui.input(placeholder="Enter Upstox Auth Code").classes("input w-full")
                            async def reconnect_upstox_action():
                                if auth_code_input.value:
                                    resp = await fetch_api(f"/auth/upstox/?auth_code={auth_code_input.value}",
                                                           method="POST")
                                    if resp:
                                        ui.notify(f"Upstox reconnected: {resp.get('message', 'Success')}",
                                                  type="positive", position="top-right")
                                        await check_and_display_status("Upstox", display_container)
                                    else:
                                        ui.notify("Failed to reconnect Upstox.", type="negative", position="top-right")
                                else:
                                    ui.notify("Auth code required.", type="warning", position="top-right")
                            ui.button("Connect", on_click=reconnect_upstox_action).classes("button-primary w-full")
                    elif broker_name == "Zerodha":
                        with ui.column().classes("space-y-4"):
                            ui.label("Broker").classes("text-subtitle1")
                            broker_select = ui.select(options=["Zerodha"], value="Zerodha").classes("input w-full")
                            ui.label("Client ID").classes("text-subtitle1")
                            req_token_input = ui.input(placeholder="Enter Zerodha Request Token").classes("input w-full")
                            async def reconnect_zerodha_action():
                                if req_token_input.value:
                                    resp = await fetch_api(f"/auth/zerodha/?request_token={req_token_input.value}",
                                                           method="POST")
                                    if resp:
                                        ui.notify(f"Zerodha reconnected: {resp.get('message', 'Success')}",
                                                  type="positive", position="top-right")
                                        await check_and_display_status("Zerodha", display_container)
                                    else:
                                        ui.notify("Failed to reconnect Zerodha.", type="negative", position="top-right")
                                else:
                                    ui.notify("Request token required.", type="warning", position="top-right")
                            ui.button("Connect", on_click=reconnect_zerodha_action).classes("button-primary w-full")

        zerodha_status_container = ui.column()
        upstox_status_container = ui.column()

        with status_area:
            with ui.expansion("Zerodha Connection", icon="link").classes("w-full"):
                await check_and_display_status("Zerodha", zerodha_status_container)
            with ui.expansion("Upstox Connection", icon="link").classes("w-full"):
                await check_and_display_status("Upstox", upstox_status_container)

        async def refresh_all_statuses():
            await asyncio.gather(
                check_and_display_status("Zerodha", zerodha_status_container),
                check_and_display_status("Upstox", upstox_status_container)
            )
            ui.notify("Connection statuses refreshed.", type="info", position="top-right")

        ui.button("Refresh All Connection Statuses", on_click=refresh_all_statuses, icon="refresh").classes("button-outline mt-6")