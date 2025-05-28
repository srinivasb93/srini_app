from nicegui import ui

@ui.page('/')
def main_page():
    ui.add_head_html('''
        <script src="https://cdn.tailwindcss.com"></script>
    ''')
    with ui.column().classes('items-center mt-10'):
        ui.html('<h1 class="text-4xl font-bold text-blue-600">Welcome to Tailwind + NiceGUI!</h1>')
        ui.html('<button class="mt-4 px-6 py-2 bg-green-500 text-white rounded hover:bg-green-600">Click Me</button>')

ui.run()
