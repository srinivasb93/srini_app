import requests

url = 'https://api.upstox.com/v3/login/auth/token/request/:4501ed1e-be0e-43b3-abf0-25c7c910f28d'
headers = {
    'Accept': 'application/json',
}

data = {
    'client_secret': 'qocrrnpqjs'
}

response = requests.post(url, headers=headers, data=data)

print(response.status_code)
print(response.json())