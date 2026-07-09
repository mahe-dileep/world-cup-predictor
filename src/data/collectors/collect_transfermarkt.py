import requests

BASE_URL="https://transfermarkt-api.fly.dev"

for endpoint in [
    "/clubs/3433/profile",
    "/clubs/3433/players"
]:
    r=requests.get(BASE_URL+endpoint)

    print("\n",endpoint)
    print("STATUS:",r.status_code)
    print(r.text[:500])