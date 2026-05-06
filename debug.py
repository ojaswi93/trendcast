# TEMPORARY DEBUG — paste this in a new cell at the bottom of app.py
# Run once, check output, then remove

import os, requests, datetime
from dotenv import load_dotenv
load_dotenv()

API_KEYS = [k for k in [
    os.getenv("YT_API_KEY_1"), os.getenv("YT_API_KEY_2"),
    os.getenv("YT_API_KEY_3"), os.getenv("YT_API_KEY_4"),
    os.getenv("YT_API_KEY_5"), os.getenv("YT_API_KEY_6"),
] if k and "your_" not in str(k)]

print(f"Keys loaded: {len(API_KEYS)}")
print(f"First key starts with: {API_KEYS[0][:8] if API_KEYS else 'NONE'}")

# Test raw API call
key = API_KEYS[0] if API_KEYS else None
if not key:
    print("NO KEY FOUND — check .env file")
else:
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "type": "video",
        "order": "date",
        "maxResults": 5,
        "key": key,
    }
    r = requests.get(url, params=params, timeout=10)
    print(f"Status code: {r.status_code}")
    print(f"Response: {r.json()}")