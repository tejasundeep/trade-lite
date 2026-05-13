import os
import requests
import time
import hmac
import hashlib
from urllib.parse import urlencode, quote
from dotenv import load_dotenv

load_dotenv()

def test_binance_auth():
    api_key = os.getenv("BINANCE_API_KEY", "").strip()
    secret = os.getenv("BINANCE_SECRET", "").strip()
    
    print(f"Testing API Key: {api_key[:6]}...{api_key[-6:]}")
    
    base_url = "https://fapi.binance.com"
    path = "/fapi/v2/balance"
    
    params = {
        "timestamp": int(time.time() * 1000),
        "recvWindow": 5000
    }
    
    query = urlencode(params, doseq=True, quote_via=quote)
    signature = hmac.new(
        secret.encode("utf-8"),
        query.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()
    
    url = f"{base_url}{path}?{query}&signature={signature}"
    headers = {"X-MBX-APIKEY": api_key}
    
    print("Sending request to Binance Futures...")
    response = requests.get(url, headers=headers)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")

    # Also test Listen Key (No signature required usually, just API Key)
    print("\nTesting Spot API (GET /api/v3/account)...")
    spot_url = "https://api.binance.com/api/v3/account"
    spot_params = {"timestamp": int(time.time() * 1000)}
    spot_query = urlencode(spot_params)
    spot_sig = hmac.new(secret.encode("utf-8"), spot_query.encode("utf-8"), hashlib.sha256).hexdigest()
    spot_res = requests.get(f"{spot_url}?{spot_query}&signature={spot_sig}", headers=headers)
    print(f"Status Code: {spot_res.status_code}")
    print(f"Response: {spot_res.text}")

if __name__ == "__main__":
    test_binance_auth()
