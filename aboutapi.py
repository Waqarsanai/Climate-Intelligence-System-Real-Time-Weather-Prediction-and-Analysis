import argparse
import json
import sys
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

def fetch_json(url):
    req = Request(url, headers={"User-Agent": "AboutAPI/1.0"})
    with urlopen(req, timeout=6) as resp:
        data = resp.read()
        return json.loads(data.decode("utf-8"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:5000")
    args = parser.parse_args()
    base = args.base.rstrip("/")
    try:
        health = fetch_json(f"{base}/health")
        status = fetch_json(f"{base}/api/system/status")
        current = fetch_json(f"{base}/api/weather/current")
        forecast = fetch_json(f"{base}/api/predict/24h")
    except (URLError, HTTPError, json.JSONDecodeError) as e:
        print("API unreachable or invalid response")
        print(str(e))
        sys.exit(1)

    print("API Base:", base)
    print("Health:", health.get("status"), health.get("timestamp"))
    print("System:", "model_loaded=" + str(status.get("model_loaded")), "trained=" + str(status.get("is_trained")))
    cw = {
        "city": current.get("city"),
        "temperature": current.get("temperature"),
        "description": current.get("description"),
        "timestamp": current.get("timestamp")
    }
    print("Current:", cw)
    preds = forecast.get("predictions") or []
    head = preds[:3]
    print("Forecast sample:")
    for p in head:
        print(str(p))
    live_flags = [bool(cw.get("timestamp")), len(preds) > 0]
    print("Condition:", "LIVE" if all(live_flags) else "DEGRADED")
    print("Status:", "OK" if health.get("status") == "healthy" else "ISSUE")

if __name__ == "__main__":
    main()
