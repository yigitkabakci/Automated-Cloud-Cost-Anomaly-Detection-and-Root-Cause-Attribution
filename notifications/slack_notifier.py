import json
import pandas as pd


def _build_payload(date: str, cost: float, root_cause: str, detected_by: str) -> dict:
    return {
        "text": "🚨 *Cloud Cost Anomaly Detected!*",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        "🚨 *Cloud Cost Anomaly Detected!*\n"
                        f"📅 *Date:* {date}\n"
                        f"💰 *Cost:* ${cost:.2f}\n"
                        f"🔍 *Root Cause:* {root_cause}\n"
                        f"⚡ *Detected by:* {detected_by}"
                    ),
                },
            }
        ],
    }


def send_slack_alert(anomalies: pd.DataFrame, webhook_url: str) -> bool:
    if not webhook_url:
        print("Error: webhook_url is empty.")
        return False

    try:
        import requests
    except ImportError:
        print("Error: 'requests' package not installed. Run: pip install requests")
        return False

    for _, row in anomalies.iterrows():
        date       = pd.to_datetime(row.get("date", "N/A")).strftime("%Y-%m-%d")
        cost       = float(row.get("cost", 0.0))
        root_cause = str(row.get("root_cause", "Unknown"))

        detected_by_parts = []
        if row.get("zscore_anomaly"):   detected_by_parts.append("Z-Score")
        if row.get("if_anomaly"):       detected_by_parts.append("Isolation Forest")
        if row.get("prophet_anomaly"):  detected_by_parts.append("Prophet")
        if row.get("sarima_anomaly"):   detected_by_parts.append("SARIMA")
        detected_by = " + ".join(detected_by_parts) if detected_by_parts else "Unknown"

        payload = _build_payload(date, cost, root_cause, detected_by)
        try:
            response = requests.post(
                webhook_url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=10,
            )
            if response.status_code != 200:
                print(f"Slack error {response.status_code}: {response.text}")
                return False
        except Exception as exc:
            print(f"Error sending Slack alert: {exc}")
            return False

    return True


def test_slack_connection(webhook_url: str) -> bool:
    if not webhook_url:
        print("Error: webhook_url is empty.")
        return False

    try:
        import requests
    except ImportError:
        print("Error: 'requests' package not installed.")
        return False

    payload = {"text": "✅ Cloud Cost Monitor connected successfully!"}
    try:
        response = requests.post(
            webhook_url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=10,
        )
        return response.status_code == 200
    except Exception as exc:
        print(f"Connection test failed: {exc}")
        return False
