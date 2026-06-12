import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pandas as pd
from notifications.slack_notifier import send_slack_alert, _build_payload


def test_notifier_import():
    from notifications import slack_notifier
    assert slack_notifier is not None


def test_send_without_url():
    df = pd.DataFrame([{"date": "2024-01-01", "cost": 999.0, "root_cause": "EC2"}])
    result = send_slack_alert(df, webhook_url="")
    assert result is False


def test_message_format():
    payload = _build_payload("2024-06-01", 312.5, "Lambda")

    assert "text" in payload
    assert "blocks" in payload
    assert len(payload["blocks"]) == 1

    block = payload["blocks"][0]
    assert block["type"] == "section"
    assert block["text"]["type"] == "mrkdwn"

    body = block["text"]["text"]
    assert "2024-06-01" in body
    assert "312.50" in body
    assert "Lambda" in body

    # Must be JSON-serialisable
    serialised = json.dumps(payload)
    assert isinstance(serialised, str)
