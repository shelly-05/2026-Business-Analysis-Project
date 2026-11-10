
import json
from datetime import datetime
from pathlib import Path

import config as cfg


def ensure_notification_folder():
    cfg.MUNICIPAL_DIR.mkdir(parents=True, exist_ok=True)


def load_notification_history():
    ensure_notification_folder()

    if not cfg.NOTIFICATION_LOG.exists():
        return []

    try:
        with open(cfg.NOTIFICATION_LOG, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return []


def save_notification_history(history):
    ensure_notification_folder()

    with open(cfg.NOTIFICATION_LOG, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)


def create_fault_notification(ticket):
    """Create and save an email notification for a maintenance ticket."""

    history = load_notification_history()

    ticket_id = ticket.get("ticket_id", "Unknown")
    
    # Avoid creating duplicate notifications for the same ticket.
    if any(item.get("ticket_id") == ticket_id for item in history):
        return False

    subject = (
        f"Streetlight Fault Alert: "
        f"{ticket.get('asset_id', 'Unknown Asset')}"
    )

    message = f"""
SEMIFORI MUNICIPAL ELECTRICITY DEPARTMENT
STREETLIGHT MAINTENANCE ALERT

A streetlight fault has been detected by SmartLight AI.

Ticket ID: {ticket_id}
Streetlight ID: {ticket.get('asset_id', 'Unknown')}
Fault Type: {ticket.get('fault_type', 'Unknown')}
Priority: {ticket.get('priority', 'Unknown')}
Confidence: {ticket.get('confidence', 'Unknown')}

Summary:
{ticket.get('summary', 'No summary available')}

Recommended Action:
{ticket.get('recommended_action', 'Inspect the streetlight')}

Please review this maintenance ticket and arrange the necessary action.

Generated automatically by SmartLight AI.
"""

    notification = {
        "ticket_id": ticket_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "recipient": cfg.MUNICIPAL_EMAIL,
        "subject": subject,
        "message": message,
        "delivery_status": "Prepared - Not Sent"
    }

    history.append(notification)
    save_notification_history(history)

    return True


def prepare_notifications_for_tickets(tickets):
    created = 0

    for ticket in tickets:
        if create_fault_notification(ticket):
            created += 1

    return created
