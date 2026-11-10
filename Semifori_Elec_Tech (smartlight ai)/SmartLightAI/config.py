
"""
SmartLight AI (Semifori Elec Tech) - central configuration.

Every other module imports its column names, fault names, paths and
thresholds from here so that nothing is hard-coded in two places.
"""

from pathlib import Path


# -----------------------------------------------------------------
# Paths
# -----------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "street_light_fault_prediction_dataset.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"

PORTAL_DIR = OUTPUT_DIR / "portal_outbox"

MODEL_FILE = BASE_DIR / "smartlight_model.joblib"

# -----------------------------------------------------------------
# Municipal portal
# -----------------------------------------------------------------

# Municipal portal storage
MUNICIPAL_DIR = OUTPUT_DIR / "municipal_portal"
MUNICIPAL_TICKETS_DIR = MUNICIPAL_DIR / "tickets"

# Notification history
NOTIFICATION_LOG = MUNICIPAL_DIR / "notification_history.json"

# Fictional municipality
MUNICIPALITY_NAME = "Semifori Municipal Electricity Department"

# Email notification settings
EMAIL_NOTIFICATIONS_ENABLED = False
MUNICIPAL_EMAIL = "maintenance@semifori-demo.local"


RANDOM_STATE = 42


# -----------------------------------------------------------------
# Columns
# -----------------------------------------------------------------

COL_ASSET = "bulb_number"
COL_TIME = "timestamp"
COL_POWER = "power_consumption (Watts)"
COL_VOLTAGE = "voltage_levels (Volts)"
COL_CURRENT = "current_fluctuations (Amperes)"
COL_TEMP = "temperature (Celsius)"
COL_ENV_CURRENT = "current_fluctuations_env (Amperes)"
COL_WEATHER = "environmental_conditions"
COL_TARGET = "fault_type"

SENSOR_COLUMNS = [
    COL_POWER,
    COL_VOLTAGE,
    COL_CURRENT,
    COL_TEMP,
    COL_ENV_CURRENT
]

# Model input, in this order
BASE_FEATURES = SENSOR_COLUMNS + [COL_WEATHER]

REQUIRED_COLUMNS = (
    [COL_ASSET, COL_TIME] + BASE_FEATURES + [COL_TARGET]
)

WEATHER_MAPPING = {
    "Clear": 0,
    "Cloudy": 1,
    "Rainy": 2
}


# -----------------------------------------------------------------
# Valid sensor ranges
# -----------------------------------------------------------------

# Readings outside these limits are treated as corrupted sensor data.

VALID_RANGES = {
    COL_POWER: (0.0, 500.0),
    COL_VOLTAGE: (100.0, 300.0),
    COL_CURRENT: (0.0, 50.0),
    COL_TEMP: (-40.0, 100.0),
    COL_ENV_CURRENT: (0.0, 50.0),
}


# -----------------------------------------------------------------
# Fault categories
# -----------------------------------------------------------------

NO_FAULT = 0
SHORT_CIRCUIT = 1
VOLTAGE_SURGE = 2
BULB_FAILURE = 3
LIGHT_FLICKERING = 4

FAULT_NAMES = {
    NO_FAULT: "No Fault",
    SHORT_CIRCUIT: "Short Circuit",
    VOLTAGE_SURGE: "Voltage Surge",
    BULB_FAILURE: "Bulb Failure",
    LIGHT_FLICKERING: "Light Flickering",
}

FAULT_LABELS = list(FAULT_NAMES)


# -----------------------------------------------------------------
# Fault information for municipal dispatch
# -----------------------------------------------------------------

FAULT_INFO = {
    NO_FAULT: {
        "priority": "None",
        "description": (
            "The streetlight is operating inside its normal range."
        ),
        "signature": "All readings are inside the normal baseline.",
        "action": "No action needed.",
    },

    SHORT_CIRCUIT: {
        "priority": "High",
        "description": (
            "A short circuit is an unintended low-resistance path in "
            "the wiring. It is a fire and safety hazard and can trip "
            "the supply."
        ),
        "signature": "Line current far above the normal 0-5 A range.",
        "action": (
            "Dispatch a crew urgently. Isolate the circuit and inspect "
            "the wiring, fuse and fixture before re-energising."
        ),
    },

    VOLTAGE_SURGE: {
        "priority": "High",
        "description": (
            "A voltage surge is a supply voltage above the safe "
            "operating level. It can damage drivers and lamps."
        ),
        "signature": "Supply voltage above about 240 V.",
        "action": (
            "Dispatch a crew urgently. Check the supply, surge "
            "protection and transformer tap, and inspect the fixture "
            "for damage."
        ),
    },

    BULB_FAILURE: {
        "priority": "Medium",
        "description": (
            "A bulb failure means the lamp has burnt out or is drawing "
            "far less power than expected, so the road may be dark."
        ),
        "signature": (
            "Power consumption below the normal operating range."
        ),
        "action": (
            "Schedule a repair visit and replace the lamp or driver."
        ),
    },

    LIGHT_FLICKERING: {
        "priority": "Medium",
        "description": (
            "Flickering is an unstable light output, usually caused "
            "by a loose connection, a failing driver or an unstable "
            "supply."
        ),
        "signature": (
            "Unstable, elevated current readings in the environment "
            "channel."
        ),
        "action": (
            "Schedule an inspection of connections and the driver."
        ),
    },
}


# -----------------------------------------------------------------
# Decision settings
# -----------------------------------------------------------------

# Below this confidence, a High-priority alert is verified before
# a crew is dispatched.

CONFIDENCE_THRESHOLD = 0.60

# A node that stays silent longer than this is considered offline.
# Communication loss is not automatically classified as a hardware fault.

HEARTBEAT_TIMEOUT_MINUTES = 150


# -----------------------------------------------------------------
# Asset identification
# -----------------------------------------------------------------

def asset_id(bulb_number) -> str:
    """
    Generate a unique asset tag for a monitored streetlight.

    Example: ASSET-SL-2026-012
    """
    return f"ASSET-SL-2026-{int(bulb_number):03d}"
