"""
Configuration parameters for the Federated Learning-Based
Dynamic Traffic Modeling for Robotaxi Simulation.
"""

import os

# ─────────────────────────────────────────────
# Directory paths
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ─────────────────────────────────────────────
# City Zoning Configuration
# ─────────────────────────────────────────────
CITY_GRID_SIZE = (5, 5)              # 5x5 grid → 25 zones
NUM_ZONES = CITY_GRID_SIZE[0] * CITY_GRID_SIZE[1]
CITY_AREA_KM = 20.0                  # City spans 20 km × 20 km

# Zone types and their characteristics
ZONE_TYPES = {
    "residential": {"density": 0.6, "demand_factor": 0.8, "peak_multiplier": 1.5},
    "commercial":  {"density": 0.9, "demand_factor": 1.5, "peak_multiplier": 2.0},
    "industrial":  {"density": 0.4, "demand_factor": 0.5, "peak_multiplier": 1.2},
    "suburban":    {"density": 0.3, "demand_factor": 0.4, "peak_multiplier": 1.1},
    "downtown":    {"density": 1.0, "demand_factor": 2.0, "peak_multiplier": 2.5},
}

# ─────────────────────────────────────────────
# Traffic Data Generation
# ─────────────────────────────────────────────
SIMULATION_HOURS = 24                 # 24-hour simulation
TIME_STEP_MINUTES = 15                # 15-minute intervals
NUM_TIME_STEPS = (SIMULATION_HOURS * 60) // TIME_STEP_MINUTES  # 96 steps
NUM_DAYS = 7                          # Generate 7 days of data
TOTAL_TIME_STEPS = NUM_TIME_STEPS * NUM_DAYS  # 672 total steps

# Peak hours (0-indexed hour of day)
MORNING_PEAK = (7, 9)                # 7 AM – 9 AM
EVENING_PEAK = (17, 19)              # 5 PM – 7 PM
LUNCH_PEAK = (12, 13)                # 12 PM – 1 PM

# Base traffic parameters
BASE_RIDE_REQUESTS = 50              # Base requests per zone per time step
BASE_CONGESTION = 0.3                # Base congestion level (0–1)
BASE_TRAFFIC_FLOW = 200              # Base vehicle flow count

# ─────────────────────────────────────────────
# Federated Learning Configuration
# ─────────────────────────────────────────────
NUM_FL_ROUNDS = 20                    # Number of federated learning rounds
LOCAL_EPOCHS = 3                      # Local training epochs per round
LEARNING_RATE = 0.001                 # Local model learning rate
BATCH_SIZE = 16                       # Training batch size
SEQUENCE_LENGTH = 8                   # Input sequence length for LSTM
PREDICTION_HORIZON = 1               # Predict 1 step ahead

# Model architecture
HIDDEN_SIZE = 64                      # LSTM hidden layer size
NUM_LAYERS = 2                        # Number of LSTM layers
INPUT_FEATURES = 3                    # ride_requests, congestion, traffic_flow
OUTPUT_FEATURES = 3                   # Predict all 3 features

# ─────────────────────────────────────────────
# Gradient Compression Configuration
# ─────────────────────────────────────────────
COMPRESSION_RATIO = 0.3              # Keep top 30% of gradients
COMPRESSION_THRESHOLD = 0.01         # Minimum gradient magnitude threshold

# ─────────────────────────────────────────────
# Correlation-Based Aggregation
# ─────────────────────────────────────────────
CORRELATION_THRESHOLD = 0.85         # Threshold for filtering redundant updates
MIN_SELECTED_CLIENTS = 3             # Minimum clients to aggregate

# ─────────────────────────────────────────────
# Robotaxi Simulation Configuration
# ─────────────────────────────────────────────
NUM_ROBOTAXIS = 100                   # Total fleet size
ROBOTAXI_SPEED_KMH = 40              # Average speed in km/h
ROBOTAXI_CAPACITY = 4                # Passengers per vehicle
MAX_WAIT_TIME_MINUTES = 15           # Maximum acceptable wait time
PICKUP_TIME_MINUTES = 2              # Time for passenger pickup
DROPOFF_TIME_MINUTES = 1             # Time for passenger dropoff

# ─────────────────────────────────────────────
# Performance Evaluation
# ─────────────────────────────────────────────
EVAL_METRICS = [
    "mae", "rmse", "mape",           # Prediction metrics
    "avg_wait_time", "fleet_utilization",
    "service_rate", "total_rides_completed"
]

# ─────────────────────────────────────────────
# Random seed for reproducibility
# ─────────────────────────────────────────────
RANDOM_SEED = 42
