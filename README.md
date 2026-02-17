# Federated Learning-Based Dynamic Traffic Modeling for Robotaxi Simulation in Intelligent Transportation Systems

## Team 5
| Reg No | Name |
|--------|------|
| 24BCE1924 | Sabareez V |
| 24BCE5420 | Jeeva N |
| 24BCE5239 | Vigneshwar Ram Mannarswamy |

## Abstract

This project uses **Federated Learning** to create a framework for dynamic traffic condition modeling within a robotaxi simulation environment. The city is divided into multiple zones, where each zone independently learns local traffic patterns using time-series data. A federated learning mechanism transfers model updates without disclosing original data, ensuring **privacy preservation**. The framework employs **gradient compression** and **correlation-driven update selection** techniques for communication efficiency and scalability.

## Algorithms Implemented

### 1. Federated Averaging (FedAvg)
Collaboratively trains a global traffic prediction model across distributed city zones without sharing raw data.

### 2. Gradient Compression Algorithm
Reduces communication overhead by transmitting only significant gradient updates (Top-K sparsification with residual memory) during model aggregation.

### 3. Correlation-Driven Update Selection Algorithm
Identifies and aggregates only informative and non-redundant model updates from zones using cosine similarity-based filtering.

> **Reference Paper:** *Gradient Compression and Correlation-Driven Federated Learning for Wireless Traffic Prediction* (IEEE Internet of Things Journal)

## Project Structure

```
├── config.py                          # Configuration parameters
├── main.py                            # Main pipeline orchestrator
├── requirements.txt                   # Python dependencies
│
├── models/
│   └── traffic_model.py               # LSTM neural network model
│
├── modules/
│   ├── city_zoning.py                 # Module 1: City Zoning
│   ├── traffic_data_generator.py      # Module 2: Traffic Data Generator
│   ├── local_training.py              # Module 3: Local Training
│   ├── gradient_compression.py        # Module 4: Gradient Compression
│   ├── correlation_aggregation.py     # Module 5: Correlation-Based Aggregation
│   ├── federated_aggregation.py       # Module 6: Federated Aggregation (FedAvg)
│   ├── traffic_prediction.py          # Module 7: Traffic Prediction
│   ├── robotaxi_simulation.py         # Module 8: Robotaxi Simulation
│   └── performance_evaluation.py      # Module 9: Performance Evaluation
│
├── utils/
│   └── helpers.py                     # Utility functions
│
└── results/                           # Generated output plots & report
```

## Modules

| # | Module | Description |
|---|--------|-------------|
| 1 | City Zoning | Divides simulated city into zones as federated clients |
| 2 | Traffic Data Generator | Generates synthetic time-series traffic data with peak/off-peak patterns |
| 3 | Local Training | LSTM-based traffic prediction model training per zone |
| 4 | Gradient Compression | Top-K sparsification to reduce communication overhead |
| 5 | Correlation-Based Aggregation | Filters redundant gradient updates via cosine similarity |
| 6 | Federated Aggregation | FedAvg algorithm for global model aggregation |
| 7 | Traffic Prediction | Forecasts next-step traffic conditions per zone |
| 8 | Robotaxi Simulation | Simulates fleet operations under dynamic vs static traffic |
| 9 | Performance Evaluation | Generates visualizations and comparison reports |

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Project
```bash
python3 main.py
```

### 3. View Results
All output plots and the evaluation report are saved in the `results/` folder:
- `training_loss.png` – FL training loss curves
- `prediction_accuracy.png` – Actual vs predicted traffic
- `compression_analysis.png` – Gradient compression effects
- `correlation_selection.png` – Client selection statistics
- `simulation_comparison.png` – Dynamic vs static comparison
- `traffic_evolution.png` – Traffic patterns over time
- `demand_heatmap.png` – City-wide demand heatmap
- `congestion_heatmap.png` – City-wide congestion heatmap
- `fleet_utilization_timeline.png` – Fleet utilization over time
- `evaluation_report.txt` – Full text report

## Expected Results

| Metric | Value |
|--------|-------|
| FL Training Loss | 0.009 (converged over 20 rounds) |
| Prediction MAE | 7.05 |
| Prediction RMSE | 13.12 |
| Gradient Compression Savings | 84.9% |
| Dynamic Service Rate | 100% |
| Static Service Rate (Baseline) | 10% |

## Tech Stack
- **Python 3.12+**
- **PyTorch** – LSTM model for traffic prediction
- **NumPy** – Numerical computations
- **Matplotlib / Seaborn** – Visualizations
- **scikit-learn** – ML utilities

## References
1. Optimization of Robotaxi Dispatch With Pick-Up/Drop-Off-Point and Boarding-Time Recommendation (IEEE)
2. Multi-Objective Optimization for Robotaxi Dispatch With Safety-Carpooling Mode in Pandemic Era (IEEE)
3. Optimization of Urban Emergency Multimodal Transportation Scheduling With UAV-Ground Traffic Coordination (IEEE)
4. Gradient Compression and Correlation-Driven Federated Learning for Wireless Traffic Prediction (IEEE IoT Journal)
