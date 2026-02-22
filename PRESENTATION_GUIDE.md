# 🚦 Project Review Presentation Guide

## 1. How to Present This (Terminal Only)
* **Set the Stage:** Begin by opening your terminal and navigating to the project directory (`/Users/sabareezv/Documents/DSC/`).
* **The Hook:** Tell the reviewers: *"Today, we are showcasing the core backend engine and algorithmic simulation of our project. The visual web dashboard is still in development and will be the focus of our next review."*
* **Execution:** Run the command `python main.py` live. As the terminal prints out Step 1 through Step 6, use the sections below to narrate what the engine is calculating in real-time.

---

## 2. What is This & What to Say
* **What is this?** *"This is a Federated Learning-Based Dynamic Traffic Modeling System for Robotaxi Simulation in Intelligent Transportation Systems (ITS) developed by Team 5."*
* **What to say:** *"Instead of relying on a centralized system that demands massive real-time data from every car and street camera, we’ve developed a decentralized AI architecture. It allows different smart-city zones to learn traffic patterns locally and securely share only their 'learnings' to optimize a fleet of autonomous Robotaxis."*

---

## 3. How Are We Using This? (The Workflow)
As the terminal scripts hit each step, explain the flow:
* **Steps 1 & 2 (City Zoning & Data Gen):** We divide a virtual city into 25 zones (commercial, downtown, residential, etc.) and generate simulated historical traffic datasets for each.
* **Step 3 (Federated Learning):** Each zone's local AI trains on its own traffic data. They compress their calculated gradients and send them to a global server.
* **Step 4 (Prediction):** We use the combined global intelligence to accurately predict the next sequence of traffic congestion and ride demands.
* **Step 5 (Robotaxi Simulation):** We deploy a fleet of 100 Robotaxis across the city, testing our **Dynamic (AI-driven)** dispatching against a **Static (Baseline)** dispatching method.
* **Step 6 (Evaluation):** The system automatically outputs performance metrics and generates result plots under the `/results` directory for our final analysis.

---

## 4. Why Are We Using This & What's The Use?
* **Privacy & Security:** By using Federated Learning, raw passenger and GPS data **never** leaves the local city zone. Only encrypted AI parameters (weights/gradients) are shared.
* **Bandwidth Efficiency:** Streaming raw continuous traffic data to a central cloud server is too heavy for IoT networks. We use **Gradient Compression** to reduce data payload sizes drastically.
* **What’s the use?** This architecture can be immediately utilized by mobility companies (Uber, Waymo, Tesla) and smart city planners to drastically reduce passenger wait times, minimize empty vehicle miles, and alleviate road congestion through predictive dispatching.

---

## 5. What is the Conclusion / What's for the Next Review?
* **Current Conclusion (Terminal Review):** *"Our terminal simulation proves that our Federated Learning approach achieves high traffic prediction accuracy while massively reducing communication bandwidth. Crucially, our Dynamic Robotaxi Dispatch outperforms the Baseline Static model, completing more rides and severely slashing average passenger wait times."*
* **Next Review Setup:** *"For the next review, our conclusion will physically manifest on our Web App. We will connect this Python backend engine to our HTML/JS frontend UI, allowing users to interactively visualize the generated heatmaps, view fleet utilization on a dynamic chart, and toggle simulation settings directly from the browser."*

---

## 6. What Features Can We Add (Future Scope)?
If the reviewers ask *"What's next?"*, you can mention:
1. **Interactive Web Dashboard:** Bringing the analytics, heatmaps, and metric charts to the `index.html` frontend.
2. **Real-World API Integration:** Permitting the simulation to pull live geodata mapping from OpenStreetMap.
3. **Weather Disturbance Modeling:** Adding factors like rain or accidents that dynamically drop road capacity and force the edge-AI to adapt its taxi dispatching patterns.

---

## 7. How to Explain the Code (Structure, Architecture, Algorithms)

When asked about the technical depth, explain it in these 3 pillars:

### A. Structure
The codebase is highly modular, split between controllers, modules, and models:
* `main.py`: The single orchestrator that triggers individual scripts located in the `/modules/` pipeline.
* `config.py`: The central nervous system where all simulation configurations and hyper-parameters are safely managed.
* `models/traffic_model.py`: Where the core Deep Learning mechanics live.
* `/modules/`: Contains separate classes handling everything from Data Generation, Local Training, Compression, Aggregation, Prediction, and final Simulation Evaluation.

### B. Architecture
It uses a **Client-Server Federated Architecture**:
* **Clients (Local Edge Training):** The 25 individual city zones act as edge-computing clients natively training the AI locally.
* **Server (Global Aggregator):** The central server node orchestrates the model distribution, collecting the gradients, and returning a globally optimal master model.

### C. Algorithms Involved
* **Deep Learning (LSTM):** *Time-Series Forecasting*. We use a Multi-layer Long Short-Term Memory (LSTM) neural network to capture temporal dependencies in historical traffic flow to predict future timestamps precisely.
* **Federated Averaging (FedAvg):** The mathematical algorithm used by the server to safely average and combine the individual LSTM models from all 25 zones into one global super-model without sharing raw data.
* **Gradient Compression (Top-K Spasification):** An optimization algorithm that only transmits the highest magnitude gradients (top 30%) to the server, solving network bandwidth bottleneck issues.
* **Correlation-Based Selection (Cosine Similarity):** It filters out redundant updates on the server-side. For example, if two similar residential zones submit identical traffic pattern updates, the algorithm rejects the duplicate to save processing time and prevent model bias.
* **Greedy Fleet Dispatch Heuristic:** The algorithm used in the Robotaxi module to locate and dispatch the absolute closest Idle Robotaxi for a dynamically predicted ride request.
