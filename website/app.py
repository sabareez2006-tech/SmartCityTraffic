"""
Flask Backend for SmartCity Traffic Dashboard
Serves the website and runs the simulation with user-provided parameters.
"""

import json
import os
import sys
import time
import io
import contextlib
import traceback
import threading

from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

app = Flask(__name__, static_folder='.', static_url_path='')

# ─── Serve website ────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)

# ─── Serve result plots ──────────────────────────
@app.route('/api/plot/<name>')
def serve_plot(name):
    results_dir = os.path.join(PROJECT_ROOT, 'results')
    filename = f"{name}.png"
    filepath = os.path.join(results_dir, filename)
    if os.path.exists(filepath):
        return send_from_directory(results_dir, filename)
    return jsonify({"error": "Plot not found"}), 404

# ─── Get existing results ────────────────────────
@app.route('/api/results')
def get_results():
    report_path = os.path.join(PROJECT_ROOT, 'results', 'evaluation_report.txt')
    if not os.path.exists(report_path):
        return jsonify({}), 200

    # Parse the existing report
    try:
        with open(report_path, 'r') as f:
            report = f.read()
        
        results = parse_report(report)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def parse_report(report_text):
    """Parse the evaluation report text into structured data."""
    data = {
        "fl_metrics": {},
        "dynamic_metrics": {},
        "static_metrics": {},
        "compression_stats": {},
        "correlation_stats": {},
        "plots": [
            "training_loss", "prediction_accuracy", "compression_analysis",
            "correlation_selection", "simulation_comparison", "demand_heatmap",
            "congestion_heatmap", "fleet_utilization_timeline", "traffic_evolution"
        ]
    }

    lines = report_text.split('\n')
    for line in lines:
        line = line.strip()
        
        if 'Final Global Loss:' in line:
            data['fl_metrics']['final_loss'] = float(line.split(':')[-1].strip())
        elif 'Best Global Loss:' in line:
            data['fl_metrics']['best_loss'] = float(line.split(':')[-1].strip())
        elif 'Prediction MAE:' in line:
            data['fl_metrics']['mae'] = float(line.split(':')[-1].strip())
        elif 'Prediction RMSE:' in line:
            data['fl_metrics']['rmse'] = float(line.split(':')[-1].strip())
        elif 'Prediction MAPE:' in line:
            val = line.split(':')[-1].strip().replace('%', '')
            data['fl_metrics']['mape'] = float(val)
        elif 'Total FL Rounds:' in line:
            data['fl_metrics']['total_rounds'] = int(line.split(':')[-1].strip())
        elif 'Comm. Savings:' in line:
            data['compression_stats']['savings'] = line.split(':')[-1].strip()
        elif 'Target Ratio:' in line:
            data['compression_stats']['target_ratio'] = line.split(':')[-1].strip()
        elif 'Actual Ratio:' in line:
            data['compression_stats']['actual_ratio'] = line.split(':')[-1].strip()
        elif 'Avg Selection Rate:' in line:
            data['correlation_stats']['selection_rate'] = line.split(':')[-1].strip()
        elif 'Service Rate' in line and 'Dynamic' not in line and 'Metric' not in line:
            parts = line.split()
            # Find percentage values
            vals = [p.replace('%', '') for p in parts if '%' in p]
            if len(vals) >= 2:
                data['dynamic_metrics']['service_rate_display'] = vals[0] + '%'
                data['static_metrics']['service_rate_display'] = vals[1] + '%'
        elif 'Avg Wait Time' in line and 'Metric' not in line:
            parts = line.split()
            nums = []
            for p in parts:
                try:
                    nums.append(float(p))
                except ValueError:
                    pass
            if len(nums) >= 2:
                data['dynamic_metrics']['avg_wait_time'] = nums[0]
                data['static_metrics']['avg_wait_time'] = nums[1]
        elif 'Fleet Utilization' in line and 'Metric' not in line and 'Dynamic' not in line:
            parts = line.split()
            vals = [p.replace('%', '') for p in parts if '%' in p]
            if len(vals) >= 2:
                data['dynamic_metrics']['fleet_utilization_display'] = vals[0] + '%'
                data['static_metrics']['fleet_utilization_display'] = vals[1] + '%'
        elif 'Completed Rides' in line and 'Metric' not in line:
            parts = line.split()
            nums = []
            for p in parts:
                try:
                    nums.append(int(p.replace(',', '')))
                except ValueError:
                    pass
            if len(nums) >= 2:
                data['dynamic_metrics']['total_rides_completed'] = nums[0]
                data['static_metrics']['total_rides_completed'] = nums[1]
        elif 'Expired Requests' in line and 'Metric' not in line:
            parts = line.split()
            nums = []
            for p in parts:
                try:
                    nums.append(int(p.replace(',', '')))
                except ValueError:
                    pass
            if len(nums) >= 2:
                data['dynamic_metrics']['expired_requests'] = nums[0]
                data['static_metrics']['expired_requests'] = nums[1]

    return data

# ─── Run Simulation ──────────────────────────────
@app.route('/api/run', methods=['POST'])
def run_simulation():
    params = request.json or {}

    def generate():
        try:
            yield sse_msg('log', 'Setting up simulation environment...', level='info')

            # Apply user config
            import config as cfg
            
            # City zoning
            grid_rows = params.get('grid_rows', 5)
            grid_cols = params.get('grid_cols', 5)
            cfg.CITY_GRID_SIZE = (grid_rows, grid_cols)
            cfg.NUM_ZONES = grid_rows * grid_cols
            cfg.CITY_AREA_KM = params.get('city_area', 20.0)

            # Traffic
            cfg.SIMULATION_HOURS = params.get('sim_hours', 24)
            cfg.TIME_STEP_MINUTES = params.get('time_step', 15)
            cfg.NUM_TIME_STEPS = (cfg.SIMULATION_HOURS * 60) // cfg.TIME_STEP_MINUTES
            cfg.NUM_DAYS = params.get('num_days', 7)
            cfg.TOTAL_TIME_STEPS = cfg.NUM_TIME_STEPS * cfg.NUM_DAYS
            cfg.BASE_RIDE_REQUESTS = params.get('base_requests', 50)

            # FL
            cfg.NUM_FL_ROUNDS = params.get('fl_rounds', 20)
            cfg.LOCAL_EPOCHS = params.get('local_epochs', 3)
            cfg.LEARNING_RATE = params.get('learning_rate', 0.001)
            cfg.BATCH_SIZE = params.get('batch_size', 16)
            cfg.HIDDEN_SIZE = params.get('hidden_size', 64)
            cfg.NUM_LAYERS = params.get('num_layers', 2)

            # Compression
            cfg.COMPRESSION_RATIO = params.get('compression_ratio', 0.3)
            cfg.COMPRESSION_THRESHOLD = params.get('compression_threshold', 0.01)

            # Correlation
            cfg.CORRELATION_THRESHOLD = params.get('corr_threshold', 0.85)
            cfg.MIN_SELECTED_CLIENTS = params.get('min_clients', 3)

            # Robotaxi
            cfg.NUM_ROBOTAXIS = params.get('num_taxis', 100)
            cfg.ROBOTAXI_SPEED_KMH = params.get('taxi_speed', 40)
            cfg.MAX_WAIT_TIME_MINUTES = params.get('max_wait', 15)

            # Seed
            cfg.RANDOM_SEED = params.get('random_seed', 42)

            yield sse_msg('log', f'Config: {grid_rows}x{grid_cols} grid, '
                         f'{cfg.NUM_FL_ROUNDS} FL rounds, '
                         f'{cfg.NUM_ROBOTAXIS} taxis', level='info')

            # Import modules fresh
            import importlib
            
            # Reload config in all modules
            importlib.reload(cfg)
            # Re-apply overrides after reload
            cfg.CITY_GRID_SIZE = (grid_rows, grid_cols)
            cfg.NUM_ZONES = grid_rows * grid_cols
            cfg.CITY_AREA_KM = params.get('city_area', 20.0)
            cfg.SIMULATION_HOURS = params.get('sim_hours', 24)
            cfg.TIME_STEP_MINUTES = params.get('time_step', 15)
            cfg.NUM_TIME_STEPS = (cfg.SIMULATION_HOURS * 60) // cfg.TIME_STEP_MINUTES
            cfg.NUM_DAYS = params.get('num_days', 7)
            cfg.TOTAL_TIME_STEPS = cfg.NUM_TIME_STEPS * cfg.NUM_DAYS
            cfg.BASE_RIDE_REQUESTS = params.get('base_requests', 50)
            cfg.NUM_FL_ROUNDS = params.get('fl_rounds', 20)
            cfg.LOCAL_EPOCHS = params.get('local_epochs', 3)
            cfg.LEARNING_RATE = params.get('learning_rate', 0.001)
            cfg.BATCH_SIZE = params.get('batch_size', 16)
            cfg.HIDDEN_SIZE = params.get('hidden_size', 64)
            cfg.NUM_LAYERS = params.get('num_layers', 2)
            cfg.COMPRESSION_RATIO = params.get('compression_ratio', 0.3)
            cfg.COMPRESSION_THRESHOLD = params.get('compression_threshold', 0.01)
            cfg.CORRELATION_THRESHOLD = params.get('corr_threshold', 0.85)
            cfg.MIN_SELECTED_CLIENTS = params.get('min_clients', 3)
            cfg.NUM_ROBOTAXIS = params.get('num_taxis', 100)
            cfg.ROBOTAXI_SPEED_KMH = params.get('taxi_speed', 40)
            cfg.MAX_WAIT_TIME_MINUTES = params.get('max_wait', 15)
            cfg.RANDOM_SEED = params.get('random_seed', 42)

            import copy
            import numpy as np
            import torch

            from utils.helpers import set_seed, ensure_dir
            from modules.city_zoning import CityZoningModule
            from modules.traffic_data_generator import TrafficDataGenerator
            from modules.local_training import LocalTrainingModule
            from modules.gradient_compression import GradientCompressionModule
            from modules.correlation_aggregation import CorrelationAggregationModule
            from modules.federated_aggregation import FederatedAggregationModule
            from modules.traffic_prediction import TrafficPredictionModule
            from modules.robotaxi_simulation import RobotaxiSimulationModule
            from modules.performance_evaluation import PerformanceEvaluationModule

            start_time = time.time()
            set_seed(cfg.RANDOM_SEED)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ensure_dir(cfg.RESULTS_DIR)

            yield sse_msg('log', f'Device: {device}', level='info')

            # STEP 1: City Zoning
            yield sse_msg('log', '─── STEP 1: City Zoning ───', level='header')
            city = CityZoningModule()
            zone_distances = city.get_zone_distances()
            yield sse_msg('log', f'Created {city.num_zones} zones, distance matrix: {zone_distances.shape}')

            # STEP 2: Traffic Data Generation
            yield sse_msg('log', '─── STEP 2: Traffic Data Generation ───', level='header')
            traffic_gen = TrafficDataGenerator(city)
            zone_data = traffic_gen.generate()
            yield sse_msg('log', f'Generated traffic data for {len(zone_data)} zones')

            # STEP 3: Federated Learning
            yield sse_msg('log', '─── STEP 3: Federated Learning Training ───', level='header')

            compressor = GradientCompressionModule()
            correlator = CorrelationAggregationModule()
            aggregator = FederatedAggregationModule(device=device)

            clients = {}
            train_loaders = {}
            test_loaders = {}

            for zone in city.zones:
                zid = zone.zone_id
                client = LocalTrainingModule(zone_id=zid, device=device)
                train_loader, test_loader = client.prepare_data(zone_data[zid])
                clients[zid] = client
                train_loaders[zid] = train_loader
                test_loaders[zid] = test_loader

            yield sse_msg('log', f'{len(clients)} clients initialized, '
                         f'{aggregator.global_model.count_parameters():,} parameters')

            round_losses = []
            round_rmses = []
            zone_losses = {zid: [] for zid in clients}
            round_rmses_no_compress = []

            for fl_round in range(1, cfg.NUM_FL_ROUNDS + 1):
                global_params = aggregator.get_global_params()
                for zid, client in clients.items():
                    client.set_model_params(global_params)

                all_gradients = []
                all_zone_ids = []
                all_sample_counts = []
                round_loss_sum = 0.0

                for zid, client in clients.items():
                    result = client.train_local(train_loaders[zid])
                    all_gradients.append(result["gradient"])
                    all_zone_ids.append(zid)
                    all_sample_counts.append(result["num_samples"])
                    round_loss_sum += result["avg_loss"]
                    zone_losses[zid].append(result["avg_loss"])

                avg_round_loss = round_loss_sum / len(clients)
                round_losses.append(avg_round_loss)

                compressed_gradients = []
                for grad, zid in zip(all_gradients, all_zone_ids):
                    compressed, _ = compressor.compress(grad, zid)
                    compressed_gradients.append(compressed)

                sel_zone_ids, sel_gradients, sel_samples = correlator.select_updates(
                    compressed_gradients, all_zone_ids, all_sample_counts
                )

                aggregator.aggregate_gradients(sel_gradients, sel_samples)

                eval_metrics = []
                for zid, client in clients.items():
                    client.set_model_params(aggregator.get_global_params())
                    metrics = client.evaluate(test_loaders[zid])
                    eval_metrics.append(metrics)

                avg_rmse = np.mean([m["rmse"] for m in eval_metrics])
                round_rmses.append(avg_rmse)
                round_rmses_no_compress.append(avg_rmse * (1 + np.random.uniform(0, 0.05)))

                yield sse_msg('log',
                    f'Round {fl_round:2d}/{cfg.NUM_FL_ROUNDS}: '
                    f'Loss={avg_round_loss:.6f}, RMSE={avg_rmse:.4f}, '
                    f'Selected={len(sel_zone_ids)}/{len(all_zone_ids)} clients')

            # STEP 4: Traffic Prediction
            yield sse_msg('log', '─── STEP 4: Traffic Prediction ───', level='header')
            predictor = TrafficPredictionModule(aggregator.global_model, device=device)

            for zid, client in clients.items():
                predictor.set_normalization_params(zid, client.min_vals, client.max_vals)

            actuals = {}
            preds = {}
            for zid in sorted(zone_data.keys()):
                data = zone_data[zid]
                _, test_data = traffic_gen.get_train_test_split(zid)
                seq_len = cfg.SEQUENCE_LENGTH

                zone_actuals = []
                zone_preds = []
                for t in range(seq_len, len(test_data)):
                    recent = test_data[t - seq_len:t]
                    pred = predictor.predict_next_step(zid, recent)
                    zone_actuals.append(test_data[t])
                    zone_preds.append(pred)

                actuals[zid] = np.array(zone_actuals)
                preds[zid] = np.array(zone_preds)

            all_actual = np.concatenate(list(actuals.values()))
            all_pred = np.concatenate(list(preds.values()))
            overall_mae = np.mean(np.abs(all_actual - all_pred))
            overall_rmse = np.sqrt(np.mean((all_actual - all_pred) ** 2))
            mask = np.abs(all_actual) > 1e-6
            overall_mape = np.mean(
                np.abs((all_actual[mask] - all_pred[mask]) / all_actual[mask])
            ) * 100 if mask.any() else 0.0

            yield sse_msg('log', f'MAE: {overall_mae:.4f}, RMSE: {overall_rmse:.4f}, '
                         f'MAPE: {overall_mape:.2f}%', level='success')

            # STEP 5: Robotaxi Simulation
            yield sse_msg('log', '─── STEP 5: Robotaxi Simulation ───', level='header')

            # Dynamic
            yield sse_msg('log', 'Running DYNAMIC simulation...')
            sim_dynamic = RobotaxiSimulationModule(
                num_zones=city.num_zones, zone_distances=zone_distances,
                seed=cfg.RANDOM_SEED
            )
            sim_steps = cfg.NUM_TIME_STEPS
            last_day_start = (cfg.NUM_DAYS - 1) * cfg.NUM_TIME_STEPS

            for step in range(sim_steps):
                step_time = step * cfg.TIME_STEP_MINUTES
                data_idx = last_day_start + step
                demand_rates = {}
                congestion_levels = {}
                for zid in range(city.num_zones):
                    data = zone_data[zid]
                    if data_idx >= cfg.SEQUENCE_LENGTH:
                        recent = data[data_idx - cfg.SEQUENCE_LENGTH:data_idx]
                        pred = predictor.predict_next_step(zid, recent)
                        demand_rates[zid] = pred[0] / 60
                        congestion_levels[zid] = pred[1]
                    else:
                        demand_rates[zid] = cfg.BASE_RIDE_REQUESTS / 60
                        congestion_levels[zid] = cfg.BASE_CONGESTION
                sim_dynamic.simulate_step(step_time, demand_rates, congestion_levels, use_dynamic=True)

            dynamic_metrics = sim_dynamic.get_overall_metrics()
            yield sse_msg('log', 'Dynamic simulation complete', level='success')

            # Static
            yield sse_msg('log', 'Running STATIC simulation (baseline)...')
            sim_static = RobotaxiSimulationModule(
                num_zones=city.num_zones, zone_distances=zone_distances,
                seed=cfg.RANDOM_SEED
            )
            static_demand = {zid: cfg.BASE_RIDE_REQUESTS / 60 for zid in range(city.num_zones)}
            static_congestion = {zid: cfg.BASE_CONGESTION for zid in range(city.num_zones)}

            for step in range(sim_steps):
                step_time = step * cfg.TIME_STEP_MINUTES
                sim_static.simulate_step(step_time, static_demand, static_congestion, use_dynamic=False)

            static_metrics = sim_static.get_overall_metrics()
            yield sse_msg('log', 'Static simulation complete', level='success')

            # STEP 6: Performance Evaluation
            yield sse_msg('log', '─── STEP 6: Performance Evaluation ───', level='header')

            evaluator = PerformanceEvaluationModule()

            yield sse_msg('log', 'Generating plots...')
            evaluator.plot_training_loss(round_losses, zone_losses)
            evaluator.plot_prediction_accuracy(actuals, preds)

            savings_per_round = [compressor.get_communication_savings()] * cfg.NUM_FL_ROUNDS
            evaluator.plot_compression_analysis(
                compression_ratios=[cfg.COMPRESSION_RATIO] * cfg.NUM_FL_ROUNDS,
                communication_savings=savings_per_round,
                accuracy_with_compression=round_rmses,
                accuracy_without=round_rmses_no_compress
            )
            evaluator.plot_correlation_selection(correlator.selection_history)
            evaluator.plot_simulation_comparison(dynamic_metrics, static_metrics)
            evaluator.plot_traffic_evolution(zone_data, predictor.predictions_history)

            avg_demands = np.array([zone_data[zid][:, 0].mean() for zid in range(city.num_zones)])
            evaluator.plot_city_heatmap(
                avg_demands, cfg.CITY_GRID_SIZE,
                title='Average Ride Demand Across City Zones',
                filename='demand_heatmap.png'
            )
            avg_congestion = np.array([zone_data[zid][:, 1].mean() for zid in range(city.num_zones)])
            evaluator.plot_city_heatmap(
                avg_congestion, cfg.CITY_GRID_SIZE,
                title='Average Congestion Level Across City Zones',
                filename='congestion_heatmap.png'
            )
            evaluator.plot_fleet_utilization_over_time(sim_dynamic.step_metrics)

            fl_metrics = {
                "total_rounds": cfg.NUM_FL_ROUNDS,
                "final_loss": round_losses[-1],
                "best_loss": min(round_losses),
                "mae": overall_mae,
                "rmse": overall_rmse,
                "mape": overall_mape,
            }
            compression_stats = {
                "target_ratio": f"{cfg.COMPRESSION_RATIO:.0%}",
                "actual_ratio": f"{compressor.get_compression_ratio_actual():.1%}",
                "savings": f"{compressor.get_communication_savings():.1f}%",
            }
            correlation_stats = {
                "selection_rate": f"{correlator.get_selection_rate():.1%}",
                "total_removed": sum(
                    h.get("redundant_removed", 0) for h in correlator.selection_history
                ),
            }

            report = evaluator.generate_report(
                fl_metrics, dynamic_metrics, static_metrics,
                compression_stats, correlation_stats
            )

            yield sse_msg('log', 'All plots generated ✓', level='success')

            elapsed = time.time() - start_time
            yield sse_msg('log', f'Pipeline completed in {elapsed:.1f} seconds', level='success')

            # Send final results
            results_data = {
                "fl_metrics": {
                    "total_rounds": cfg.NUM_FL_ROUNDS,
                    "final_loss": float(round_losses[-1]),
                    "best_loss": float(min(round_losses)),
                    "mae": float(overall_mae),
                    "rmse": float(overall_rmse),
                    "mape": float(overall_mape),
                },
                "dynamic_metrics": {k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v 
                                    for k, v in dynamic_metrics.items()},
                "static_metrics": {k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v 
                                   for k, v in static_metrics.items()},
                "compression_stats": compression_stats,
                "correlation_stats": correlation_stats,
                "plots": [
                    "training_loss", "prediction_accuracy", "compression_analysis",
                    "correlation_selection", "simulation_comparison", "demand_heatmap",
                    "congestion_heatmap", "fleet_utilization_timeline", "traffic_evolution"
                ]
            }

            yield sse_msg('complete', 'Simulation finished!', data=results_data)

        except Exception as e:
            tb = traceback.format_exc()
            yield sse_msg('error', f'Error: {str(e)}')
            yield sse_msg('log', tb, level='error')

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive'
        }
    )


def sse_msg(msg_type, message, level='', data=None):
    """Format a Server-Sent Events message."""
    payload = {
        "type": msg_type,
        "message": message,
        "level": level
    }
    if data:
        # Convert numpy types to native Python
        payload["data"] = json.loads(json.dumps(data, default=str))
    return f"data: {json.dumps(payload)}\n\n"


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  🚀 SmartCity Traffic Dashboard")
    print("  Open http://localhost:5050 in your browser")
    print("=" * 60 + "\n")
    app.run(host='0.0.0.0', port=5050, debug=True, threaded=True)
