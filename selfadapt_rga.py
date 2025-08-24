import numpy as np
import time
import threading
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.metrics import precision_score, recall_score, f1_score
import random
from datetime import datetime

# ———————————————————————————————————————————————————————————————————————
# 🔬 SELFADAPT-RGA v2: Self-Adaptive Recurrent Graph Anomaly Detector
# ———————————————————————————————————————————————————————————————————————
class SELFADAPT_RGA:
    def __init__(self, node_features_dim=5, memory_dim=10, decay_rate=0.01, threshold_multiplier=2.0):
        self.graph = defaultdict(dict)
        self.node_features_dim = node_features_dim
        self.memory_dim = memory_dim
        self.decay_rate = decay_rate
        self.threshold_multiplier = threshold_multiplier

        # Learnable weight matrix (instead of random projection)
        self.W_combine = np.random.randn(node_features_dim + memory_dim, memory_dim) * 0.1
        self.kernel_state = np.zeros(memory_dim)  # Adaptive centroid

        self.node_id_counter = 0
        self.scores_log = deque(maxlen=200)        # Sliding window
        self.predictions_log = deque(maxlen=200)
        self.true_labels_log = deque(maxlen=200)

    def add_node(self, node_id, features, timestamp):
        memory = np.zeros(self.memory_dim)
        self.graph[node_id] = {
            'features': np.array(features),
            'timestamp': timestamp,
            'memory': memory
        }

    def update_node(self, node_id, features, timestamp):
        if node_id not in self.graph:
            self.add_node(node_id, features, timestamp)
        else:
            prev_memory = self.graph[node_id]['memory']
            prev_timestamp = self.graph[node_id]['timestamp']
            delta_t = timestamp - prev_timestamp

            # Decay old memory
            decay = np.exp(-self.decay_rate * delta_t)

            # Combine old memory and new features
            combined = np.concatenate([prev_memory, np.array(features)])
            hidden = np.tanh(combined @ self.W_combine)

            # Update memory: leaky integration
            new_memory = decay * prev_memory + (1 - decay) * hidden

            self.graph[node_id].update({
                'features': np.array(features),
                'timestamp': timestamp,
                'memory': new_memory
            })

    def compute_anomaly_score(self, node_id):
        memory = self.graph[node_id]['memory']
        score = np.linalg.norm(memory - self.kernel_state)
        return score

    def adapt_kernel(self, threshold_multiplier=3.0):
        if len(self.graph) == 0:
            return

        scores = {n: self.compute_anomaly_score(n) for n in self.graph}
        score_values = list(scores.values())

        median = np.median(score_values)
        mad = np.median(np.abs(np.array(score_values) - median))
        cutoff = median + threshold_multiplier * mad

        # Only use non-anomalous nodes to update kernel
        non_anomalous_memories = [
            self.graph[n]['memory'] for n in self.graph if scores[n] <= cutoff
        ]

        if non_anomalous_memories:
            self.kernel_state = np.mean(non_anomalous_memories, axis=0)

    def get_dynamic_threshold(self):
        if len(self.scores_log) < 10:
            return 3.0  # default fallback
        median = np.median(self.scores_log)
        mad = np.median(np.abs(np.array(self.scores_log) - median))
        return median + self.threshold_multiplier * mad

    def detect(self, node_id, features, timestamp, label=None):
        self.update_node(node_id, features, timestamp)
        self.adapt_kernel()

        score = self.compute_anomaly_score(node_id)
        threshold = self.get_dynamic_threshold()

        is_anomaly = score > threshold

        # Log for metrics
        self.scores_log.append(score)
        self.predictions_log.append(1 if is_anomaly else 0)
        if label is not None:
            self.true_labels_log.append(label)

        return is_anomaly, score, threshold


# ———————————————————————————————————————————————————————————————————————
# 🧪 Realistic Synthetic Data Stream (e.g., IoT Sensor or Network Node)
# ———————————————————————————————————————————————————————————————————————
def generate_normal_data(dim=5):
    # Simulate stable sensor readings (e.g., temperature, CPU, velocity)
    base = np.array([0.5, 1.0, 0.0, -0.5, 2.0][:dim])
    noise = np.random.normal(0, 0.1, dim)
    return base + noise

def generate_anomaly_data(dim=5, severity='high'):
    # Simulate fault, intrusion, or spike
    if severity == 'high':
        offset = np.random.normal(5, 2, dim)
    else:
        offset = np.random.normal(2, 0.5, dim)
    return generate_normal_data(dim) + offset

class DataStream:
    def __init__(self, dim=5, anomaly_probability=0.05, burst_anomaly_step=50):
        self.dim = dim
        self.anomaly_probability = anomaly_probability
        self.burst_anomaly_step = burst_anomaly_step
        self.step = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.step += 1
        timestamp = time.time()

        # Random anomalies
        if random.random() < self.anomaly_probability:
            features = generate_anomaly_data(self.dim)
            label = 1
        # Scheduled burst anomaly
        elif self.step == self.burst_anomaly_step:
            features = generate_anomaly_data(self.dim, severity='high')
            label = 1
        else:
            features = generate_normal_data(self.dim)
            label = 0

        # Simulate multiple nodes
        node_id = random.choice(['sensor_01', 'sensor_02', 'gateway_A', 'device_X'])

        time.sleep(0.1)  # Simulate real-time delay
        return node_id, features, timestamp, label


# ———————————————————————————————————————————————————————————————————————
# 📊 Real-Time Dashboard
# ———————————————————————————————————————————————————————————————————————
class RealTimeDashboard:
    def __init__(self, detector):
        self.detector = detector
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.fig.suptitle("SELFADAPT-RGA v2: Real-Time Anomaly Detection Dashboard", fontsize=14)

        # Plot 1: Anomaly Score + Threshold
        self.x_data = []
        self.y_scores = []
        self.y_threshold = []
        self.anomaly_points_x = []
        self.anomaly_points_y = []

        self.line1, = self.ax1.plot([], [], label="Anomaly Score", color="blue")
        self.threshold_line = self.ax1.axhline(y=0, color="orange", linestyle="--", label="Dynamic Threshold")
        self.anomaly_scatter = self.ax1.scatter([], [], color="red", marker="x", label="Anomaly Detected")

        self.ax1.set_xlim(0, 200)
        self.ax1.set_ylim(0, 10)
        self.ax1.set_ylabel("Anomaly Score")
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)

        # Plot 2: Metrics (Precision, Recall, F1)
        self.metrics_x = []
        self.precision_y = []
        self.recall_y = []
        self.f1_y = []

        self.p_line, = self.ax2.plot([], [], label="Precision", color="green")
        self.r_line, = self.ax2.plot([], [], label="Recall", color="orange")
        self.f1_line, = self.ax2.plot([], [], label="F1-Score", color="purple")

        self.ax2.set_xlim(0, 200)
        self.ax2.set_ylim(0, 1.1)
        self.ax2.set_xlabel("Time Step")
        self.ax2.set_ylabel("Score")
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.3)

        self.step = 0
        self.window_size = 50  # rolling window for metrics

    def update_metrics(self):
        if len(self.detector.true_labels_log) < 5:
            return 0.0, 0.0, 0.0
        try:
            p = precision_score(self.detector.true_labels_log, self.detector.predictions_log)
            r = recall_score(self.detector.true_labels_log, self.detector.predictions_log)
            f = f1_score(self.detector.true_labels_log, self.detector.predictions_log)
            return p, r, f
        except:
            return 0.0, 0.0, 0.0

    def animate(self, frame):
        # This is called every 200ms by FuncAnimation
        try:
            node_id, features, timestamp, label = next(data_stream)
            is_anomaly, score, threshold = detector.detect(node_id, features, timestamp, label=label)

            self.step += 1

            # Update score plot
            self.x_data.append(self.step)
            self.y_scores.append(score)
            self.y_threshold.append(threshold)

            if is_anomaly:
                self.anomaly_points_x.append(self.step)
                self.anomaly_points_y.append(score)

            # Keep only last N points
            if len(self.x_data) > 200:
                self.x_data.pop(0)
                self.y_scores.pop(0)
                self.y_threshold.pop(0)
            if len(self.anomaly_points_x) > 50:
                self.anomaly_points_x.pop(0)
                self.anomaly_points_y.pop(0)

            # Update main score line
            self.line1.set_data(self.x_data, self.y_scores)
            self.threshold_line.set_ydata([threshold] * 2)  # update threshold line
            self.anomaly_scatter.set_offsets(np.c_[self.anomaly_points_x, self.anomaly_points_y])

            # Dynamic axis scaling
            max_score = max(self.y_scores + [threshold]) if self.y_scores else 1
            self.ax1.set_ylim(0, max_score * 1.3)

            # Update metrics
            p, r, f = self.update_metrics()
            self.metrics_x.append(self.step)
            self.precision_y.append(p)
            self.recall_y.append(r)
            self.f1_y.append(f)

            if len(self.metrics_x) > 200:
                self.metrics_x.pop(0)
                self.precision_y.pop(0)
                self.recall_y.pop(0)
                self.f1_y.pop(0)

            self.p_line.set_data(self.metrics_x, self.precision_y)
            self.r_line.set_data(self.metrics_x, self.recall_y)
            self.f1_line.set_data(self.metrics_x, self.f1_y)

            # Update title with real-time stats
            current_kernel_norm = np.linalg.norm(detector.kernel_state)
            self.fig.suptitle(
                f"SELFADAPT-RGA v2 | Step: {self.step} | "
                f"Score: {score:.3f} | Threshold: {threshold:.3f} | "
                f"Kernel Norm: {current_kernel_norm:.3f}",
                fontsize=14
            )

            return self.line1, self.threshold_line, self.anomaly_scatter, self.p_line, self.r_line, self.f1_line

        except StopIteration:
            plt.close()
            return


# ———————————————————————————————————————————————————————————————————————
# 🚀 Main Execution
# ———————————————————————————————————————————————————————————————————————
if __name__ == "__main__":
    print("🚀 Initializing SELFADAPT-RGA v2 for Real-Time Anomaly Detection...")

    # Initialize detector
    detector = SELFADAPT_RGA(node_features_dim=5, memory_dim=10, decay_rate=0.01)

    # Initialize data stream
    data_stream = iter(DataStream(dim=5, anomaly_probability=0.08, burst_anomaly_step=60))

    # Initialize dashboard
    dashboard = RealTimeDashboard(detector)

    # Set up animation (real-time update every 200ms)
    ani = FuncAnimation(
        dashboard.fig,
        dashboard.animate,
        interval=200,  # ms
        cache_frame_data=False
    )

    print("🟢 Dashboard starting... Close window to stop.")
    try:
        plt.show()
    except KeyboardInterrupt:
        print("\n👋 Stopped by user.")
    finally:
        print(f"✅ Detection ended at step {dashboard.step}.")