import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
n_steps = 50
dt = 1.0  # time step

# True state: position, velocity, acceleration (with time-varying acceleration)
true_states = np.zeros((n_steps, 3))  # [position, velocity, acceleration]
true_states[0] = [0, 0, 0.0]

# Simulate changing acceleration (e.g., sinusoidal)
for t in range(1, n_steps):
    a_t = 1.0 * np.sin(0.2 * t)
    true_states[t, 2] = a_t
    true_states[t, 1] = true_states[t-1, 1] + a_t * dt
    true_states[t, 0] = true_states[t-1, 0] + true_states[t-1, 1] * dt + 0.5 * a_t * dt**2

# Noisy measurements (only position is observed)
measurement_noise_std = 2.0
measurements = true_states[:, 0] + np.random.normal(0, measurement_noise_std, n_steps)

# Kalman filter setup
x_est = np.zeros((n_steps, 3))  # estimated [position, velocity, acceleration]
P = np.eye(3) * 10  # initial covariance

# Measurement matrix
H = np.array([[1, 0, 0]])
R = np.array([[measurement_noise_std**2]])

# Initial estimate
x_est[0] = [0, 0, 0]

for t in range(1, n_steps):
    A = np.array([
        [1, dt, 0.5 * dt**2],
        [0, 1, dt],
        [0, 0, 1]
    ])
    
    # Time-varying process noise
    sigma_a = 0.1 + 0.05 * np.abs(np.sin(0.1 * t))
    G = np.array([[0.5 * dt**2], [dt], [1]])
    Q = (sigma_a**2) * (G @ G.T)

    # Predict
    x_pred = A @ x_est[t-1]
    P_pred = A @ P @ A.T + Q

    # Update
    y = measurements[t] - H @ x_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_est[t] = x_pred + K.flatten() * y
    P = (np.eye(3) - K @ H) @ P_pred

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(true_states[:, 0], label="True Position", linestyle="--")
plt.plot(measurements, label="Measurements", linestyle=":", marker="o", alpha=0.5)
plt.plot(x_est[:, 0], label="Estimated Position", linewidth=2)
plt.fill_between(range(n_steps),
                 x_est[:, 0] - 1.96*np.sqrt(P[0, 0]),
                 x_est[:, 0] + 1.96*np.sqrt(P[0, 0]),
                 alpha=0.2, color='green', label="±1.96σ")
plt.title("Kalman Filter with Changing Acceleration")
plt.xlabel("Time Step")
plt.ylabel("Position")
plt.legend()
plt.grid(True)
plt.show()
