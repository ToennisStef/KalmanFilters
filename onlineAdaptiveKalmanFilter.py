from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

# Funktion zur Berechnung des negativen Log-Likelihoods für gegebene Q und R
def kalman_log_likelihood(params, measurements):
    Q, R = params

    # Verhindere nicht-positive Kovarianzen
    if Q <= 0 or R <= 0:
        return np.inf

    x_est = 0.0
    P = 1.0
    A = 1
    H = 1

    log_likelihood = 0.0

    for z in measurements:
        x_pred = A * x_est
        P_pred = A * P * A + Q

        S = H * P_pred * H + R
        K = P_pred * H / S

        # Innovation (Messfehler)
        innovation = z - H * x_pred

        # Log-Likelihood des Messwerts unter Normalverteilung
        log_likelihood += -0.5 * (np.log(2 * np.pi * S) + (innovation**2) / S)

        x_est = x_pred + K * innovation
        P = (1 - K * H) * P_pred

    return -log_likelihood  # Minimieren



# Simulationsparameter
np.random.seed(42)
n_steps = 50
true_velocity = 1.0
measurement_noise_std = 2.0
process_noise_std = 0.6

# Wahre Position (lineare Bewegung)
true_positions = np.cumsum(np.ones(n_steps) * true_velocity)

# Messungen mit Rauschen
measurements = true_positions + np.random.normal(0, measurement_noise_std, n_steps)


# Online adaptive Kalman filtering: Optimize Q, R at each time step based on measurements so far
adaptive_estimates = []
adaptive_std_devs = []
adaptive_Qs = []
adaptive_Rs = []

A = 1
H = 1

# Initial state and covariance
x_est = 0.0
P = 1.0

# Starte mit Schätzungen in der Nähe der wahren Werte
initial_guess = [0.1, 4.0]


for t in range(1, n_steps + 1):
    # Use only measurements up to current time step
    measurements_so_far = measurements[:t]

    # Optimize Q and R for current subset of data
    result = minimize(kalman_log_likelihood, initial_guess, args=(measurements_so_far,), bounds=[(1e-5, 10), (1e-5, 10)])
    Q_t, R_t = result.x
    adaptive_Qs.append(Q_t)
    adaptive_Rs.append(R_t)

    # Apply Kalman filter just for current step with optimized Q_t, R_t
    # Reset estimate from previous step
    x_pred = A * x_est
    P_pred = A * P * A + Q_t

    K = P_pred * H / (H * P_pred * H + R_t)
    x_est = x_pred + K * (measurements[t - 1] - H * x_pred)
    P = (1 - K * H) * P_pred

    adaptive_estimates.append(x_est)
    adaptive_std_devs.append(np.sqrt(P))

# Convert to arrays for plotting
adaptive_estimates = np.array(adaptive_estimates)
adaptive_std_devs = np.array(adaptive_std_devs)
adaptive_Qs = np.array(adaptive_Qs)
adaptive_Rs = np.array(adaptive_Rs)

# Plot adaptive Kalman estimates
plt.figure(figsize=(10, 5))
plt.plot(true_positions, label="Wahrer Zustand", linestyle="--")
plt.plot(measurements, label="Messungen", linestyle=":", marker="o", alpha=0.5)
plt.plot(adaptive_estimates, label="Adaptive Kalman-Schätzung", linewidth=2)
plt.fill_between(range(n_steps), adaptive_estimates - adaptive_std_devs, adaptive_estimates + adaptive_std_devs,
                 color='orange', alpha=0.3, label="±1σ (adaptiv)")
plt.xlabel("Zeit")
plt.ylabel("Position")
plt.title("Online Adaptive Kalman-Filter mit iterativem Q & R Update")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()