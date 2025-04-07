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



# Starte mit Schätzungen in der Nähe der wahren Werte
initial_guess = [0.1, 4.0]

# Optimieren der Parameter
result = minimize(kalman_log_likelihood, initial_guess, args=(measurements,), bounds=[(1e-5, 10), (1e-5, 10)])
optimal_Q, optimal_R = result.x

# Kalman-Filter erneut mit optimierten Parametern anwenden
x_est = 0.0
P = 1.0
A = 1
H = 1

estimates_opt = []
std_devs_opt = []

for z in measurements:
    x_pred = A * x_est
    P_pred = A * P * A + optimal_Q

    K = P_pred * H / (H * P_pred * H + optimal_R)

    x_est = x_pred + K * (z - H * x_pred)
    P = (1 - K * H) * P_pred

    estimates_opt.append(x_est)
    std_devs_opt.append(np.sqrt(P))

# In NumPy-Arrays umwandeln
estimates_opt = np.array(estimates_opt)
std_devs_opt = np.array(std_devs_opt)

# Plot mit ±1σ Konfidenzintervall nach Optimierung
plt.figure(figsize=(10, 5))
plt.plot(true_positions, label="Wahrer Zustand", linestyle="--")
plt.plot(measurements, label="Messungen", linestyle=":", marker="o", alpha=0.6)
plt.plot(estimates_opt, label="Optimierte Kalman-Schätzung", linewidth=2)
plt.fill_between(range(n_steps), estimates_opt - std_devs_opt, estimates_opt + std_devs_opt,
                 color='green', alpha=0.3, label="±1σ (optimiert)")
plt.xlabel("Zeit")
plt.ylabel("Position")
plt.title(f"Kalman-Filter mit optimierten Parametern\nQ={optimal_Q:.4f}, R={optimal_R:.4f}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
