import numpy as np
import matplotlib.pyplot as plt

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

# Kalman-Filter-Initialisierung
x_est = 0.0              # Initiale Zustandsschätzung (Position)
P = 1.0                  # Anfangs-Kovarianz
Q = process_noise_std**2 # Prozessrauschen
R = measurement_noise_std**2  # Messrauschen

# Matrizendarstellung für 1D-System
A = 1  # Systemmatrix (keine Änderung in der Dynamik)
H = 1  # Messmatrix

# Arrays zum Speichern
estimates = []
uncertainties = []

# Kalman-Filter Schleife
for z in measurements:
    # Vorhersage
    x_pred = A * x_est
    P_pred = A * P * A + Q

    # Kalman-Gewinn
    K = P_pred * H / (H * P_pred * H + R)

    # Update
    x_est = x_pred + K * (z - H * x_pred)
    P = (1 - K * H) * P_pred

    # Schätzung speichern
    estimates.append(x_est)
    uncertainties.append(P)
    
# Plot
plt.figure(figsize=(10, 5))
plt.plot(true_positions, label="Wahrer Zustand", linestyle="--")
plt.plot(measurements, label="Messungen", linestyle=":", marker="o", alpha=0.6)
plt.plot(estimates, label="Kalman-Schätzung", linewidth=2)
plt.plot(np.sqrt(uncertainties), label="Unsicherheit (Std. Abw.)", linestyle="--")
plt.fill_between(range(n_steps), 
                 np.array(estimates) - 3*np.sqrt(uncertainties), 
                 np.array(estimates) + 3*np.sqrt(uncertainties), 
                 color='gray', alpha=0.5, label="Unsicherheitsbereich")
plt.axhline(0, color='black', lw=0.5)
plt.axvline(0, color='black', lw=0.5)
plt.xlabel("Zeit")
plt.ylabel("Position")
plt.title("Einfaches Kalman-Filter Beispiel (1D-Position)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
