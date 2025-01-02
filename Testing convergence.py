import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# ============================
# Define ODEs for Testing
# ============================

def linear_ode(x, y):
    """Linear ODE: dy/dx = 2x - y"""
    return 2 * x - y


def oscillatory_ode(x, y):
    """Oscillatory ODE: dy/dx = sin(5x) - y"""
    return np.sin(5 * x) - y


def piecewise_ode(x, y):
    """Piecewise ODE: dy/dx = x^2 - cos(x) (x > 0), or sin(2x) (x <= 0)"""
    return np.where(x > 0, x ** 2 - np.cos(x), np.sin(2 * x))


def highly_oscillatory_ode(x, y):
    """Highly Oscillatory ODE: dy/dx = sin(50x) - y"""
    return np.sin(50 * x) - y


def sharp_feature_ode(x, y):
    """Sharp-Feature ODE: dy/dx = atan(100(x-5))"""
    return np.arctan(100 * (x - 5))

def highly_volatile_ode(x, y):
    """Highly Volatile ODE: dy/dx = exp(0.1x) * sin(100x) - y"""
    return np.exp(0.1 * x) * np.sin(100 * x) - y



# ============================
# Solve the ODE
# ============================

def actual_solution(ode_function, x):
    """
    Solves the given ODE numerically for the provided points `x`.
    If `x` is a single value, interpolate from a broader solution.
    """
    if isinstance(x, (float, int)):
        global_solution_x = np.linspace(-10, 10, 1000)
        global_solution_y = solve_ivp(
            ode_function, [global_solution_x[0], global_solution_x[-1]], [1],
            t_eval=global_solution_x, method='RK45', atol=1e-8, rtol=1e-8).y[0]
        return np.interp(x, global_solution_x, global_solution_y)

    sol = solve_ivp(ode_function, [x[0], x[-1]], [1], t_eval=x, method='RK45', atol=1e-8, rtol=1e-8)
    if len(sol.y) > 0:
        return sol.y[0]
    else:
        raise ValueError("solve_ivp failed to produce results for the given input.")


# ============================
# Initial Guess
# ============================

def initial_guess(x, ode_function):
    y_start, y_end = actual_solution(ode_function, [x[0], x[-1]])
    slope = (y_end - y_start) / (x[-1] - x[0])
    return slope * (x - x[0]) + y_start


# ============================
# Residual Polynomial Fitting
# ============================

def compute_residuals_with_polynomial(x, y_guess, y_actual, degree=3):
    residuals = y_guess - y_actual
    residual_polynomials = []

    for i in range(len(x) - degree):  # Ensure enough points for the polynomial fit
        x_segment = x[i:i + degree + 1]  # At least `degree+1` points
        residual_segment = residuals[i:i + degree + 1]
        poly_coeff = np.polyfit(x_segment, residual_segment, degree)
        residual_polynomials.append(np.poly1d(poly_coeff))

    def residual_spline(x_eval):
        y_residual = np.zeros_like(x_eval)
        for i in range(len(x) - degree):
            mask = (x_eval >= x[i]) & (x_eval <= x[i + degree])  # Apply only in this segment
            y_residual[mask] = residual_polynomials[i](x_eval[mask])
        return y_residual

    return residual_spline


# ============================
# Recursive Sampling with Initial Guess Plotting
# ============================

def recursive_sampling_with_initial_guess(
    x, ode_function, threshold_error=1, avg_error_threshold=1,
    degree=3, max_x_distance_factor=3.0
):
    iterations = 0
    while True:
        y_actual = actual_solution(ode_function, x)
        y_guess = initial_guess(x, ode_function)

        R_interp = compute_residuals_with_polynomial(x, y_guess, y_actual, degree=degree)
        x_fine = np.linspace(x[0], x[-1], 800)
        y_approx = initial_guess(x_fine, ode_function) - R_interp(x_fine)

        y_actual_fine = actual_solution(ode_function, x_fine)
        error = np.abs(y_actual_fine - y_approx)
        max_error = np.max(error)
        average_error = np.mean(error)

        fig, ax = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})

        # Plot actual solution and approximated curve
        ax[0].plot(x_fine, y_actual_fine, '--', label="Actual Solution", linewidth=2)
        ax[0].plot(x_fine, y_approx, label="Approximated Curve", linewidth=2)
        ax[0].scatter(x, y_actual, color='red', zorder=5, label="Sampled Points (Actual Curve)")

        # Add initial guess for the first iteration
        if iterations == 0:
            y_initial_guess = initial_guess(x_fine, ode_function)
            ax[0].plot(x_fine, y_initial_guess, ':', label="Initial Guess", color='orange', linewidth=2)

        ax[0].set_title(f"Iteration {iterations + 1} - Residual-Based Curve Approximation")
        ax[0].set_ylabel("y")
        ax[0].legend()
        ax[0].grid()
        ax[0].text(0.95, 0.05, f"Max Error: {max_error:.4e}\nAverage Error: {average_error:.4e}",
                   transform=ax[0].transAxes, fontsize=10,
                   verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))

        # Plot error graph
        ax[1].plot(x_fine, error, color='purple', label="Error", linewidth=2)
        ax[1].set_title("Error at Each Step")
        ax[1].set_xlabel("x")
        ax[1].set_ylabel("Error")
        ax[1].grid()
        ax[1].legend()

        plt.tight_layout()
        plt.show()

        # Check convergence
        if max_error < threshold_error:
            print(f"Max Error ({max_error:.4e}) is below the threshold ({threshold_error}). Stopping.")
            break
        elif average_error < avg_error_threshold:
            print(f"Average Error ({average_error:.4e}) is below the threshold ({avg_error_threshold}). Stopping.")
            break

        # Adaptive sampling based on residuals and point distances
        new_points = []
        x_distances = np.diff(x)
        avg_x_distance = np.mean(x_distances)
        for i in range(len(x) - 1):
            if x_distances[i] > max_x_distance_factor * avg_x_distance:
                x_mid = (x[i] + x[i + 1]) / 2
                new_points.append(x_mid)

        residuals = y_guess - y_actual
        for i in range(len(x) - 1):
            if abs(residuals[i]) > threshold_error:
                new_points.append((x[i] + x[i + 1]) / 2)

        if not new_points:
            print("No more points to refine. Stopping.")
            break

        x = np.sort(np.unique(np.concatenate((x, new_points))))
        iterations += 1

    print(f"Converged in {iterations + 1} iterations.")
    return x


# ============================
# Main Program
# ============================

ode_options = {
    "1": ("Linear ODE", linear_ode),
    "2": ("Oscillatory ODE", oscillatory_ode),
    "3": ("Piecewise ODE", piecewise_ode),
    "4": ("Highly Oscillatory ODE", highly_oscillatory_ode),
    "5": ("Sharp-Feature ODE", sharp_feature_ode),
    "6": ("Highly Volatile ODE", highly_volatile_ode)
}

print("Select the ODE to test:")
for key, (name, _) in ode_options.items():
    print(f"{key}: {name}")

choice = input("Enter the number corresponding to your choice: ").strip()
if choice in ode_options:
    ode_name, ode_function = ode_options[choice]
    print(f"Testing {ode_name}...")
    x_initial = np.linspace(-10, 10, 10)
    recursive_sampling_with_initial_guess(
        x_initial,
        ode_function=ode_function,
        threshold_error=1e-6,
        avg_error_threshold=1e-8,
        degree=3,
        max_x_distance_factor=3.0
    )
else:
    print("Invalid choice. Exiting.")

