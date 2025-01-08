import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from sklearn.model_selection import train_test_split


def initial_guess(x_sampled, y_sampled):
    """
    Create an initial polynomial guess using a quadratic fit.
    """
    poly_coeff = np.polyfit(x_sampled, y_sampled, deg=2)  # Quadratic fit
    return lambda x: np.polyval(poly_coeff, x)


def recursive_regression_with_adaptive_sampling(
    x_train, y_train, degree=2, max_iterations=10, sampling_factor=1.5, max_new_points=10
):
    """
    Recursive regression with polynomial splines and adaptive sampling.

    Parameters:
        x_train, y_train: Training data.
        degree: Degree of polynomial splines.
        max_iterations: Maximum number of iterations.
        sampling_factor: Threshold factor for adaptive sampling.
        max_new_points: Max number of new points per iteration.

    Returns:
        Callable function for the fitted model.
    """
    # Initial sampling (only from training data)
    indices = np.linspace(0, len(x_train) - 1, 10, dtype=int)
    x_sampled = x_train[indices]
    y_sampled = y_train[indices]

    for iteration in range(max_iterations):
        # Ensure sampled points come only from training data
        unique_x, inverse_indices = np.unique(x_sampled, return_inverse=True)
        averaged_y = np.array([y_sampled[inverse_indices == i].mean() for i in range(len(unique_x))])
        x_sampled, y_sampled = unique_x, averaged_y

        # Initial guess: Quadratic fit
        linear_model = initial_guess(x_sampled, y_sampled)
        y_guess = linear_model(x_sampled)

        # Compute residuals
        residuals = y_sampled - y_guess

        # Fit spline to residuals without smoothing
        spline = UnivariateSpline(x_sampled, residuals, k=degree, s=0)

        # Combined model: initial guess + residual spline
        model = lambda x: linear_model(x) + spline(x)

        # Compute residuals on the entire training set
        y_pred = model(x_train)
        full_residuals = y_train - y_pred

        # Adaptive sampling: select high-residual points
        residual_threshold = sampling_factor * np.std(full_residuals)
        new_indices = np.where(np.abs(full_residuals) > residual_threshold)[0]

        # Limit the number of new points added
        if len(new_indices) > max_new_points:
            new_indices = np.random.choice(new_indices, max_new_points, replace=False)

        if len(new_indices) == 0:
            print(f"Converged at iteration {iteration + 1}.")
            break

        # Add new points strictly from the training set
        x_sampled = np.unique(np.concatenate((x_sampled, x_train[new_indices])))
        y_sampled = np.array([y_train[np.where(x_train == xi)[0][0]] for xi in x_sampled])

        print(f"Iteration {iteration + 1}: Sampled points in training = {len(x_sampled)}, "
              f"total points in training set = {len(x_train)}, "
              f"proportion sampled/available = {len(x_sampled) / len(x_train):.4f}")

        # Plot progress
        plt.figure(figsize=(10, 6))
        plt.scatter(x_train, y_train, label="All Training Data", color="red")
        plt.scatter(x_sampled, y_sampled, label="Sampled Points", color="blue")
        plt.plot(
            np.sort(x_train), model(np.sort(x_train)),
            label=f"Iteration {iteration + 1}", color="purple"
        )
        plt.title(f"Iteration {iteration + 1}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid()
        plt.show()

    # Identify slope change points (green and yellow)
    x_fine = np.linspace(x_train.min(), x_train.max(), 1000)
    y_fine = model(x_fine)
    dydx = np.gradient(y_fine, x_fine)

    green_points = []
    yellow_points = []

    for i in range(1, len(dydx)):
        if dydx[i - 1] > 0 and dydx[i] <= 0:
            green_points.append((x_fine[i], y_fine[i]))
        elif dydx[i - 1] < 0 and dydx[i] >= 0:
            yellow_points.append((x_fine[i], y_fine[i]))

    green_points = np.array(green_points)
    yellow_points = np.array(yellow_points)

    # Fit global polynomials through green and yellow points
    green_poly_coeff = np.polyfit(green_points[:, 0], green_points[:, 1], deg=8)
    yellow_poly_coeff = np.polyfit(yellow_points[:, 0], yellow_points[:, 1], deg=8)

    green_poly = lambda x: np.polyval(green_poly_coeff, x)
    yellow_poly = lambda x: np.polyval(yellow_poly_coeff, x)

    # Average the two global polynomials to create the final model
    averaged_model = lambda x: 0.5 * (green_poly(x) + yellow_poly(x))

    # Plot final model with error bounds
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, label="All Training Data", color="red")
    plt.plot(x_fine, y_fine, label="Final Iteration Spline Model", color="purple")
    plt.scatter(green_points[:, 0], green_points[:, 1], label="Slope Change (+ to -)", color="green")
    plt.scatter(yellow_points[:, 0], yellow_points[:, 1], label="Slope Change (- to +)", color="yellow")
    plt.plot(x_fine, green_poly(x_fine), label="Global Polynomial (Green Points)", color="green", linestyle="--")
    plt.plot(x_fine, yellow_poly(x_fine), label="Global Polynomial (Yellow Points)", color="yellow", linestyle="--")
    plt.plot(x_fine, averaged_model(x_fine), label="Averaged Model (New Curve)", color="blue")
    plt.title("Final Model with Global Polynomial Error Bounds")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.show()

    return averaged_model


def generate_datasets(option=1):
    """
    Generate synthetic datasets based on the selected option.
    """
    np.random.seed(42)
    x = np.linspace(0, 10, 500)
    if option == 1:
        y = np.sin(x) + 0.1 * np.random.randn(len(x))
    elif option == 2:
        y = x**2 + 0.5 * np.random.randn(len(x))
    elif option == 3:
        y = np.where(x < 5, x**3, x**2) + 0.2 * np.random.randn(len(x))
    elif option == 4:
        y = np.sin(10 * x) + 0.1 * np.random.randn(len(x))
    elif option == 5:
        y = np.where(x < 5, x, np.log(x + 1)) + 0.2 * np.random.randn(len(x))
    else:
        raise ValueError("Invalid dataset option.")
    return x, y


if __name__ == "__main__":
    print("Select a dataset option:")
    print("1: Sinusoidal data with noise")
    print("2: Quadratic data with noise")
    print("3: Piecewise polynomial data")
    print("4: Highly oscillatory data")
    print("5: Mixed behavior data")

    option = int(input("Enter the dataset option (1-5): ").strip())

    # Generate the selected dataset
    x_full, y_full = generate_datasets(option)

    # Split the data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, test_size=0.2, random_state=42)

    # Apply the recursive regression model with adaptive sampling
    model_func = recursive_regression_with_adaptive_sampling(
        x_train, y_train, degree=3, max_iterations=10
    )

    # Evaluate the model on the test set
    y_test_pred = model_func(x_test)
    mse = np.mean((y_test - y_test_pred) ** 2)
    print(f"Test Set Mean Squared Error (MSE): {mse:.4f}")

    # Final plot with test data
    x_fine = np.linspace(x_full.min(), x_full.max(), 1000)
    y_fine = model_func(x_fine)

    plt.figure(figsize=(10, 6))
    plt.scatter(x_full, y_full, label="All Data Points", color="red")
    plt.plot(x_fine, y_fine, label="Final Fitted Model", color="blue")
    plt.scatter(x_test, y_test, label="Test Data", color="orange")
    plt.title("Final Fitted Model with Test Data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.show()
