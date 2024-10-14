import numpy as np
import numpy.linalg as linalg


class InteriorMethod:
    def solve(
        self,
        problem_type: str,
        c: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        accuracy: float,
        x_init: np.ndarray,
        alpha: float = 0.5,
    ) -> tuple[np.ndarray, float]:
        """
        Solves the linear programming problem using the interior method.

        Args:
            problem_type (str): 'min' or 'max'.
            c (np.ndarray): Coefficients of the objective function.
            A (np.ndarray): Coefficients of the constraints.
            b (np.ndarray): Right-hand side of the constraints.
            accuracy (float): Accuracy of the solution.
            x_init (np.ndarray): Initial guess for the solution.
            alpha (float): Step size for the method.

        Returns:
            tuple[np.ndarray, float]: Solution and objective value.
        """
        if problem_type not in ["min", "max"]:
            raise ValueError("problem_type must be 'min' or 'max'")

        c = np.pad(c, (0, A.shape[1] - len(c)), "constant")

        if problem_type == "min":
            c = -c

        x = x_init

        try:
            while True:
                D = np.diag(x)

                A_tilde = A @ D
                c_tilde = D @ c

                P = (
                    np.eye(len(x))
                    - A_tilde.T @ linalg.pinv(A_tilde @ A_tilde.T) @ A_tilde
                )
                c_p = P @ c_tilde

                v = np.abs(np.max(-c_p))

                x_tilde = D @ (np.ones_like(x) + (alpha / v) * c_p)

                if linalg.norm(x_tilde - x) < accuracy:
                    break

                x = x_tilde

            objective_value = float(c @ x)

            return x, -objective_value if problem_type == "min" else objective_value
        except linalg.LinAlgError:
            raise Exception("The method is not applicable!")
