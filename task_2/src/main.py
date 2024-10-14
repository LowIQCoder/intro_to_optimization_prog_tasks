import numpy as np
from task_2.src.interior_method import InteriorMethod
from task_1.src.simplex_method import SimplexMethod


def parse_input() -> tuple[str, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    problem_type = input("Enter the problem type (max or min): ")
    if problem_type not in ["max", "min"]:
        raise ValueError("Invalid problem type. Please enter 'max' or 'min'.")

    C = np.array(
        list(
            map(
                float,
                input(
                    "Enter coefficients of the objective function (C) separated by spaces: "
                ).split(),
            )
        )
    )

    m = int(input("Enter the number of constraints: "))

    A = []
    for i in range(m):
        constraint = list(
            map(
                float,
                input(
                    f"Enter coefficients of constraint {i+1} (including slack variables) separated by spaces: "
                ).split(),
            )
        )
        if len(constraint) != len(C) + m:
            raise ValueError(
                "Number of coefficients does not match the number of variables plus slack variables."
            )
        A.append(constraint)

    A = np.array(A)

    b = np.array(
        list(
            map(
                float,
                input(
                    "Enter the right-hand side values for constraints separated by spaces: "
                ).split(),
            )
        )
    )

    if len(b) != m:
        raise ValueError("Number of b values does not match the number of constraints.")

    accuracy = float(input("Enter the accuracy of the solution: "))

    x_init = np.array(
        list(
            map(
                float,
                input(
                    "Enter initial values for decision variables separated by spaces: "
                ).split(),
            )
        )
    )
    return problem_type, C, A, b, accuracy, x_init


def main():
    """
    Main function to execute the Simplex method solver.
    """
    try:
        problem_type, C, A, b, accuracy, x_init = parse_input()
        interior_method = InteriorMethod()
        solution, optimal_value = interior_method.solve(
            problem_type, C, A, b, accuracy, x_init, alpha=0.5
        )
        print("\nSolution for a = 0.5:")
        print("A vector of decision variables")
        for idx, val in enumerate(solution):
            print(f"x{idx + 1} = {val}")
        print(f"Optimal value of the objective function: {optimal_value}")

        solution, optimal_value = interior_method.solve(
            problem_type, C, A, b, accuracy, x_init, alpha=0.9
        )
        print("\nSolution for a = 0.9:")
        print("A vector of decision variables")
        for idx, val in enumerate(solution):
            print(f"x{idx + 1} = {val}")
        print(f"Optimal value of the objective function: {optimal_value}")

        if np.min(A[:, len(C) :]) == -1 or problem_type == "min":
            print("Simplex method from Task 1 is not applicable")
        else:
            A = A[:, : len(C)].tolist()
            C = C.tolist()
            b = b.tolist()
            print(A)
            print(b)
            print(C)
            solution, optimal_value = SimplexMethod().solve(C, A, b, accuracy)
            print("\nSolution of Simplex method from Task 1:")
            print("A vector of decision variables")
            for idx, val in enumerate(solution):
                print(f"x{idx + 1} = {val}")
            print(f"Optimal value of the objective function: {optimal_value}")

    except Exception as exc:
        print(str(exc))


if __name__ == "__main__":
    main()
