from simplex_method import SimplexMethod


def parse_input() -> tuple[list[float], list[list[float]], list[float], float]:
    C: list[float] = list(
        map(
            float,
            input(
                "Enter coefficients of the objective function (C) separated by spaces: "
            ).split(),
        )
    )

    m = int(input("Enter the number of constraints: "))

    A: list[list[float]] = []
    b: list[float] = []
    for i in range(m):
        constraint: list[float] = list(
            map(
                float,
                input(
                    f"Enter coefficients of constraint {i+1} separated by spaces: "
                ).split(),
            )
        )
        if len(constraint) != len(C):
            raise ValueError(
                "Number of coefficients does not match the number of variables."
            )
        A.append(constraint)
        b_value: float = float(
            input(f"Enter the right-hand side value for constraint {i+1}: ")
        )
        b.append(b_value)

    accuracy: float = float(input("Enter the accuracy of the solution: "))

    return C, A, b, accuracy


def print_problem(C: list[float], A: list[list[float]], b: list[float]):
    print("Optimization problem:")
    objective = " + ".join([f"{c:.2f} * x{i+1}" for i, c in enumerate(C)])
    print(f"max z = {objective}")
    print("subject to:")
    for i, constraint in enumerate(A):
        constraint_str = " + ".join(
            [f"{coef:.2f} * x{j+1}" for j, coef in enumerate(constraint)]
        )
        print(f"  {constraint_str} <= {b[i]:.2f}")

def main():
    """
    Main function to execute the Simplex method solver.
    """
    try:
        C, A, b, accuracy = parse_input()
        print_problem(C, A, b)
        simplex_method = SimplexMethod()
        solution, optimal_value = simplex_method.solve(C, A, b, accuracy)
        print("\nSolution:")
        print("A vector of decision variables")
        for idx, val in enumerate(solution):
            print(f"x{idx + 1} = {val}")
        print(f"Maximum value of the objective function: {optimal_value}")

    except Exception as exc:
        print(str(exc))


if __name__ == "__main__":
    main()
