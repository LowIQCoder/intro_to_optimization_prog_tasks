from simplex_method import SimplexMethod


def parse_input() -> tuple[list[float], list[list[float]], list[float], int]:
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

    accuracy: int = int(input("Enter the number of decimal places for output: "))

    return C, A, b, accuracy


def main():
    """
    Main function to execute the Simplex method solver.
    """
    try:
        C, A, b, accuracy = parse_input()
        simplex_method = SimplexMethod()
        solution, optimal_value = simplex_method.solve(C, A, b)
        print("A vector of decision variables")
        for idx, val in enumerate(solution):
            print(f"x{idx + 1} = {val:.{accuracy}f}")
        print(f"Maximum value of the objective function: {optimal_value:.{accuracy}f}")

    except Exception as exc:
        print(str(exc))


if __name__ == "__main__":
    main()
