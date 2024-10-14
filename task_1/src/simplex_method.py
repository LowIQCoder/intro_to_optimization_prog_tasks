class SimplexMethod:
    def solve(
        self, C: list[float], A: list[list[float]], b: list[float], accuracy: float
    ) -> tuple[list[float], float]:
        """
        Solves the Linear Programming problem using the Simplex method.
        Returns:
            A tuple containing variables optimal values and the optimal value of the objective function.
        Raises:
            Exception: If the method is not applicable.
        """
        self.C = C
        self.A = A
        self.b = b
        self.accuracy = accuracy
        self.num_constraints = len(A)
        self.num_variables = len(C)
        self.table = self.__initialize_table()

        if any(rhs < 0 for rhs in self.b):
            raise Exception("The method is not applicable!")

        while True:
            pivot_col = self.__find_pivot_col()
            if pivot_col is None:
                break

            pivot_row = self.__find_pivot_row(pivot_col)
            if pivot_row is None:
                raise Exception("The method is not applicable!")

            self.__pivot(pivot_row, pivot_col)

        solution = self.__extract_solution()
        return solution, self.table[-1][-1]

    def __initialize_table(self) -> list[list[float]]:
        """Initializes the simplex table with slack variables."""
        table = [
            [
                0.0 if -self.accuracy <= element <= self.accuracy else element
                for element in row
            ]
            + [0] * self.num_constraints
            + [rhs]
            for row, rhs in zip(self.A, self.b)
        ]
        for i in range(self.num_constraints):
            table[i][self.num_variables + i] = 1
        table.append([-c for c in self.C] + [0] * self.num_constraints + [0])
        return table

    def __find_pivot_col(self) -> int | None:
        """
        Finds the pivot column using the Bland's rule to prevent cycling.

        Returns:
            The index of the pivot column or None if optimal.
        """
        for j in range(len(self.table[-1]) - 1):
            if self.table[-1][j] < 0:
                return j
        return None

    def __find_pivot_row(self, pivot_col: int) -> int | None:
        """
        Finds the pivot row based on the minimum ratio test.

        Parameters:
            pivot_col: The index of the pivot column.

        Returns:
            The index of the pivot row or None if unbounded.
        """
        min_ratio = float("inf")
        pivot_row = None
        for i in range(self.num_constraints):
            element = self.table[i][pivot_col]
            if -self.accuracy <= element <= self.accuracy:
                continue
            ratio = self.table[i][-1] / element
            if ratio > 0 and ratio < min_ratio:
                min_ratio = ratio
                pivot_row = i
        return pivot_row

    def __pivot(self, pivot_row: int, pivot_col: int):
        """
        Performs the pivot operation to update the table.

        Parameters:
            pivot_row: The index of the pivot row.
            pivot_col: The index of the pivot column.
        """
        pivot_element = self.table[pivot_row][pivot_col]
        self.table[pivot_row] = [x / pivot_element for x in self.table[pivot_row]]
        for i in range(len(self.table)):
            if i != pivot_row:
                factor = self.table[i][pivot_col]
                self.table[i] = [
                    x - factor * y for x, y in zip(self.table[i], self.table[pivot_row])
                ]

    def __extract_solution(self) -> list[float]:
        """
        Extracts the solution from the final table.

        Returns:
            A list of decision variable values.
        """
        solution: list[float] = [0] * self.num_variables
        for i in range(self.num_constraints):
            for j in range(self.num_variables):
                if self.table[i][j] == 1 and all(
                    self.table[k][j] == 0 for k in range(self.num_constraints) if k != i
                ):
                    solution[j] = self.table[i][-1]
                    break
        return solution
