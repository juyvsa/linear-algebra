from augmented_matrix import AugmentedMatrix
from matrix import format_sub_number
import re
import numpy as np


class LinearSystem:  # add constructor with LinearEquations
    def __init__(self, equations, variables):
        self.eqns = equations
        self.variables = variables
        self.rhs_index = len(self.variables)

    @property
    def coeff_matrix(self):
        mat = AugmentedMatrix(np.zeros((len(self.eqns), len(self.variables) + 1)), len(self.variables) + 1)

        for j in range(len(self.variables)):
            for i in range(len(self.eqns)):
                pattern = re.compile("-?\d*" + self.variables[j])  # assuming all vars are on left side of eqn
                pattern.findall(self.eqns[i])
                mat[i,j] += float(self.eqns[i][:-len(self.variables[j])])  # puts coeff in spot

        for i in range(len(self.eqns)):
            sides = self.eqns[i].split("=")
            mat[i][mat.augment_index] = float(sides[-1])  # RHS
        return mat

    @property
    def solved_matrix(self):
        return self.coeff_matrix.rref

    @property
    def is_consistent(self):
        return self.solved_matrix.is_consistent

    @property
    def is_homogeneous(self):
        return self.solved_matrix.is_homogeneous

    @property
    def fixed_var_cols(self):
        return self.solved_matrix.pivots[:,1]

    @property
    def fixed_vars(self):
        for index in self.fixed_var_cols:
            yield self.variables[index]

    @property
    def free_var_cols(self):
        matrix = self.solved_matrix
        return np.array(set(range(matrix.shape[1])) - set(matrix.pivots[:,1])) # all cols - piv cols

    @property
    def free_vars(self):
        for index in self.free_var_cols:
            yield self.variables[index]

    def sol_set(self):
        matrix = self.solved_matrix
        for var_num in range(len(self.variables)):
            var_row = matrix.get_col_pivot(var_num)
            if var_row is not None:
                for col_j in np.nonzero(matrix[var_row])[1:]:
                    free_index = np.where(self.free_var_cols == col_j)[0][0]  # which of free vars is it (for format)
                    str(matrix[var_row, col_j] * -1) + "s" + format_sub_number(free_index)






