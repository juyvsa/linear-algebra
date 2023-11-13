from matrix import Matrix
import numpy as np


class AugmentedMatrix(Matrix):
    def __new__(cls, *args):  # from array to augmented matrix
        if len(args) > 2:
            mat = super().__new__(cls, np.column_stack(args[-1]))
            mat.delimiter_index = args[-1]

        elif isinstance(args[1], (int, np.int_)):  # args are input array and augment index
            mat = super().__new__(cls, args[0])
            mat.delimiter_index = args[1]

        elif isinstance(args[1], (np.ndarray, list, tuple)):
            if not isinstance(args[1][0], (np.ndarray, list, tuple)):
                rhs = np.reshape(args[1], (-1, 1))
            else:
                rhs = args[1]
            mat = super().__new__(cls, np.concatenate((args[0], rhs), axis=1))
            mat.delimiter_index = args[0].shape[0]

        else:
            raise TypeError()

        mat.left = mat[:,:-mat.delimiter_index]
        mat.right = mat[:,mat.delimiter_index:]
        return mat

    @property
    def is_consistent(self):
        if not self.is_echelon_form:
            pivs = self.echelon_form.pivots()
        else:
            pivs = self.pivots()
        for _, piv_col in pivs:
            if piv_col >= self.delimiter_index:  # pivot is after the delimiter
                return False
        return True

    @property
    def is_homogenous(self):
        return len(np.nonzero(self.right)) == 0
