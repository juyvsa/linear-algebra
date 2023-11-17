import numpy as np
from fractions import Fraction


def format_sub_number(s):  # turns row numbers into subscripts for clarity in printing
    s += 1
    s = str(s)
    sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return s.translate(sub)


class Matrix(np.ndarray):
    def __new__(cls, *args):
        if len(args) > 1:
            input_arr = np.column_stack(args)
        else:
            input_arr = args[0]
        mat = np.asarray(input_arr, dtype=np.float64).view(cls)

        return mat

    """ELEMENTARY ROW OPERATIONS (EXCEPT MY ELIMINATION IS SCALAR MULT AND ELIMINATION IN ONE)"""
    def swap_rows(self, rowi_index, rowj_index):  # Ri <--> Rj
        assert 0 <= rowi_index < self.shape[0] and 0 <= rowj_index < self.shape[0]

        r1 = self[rowi_index]
        self[rowi_index] = self[rowj_index]
        self[rowj_index] = r1
        print("R{} ↔ R{}".format(format_sub_number(rowi_index), format_sub_number(rowj_index)))
        print(self)

    def eliminate_row(self, r_loc, c1, rowi_index, c2, rowj_index): # R_loc --> c1Ri + c2Rj
        assert c1 != 0 and c2 != 0
        assert 0 <= r_loc < self.shape[0] and 0 <= rowi_index < self.shape[0] and 0 <= rowj_index < self.shape[0]

        self[r_loc] = c1 * self[rowi_index] + c2 * self[rowj_index]
        print("R{} ↦ ({})R{} + ({})R{}".format(format_sub_number(r_loc), c1, format_sub_number(rowi_index), c2,
                                               format_sub_number(rowj_index)))
        print(self)

    def scalar_mult(self, row_index, c):  # Ri --> cRi
        assert 0 <= row_index < self.shape[0]
        assert c != 0

        self[row_index] = c * self[row_index]

        if isinstance(c, int):
            print("R{} ↦ {}R{}".format(format_sub_number(row_index), c, format_sub_number(row_index)))
        else:
            c_frac = Fraction(c).limit_denominator()
            print("R{} ↦ {}/{} R{}".format(format_sub_number(row_index), c_frac.numerator, c_frac.denominator,
                                           format_sub_number(row_index)))
        print(self)

    """PIVOT FINDING/ANALYZING"""
    def find_new_pivot(self, row_index):  # returns the entry we are going to make into a pivot
        assert 0 <= row_index < self.shape[0]

        if row_index == 0:  # first row
            nonzero_indices = np.nonzero(self[row_index])[0]  # all nonzero indices in first row
            if len(nonzero_indices) > 0:  # there are nonzero indices in first row
                return nonzero_indices[0]  # first nonzero index of first row
            else:  # first row all zero
                for other_row_index in reversed(range(1, self.shape[0])):
                    other_nonzero_indices = np.nonzero(self[other_row_index])
                    if len(other_nonzero_indices) > 0:    # find nonzero row
                        self.swap_rows(0, other_row_index)  # make it first row
                        return other_nonzero_indices[0]  # return the first nonzero index of new first row
                return  # matrix is all zeros

        else: # not first row
            prev_piv_index = self.get_row_pivot(row_index - 1)  # index of first nonzero entry of previous row

            for i in range(prev_piv_index + 1, self.shape[1]):  # all cols from one after previous pivot
                nonzero_indices = np.nonzero(self[:, i])[0]  # nonzero elements of ith col of self
                if len(nonzero_indices) != 0:  # there are nonzero entries in this col, in current row or after
                    if nonzero_indices[0] > prev_piv_index + 1:
                        # zero in this row but nonzero entry later in column
                        self.swap_rows(row_index, nonzero_indices[0])  # put nonzero row in current row
                    return i  # there is in the ith column of this row, whether i had to swap or not
            return  # if you're still here, the rows at or below the current row are all zero

    def get_row_pivot(self, row_index):  # returns the first nonzero entry of this row
        assert 0 <= row_index < self.shape[0]

        nonzero_indices = np.nonzero(self[row_index])[0]
        return nonzero_indices[0] if len(nonzero_indices) > 0 else None

    def get_col_pivot(self, col_index): # returns the last nonzero entry of col
        assert 0 <= col_index < self.shape[1]

        nonzero_indices = np.nonzero(self[:, col_index])[0]
        return nonzero_indices[-1] if len(nonzero_indices) > 0 else None

    @property
    def pivots(self):  # returns all pivots of echelon form matrix (if not echelon, this would not be unique
        assert self.is_echelon_form

        for row_index in range(self.shape[0]):
            piv = self.get_row_pivot(row_index)
            if piv is not None:
                yield row_index, piv

    @property
    def are_all_rows_pivs(self):  # returns if all rows have pivots (aka are pivot rows)
        return len(list(self.echelon_form.pivots)) == self.shape[0]

    @property
    def are_all_cols_pivs(self):  # returns if all columns have pivots (aka are pivot rows)
        return len(list(self.echelon_form.pivots)) == self.shape[1]

    """GAUSSIAN ELIMINATION"""
    def eliminate_below(self, row_index, col_index):  # row_index is row we are turning into a pivot row
        assert 0 <= row_index < self.shape[0] and 0 <= col_index < self.shape[1]

        col = self[:, col_index]
        nonzero_indices = np.nonzero(col)[0]  # nonzero elements of nth col of self

        for index in nonzero_indices:  # want to eliminate each nonzero entry
            if index <= row_index:
                continue
            lcm = np.lcm(int(col[index]), int(col[row_index]))
            c1 = lcm // abs(col[index])  # r1 = index is row we're eliminating
            c2 = lcm // abs(col[row_index])  # r2 = row_index is pivot row

            if (col[row_index] > 0) == (col[index] > 0):  # same sign
                if (col_index + 1 == self.shape[1]
                        or c1 * self[index, col_index + 1] - c2 * self[row_index, col_index + 1] > 0):
                    # ideally we would have c2 be negative, so if it doesn't matter (bc this is last index) we do that
                    c2 *= -1
                else:
                    c1 *= -1

            self.eliminate_row(index, c1, index, c2, row_index)
            self.check_row_factors(index)

    def check_row_factors(self, index):  # checks if a row has common factors and removes them if they exist
        c = 1

        row_gcd = int(np.gcd.reduce(self[index].astype(int)))  # gcd inputs have to be ints
        if row_gcd > 1:
            c /= row_gcd

        piv = self.get_row_pivot(index)
        if piv is not None:
            if self[index, piv] < 0:
                c *= -1

        if c != 1:
            self.scalar_mult(index, c)

    def to_echelon_form(self):  # performs Gaussian elimination on a matrix until it is in echelon form
        assert 1 < len(self.shape) <= 2  # must be 2D. row vectors must be written as 2 x 1 matrices

        if self.is_echelon_form:
            return

        for row_index in range(self.shape[0]):
            piv_col_index = self.find_new_pivot(row_index)
            if piv_col_index is None:
                break
            self.eliminate_below(row_index, piv_col_index)

    @property
    def echelon_form(self):  # returns the echelon form of a matrix without modifying the original
        m = self.copy()
        m.to_echelon_form()
        return m

    @property
    def is_echelon_form(self):  # returns True if the matrix is in echelon form
        prev_piv = self.get_row_pivot(0)
        for row_index in range(1, self.shape[0]):
            this_piv = self.get_row_pivot(row_index)

            if prev_piv is None and this_piv is not None:  # prev row all zeros,  but not this row
                return False
            elif this_piv is None:
                prev_piv = this_piv
            elif this_piv <= prev_piv:  # if this row's pivot is to the left of the previous'
                return False
        return True

    def eliminate_above(self, row_index, col_index):  # eliminates all entries above a pivot
        assert 0 <= row_index < self.shape[0] and 0 <= col_index < self.shape[1]

        col = self[:, col_index]
        nonzero_indices = np.nonzero(col[:row_index])[0]  # nonzero elements of nth col of self above pivot

        for index in nonzero_indices:  # want to eliminate each nonzero entry
            lcm = np.lcm(int(col[index]), int(col[row_index]))
            c1 = lcm // abs(col[index])  # r1 = index is row we're eliminating
            c2 = lcm // abs(col[row_index])  # r2 = row_index is pivot row

            if (col[row_index] > 0) == (col[index] > 0):  # same sign
                piv_index = self.get_row_pivot(index)  # piv of row we're reducing
                if self[index, piv_index] > 0:  # should not == 0
                    c2 *= -1
                else:
                    c1 *= -1

            self.eliminate_row(index, c1, index, c2, row_index)
            self.check_row_factors(index)

    def ef_to_rref(self):  # takes a matrix in echelon form into a matrix in reduced row echelon form
        assert 1 < len(self.shape) <= 2
        assert self.is_echelon_form

        if self.is_rref:
            return

        for row_index in reversed(range(1, self.shape[0])):  # start from bottom
            piv_col_index = self.get_row_pivot(row_index)
            if piv_col_index is None:
                continue
            self.eliminate_above(row_index, piv_col_index)  # if there is a pivot in the row, use it to elim above it

        for piv_loc in self.pivots:  # making sure that all pivots = 1
            row, col = piv_loc
            if self[row, col] != 1:
                self.scalar_mult(row, 1 / self[row, col])  # there should not be any decimals until now

    def to_rref(self):  # takes a matrix from any state to reduced row echelon form with Gaussian elimination
        if not self.is_echelon_form:
            self.to_echelon_form()
        if not self.is_rref:
            self.ef_to_rref()

    @property
    def rref(self):  # returns the reduced row echelon form of a matrix without modifying the original
        m = self.copy()
        m.to_rref()
        return m

    @property
    def is_rref(self):  # returns True if a matrix is in reduced row echelon form
        if not self.is_echelon_form:  # rref form is also an echelon form
            return False
        else:
            for pivot_loc in self.pivots:
                row, col = pivot_loc
                if self[row, col] != 1:  # all pivots = 1
                    return False
                elif not np.array_equal(np.nonzero(self[:, col])[0], [row]):  # only nonzero entry should be pivot
                    return False
        return True

    @property
    def inverse(self):
        from augmented_matrix import AugmentedMatrix

        assert self.shape[0] == self.shape[1]
        aug_mat = AugmentedMatrix(self, np.identity(self.shape[0]))
        aug_mat.to_rref()
        if np.array_equal(aug_mat.left, np.identity(self.shape[0])):
            return aug_mat.right
        return

