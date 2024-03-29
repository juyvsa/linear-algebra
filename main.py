from matrix import Matrix
from augmented_matrix import AugmentedMatrix
from linear_equation import LinearEquation
import numpy as np


def is_linear_combination(vect, *a):  # vect is vector, a is collection (list) of vectors
    assert len(a[0]) == len(vect)  # a and v are both n-vectors. technically i should check all a_i

    mat = AugmentedMatrix(np.column_stack((*a, vect)), len(a))
    print(mat)
    return mat.is_consistent


def is_in_span(vect, *a):  # check if vector is in the span of vectors {a1, ... , an}
    return is_linear_combination(vect, *a)


def does_span_rn(*a):  # a is collection of k n-vectors
    if len(a) < len(a[0]): # if k < n, a_1,..,a_n cannot span Rn
        return False

    mat = Matrix(np.column_stack(a))
    return mat.are_all_rows_pivs


# TESTING
"""
u1 = [1,1,-2]
u2 = [2,3,-3]
u3 = [0,1,1]
u4 = [-1,3,4]

print(does_span_rn(u1, u2, u3, u4))


e = LinearEquation(["x","y","z","w"],[0,-4,3,0],4)
print(str(e))
print(e)


m = Matrix([[-1,1,8,2],[3,8,2,-5],[4,9,-3,1],[5,13,7,6]])
print(m.echelon_form)

mat = Matrix([[-5,-4,-4],[4,3,4],[2,2,1]])
print(mat.inverse)

mat16 = AugmentedMatrix([[1,1,1,1],[1,2,3,3],[1,2,2,2],[1,2,3,4]], np.identity(4))
print(mat16.rref)


mat3 = Matrix([[1,1,-1,3],[1,2,2,2],[1,2,1,3],[1,3,4,1]])
print(mat3.echelon_form)

mat4 = Matrix([[1,2,1],[-1,-1,-2],[0,1,0],[2,3,3]])
print(mat4.echelon_form)
"""

mat7 = Matrix([[1,-10,8],[4,-18,13],[4,-20,15]])
print(mat7.echelon_form)