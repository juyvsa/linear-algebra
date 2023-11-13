class LinearEquation:
    def __init__(self, variables, coeffs, rhs=0):
        assert len(variables) == len(coeffs)

        self.variables = variables
        self.coefficients = coeffs
        self.rhs = rhs

    def __str__(self):
        s = ""
        for i in range(len(self.coefficients)):
            coeff_i = self.coefficients[i]
            if coeff_i == 0:
                continue
            elif coeff_i > 0 and s != "":
                s += "+"

            if coeff_i == -1:
                s += "-"
            elif coeff_i != 1:
                s += str(coeff_i)

            s += self.variables[i]

        s += ("=" + str(self.rhs))
        return s
