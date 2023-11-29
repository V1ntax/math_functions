# Math functions

def round(x):
    if ((x % 1) < 1):
        if ((x % 1) < 0.5):
            x = x - x % 1
        else:
            x = x + (1 - x % 1)
    return x

def max(list):
    result = 0
    for num in list:
        if num > result:
            result = num
    return result

def min(list):
    result = list[0]
    for num in list:
        if num < result:
            result = num
    return result

def sum(list):
    res = 0
    for i in list:
        res += i
    return res

def arithmeticMean(list):
    res = 0
    for i in list:
        res += i
    return res / len(list)

def power(x, powNum):
    res = 1
    for i in range(int(powNum)):
        res = res * x
    return res

def abs(x):
    if x < 0:
        return -x
    return x

def factorial(x):
    if x < 0:
        return "Invalid number"
    res = 1
    for i in range(1, x+1):
        res = i * res
    return res

# Another method of factorial executes through recursion but has a smaller limit than first method (at least in 2 times)
# def factorial(x):
#     if x == 0 or x == 1:
#         return 1
#     return x * fact(x-1)

def doubFactorial(x):
    if x < 0:
        return "Invalid number"
    res = 1
    if x % 2 == 0:
        for i in range(2, x + 1, 2):
            res = i * res
    else:
        for i in range(1, x + 1, 2):
            res = i * res
    return res

# for future improvements: to add  power of root as second required variable
def sqrt(x, pow = 2):
    if type(x) is not int or x < 0:
        return "Invalid value"

    dvs = []
    cd = 2
    # Finds all dividers
    while x >= cd:
        if x % cd == 0:
            dvs.append(cd)
            x /= cd
            cd = 2
        else:
            cd += 1
    dvs.append(x)

    res = 1
    excess = 1

    while len(dvs) > 0:
        x = dvs[0]
        quant = dvs.count(x)
        # Finds dividers that is != x (current number)
        dvs = [i for i in dvs if i != x]

        # if quant (quantity of dividers of current number) is even num than we don't have excess, else we do
        if quant % 2 == 0:
            res *= power(x, quant / 2)
        else:
            res *= power(x, round(quant / 2) - 1)
            excess *= int(x)

    if excess == 1:
        return res
    return f"{res}√{excess}"

def sqrtApprox(number):
    if type(number) is not int or number < 0:
        return "Invalid value"
    if number == 0 or number == 1:
        return number

    # Initial guess for the square root
    x = number
    y = (x + number / x) / 2

    # Keep iterating until a good enough approximation is achieved
    while y < x:
        x = y
        y = (x + number / x) / 2

    return x

def median(list):
    for i in range(1, len(list)):
        value_to_sort = list[i]

        while list[i - 1] > value_to_sort and i > 0:
            list[i], list[i - 1] = list[i - 1], list[i]
            i = i - 1

    # Median for odd quantity
    if len(list) % 2 == 1:
        return list[int((len(list) / 2) + 0.5)]
    # Median for even quantity
    else:
        return (list[int(len(list)/2 - 1)] + list[int(len(list)/2)]) / 2

# def exp(x):
def exp(x, terms=20):
    result = 0
    for n in range(terms):
        result += x ** n / factorial(n)
    return result

# Logarithm for natural numbers
#logₐb = y; aʸ = b
def log(a, b):
    if a <= 0 or b <= 0:
        return "Invalid value"
    for y in range(2, 100):
        if a ** y == b:
            return y
    return "Result is non-natural number"

#ln(x) = log e(x)
def ln(x, terms = 100):
    if x <= 0:
        return "Non-natural number is entered"

    result = 0
    if x > 2:  # To improve convergence, we use ln(x) = -ln(1/x) for x > 2
        x = 1 / x
        result -= ln(x) # Use ln(x) = -ln(1/x) for x > 2
        return result

    # Here we use the Taylor series expansion of ln(x)
    for n in range(1, terms + 1):
        result += ((-1) ** (n - 1)) * ((x - 1) ** n) / n
    return result


# Operations with matrices
import numpy as np

# try:
#     a, b, c = [int(x) for x in input("Enter three values of the first row: ").split()]
#     x, y, z = [int(x) for x in input("Enter three values of the second row: ").split()]
#     n, m, p = [int(x) for x in input("Enter three values of the third row: ").split()]
# except:
#     print("You have entered invalid symbol or incorrect amount of numbers!")
#     exit()

arr = np.array([[1, 3, 12], [10, 14, 6], [-3, 4, 7]])
arr2 = np.array([[3, 2, 5], [8, 4, 9], [1, -2, 8]])
array1 = np.array([[2, 4, 5, 1], [0, 11, 53, -5], [1, 4, 7, 21], [1, 8, 49, -12]])
array2 = np.array([[0, 1, 3], [8, 4, 7]])
# arr = np.array([[a, b, c], [x, y, z], [n, m, p]])
# print("Your array :\n"+arrLook)

# MatrixView
def matrixView(matrix):
    # Find the maximum width for each column
    column_widths = [max(map(lambda x: len(str(x)), col)) for col in zip(*matrix)]

    # Format and print the array
    for row in matrix:
        formatted_row = "("+' '.join(f'{value:<{width}}' for value, width in zip(row, column_widths))+")"
        print(formatted_row)

# Multiplying a matrix by a number
def matrixMult(multiplier, array):
    if type(multiplier) != int:
        print("You have entered invalid multiplier!")
        exit()

    for i in range(array.shape[1]):
        array[[i]] *= multiplier

# Adding, subtracting matrices
def matrixAdd(arr1, arr2):
    row, column = 0, 0
    arr3 = []
    if arr1.shape == arr2.shape:
        while row < arr1.shape[0]:
            while column < arr1.shape[1]:
                # Append the sum of corresponding elements to arr3
                arr3.append(arr1[row][column] + arr2[row][column])
                column += 1

            # Move to the next row and reset the column index
            row += 1
            column = 0

    # Convert the list to a NumPy array
    arr3 = np.array(arr3)
    # Reshape to the shape of the original arrays
    arr3 = arr3.reshape(arr1.shape)

    return arr3

def matrixSubt(arr1, arr2):
    row, column = 0, 0
    arr3 = []
    if arr1.shape == arr2.shape:
        while row < arr1.shape[0]:
            while column < arr1.shape[1]:
                # Append the subtraction of corresponding elements to arr3
                arr3.append(arr1[row][column] - arr2[row][column])
                column += 1

            # Move to the next row and reset the column index
            row += 1
            column = 0

    # Convert the list to a NumPy array
    arr3 = np.array(arr3)
    # Reshape to the shape of the original arrays
    arr3 = arr3.reshape(arr1.shape)

    return arr3

# Matrix product
def matrixProd(arr1, arr2):
    # initialize rows and columns
    row, col = 0, 0
    row2, col2 = 0, 0
    # arr3 is final array; sums is the result of multiplying the elements of two matrices
    arr3 = []
    sums = []
    # Check for matrix dimensionality
    if arr1.shape == arr2.shape:
        # Passing through the rows of the matrix(arr1)
        while row < arr1.shape[0]:
            # Passing through the columns of the second matrix(arr2)
            while col2 < arr1.shape[0]:
                while row2 < arr1.shape[1]:
                    sums.append(arr1[row][col] * arr2[row2][col2])

                    col += 1
                    row2 += 1
                # Appends to arr3 list : sum of n-amount numbers from the list sums(multiplied elements of two matrices)
                # In fact, it appends one element of new matrix
                arr3.append(sum(sums))
                sums.clear()

                col, row2 = 0, 0
                col2 += 1

            row += 1
            col, row2, col2 = 0, 0, 0

    # Convert the list to a NumPy array
    arr3 = np.array(arr3)
    # Reshape to the shape of the original arrays
    arr3 = arr3.reshape(arr1.shape)

    return arr3

# Transpose a matrix
def transpMatrix(array):
    # initialize rows and columns
    rows, cols = len(array), len(array[0])
    transposed_matrix = [[0] * rows for _ in range(cols)]

    # Passing through the rows
    for row in range(rows):
        # Passing through the columns
        for col in range(cols):
            transposed_matrix[col][row] = array[row][col]

    return transposed_matrix


# Calculating the third-order determinant of a matrix (detA)


# Inverse matrices

arr3 = transpMatrix(array2)
matrixView(arr3)