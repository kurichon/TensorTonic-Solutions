import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    arr = np.array(A)
    (rows,cols) = arr.shape

    transpose = np.zeros((cols,rows))

    row_count = 0
    col_count = 0
    for i in arr: # arr_cols into transpose rows
        for j in i:
            #arr_rows into transpose_cols
            transpose[row_count,col_count] = j
            row_count+=1
    
        row_count = 0
        col_count+=1
    return transpose
    # Write code here
    pass
