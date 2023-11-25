#include <omp.h>
#include <bits/stdc++.h>

using namespace std;


/**
 * @brief Prints a square matrix to the standard output.
 * 
 * This function iterates over each row and column of a given 
 * square matrix and prints its elements. Each row of elements 
 * is printed on a new line and elements in a row are separated 
 * by a space. An additional newline is printed at the end for 
 * better visual separation.
 * 
 * @param n    The size of the matrix (number of rows/columns).
 * @param mat  A pointer to a 2D array representing the matrix to be printed.
 */
void print(int n, int** mat)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << mat[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}


/**
 * @brief Allocates memory for a square matrix of size n x n.
 * 
 * This function dynamically allocates memory for a square matrix
 * of size n x n using a contiguous memory layout. It creates an 
 * array of pointers (array) to represent the rows, and a contiguous 
 * block of memory (data) to store the elements. Each row pointer 
 * in array points to the corresponding row in data.
 * 
 * It is the caller's responsibility to free the allocated memory 
 * using a corresponding deallocation function.
 *
 * @param n    The size of the matrix (number of rows and columns).
 * @return     A pointer to the 2D array representing the allocated matrix.
 */
int** allocateMatrix(int n)
{
    int* data = (int*)malloc(n * n * sizeof(int));
    int** array = (int**)malloc(n * sizeof(int*));
    for (int i = 0; i < n; i++)
    {
        array[i] = &(data[n * i]);
    }
    return array;
}


/**
 * @brief Fills a square matrix with random values.
 * 
 * This function iterates over each row and column of a given
 * square matrix and assigns each element a random value between
 * 0 and 4 (inclusive). The random values are generated using
 * the standard `rand()` function.
 * 
 * @param n    The size of the matrix (number of rows and columns).
 * @param mat  A reference to a pointer to a 2D array representing 
 *             the matrix to be filled.
 */
void fillMatrix(int n, int**& mat)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            mat[i][j] = rand() % 5;
        }
    }
}


/**
 * @brief Deallocates memory used by a square matrix.
 * 
 * This function deallocates memory that was previously allocated 
 * for a square matrix of size n x n. It assumes that the matrix 
 * was allocated using a contiguous memory layout, where the actual
 * data (elements of the matrix) and the array of pointers (representing 
 * the rows) were allocated separately.
 * 
 * The function first frees the block of memory containing the matrix data 
 * (mat[0]) and then frees the block containing the row pointers (mat).
 *
 * @param n    The size of the matrix (number of rows and columns).
 * @param mat  A pointer to a 2D array representing the matrix to be freed.
 */
void freeMatrix(int n, int** mat)
{
    free(mat[0]);
    free(mat);
}


/**
 * @brief Extracts a square sub-matrix from a given matrix.
 * 
 * This function extracts a sub-matrix of size n/2 x n/2 from a given square matrix of size n x n.
 * The sub-matrix starts at position (offseti, offsetj) in the original matrix.
 * 
 * @param n        Size of the input square matrix (n x n).
 * @param mat      Pointer to the input square matrix.
 * @param offseti  Row offset indicating the start of the sub-matrix in the original matrix.
 * @param offsetj  Column offset indicating the start of the sub-matrix in the original matrix.
 * @return         Pointer to the extracted sub-matrix of size (n/2) x (n/2).
 */
int** getSlice(int n, int** mat, int offseti, int offsetj)
{
    int m = n / 2;
    int** slice = allocateMatrix(m);
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < m; j++)
        {
            slice[i][j] = mat[offseti + i][offsetj + j];
        }
    }
    return slice;
}


/**
 * @brief Performs element-wise addition or subtraction of two square matrices.
 * 
 * This function takes two square matrices of the same size and performs
 * element-wise addition or subtraction based on the 'add' parameter.
 * 
 * @param n     Size of the input square matrices (n x n).
 * @param mat1  Pointer to the first input square matrix.
 * @param mat2  Pointer to the second input square matrix.
 * @param add   Boolean flag indicating the operation to perform:
 *              true  - Addition (result[i][j] = mat1[i][j] + mat2[i][j])
 *              false - Subtraction (result[i][j] = mat1[i][j] - mat2[i][j])
 * @return      Pointer to the resulting square matrix of size (n x n).
 */
int** addMatrices(int n, int** mat1, int** mat2, bool add)
{
    int** result = allocateMatrix(n);
    // Perform element-wise addition or subtraction based on 'add' parameter
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (add)
                result[i][j] = mat1[i][j] + mat2[i][j];
            else
                result[i][j] = mat1[i][j] - mat2[i][j];
        }
    }

    return result;
}


/**
 * @brief Combines four square matrices into a single larger square matrix.
 * 
 * This function takes four square matrices (c11, c12, c21, c22) of size (m x m)
 * and combines them to form a single larger square matrix of size (2m x 2m).
 * The input matrices are positioned in the resulting matrix as follows:
 * 
 *     | c11  c12 |
 *     |          |
 *     | c21  c22 |
 * 
 * @param m    Size of the input square matrices (m x m).
 * @param c11  Pointer to the top-left matrix to be combined.
 * @param c12  Pointer to the top-right matrix to be combined.
 * @param c21  Pointer to the bottom-left matrix to be combined.
 * @param c22  Pointer to the bottom-right matrix to be combined.
 * @return     Pointer to the resulting combined square matrix of size (2m x 2m).
 */
int** combineMatrices(int m, int** c11, int** c12, int** c21, int** c22)
{
    int n = 2 * m;
    int** result = allocateMatrix(n);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i < m && j < m)
                result[i][j] = c11[i][j]; // Top-left quadrant
            else if (i < m)
                result[i][j] = c12[i][j - m]; // Top-right quadrant
            else if (j < m)
                result[i][j] = c21[i - m][j]; // Bottom-left quadrant
            else
                result[i][j] = c22[i - m][j - m]; // Bottom-right quadrant
        } 
    }

    return result;
}


/**
 * @brief Performs matrix multiplication using a naive algorithm and parallelizes it using OpenMP.
 * 
 * This function multiplies two square matrices `mat1` and `mat2` of size n x n using a naive 
 * cubic complexity algorithm. The computation is parallelized using OpenMP to potentially 
 * speed up the operation on multi-core systems.
 * 
 * The function first allocates memory for the resulting matrix `prod` using `allocateMatrix`, 
 * then uses nested loops to perform the matrix multiplication. The outer two loops iterate 
 * over each row and column of the matrices, and the innermost loop computes the dot product 
 * of the corresponding row from `mat1` and column from `mat2`.
 * 
 * The `#pragma omp parallel for collapse(2)` directive is used to collapse the two outer loops
 * into one, allowing for better load balancing among threads.
 * 
 * @param n     The size of the matrices (number of rows and columns).
 * @param mat1  A pointer to a 2D array representing the first matrix.
 * @param mat2  A pointer to a 2D array representing the second matrix.
 * @return      A pointer to a 2D array representing the product of `mat1` and `mat2`.
 */
int** naive(int n, int** mat1, int** mat2)
{
    int** prod = allocateMatrix(n);

    int i, j;

    #pragma omp parallel for collapse(2)
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            prod[i][j] = 0;
            for (int k = 0; k < n; k++)
            {
                prod[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }

    return prod;
}


/**
 * @brief Implements the Strassen algorithm for matrix multiplication.
 * 
 * This function performs matrix multiplication using the Strassen algorithm,
 * which divides each matrix into four submatrices and recursively calculates
 * seven products of submatrices, reducing the number of multiplications required.
 * For small matrices (n <= 32), it falls back to naive multiplication.
 * OpenMP is used for parallelizing the computation.
 *
 * @param n     Size of the input square matrices (n x n).
 * @param mat1  Pointer to the first input matrix.
 * @param mat2  Pointer to the second input matrix.
 * @return      Pointer to the resulting matrix obtained by multiplying mat1 and mat2.
 */
int** strassen(int n, int** mat1, int** mat2)
{
    // For small matrices, fallback to naive multiplication
    if (n <= 32)
    {
        return naive(n, mat1, mat2);
    }

    // Divide the input matrices into four submatrices each
    int m = n / 2;
    int** a = getSlice(n, mat1, 0, 0);
    int** b = getSlice(n, mat1, 0, m);
    int** c = getSlice(n, mat1, m, 0);
    int** d = getSlice(n, mat1, m, m);
    int** e = getSlice(n, mat2, 0, 0);
    int** f = getSlice(n, mat2, 0, m);
    int** g = getSlice(n, mat2, m, 0);
    int** h = getSlice(n, mat2, m, m);

    // Calculate the seven products of submatrices using Strassen's formulas
    int** s1;
    #pragma omp task shared(s1)
    {
        int** bds = addMatrices(m, b, d, false);
        int** gha = addMatrices(m, g, h, true);
        s1 = strassen(m, bds, gha);
        freeMatrix(m, bds);
        freeMatrix(m, gha);
    }

    int** s2;
    #pragma omp task shared(s2)
    {
        int** ada = addMatrices(m, a, d, true);
        int** eha = addMatrices(m, e, h, true);
        s2 = strassen(m, ada, eha);
        freeMatrix(m, ada);
        freeMatrix(m, eha);
    }

    int** s3;
    #pragma omp task shared(s3)
    {
        int** acs = addMatrices(m, a, c, false);
        int** efa = addMatrices(m, e, f, true);
        s3 = strassen(m, acs, efa);
        freeMatrix(m, acs);
        freeMatrix(m, efa);
    }

    int** s4;
    #pragma omp task shared(s4)
    {
        int** aba = addMatrices(m, a, b, true);
        s4 = strassen(m, aba, h);
        freeMatrix(m, aba);
    }

    int** s5;
    #pragma omp task shared(s5)
    {
        int** fhs = addMatrices(m, f, h, false);
        s5 = strassen(m, a, fhs);
        freeMatrix(m, fhs);
    }

    int** s6;
    #pragma omp task shared(s6)
    {
        int** ges = addMatrices(m, g, e, false);
        s6 = strassen(m, d, ges);
        freeMatrix(m, ges);
    }

    int** s7;
    #pragma omp task shared(s7)
    {
        int** cda = addMatrices(m, c, d, true);
        s7 = strassen(m, cda, e);
        freeMatrix(m, cda);
    }

    #pragma omp taskwait

    freeMatrix(m, a);
    freeMatrix(m, b);
    freeMatrix(m, c);
    freeMatrix(m, d);
    freeMatrix(m, e);
    freeMatrix(m, f);
    freeMatrix(m, g);
    freeMatrix(m, h);

    int** c11;
    #pragma omp task shared(c11)
    {
        int** s1s2a = addMatrices(m, s1, s2, true);
        int** s6s4s = addMatrices(m, s6, s4, false);
        c11 = addMatrices(m, s1s2a, s6s4s, true);
        freeMatrix(m, s1s2a);
        freeMatrix(m, s6s4s);
    }

    int** c12;
    #pragma omp task shared(c12)
    {
        c12 = addMatrices(m, s4, s5, true);
    }

    int** c21;
    #pragma omp task shared(c21)
    {
        c21 = addMatrices(m, s6, s7, true);
    }

    int** c22;
    #pragma omp task shared(c22)
    {
        int** s2s3s = addMatrices(m, s2, s3, false);
        int** s5s7s = addMatrices(m, s5, s7, false);
        c22 = addMatrices(m, s2s3s, s5s7s, true);
        freeMatrix(m, s2s3s);
        freeMatrix(m, s5s7s);
    }

    #pragma omp taskwait

    freeMatrix(m, s1);
    freeMatrix(m, s2);
    freeMatrix(m, s3);
    freeMatrix(m, s4);
    freeMatrix(m, s5);
    freeMatrix(m, s6);
    freeMatrix(m, s7);

    // Combine the results into the final product matrix
    int** prod = combineMatrices(m, c11, c12, c21, c22);

    freeMatrix(m, c11);
    freeMatrix(m, c12);
    freeMatrix(m, c21);
    freeMatrix(m, c22);

    return prod;
}


/**
 * @brief Compares two square matrices for equality.
 * 
 * This function iterates through each element of the two input matrices,
 * checking if the corresponding elements are equal. If any pair of elements 
 * are not equal, the function returns false. If all pairs are equal, it returns true.
 *
 * @param n     Size of the input square matrices (n x n).
 * @param prod1 Pointer to the first matrix to be compared.
 * @param prod2 Pointer to the second matrix to be compared.
 * @return      Returns true if both matrices are identical, otherwise false.
 */
bool check(int n, int** prod1, int** prod2)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (prod1[i][j] != prod2[i][j])
                return false;
        }
    }
    return true;
}


int main()
{
    int n;
    cout << "\nEnter matrix Dimension: ";
    cin >> n;

    // int threads;
    // cout << "\nEnter number of thread: ";
    // cin >> threads;

    int** mat1 = allocateMatrix(n);
    fillMatrix(n, mat1);

    int** mat2 = allocateMatrix(n);
    fillMatrix(n, mat2);

    double startTime = omp_get_wtime();
    int** prod;
    omp_set_dynamic(0); // Disable dynamic adjustment of threads
    omp_set_num_threads(8);

    #pragma omp parallel
    {
    #pragma omp single
        {
            prod = strassen(n, mat1, mat2);
        }
    }
    double endTime = omp_get_wtime();
    cout << "\nParallel Strassen Runtime (OMP) with threads : "<< threads;
    cout << setprecision(5) << endTime - startTime << endl;

    cout << endl;

    return 0;
}
