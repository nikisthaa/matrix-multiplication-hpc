#include <mpi.h>
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

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
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
    #pragma omp parallel for num_threads(1) collapse(2)
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
    #pragma omp parallel for num_threads(1)
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i < m && j < m)
                result[i][j] = c11[i][j];
            else if (i < m)
                result[i][j] = c12[i][j - m];
            else if (j < m)
                result[i][j] = c21[i - m][j];
            else
                result[i][j] = c22[i - m][j - m];
        }
    }

    return result;
}


/**
 * Implements the Strassen algorithm for multiplying two square matrices.
 * The Strassen algorithm is a divide-and-conquer method that performs matrix 
 * multiplication more efficiently than the naive method.
 *
 * @param n: The order of the square matrices mat1 and mat2.
 * @param mat1: The first input matrix to be multiplied.
 * @param mat2: The second input matrix to be multiplied.
 * @return: A pointer to the resulting matrix after multiplication.
 */

int** strassen(int n, int** mat1, int** mat2)
{
    // Base case: If matrix size is small, use naive multiplication
    if (n <= 32)
    {
        return naive(n, mat1, mat2);
    }

    // Divide matrices into 4 submatrices each
    int m = n / 2;

    int** a = getSlice(n, mat1, 0, 0);
    int** b = getSlice(n, mat1, 0, m);
    int** c = getSlice(n, mat1, m, 0);
    int** d = getSlice(n, mat1, m, m);
    int** e = getSlice(n, mat2, 0, 0);
    int** f = getSlice(n, mat2, 0, m);
    int** g = getSlice(n, mat2, m, 0);
    int** h = getSlice(n, mat2, m, m);


    // Compute seven products, recursively, using Strassen's formulas
    int** bds = addMatrices(m, b, d, false);
    int** gha = addMatrices(m, g, h, true);
    int** s1 = strassen(m, bds, gha);
    freeMatrix(m, bds);
    freeMatrix(m, gha);

    int** ada = addMatrices(m, a, d, true);
    int** eha = addMatrices(m, e, h, true);
    int** s2 = strassen(m, ada, eha);
    freeMatrix(m, ada);
    freeMatrix(m, eha);

    int** acs = addMatrices(m, a, c, false);
    int** efa = addMatrices(m, e, f, true);
    int** s3 = strassen(m, acs, efa);
    freeMatrix(m, acs);
    freeMatrix(m, efa);

    int** aba = addMatrices(m, a, b, true);
    int** s4 = strassen(m, aba, h);
    freeMatrix(m, aba);
    freeMatrix(m, b);

    int** fhs = addMatrices(m, f, h, false);
    int** s5 = strassen(m, a, fhs);
    freeMatrix(m, fhs);
    freeMatrix(m, a);
    freeMatrix(m, f);
    freeMatrix(m, h);

    int** ges = addMatrices(m, g, e, false);
    int** s6 = strassen(m, d, ges);
    freeMatrix(m, ges);
    freeMatrix(m, g);

    int** cda = addMatrices(m, c, d, true);
    int** s7 = strassen(m, cda, e);
    freeMatrix(m, cda);
    freeMatrix(m, c);
    freeMatrix(m, d);
    freeMatrix(m, e);

    int** s1s2a = addMatrices(m, s1, s2, true);
    int** s6s4s = addMatrices(m, s6, s4, false);
    int** c11 = addMatrices(m, s1s2a, s6s4s, true);
    freeMatrix(m, s1s2a);
    freeMatrix(m, s6s4s);
    freeMatrix(m, s1);

    int** c12 = addMatrices(m, s4, s5, true);
    freeMatrix(m, s4);

    int** c21 = addMatrices(m, s6, s7, true);
    freeMatrix(m, s6);

    int** s2s3s = addMatrices(m, s2, s3, false);
    int** s5s7s = addMatrices(m, s5, s7, false);
    int** c22 = addMatrices(m, s2s3s, s5s7s, true);
    freeMatrix(m, s2s3s);
    freeMatrix(m, s5s7s);
    freeMatrix(m, s2);
    freeMatrix(m, s3);
    freeMatrix(m, s5);
    freeMatrix(m, s7);

    int** prod = combineMatrices(m, c11, c12, c21, c22);

    freeMatrix(m, c11);
    freeMatrix(m, c12);
    freeMatrix(m, c21);
    freeMatrix(m, c22);

    return prod;
}


// Uncomment code below if you want to run in 8 cores
// /**
//  * Function: strassen
//  * -------------------
//  * Performs the multiplication of two square matrices (mat1 and mat2) of size 'n x n' using Strassen's algorithm and MPI
//  * for parallel computation. The result is stored in the 'prod' matrix.
//  *
//  * @param n: The dimension of the input square matrices (mat1 and mat2).
//  * @param mat1: The first input matrix to be multiplied.
//  * @param mat2: The second input matrix to be multiplied.
//  * @param prod: The resulting product matrix after multiplication.
//  * @param rank: The rank (ID) of the current process.
//  */
// void strassen(int n, int** mat1, int** mat2, int**& prod, int rank)
// {
//     // Base case: when the matrix size is 1x1
//     if (n == 1)
//     {
//         prod = allocateMatrix(1);
//         prod[0][0] = mat1[0][0] * mat2[0][0];
//     }

//     int m = n / 2;

//     // Divide the input matrices into 4 submatrices each
//     int** a = getSlice(n, mat1, 0, 0);
//     int** b = getSlice(n, mat1, 0, m);
//     int** c = getSlice(n, mat1, m, 0);
//     int** d = getSlice(n, mat1, m, m);
//     int** e = getSlice(n, mat2, 0, 0);
//     int** f = getSlice(n, mat2, 0, m);
//     int** g = getSlice(n, mat2, m, 0);
//     int** h = getSlice(n, mat2, m, m);

//     // Allocate memory for the 7 products (s1 to s7) required by Strassen's algorithm
//     int** s1 = allocateMatrix(m);
//     int** s2 = allocateMatrix(m);
//     int** s3 = allocateMatrix(m);
//     int** s4 = allocateMatrix(m);
//     int** s5 = allocateMatrix(m);
//     int** s6 = allocateMatrix(m);
//     int** s7 = allocateMatrix(m);

//     // Parallel computation: each process calculates one of the products (s1 to s7)
//     // The master process (rank 0) receives the computed products from other processes
//     if (rank == 0)
//     {   
//         // Receive computed products from worker processes (1 to 7)
//         MPI_Recv(&(s1[0][0]), m * m, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//         MPI_Recv(&(s2[0][0]), m * m, MPI_INT, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//         MPI_Recv(&(s3[0][0]), m * m, MPI_INT, 3, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//         MPI_Recv(&(s4[0][0]), m * m, MPI_INT, 4, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//         MPI_Recv(&(s5[0][0]), m * m, MPI_INT, 5, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//         MPI_Recv(&(s6[0][0]), m * m, MPI_INT, 6, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//         MPI_Recv(&(s7[0][0]), m * m, MPI_INT, 7, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//     }

//     // Worker processes (1 to 7) compute one product each and send it to the master process
//     if (rank == 1)
//     {
//         int** bds = addMatrices(m, b, d, false);
//         int** gha = addMatrices(m, g, h, true);
//         s1 = strassen(m, bds, gha);
//         freeMatrix(m, bds);
//         freeMatrix(m, gha);
//         MPI_Send(&(s1[0][0]), m * m, MPI_INT, 0, 0, MPI_COMM_WORLD);
//     }

//     if (rank == 2)
//     {
//         int** ada = addMatrices(m, a, d, true);
//         int** eha = addMatrices(m, e, h, true);
//         s2 = strassen(m, ada, eha);
//         freeMatrix(m, ada);
//         freeMatrix(m, eha);
//         MPI_Send(&(s2[0][0]), m * m, MPI_INT, 0, 0, MPI_COMM_WORLD);
//     }

//     if (rank == 3)
//     {
//         int** acs = addMatrices(m, a, c, false);
//         int** efa = addMatrices(m, e, f, true);
//         s3 = strassen(m, acs, efa);
//         freeMatrix(m, acs);
//         freeMatrix(m, efa);
//         MPI_Send(&(s3[0][0]), m * m, MPI_INT, 0, 0, MPI_COMM_WORLD);
//     }

//     if (rank == 4)
//     {
//         int** aba = addMatrices(m, a, b, true);
//         s4 = strassen(m, aba, h);
//         freeMatrix(m, aba);
//         MPI_Send(&(s4[0][0]), m * m, MPI_INT, 0, 0, MPI_COMM_WORLD);
//     }
//     freeMatrix(m, b);

//     if (rank == 5)
//     {
//         int** fhs = addMatrices(m, f, h, false);
//         s5 = strassen(m, a, fhs);
//         freeMatrix(m, fhs);
//         MPI_Send(&(s5[0][0]), m * m, MPI_INT, 0, 0, MPI_COMM_WORLD);
//     }
//     freeMatrix(m, a);
//     freeMatrix(m, f);
//     freeMatrix(m, h);

//     if (rank == 6)
//     {
//         int** ges = addMatrices(m, g, e, false);
//         s6 = strassen(m, d, ges);
//         freeMatrix(m, ges);
//         MPI_Send(&(s6[0][0]), m * m, MPI_INT, 0, 0, MPI_COMM_WORLD);
//     }
//     freeMatrix(m, g);

//     if (rank == 7)
//     {
//         int** cda = addMatrices(m, c, d, true);
//         s7 = strassen(m, cda, e);
//         freeMatrix(m, cda);
//         MPI_Send(&(s7[0][0]), m * m, MPI_INT, 0, 0, MPI_COMM_WORLD);
//     }
//     freeMatrix(m, c);
//     freeMatrix(m, d);
//     freeMatrix(m, e);

//     // Ensure all processes reach this point before proceeding
//     MPI_Barrier(MPI_COMM_WORLD);

//     // Master process computes the final result using the 7 products
//     if (rank == 0)
//     {
//         // Compute the final submatrices (c11, c12, c21, c22) using Strassen's formulas
//         int** s1s2a = addMatrices(m, s1, s2, true);
//         int** s6s4s = addMatrices(m, s6, s4, false);
//         int** c11 = addMatrices(m, s1s2a, s6s4s, true);
//         freeMatrix(m, s1s2a);
//         freeMatrix(m, s6s4s);

//         int** c12 = addMatrices(m, s4, s5, true);

//         int** c21 = addMatrices(m, s6, s7, true);

//         int** s2s3s = addMatrices(m, s2, s3, false);
//         int** s5s7s = addMatrices(m, s5, s7, false);
//         int** c22 = addMatrices(m, s2s3s, s5s7s, true);
//         freeMatrix(m, s2s3s);
//         freeMatrix(m, s5s7s);

//         // Combine the submatrices to get the final product matrix
//         prod = combineMatrices(m, c11, c12, c21, c22);

//         // Deallocate memory
//         freeMatrix(m, c11);
//         freeMatrix(m, c12);
//         freeMatrix(m, c21);
//         freeMatrix(m, c22);
//     }

//     // Deallocate memory
//     freeMatrix(m, s1);
//     freeMatrix(m, s2);
//     freeMatrix(m, s3);
//     freeMatrix(m, s4);
//     freeMatrix(m, s5);
//     freeMatrix(m, s6);
//     freeMatrix(m, s7);
// }


/**
 * Function: strassen
 * -------------------
 * Performs the multiplication of two square matrices (mat1 and mat2) of size 'n x n' using Strassen's algorithm and MPI
 * for parallel computation. The result is stored in the 'prod' matrix.
 *
 * @param n: The dimension of the input square matrices (mat1 and mat2).
 * @param mat1: The first input matrix to be multiplied.
 * @param mat2: The second input matrix to be multiplied.
 * @param prod: The resulting product matrix after multiplication.
 * @param rank: The rank (ID) of the current process.
 */
void strassen(int n, int** mat1, int** mat2, int**& prod, int rank)
{
    // Base case: when the matrix size is 1x1
    if (n == 1)
    {
        prod = allocateMatrix(1);
        prod[0][0] = mat1[0][0] * mat2[0][0];
    }

    int m = n / 2;

    // Divide the input matrices into 4 submatrices each
    int** a = getSlice(n, mat1, 0, 0);
    int** b = getSlice(n, mat1, 0, m);
    int** c = getSlice(n, mat1, m, 0);
    int** d = getSlice(n, mat1, m, m);
    int** e = getSlice(n, mat2, 0, 0);
    int** f = getSlice(n, mat2, 0, m);
    int** g = getSlice(n, mat2, m, 0);
    int** h = getSlice(n, mat2, m, m);

    // Allocate memory for the 7 products (s1 to s7) required by Strassen's algorithm
    int** s1 = allocateMatrix(m);
    int** s2 = allocateMatrix(m);
    int** s3 = allocateMatrix(m);
    int** s4 = allocateMatrix(m);
    int** s5 = allocateMatrix(m);
    int** s6 = allocateMatrix(m);
    int** s7 = allocateMatrix(m);

    // Parallel computation: each process calculates one of the products (s1 to s7)
    // The master process (rank 0) receives the computed products from other processes
    if (rank == 0)
    {   
        // Receive computed products from worker processes (1 to 3)
        MPI_Recv(&(s1[0][0]), m * m, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&(s2[0][0]), m * m, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&(s3[0][0]), m * m, MPI_INT, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&(s4[0][0]), m * m, MPI_INT, 2, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&(s5[0][0]), m * m, MPI_INT, 3, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&(s6[0][0]), m * m, MPI_INT, 3, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&(s7[0][0]), m * m, MPI_INT, 3, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Worker processes (1 to 7) compute one product each and send it to the master process
    if (rank == 1)
    {
        // Worker 1 computes s1 and s2
        int** bds = addMatrices(m, b, d, false);
        int** gha = addMatrices(m, g, h, true);
        s1 = strassen(m, bds, gha);
        freeMatrix(m, bds);
        freeMatrix(m, gha);
        MPI_Send(&(s1[0][0]), m * m, MPI_INT, 0, 0, MPI_COMM_WORLD);

        int** ada = addMatrices(m, a, d, true);
        int** eha = addMatrices(m, e, h, true);
        s2 = strassen(m, ada, eha);
        freeMatrix(m, ada);
        freeMatrix(m, eha);
        MPI_Send(&(s2[0][0]), m * m, MPI_INT, 0, 1, MPI_COMM_WORLD);
    }

    if (rank == 2)
    {
        // Worker 2 computes s3 and s4
        int** acs = addMatrices(m, a, c, false);
        int** efa = addMatrices(m, e, f, true);
        s3 = strassen(m, acs, efa);
        freeMatrix(m, acs);
        freeMatrix(m, efa);
        MPI_Send(&(s3[0][0]), m * m, MPI_INT, 0, 0, MPI_COMM_WORLD);

        int** aba = addMatrices(m, a, b, true);
        s4 = strassen(m, aba, h);
        freeMatrix(m, aba);
        MPI_Send(&(s4[0][0]), m * m, MPI_INT, 0, 1, MPI_COMM_WORLD);
    }
    freeMatrix(m, b);

    if (rank == 3)
    {
        // Worker 3 computes s5, s6, and s7
        int** fhs = addMatrices(m, f, h, false);
        s5 = strassen(m, a, fhs);
        freeMatrix(m, fhs);
        MPI_Send(&(s5[0][0]), m * m, MPI_INT, 0, 0, MPI_COMM_WORLD);
        int** ges = addMatrices(m, g, e, false);
        s6 = strassen(m, d, ges);
        freeMatrix(m, ges);
        MPI_Send(&(s6[0][0]), m * m, MPI_INT, 0, 1, MPI_COMM_WORLD);

        int** cda = addMatrices(m, c, d, true);
        s7 = strassen(m, cda, e);
        freeMatrix(m, cda);
        MPI_Send(&(s7[0][0]), m * m, MPI_INT, 0, 2, MPI_COMM_WORLD);
    }
\
    freeMatrix(m, c);
    freeMatrix(m, d);
    freeMatrix(m, e);
    freeMatrix(m, a);
    freeMatrix(m, f);
    freeMatrix(m, h);

    // Ensure all processes reach this point before proceeding
    MPI_Barrier(MPI_COMM_WORLD);

    // Master process computes the final result using the 7 products
    if (rank == 0)
    {
        // Compute the final submatrices (c11, c12, c21, c22) using Strassen's formulas
        int** s1s2a = addMatrices(m, s1, s2, true);
        int** s6s4s = addMatrices(m, s6, s4, false);
        int** c11 = addMatrices(m, s1s2a, s6s4s, true);
        freeMatrix(m, s1s2a);
        freeMatrix(m, s6s4s);

        int** c12 = addMatrices(m, s4, s5, true);

        int** c21 = addMatrices(m, s6, s7, true);

        int** s2s3s = addMatrices(m, s2, s3, false);
        int** s5s7s = addMatrices(m, s5, s7, false);
        int** c22 = addMatrices(m, s2s3s, s5s7s, true);
        freeMatrix(m, s2s3s);
        freeMatrix(m, s5s7s);

        // Combine the submatrices to get the final product matrix
        prod = combineMatrices(m, c11, c12, c21, c22);

        // Deallocate memory
        freeMatrix(m, c11);
        freeMatrix(m, c12);
        freeMatrix(m, c21);
        freeMatrix(m, c22);
    }

    // Deallocate memory
    freeMatrix(m, s1);
    freeMatrix(m, s2);
    freeMatrix(m, s3);
    freeMatrix(m, s4);
    freeMatrix(m, s5);
    freeMatrix(m, s6);
    freeMatrix(m, s7);
}

int main(int argc, char* argv[])
{
    int p_rank;
    int num_process;
    int provided;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if(provided < MPI_THREAD_FUNNELED) {
        printf("Thread support not available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &p_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_process);

    int n;
    if (p_rank == 0)
    {
        cout << endl;
        cout << "Enter the dimensions of the matrix: ";
        cin >> n;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int** mat1 = allocateMatrix(n);
    int** mat2 = allocateMatrix(n);

    if (p_rank == 0)
    {
        fillMatrix(n, mat1);
        fillMatrix(n, mat2);
    }

    MPI_Bcast(&(mat1[0][0]), n * n, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&(mat2[0][0]), n * n, MPI_INT, 0, MPI_COMM_WORLD);

    double startTime = MPI_Wtime();

    int** prod;
    strassen(n, mat1, mat2, prod, p_rank);

    double endTime = MPI_Wtime();

    if (p_rank == 0)
    {
        cout << "\nParallel Strassen Runtime (MPI): ";
        cout << setprecision(5) << endTime - startTime << endl;
        cout << endl;
    }

    MPI_Finalize();

    return 0;
}
