# Matrix Inversion
### `main()` function
**description:**
call GPU functions to compute inverse matrix, and record the running time.


### GPU functions

    double *gpuMatrixInverse(double *inputMatrix, const int rows, const int cols)
**description:** <br>
@param `inputMatrix`  a pointer to an address which stores the input matrix on host<br>
@param `rows`  num of rows in input matrix <br>
@param `cols`  num of cols in input matrix <br>
This function initializes the variables on GPU, and translates the data to GPU. Do Gaussian Elimination on each row of the matrix. Then translate the data back to CPU. <br>
___

    __global__ void augmentMatrixKernel(double *d_augmentedMatrix, double *d_inputMatrix, const int rows, const int cols)
**description:** <br>
@param `d_augmentedMatrix`  a pointer to an address which stores the augmented Matrix <br>
@param `d_inputMatrix`  a pointer to an address where stores the input matrix <br>
@param `rows`  num of rows in augmented matrix <br>
@param `cols`  num of cols in augmented matrix <br>
This function translates the input matrix to augmented matrix <br>
___


    __global__ void computeRowsKernel(double *d_augmentedMatrix, const int rowId, const int size)
**description:** <br>
@param `d_augmentedMatrix`  a pointer to an address which stores the augmented Matrix <br>
@param `rowId`  the ID of row to be divided by its pivot <br>
@param `size`  the size of matrix <br>
the row which `rowId` points is divided by its pivot (whoes row ID equals col ID) <br>
___

    __global__ void harnessZeroKernel(double *d_augmentedMatrix, const int rowId1, const int rowId2, const int size)
**description**
@param `d_augmentedMatrix`  a pointer to an address which stores the augmented Matrix <br>
@param `rowId1`  the ID of row 1 <br>
@param `rowId2`  the ID of row 2 <br>
@param `size`  the size of matrix <br>
if a row's pivot is equal to zero, we can't do row operation, we need add another non-zero row to the current row <br>
___

    __global__ void computeColsKernel(double *d_augmentedMatrix, const int colId, const int size)
**description**
@param `d_augmentedMatrix`  a pointer to an address which stores the augmented Matrix <br>
@param `colId`  ID of col which needs to be zero, except the pivot is 1 <br>
@param `size`  the size of matrix <br>
Other rows except `colId` row are subtracted by the corresponding coefficient.
___

    __global__ void getInverseMatrixKernel(double *d_augmentedMatrix, double *d_inverseMatrix, const int rows, const int cols)
**description**
@param `d_augmentedMatrix`  a pointer to an address which stores the augmented Matrix <br>
@param `d_inverseMatrix`  a pointer to an address which stores the final inversed matrix <br>
@param `rows`  num of rows in augmented matrix <br>
@param `cols`  num of cols in augmented matrix <br>
Get the final inversed matrix
___
