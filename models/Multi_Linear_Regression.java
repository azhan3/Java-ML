package models;

import java.util.Arrays;

public class Multi_Linear_Regression {
    double[][] x, y, Y_mat, X_mat;
    int r, c;

    public Multi_Linear_Regression(double[][] x, double[] y) {
        this.r = y.length;
        this.c = x.length;
        this.Y_mat = new double[this.r][1];
        this.X_mat = transpose(x);
        this.X_mat = addOnesColumn(this.X_mat);

        for (int row = 0; row < this.r; ++row) {
            this.Y_mat[row][0] = y[row];

        }
    }

    public static double[][] transpose(double[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;
        double[][] result = new double[n][m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result[j][i] = matrix[i][j];
            }
        }
        return result;
    }
    
    public static double[][] addOnesColumn(double[][] matrix) {
        int numRows = matrix.length;
        int numCols = matrix[0].length;
    
        // Create a new matrix with an extra column for the ones
        double[][] newMatrix = new double[numRows][numCols + 1];
    
        // Add the ones to the first column of the new matrix
        for (int i = 0; i < numRows; i++) {
            newMatrix[i][0] = 1.0;
        }
    
        // Copy the original matrix into the rest of the new matrix
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                newMatrix[i][j + 1] = matrix[i][j];
            }
        }
    
        return newMatrix;
    }   

    public double[][] predict(double[][] w) {
        return matrixMultiplication(this.X_mat, w);
    }

    public static double[][] matrixMultiplication(double[][] A, double[][] B) throws IllegalArgumentException {
        if (A[0].length != B.length) {
            throw new IllegalArgumentException("Invalid dimensions for matrix multiplication.");
        }
        int n = A.length, m = B[0].length, p = B.length;
        double[][] C = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                for (int k = 0; k < p; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        for (double[]i : C) System.out.println(Arrays.toString(i));
        return C;
    }
    
    

    public void dbg() {
        for (int i = 0 ; i < this.r; ++i) {
            System.out.println(Arrays.toString(this.X_mat[i])+ " " + Arrays.toString(this.Y_mat[i]));
        }
    }

}
