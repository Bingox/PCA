package com.test.pca;

import Jama.EigenvalueDecomposition;
import Jama.Matrix;
import Jama.SingularValueDecomposition;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

public class PCA {
	
	private static final String FILE_IN_PATH = "./data/data.txt";
	private static final String FILE_OUT_PATH = "./data/";
	private static final int ROW = 150, COL = 4;
	
	static double[][] matrix = new double[ROW][COL];
	
	public static void readMatrixByLine(String filePath, String sp){
		File file = new File(filePath);
		BufferedReader buf = null;
		try {
			buf = new BufferedReader(new FileReader(file));
			String tmp = null;
			int index = 0;
			while((tmp = buf.readLine()) != null) {	
				String[] raws = tmp.split(sp);
				int cnt = 0;
				for (String str:raws) {
					matrix[index][cnt++] = Double.parseDouble(str);
				}
				index++;
			}
		} catch (Exception e) {
			e.getStackTrace();
		} finally {
			if (buf != null) {
				try {
					buf.close();
				} catch (IOException e) {
					e.getStackTrace();
				}
			}
		}
	}
	
	public static double[] getColAverage(double[][] mat) {
		double[] avr = new double[COL];
		double sum = 0;
		for (int j=0;j<COL;j++){
			for (int i=0;i<ROW;i++){
				sum += mat[i][j];
			}
			avr[j] = sum;
			sum = 0;
		}
		for (int j=0;j<COL;j++){
			avr[j] /= ROW;
		}
		return avr;
	}
	
	public static Matrix deviation(double[][] mat) {
		double[] avr = getColAverage(mat);
		Matrix m = new Matrix(ROW, COL);
		for (int j=0;j<COL;j++){
			for (int i=0;i<ROW;i++){
				m.set(i, j, mat[i][j] -avr[j]);
			}
		}
		return m;
	}
	
	public static Matrix pca1(double[][] matrix, double co) {
		Matrix mat = deviation(matrix);		
		SingularValueDecomposition svd = mat.svd();//svd
		double[] lambda = svd.getSingularValues();	
		double sum = 0,t = 0;
		for (int i=0;i<COL;i++) {
			t = lambda[i]*lambda[i]/(ROW-1);
			sum += t;
			lambda[i] = t;
		}
		int i = 0;
		t = lambda[0]/sum;
		while (t <= co) {//co : contribution rate
			t += lambda[++i]/sum; 
		}
		return svd.getV().getMatrix(0, COL-1, 0, i);
	}
	
	public static Matrix pca2(double[][] matrix, double co) {
		Matrix mat = deviation(matrix);
		Matrix tmat = mat.transpose();
		Matrix cov = tmat.times(mat).times(1.0/(ROW-1));//cov = A'A/n
		EigenvalueDecomposition ec = cov.eig();
		Matrix d = ec.getD();	
		double t = 0;
		double sum = d.trace();
		int i = COL;
		while (t <= co) {
			i--;
			t += d.get(i, i)/sum;
		}
		int [] a = new int[COL-i];
		for (int k=0;k<COL-i;k++){
			a[k] = COL-1-k;
		}
		return ec.getV().getMatrix(0, COL-1, a);
	}
	
	public static void main (String[] args){
		readMatrixByLine(FILE_IN_PATH, "     ");
		Matrix mat = new Matrix(matrix);
		Matrix p1 = pca1(matrix,0.98);
		Matrix mat1 = mat.times(p1);
		//mat1.print(2, 6);
		Matrix rmat1 = mat1.times(p1.transpose());
		rmat1.print(4, 6);
		//Matrix p2 = pca2(matrix,0.98);
		//Matrix mat2 = mat.times(p2);
		//mat2.print(2, 6);
		//Matrix rmat2 = mat2.times(p2.transpose());
		//rmat2.print(4, 6);
		
		File out = new File(FILE_OUT_PATH + "out1.txt");
		try {
			PrintWriter pw = new PrintWriter(new FileWriter(out));
			mat1.print(pw, 2, 6);
			pw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
