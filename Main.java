import java.io.File;
import java.io.FileNotFoundException;
import java.math.BigDecimal;
import java.util.Scanner;
import java.util.Arrays;

import models.Single_Linear_Regression;
import models.Multi_Linear_Regression;

public class Main {
    
    public static void main(String[] args) {
        double[][] data_1 = loadtxt("temp.txt", 1);
        // double [] x = data_1[0];
        // double [] y = data_1[1];

        Single_Linear_Regression sReg_model = new Single_Linear_Regression(data_1[0], data_1[1]);
        // System.out.println(Arrays.toString(sReg_model.train(150000, 0.0001, false)));
        


        double[][] data_2 = loadtxt("data.txt", 1);

        Multi_Linear_Regression mReg_model = new Multi_Linear_Regression(Arrays.copyOfRange(data_2, 0, 3), data_2[3]);
        mReg_model.dbg();

        double[][]fake_weights = {
            {0.1},
            {0.2},
            {0.3},
            {0.4}
        };

        //mReg_model.predict(fake_weights);

        System.out.println("Test Loss => " + mReg_model.loss(fake_weights));

        double[][]x=mReg_model.gradient(fake_weights);
        for (double[]i : x) System.out.println(Arrays.toString(i));
        double[][]w = mReg_model.train(250000, 0.0001);

        System.out.println("Weights => [");
        for (double[] i : w) System.out.println(Arrays.toString(i));
        System.out.println("]");

        System.out.println("Loss => " + mReg_model.loss(w));
        
    }


    public static int countRows(File file) throws FileNotFoundException {
        Scanner in = new Scanner(file);
        int rownum = 0;
        while (in.hasNextLine()) {
            rownum++;
            in.nextLine();
        }
        in.close();
        return rownum;
    }


    public static double[][] loadtxt(String name, int skiprows) {
        File inputFile = new File(name);
        double [][] ret;
        try {
            Scanner in = new Scanner(inputFile);
            int columnnum=0;
            for (int i = 0 ; i < skiprows ; ++i) {
                String[] tmp = in.nextLine().split("\\s+");
                columnnum = tmp.length;
            }

            int rownum = countRows(inputFile) - 1;

            ret = new double[columnnum][rownum];

            int idx = 0;
            while (in.hasNextLine()) {
                String line = in.nextLine();
                
                String[] columns = line.split("\\s+");
                
                for (int i = 0 ; i < columnnum ; ++i) {
                    ret[i][idx] = Double.parseDouble(columns[i]);
                }
                idx++;
            }
            
            in.close();
            
        } catch(FileNotFoundException e) {
            e.printStackTrace();
            ret = new double[0][0];
        }
        return ret;
    }
}