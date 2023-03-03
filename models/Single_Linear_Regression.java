package models;

import java.util.Arrays;
import java.math.BigDecimal;
import java.math.*;

public class Single_Linear_Regression {

    // instance variables

    double[] x, y;
    double m;
    BigDecimal w, b;

    // constructor

    public Single_Linear_Regression(double[] col1, double[] col2) {
        this.x = col1;
        this.y = col2;
        this.m = col1.length;
    }

    public void dbg() {
        System.out.println("X: " + Arrays.toString(this.x));
        System.out.println("y: " + Arrays.toString(this.y));
    }

    // method to predict output given input and model coefficients (double version)

    public double[] predict(double x, double w, double b) {
        double[] prediction = new double[1];
        prediction[0] = w * x + b;
        return prediction;

    }

    public BigDecimal[] predict(double[] x, BigDecimal w, BigDecimal b) {
        BigDecimal[] prediction = new BigDecimal[x.length];
        for (int i = 0; i < x.length; ++i) {
            prediction[i] = BigDecimal.valueOf(x[i]).multiply(w).add(b);
        }
        return prediction;
    }

    public BigDecimal loss(BigDecimal w, BigDecimal b) {
        BigDecimal loss = BigDecimal.ZERO;
        BigDecimal[] predictions = predict(this.x, w, b);
        for (int i = 0; i < predictions.length; ++i) {
            BigDecimal diff = predictions[i].subtract(BigDecimal.valueOf(y[i]));
            loss = loss.add(diff.multiply(diff));
        }
        return loss.divide(BigDecimal.valueOf(predictions.length), MathContext.DECIMAL32);
    }

    public BigDecimal[] gradient(BigDecimal w, BigDecimal b) {
        BigDecimal[] grads = new BigDecimal[2];
        BigDecimal w_grad = new BigDecimal(0), b_grad = new BigDecimal(0);
        BigDecimal der = new BigDecimal(2);
        for (int i = 0; i < this.m; ++i) {
            w_grad = w_grad.add(der.multiply(BigDecimal.valueOf(this.x[i]))
                    .multiply(BigDecimal.valueOf(predict(this.x[i], w.doubleValue(), b.doubleValue())[0] - this.y[i])));
            b_grad = b_grad.add(der
                    .multiply(BigDecimal.valueOf(predict(this.x[i], w.doubleValue(), b.doubleValue())[0] - this.y[i])));
        }
        grads[0] = w_grad.divide(BigDecimal.valueOf(this.m), new MathContext(10, RoundingMode.HALF_UP));
        grads[1] = b_grad.divide(BigDecimal.valueOf(this.m), new MathContext(10, RoundingMode.HALF_UP));
        return grads;
    }

    public BigDecimal[] train(int itr, double lr, boolean print_out) {
        BigDecimal[] weights = new BigDecimal[2];
        BigDecimal[] gradients;
        BigDecimal w = new BigDecimal(0), b = new BigDecimal(0);
        int interval = itr / 200;
        for (int i = 0; i < itr; ++i) {
            gradients = gradient(w, b);
            w = w.subtract(gradients[0].multiply(BigDecimal.valueOf(lr)));
            b = b.subtract(gradients[1].multiply(BigDecimal.valueOf(lr)));
            if (i % interval == 0 && print_out) {
                System.out.println("Iteration => " + i);
                System.out.println("W Gradient: " + gradients[0] + " " + "B Gradient: " + gradients[1]);
                System.out.println("W: " + w + " " + "B: " + b);
                System.out.println("Loss => " + loss(w, b));
            }

        }
        weights[0] = w;
        weights[1] = b;

        this.w = w;
        this.b = b;
        return weights;
    }
}
