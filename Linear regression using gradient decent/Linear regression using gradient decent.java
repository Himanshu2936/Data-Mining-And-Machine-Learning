import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.PriorityQueue;
import java.util.Comparator;
import java.util.*;

class LRWGD {

    public static void main(String[] args) throws FileNotFoundException {
        
        double[][] X=new double[506][14];					// training set
        double[][] Y=new double[506][1];					//result of given data
        double[] theta=new double[14];
        double testdata1[]={1,0.0101, 30, 5.19, 0, 0.0493, 6.059,37.3, 4.8122,1, 430,19.6, 375.21, 8.51 };
        double testdata2[]={1,0.02501, 35, 4.15, 1, 0.77, 8.78, 81.3, 2.5051, 24, 666, 17, 382.8, 11.48};
        double testdata3[]={1,3.67822, 0, 18.1, 1, 0.7, 6.649, 98.8, 1.1742, 24, 711, 20.2, 398.28, 18.07 };
        double cost,iterations=0,alpha=1;
        
        for (int i = 0; i < 506; i++) {
            X[i][0]=1;										//setting x0
        }
        
        
        Scanner s=new Scanner(new File("boston_housing.csv"));
        s.nextLine();
        int rowno=0;
        while(s.hasNext()){
            String str[]=s.nextLine().split(",");			//reading file
            for (int i = 0; i < str.length-1; i++) {
                X[rowno][i+1]=Double.parseDouble(str[i]);
            }
            Y[rowno][0]=Double.parseDouble(str[str.length-1]);
            rowno++;
        }
        
        for (int i = 1; i < 14; i++) {						//normalization of input set
            double min=Integer.MAX_VALUE,max=Integer.MIN_VALUE,mean=0;
            for (int j = 0; j < 506; j++) {
                if(X[j][i]<min){
                    min=X[j][i];
                }
                if(X[j][i]>max){
                    max=X[j][i];
                }
                mean=mean+X[j][i];
            }
            mean=mean/506;
            double temp=max-min;
            if(temp==0)
                temp=1;
            for (int j = 0; j < 506; j++) {
                X[j][i]=((X[j][i]-mean)/temp);
            }
            testdata1[i]=((testdata1[i]-mean)/temp);
            testdata2[i]=((testdata2[i]-mean)/temp);
            testdata3[i]=((testdata3[i]-mean)/temp);
        }
        
        System.out.println("For Learning Rate Alpha : "+alpha);
        
        cost=cost(X,Y,theta);
        double prevcost=0;
        while(iterations<100000){							//iteration limit is 100000(if not converged in this means diverged)
            double[] grad=getgrad(X,Y,theta);
            for (int i = 0; i < 14; i++) {
                theta[i]=theta[i]-(alpha*grad[i]);
            }
            cost=cost(X,Y,theta);
            if(Math.abs(cost-prevcost)<0.000001)
                break;
            iterations++;
            prevcost=cost;
        }
        
        System.out.println("No of Iterations : "+iterations);
        System.out.println("Value of cost function : "+cost);
        System.out.println("Learned weight vector (theta)");
        for (int i = 0; i < 14; i++) {
            System.out.println(" Theta "+(i+1)+" : "+theta[i]+"  ");
        }
        double ans1=0,ans2=0,ans3=0;
        for (int i = 0; i < 14; i++) {
            ans1=ans1+theta[i]*testdata1[i];
            ans2=ans2+theta[i]*testdata2[i];
            ans3=ans3+theta[i]*testdata3[i];
        }
        
        
        System.out.println("The Predicted Price of House 1 is : "+ans1);
        
        System.out.println("The Predicted Price of House 2 is : "+ans2);
        
        System.out.println("The Predicted Price of House 3 is : "+ans3);
    }
    
    public static double cost(double[][] X,double[][] Y,double[] theta){		//returns cost for given values
        double cost=0;
        for (int i = 0; i < 506; i++) {
            double sum=0;
            for (int j = 0; j < 14; j++) {
                sum=sum+X[i][j]*theta[j];
            }
            sum=sum-Y[i][0];
            cost=cost+(sum*sum);
        }
        return (cost/1012);
    }
    public static double[] getgrad(double[][] X,double[][] Y,double[] theta){	//returns gradient for decent
        double[] error=new double[506];
        for (int i = 0; i < 506; i++) {
            double sum=0;
            for (int j = 0; j < 14; j++) {
                sum=sum+X[i][j]*theta[j];
            }
            sum=sum-Y[i][0];
            error[i]=sum;
        }
        double grad[]=new double[14];
        for (int i = 0; i < 14; i++) {
            for (int j = 0; j < 506; j++) {
                grad[i]+=X[j][i]*error[j];
            }
            grad[i]/=506;
        }
        return grad;
    }
    
}