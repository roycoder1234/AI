import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Network {
    Neuron first = new Neuron();
    Neuron second = new Neuron();
    Neuron third = new Neuron();
    Neuron fourth = new Neuron();
    Neuron out = new Neuron();

     //Data flows: inputs → hidden layer → output layer
    public double predict(double x1, double x2) {
        // Pass inputs into hidden neurons
        double firstOut = first.predict(x1, x2);
        double secondOut = second.predict(x1, x2);

        //pass inputs into 2nd layer
        double thirdOut = third.predict(firstOut, secondOut);
        double fourthOut = fourth.predict(firstOut, secondOut);

        // New outputs become inputs to final neuron
        return out.predict(thirdOut, fourthOut);
    }

    public String predictWin(double percentage){
        if(percentage > 0.5){
            return "Team 1 Won";
        }
        else if(percentage < 0.5){
            return "Team 1 Lost";
        }
        return "Draw";
    }

    //TRAINING FUNCTION
    //Uses gradient descent + backpropagation to adjust weights and reduce error
    public void train(ArrayList<double[]> data, ArrayList<Double> answers) {
        
        double lr = 0.1; // learning rate

        // Repeat many times so the network can improve
        for (int epoch = 0; epoch < 10000; epoch++) {

            double totalLoss = 0;

            // Go through each training example
            for (int i = 0; i < data.size(); i++) {
                double x1 = data.get(i)[0];
                double x2 = data.get(i)[1]; 
                double correctAnswer = answers.get(i);   

                // =========================
                // 1. FORWARD PASS
                // =========================

                // Hidden neuron 1
                double z1 = first.rawValue(x1, x2);     
                double firstOut = first.predict(x1, x2); 

                // Hidden neuron 2
                double z2 =  second.rawValue(x1, x2); 
                double secondOut = second.predict(x1, x2);

                // Output neuron
                double z3 =  out.rawValue(firstOut, secondOut); 
                double pred = out.predict(firstOut, secondOut);

                // =========================
                // 2. LOSS (error)
                // =========================

                double error = pred - correctAnswer;

                // Mean Squared Error contribution
                totalLoss += error * error;

                // =========================
                // 3. BACKPROPAGATION
                // =========================
                // We compute how much each weight contributed to the error.

                // ----- Output neuron -----

                double predictionLoss = 2 * error;
                double predDerivitive = Neuron.sigmoidDerivative(z3);

                // Gradients for output neuron weights
                double grad_out_w1 = predictionLoss * predDerivitive * firstOut;
                double grad_out_w2 = predictionLoss * predDerivitive * secondOut;
                double grad_out_b  = predictionLoss * predDerivitive;

                // ----- Hidden neurons -----
                // Activation derivatives
                double dz1 = Neuron.sigmoidDerivative(z1);
                double dz2 = Neuron.sigmoidDerivative(z2);

                // Gradients for hidden neuron 1
                double grad_h1_w1 = predictionLoss * predDerivitive * out.w1 * dz1 * x1;
                double grad_h1_w2 = predictionLoss * predDerivitive * out.w1 * dz1 * x2;
                double grad_h1_b  = predictionLoss * predDerivitive * out.w1 * dz1;

                // Gradients for hidden neuron 2
                double grad_h2_w1 = predictionLoss * predDerivitive * out.w2 * dz2 * x1;
                double grad_h2_w2 = predictionLoss * predDerivitive * out.w2 * dz2 * x2;
                double grad_h2_b  = predictionLoss * predDerivitive * out.w2 * dz2;

                // =========================
                // 4. UPDATE WEIGHTS
                // =========================
                // Each neuron adjusts its own weights

                out.update(grad_out_w1, grad_out_w2, grad_out_b, lr);

                first.update(grad_h1_w1, grad_h1_w2, grad_h1_b, lr);
                second.update(grad_h2_w1, grad_h2_w2, grad_h2_b, lr);
            }

            // Print progress every 1000 epochs
            if (epoch % 1000 == 0) {
                System.out.println("Epoch " + epoch +
                        " Loss: " + (totalLoss / data.size()));
            }
        }
    }

     //Reads CSV file into a 2D list of strings
    public static ArrayList<String[]> readCSV(String filePath) {
        ArrayList<String[]> data = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            br.readLine();
            while ((line = br.readLine()) != null) {
                String[] row = line.split(",");
                data.add(row);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return data;
    }

    //Loads training data
    public static void loadTrainingData(ArrayList<Double> answers, ArrayList<double[]> data, String filepath){
        ArrayList<String[]> file = readCSV(filepath);

        for(String[] row: file){

            double possession1 = Double.parseDouble(row[3]);
            double possession2 = Double.parseDouble(row[4]);

            double goals1 = Double.parseDouble(row[6]);
            double goals2 = Double.parseDouble(row[7]);

            double[] possessionData = new double[2];
            possessionData[0] = possession1/100.0;
            possessionData[1] = possession2/100.0;
            data.add(possessionData);

            if(goals1 > goals2){
                answers.add(1.0);
            }
            else if(goals1 < goals2){
                answers.add(0.0);
            }
            else{
                answers.add(0.5);
            }
        }
    }

    public static void main(String[] args) {   

        ArrayList<double[]> data = new ArrayList<double[]>();
        ArrayList<Double> answers = new ArrayList<Double>();
 	    String filepath = "/workspaces/AI/FIFA World Cup 2022.csv";
       
        loadTrainingData(answers, data, filepath);

    	Network network = new Network();
    	network.train(data, answers);

    	//Try making some predictions
        System.out.println(network.predict(0.9, 0.3));
    }

}

