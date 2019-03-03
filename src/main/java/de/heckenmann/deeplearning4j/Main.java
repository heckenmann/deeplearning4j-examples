package de.heckenmann.deeplearning4j;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Original:
 * https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/feedforward/regression/RegressionSum.java
 */
public class Main {

  // Random number generator seed, for reproducability
  public static final int seed = 12345;
  // Number of epochs (full passes of the data)
  public static final int nEpochs = 200;
  // Number of data points
  public static final int nSamples = 1000;
  // Batch size: i.e., each epoch has nSamples/batchSize parameter updates
  public static final int batchSize = 100;
  // Network learning rate
  public static final double learningRate = 0.1;
  // The range of the sample data, data in range (0-1 is sensitive for NN, you can try other ranges
  // and see how it effects the results
  // also try changing the range along with changing the activation function
  public static int MIN_RANGE = 0;
  public static int MAX_RANGE = 1;
  public static final Random RNG = new Random(seed);

  public static void main(String... args) {

    // Schema schema = new Schema.Builder().addColumnInteger("x").addColumnInteger("y").build();

    // Create the network
    int numInput = 1;
    int numOutputs = 1;
    int nHidden = 10;
    MultiLayerConfiguration conf =
        new NeuralNetConfiguration.Builder()
            .seed(seed)
            .weightInit(WeightInit.XAVIER)
            .updater(new Nesterovs(learningRate, 0.9))
            .list()
            .layer(
                0,
                new DenseLayer.Builder()
                    .nIn(numInput)
                    .nOut(nHidden)
                    .activation(Activation.TANH)
                    .build())
            .layer(
                1,
                new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                    .activation(Activation.IDENTITY)
                    .nIn(nHidden)
                    .nOut(numOutputs)
                    .build())
            .build();

    MultiLayerNetwork net = new MultiLayerNetwork(conf);
    net.init();
    net.setListeners(new ScoreIterationListener(100));

    DataSetIterator iterator = getTrainingData(batchSize, RNG);

    // Train the network on the full data set, and evaluate in periodically
    for (int i = 0; i < nEpochs; i++) {
      iterator.reset();
      net.fit(iterator);
    }

    final INDArray input = Nd4j.create(new double[] {0.25});
    INDArray out = net.output(input, true);
    System.out.println(out);
  }

  private static DataSetIterator getTrainingData(int batchSize, Random rand) {
    double[] sum = new double[nSamples];
    double[] input1 = new double[nSamples];
    for (int i = 0; i < nSamples; i++) {
      input1[i] = MIN_RANGE + (MAX_RANGE - MIN_RANGE) * rand.nextDouble();

      // Beliebige Funktion
      sum[i] = input1[i] * 2;
    }
    INDArray inputNDArray1 = Nd4j.create(input1, new int[] {nSamples, 1});
    INDArray inputNDArray = Nd4j.hstack(inputNDArray1);
    INDArray outPut = Nd4j.create(sum, new int[] {nSamples, 1});
    DataSet dataSet = new DataSet(inputNDArray, outPut);
    List<DataSet> listDs = dataSet.asList();
    Collections.shuffle(listDs, RNG);
    return new ListDataSetIterator(listDs, batchSize);
  }
}
