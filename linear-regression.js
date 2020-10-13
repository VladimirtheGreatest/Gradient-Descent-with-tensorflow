const tf = require("@tensorflow/tfjs");
const _ = require("lodash");

class LinearRegression {
  constructor(features, labels, options) {
    this.features = tf.tensor(features);
    this.labels = tf.tensor(labels);

    //generates an extra column so we can use the matrix multiplication
    //ones([shape]) shape = features row, one column,    1 for concatenation axis
    this.features = tf
      .ones([this.features.shape[0], 1])
      .concat(this.features, 1);

    //if we dont provide learning rate in options the default rate will be 0.1
    this.options = Object.assign(
      { learningRate: 0.1, iterations: 1000 },
      options
    );

    //initial guesses, M and B previously
    this.weights = tf.zeros([2, 1]);
  }

  gradientDescent() {
    //mathmul matrix multiplication
    const currentGuesses = this.features.matMul(this.weights);
    const differences = currentGuesses.sub(this.labels);

    const slopes = this.features
      .transpose() // RESHAPING TENSOR so we can match the shape of differences
      .matMul(differences)
      .div(this.features.shape[0]);

      this.weights = this.weights.sub(slopes.mul(this.options.learningRate)); //this.m = this.m - mSlope * this.options.learningRate;
  }

  train() {
    for (let index = 0; index < this.options.iterations; index++) {
      this.gradientDescent();
    }
  }
  test(testFeatures,testLabels){
    testFeatures = tf.tensor(testFeatures);
    testLabels = tf.tensor(testLabels);

    testFeatures = tf.ones([testFeatures.shape[0], 1]).concat(testFeatures, 1);

    const predictions = testFeatures.matMul(this.weights);

    //coefficient of determination  R2 = 1 - total sum of squares / sum of squares of residuals  check notes, aka gauging accuracy of our prediction

    //sum of squares of residuals
    const res = testLabels.sub(predictions)
    .pow(2)
    .sum() // we dont have to provide axis for this
    .get()
    //total sum of squares
    const tot = testLabels
    .sub(testLabels.mean())
    .pow(2)
    .sum()
    .get();

    return 1 - res / tot;
  }
}

module.exports = LinearRegression;
