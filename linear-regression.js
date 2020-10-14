const tf = require("@tensorflow/tfjs");
const _ = require("lodash");

class LinearRegression {
  constructor(features, labels, options) {
    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels);

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
    testFeatures = this.processFeatures(testFeatures);
    testLabels = tf.tensor(testLabels);

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

  processFeatures(features){
    //generates an extra column so we can use the matrix multiplication
    //ones([shape]) shape = features row, one column,    1 for concatenation axis
    features = tf.tensor(features);

    //we have to reapply mean and variance for our test set if it is second time
    if(this.mean && this.variance){
      features =  features.sub(this.mean).div(this.variance.pow(0.5));
      //we use our helper function for the first case
    } else {
      features = this.standardize(features);
    }
    //column of ones needs to come after standardization so we do not change the ones values
    features = tf.ones([features.shape[0], 1]).concat(features, 1);

    return features;
  }

  standardize(features){
    const {mean, variance} = tf.moments(features, 0);

    this.mean = mean;
    this.variance = variance;

    return features.sub(mean).div(variance.pow(0.5));
  }
}

module.exports = LinearRegression;
