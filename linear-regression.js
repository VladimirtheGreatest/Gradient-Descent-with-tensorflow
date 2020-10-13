const tf = require("@tensorflow/tfjs");
const _ = require("lodash");
import * as GradientDescentOld from './gradientDescentOld';

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
}

module.exports = LinearRegression;
