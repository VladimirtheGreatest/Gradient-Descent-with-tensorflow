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
  }

  //old implementation
  /* gradientDescent() {
    const currentGuessesForMPG = this.features.map((row) => {
      //inner array calculation or features * weights
      return this.m * row[0] + this.b;
    });

    //formula   sum of ((m * feature) + b) - label(actual value)
    const bSlope =
      (_.sum(
        currentGuessesForMPG.map((guess, i) => {
          return guess - this.labels[i][0];   //inner array - labels
        })
      ) *
        2) /
      this.features.length; //  * derivative / number of observation can be either length of features or labels

    const mSlope =
      (_.sum(
        currentGuessesForMPG.map((guess, i) => {
          return -1 * this.features[i][0] * (this.labels[i][0] - guess);
        })
      ) *
        2) /
      this.features.length;

      this.m = this.m - mSlope * this.options.learningRate;
      this.b = this.b - bSlope * this.options.learningRate;
  } */

  train() {
    for (let index = 0; index < this.options.iterations; index++) {
      this.gradientDescent();
    }
  }
}

module.exports = LinearRegression;
