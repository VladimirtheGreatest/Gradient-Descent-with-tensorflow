const tf = require("@tensorflow/tfjs");
const _ = require("lodash");

class LinearRegression {
  constructor(features, labels, options) {
    this.features = features;
    this.labels = labels;
    //if we dont provide learning rate in options the default rate will be 0.1
    this.options = Object.assign(
      { learningRate: 0.1, iterations: 1000 },
      options
    );

    //initial guesses
    this.m = 0;
    this.b = 0;
  }

  gradientDescent() {
    const currentGuessesForMPG = this.features.map((row) => {
      //inner array calculation
      return this.m * row[0] + this.b;
    });

    //formula   sum of ((m * feature) + b) - label(actual value)
    const bSlope =
      (_.sum(
        currentGuessesForMPG.map((guess, i) => {
          return guess - this.labels[i][0];
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
  }
  
  train() {
    for (let index = 0; index < this.options.iterations; index++) {
      this.gradientDescent();
    }
  }
}

module.exports = LinearRegression;
