require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv");
const LinearRegression = require('./linear-regression');
const plot = require('node-remote-plot');

let { features, labels, testFeatures, testLabels } = loadCSV("./cars.csv", {
  shuffle: true,
  splitTest: 50,
  dataColumns: ["horsepower", "weight", "displacement"],
  labelColumns: ["mpg"],
});

const regression = new LinearRegression(features, labels, {
  learningRate : 0.1,
  iterations: 100
});


regression.train();
const r2 = regression.test(testFeatures, testLabels);

plot({
 x: regression.mseHistory.reverse(), //we have to reverse iteration 
 xLabel : 'Iteration Number',
 yLabel : 'Mean Squared Error'
});


console.log('Coefficient of determination is', r2);
//negative value means bad result, we need to improve accuracy of our analysis

//console.log('Updated M is:', regression.weights.get(1,0), 'Updated B is:', regression.weights.get(0,0));