import {DataFeatures2, DataMovementMatrix} from "./data";
import {Model} from "./recording";

export * as data from "./data";
export * from "./recording";

// Pre-trained model, easy to use with class Model and Recorder.
export const Models = Object.freeze({
  rnn3: new Model(
    "https://raw.githubusercontent.com/chrisgdt/DELBOT-Mouse/master/trained-models/rnn3/model-rnn3-features2.json",
    new DataFeatures2({
      // Default parameters
      numClasses:1,
      xSize:24,
      shouldCompleteXSize:false
    })),
  rnn1: new Model(
    "https://raw.githubusercontent.com/chrisgdt/DELBOT-Mouse/master/trained-models/rnn1/model-rnn1-features2.json",
    new DataFeatures2({
      // Default parameters
      numClasses:1,
      xSize:24,
      shouldCompleteXSize:false
    })),
  denseMatrix: new Model(
    "https://raw.githubusercontent.com/chrisgdt/DELBOT-Mouse/master/trained-models/dense_matrix/model-dense-matrix.json",
    new DataMovementMatrix({
      // Default parameters
      numClasses:1,
      xMinMov: -25, xMaxMov: 25,
      yMinMov: -25, yMaxMov: 25,
      steps: [25, 50, 100, 150, 200, 250]
    })),
  convolutional: new Model(
    "https://raw.githubusercontent.com/chrisgdt/DELBOT-Mouse/master/trained-models/convolutional_matrix/model-conv-matrix.json",
    new DataMovementMatrix({
      // Default parameters
      numClasses:1,
      xMinMov: -25, xMaxMov: 25,
      yMinMov: -25, yMaxMov: 25,
      steps: [25, 50, 100, 150, 200, 250]
    })),
});
