import {DataFeatures2, DataMovementMatrix} from "./data";
import {RandomForestModel, TensorFlowModel, Model, Recorder} from "./recording";

// TODO: Something is causing a compile error on ml-random-forest, modify type.d.ts
//       manually works as temporary solution (https://github.com/mljs/random-forest/issues/33)
export { RandomForestClassifier, RandomForestBaseOptions } from 'ml-random-forest';
export * as utils from "./utils";
export * as data from "./data";
export * from "./recording";

/**
 * Contains pre-trained models for an easy use of {@link Model} class with {@link Recorder}.
 * The best model so far is rnn1.
 */
export const Models = Object.freeze({
  randomForest: new RandomForestModel(
    "https://raw.githubusercontent.com/chrisgdt/DELBOT-Mouse/master/trained-models/random-forest/random-forest-features2_1024.txt",
    new DataFeatures2({ // Default parameters of feature datas are numClasses: 2, xSize: 24, shouldCompleteXSize: false
      numClasses: 1,
      xSize: 24,
      shouldCompleteXSize: false
    })),
  rnn3: new TensorFlowModel(
    "https://raw.githubusercontent.com/chrisgdt/DELBOT-Mouse/master/trained-models/rnn3/model-rnn3-features2.json",
    new DataFeatures2({
      numClasses: 2,
      xSize: 24,
      shouldCompleteXSize: false
    })),
  rnn1: new TensorFlowModel(
    "https://raw.githubusercontent.com/chrisgdt/DELBOT-Mouse/master/trained-models/rnn1/model-rnn1-features2.json",
    new DataFeatures2({
      numClasses: 2,
      xSize: 24,
      shouldCompleteXSize: false
    })),
  denseMatrix: new TensorFlowModel(
    "https://raw.githubusercontent.com/chrisgdt/DELBOT-Mouse/master/trained-models/dense_matrix/model-dense-matrix.json",
    new DataMovementMatrix({
      // Default parameters
      numClasses: 2,
      xMinMov: -25, xMaxMov: 25,
      yMinMov: -25, yMaxMov: 25,
      steps: [25, 50, 100, 150, 200, 250]
    })),
  convolutional: new TensorFlowModel(
    "https://raw.githubusercontent.com/chrisgdt/DELBOT-Mouse/master/trained-models/convolutional_matrix/model-conv-matrix.json",
    new DataMovementMatrix({
      // Default parameters
      numClasses: 2,
      xMinMov: -25, xMaxMov: 25,
      yMinMov: -25, yMaxMov: 25,
      steps: [25, 50, 100, 150, 200, 250]
    })),
});
