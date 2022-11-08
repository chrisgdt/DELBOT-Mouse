import {DataFeatures2} from "./data";
import {Model} from "./recording";

export * as data from "./data";
export * from "./recording";

// Pre-trained model, easy to use with class Model and Recorder.
export const Models = Object.freeze({
  rnn1: new Model(
    "https://raw.githubusercontent.com/chrisgdt/mouse-drawing/main/model_lstm_rnn1/model-rnn1-features2.json",
    new DataFeatures2({
      numClasses:1,
      xSize:24,
      shouldCompleteXSize:false
    })),
  rnn2: new Model(
    "https://raw.githubusercontent.com/chrisgdt/mouse-drawing/main/model_lstm_rnn2/model-rnn2-features2.json",
    new DataFeatures2({
      numClasses:1,
      xSize:24,
      shouldCompleteXSize:false
    })),
  rnn3: new Model(
    "https://raw.githubusercontent.com/chrisgdt/mouse-drawing/main/model_lstm_rnn3/model-rnn3-features2.json",
    new DataFeatures2({
      numClasses:1,
      xSize:24,
      shouldCompleteXSize:false
    })),
});
