import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import * as delbot from "@chrisgdt/delbot-mouse";
import {DataTraining} from "./dataTraining";


export interface ModelProperties {
  dataTraining: DataTraining;
  nameToSave?: string;
  nameToLoad?: string;
  useTfjsVis?: boolean;
  consoleInfo?: boolean;
  epoch?: number;
  batchSize?: number;
}

export abstract class Model {
  protected readonly data: DataTraining;
  protected readonly nameToSave: string;
  protected readonly nameToLoad: string;
  protected readonly useTfjsVis: boolean;
  protected readonly consoleInfo: boolean;
  protected readonly epoch: number;
  protected readonly batchSize: number;

  protected constructor(args: ModelProperties | DataTraining) {
    if (args instanceof DataTraining) args = {dataTraining: args};

    this.data = args.dataTraining;
    this.useTfjsVis = args.useTfjsVis == null ? false : args.useTfjsVis;
    this.consoleInfo = args.consoleInfo == null ? false : args.consoleInfo;
    this.nameToSave = args.nameToSave;
    this.nameToLoad = args.nameToLoad;

    this.epoch = args.epoch == null ? 10 : args.epoch;
    this.batchSize = args.batchSize == null ? 128 : args.batchSize;
  }

  async testingModel(predicts: tf.Tensor1D, labels: tf.Tensor1D, name: string = "validation") {
    const classAccuracy = await this.getClassAccuracy(predicts, labels);
    const confusionMatrix = await this.getConfusionMatrix(predicts, labels);
    labels.dispose();
    predicts.dispose();
    if (this.useTfjsVis) {
      await tfvis.show.perClassAccuracy({name: `Accuracy ${name}`, tab: 'Evaluation'}, classAccuracy, ['human', 'bot']);
      await tfvis.render.confusionMatrix({name: `Confusion Matrix ${name}`, tab: 'Evaluation'}, {
        values: confusionMatrix,
        tickLabels: ['human', 'bot']
      });
    }
    if (this.consoleInfo) {
      console.log("Class Accuracy", confusionMatrix);
      console.log("Confusion Matrix", confusionMatrix);
    }
  }

  /**
   * Call {@link doPredictionBatch} and compare the model output with the
   * real data labels, then show the accuracy with tfjs-vis or directly to the console.
   * @param predicts
   * @param labels
   */
  async getClassAccuracy(predicts: tf.Tensor1D, labels: tf.Tensor1D): Promise<{ accuracy: number, count: number }[]> {
    return await tfvis.metrics.perClassAccuracy(labels, predicts);
  }

  /**
   * Call {@link doPredictionBatch} and compare the model output with the
   * real data labels, then show the confusion matrix with tfjs-vis or directly to the console.
   * @param predicts
   * @param labels
   */
  async getConfusionMatrix(predicts: tf.Tensor1D, labels: tf.Tensor1D): Promise<number[][]> {
    return await tfvis.metrics.confusionMatrix(labels, predicts);
  }

  abstract run();

  abstract train();

  abstract saveModel();

  abstract loadExistingModel();

  /**
   * Gets a sample of test datas and returns the model output, with some basic operations
   * to reshape the output as 1d tensor.
   * @param size The batch size for the sample.
   * @return A list of two elements that are 1d tensors corresponding to a list [b1, b2, ...].
   *         These two list are 0-1 lists for each batch element to be a bot trajectory (1 means bot).
   */
  abstract doPredictionBatch(size: number): tf.Tensor1D[];
}

export interface RandomForestProperties extends ModelProperties {
  nEstimators?: number;
  maxDepth?: number;
  maxFeatures?: 'auto' | 'sqrt' | 'log2' | number;
  minSampleLeaf?: number;
  minInfoGain?: 0;
}

export class RandomForestModel extends Model {
  private randomForest: delbot.RandomForestClassifier;

  constructor(args: RandomForestProperties | DataTraining) {
    super(args);
    this.randomForest = new delbot.RandomForestClassifier({
    });
    if (this.data.data.numClasses !== 1) {
      throw Error("Random Forest should have one class as 'numClasses' !");
    }
  }

  async loadExistingModel() {
    const model = await delbot.utils.loadFile(this.nameToLoad);
    this.randomForest = delbot.RandomForestClassifier.load(JSON.parse(model));
  }

  saveModel() {
    delbot.utils.download(JSON.stringify(this.randomForest.toJSON()), this.nameToSave);
  }

  async run() {
    await this.data.load(this.consoleInfo);

    if (this.nameToLoad == null) {
      if (this.consoleInfo) console.log("Training of RandomForest...");
      await this.train();
      if (this.consoleInfo) { // @ts-ignore - TODO: featureImportance is not in type file but is in source code...
        console.log("Train finished ! Feature importance :", this.randomForest.featureImportance());
      }

      if (this.nameToSave != null) {
        await this.saveModel();
        if (this.consoleInfo) console.log("Saved !");
      }

    } else {
      await this.loadExistingModel();
    }

    if (this.useTfjsVis || this.consoleInfo) {
      const [predicts, labels] = this.doPredictionBatch();
      await this.testingModel(predicts, labels);

      if (this.randomForest.oobResults && this.randomForest.oobResults.length > 0) {
        const oobPred = tf.tensor1d(this.randomForest.predictOOB());
        const oobLabel = tf.tensor1d(this.randomForest.oobResults.map((v) => v.true));
        await this.testingModel(oobPred, oobLabel, "OOB");
      }
    }

    if (this.consoleInfo) console.log("-- end --");
  }

  async train() {
    const test = this.data.nextTestBatchRaw(this.data.nbrTestElements);
    const testX = await tf.reshape(test.xs, [test.xs.length, test.xs[0].length*test.xs[0][0].length]).array() as number[][];
    const testY = tf.tensor1d(Array.prototype.slice.call(tf.util.flatten(test.ys)));

    if (this.consoleInfo) console.log("Test data", testX, testY);

    const history = [];

    console.time('Time sync');
    for (let epoch = 0; epoch < this.epoch; epoch++) {
      const train = this.data.nextTrainBatchRaw(this.batchSize);
      const trainX = await tf.reshape(train.xs, [train.xs.length, train.xs[0].length*train.xs[0][0].length]).array() as number[][];
      const trainY = Array.prototype.slice.call(tf.util.flatten(train.ys));

      this.randomForest.train(trainX, trainY);

      if (this.useTfjsVis) {
        const predY = tf.tensor1d(this.randomForest.predict(testX));
        history.push({
          loss: await tf.metrics.binaryCrossentropy(testY, predY).array(),
          acc: await tf.metrics.binaryAccuracy(testY, predY).array()
        });
        await tfvis.show.history({name: 'Validation data', tab: 'Training'}, history, ['loss', 'acc']);
      }
    }
    console.timeEnd('Time sync');
    testY.dispose();
  }

  doPredictionBatch(size: number = 1000): tf.Tensor1D[] {
    const testData = this.data.nextTestBatchRaw(size);
    const xs = tf.reshape(testData.xs, [testData.xs.length, testData.xs[0].length*testData.xs[0][0].length]).arraySync() as number[][];
    return [tf.tensor1d(this.randomForest.predict(xs)), tf.tensor1d(Array.prototype.slice.call(tf.util.flatten(testData.ys)))];
  }
}


export interface TensorflowModelProperties extends ModelProperties {
  epoch?: number;
  batchSize?: number;
  trainingRatio?: number;
  testingRatio?: number;
  optimizer?: tf.Optimizer;
}

/**
 * Abstract class used to create a new TensorFlow.js model and train it
 * from a dataset {@link dataTraining!DataTraining}. Call
 * <br>
 * The constructor takes an object with parameters :
 * <ul>
 *   <li>dataTraining : a {@link dataTraining!DataTraining} instance to load datas.</li>
 *   <li>epoch : integer number of times to iterate over the training data arrays. Default to 10</li>
 *   <li>batchSize : number of samples per gradient update. If unspecified, it will default to 128.</li>
 *   <li>trainingRatio : a number between 0 and 1, default to 1, the %/100 of the training set used during training.</li>
 *   <li>testingRatio : a number between 0 and 1, default to 0.9, the %/100 of the testing set used for validation.</li>
 *   <li>nameToSave : if specified, the model will be saved as this name with {@link tf.LayersModel.save} after
 *                    training, e.g. 'downloads://my-model'.</li>
 *   <li>nameToLoad : if specified, the model won't be trained but loaded with {@link tf.loadLayersModel}.</li>
 *   <li>useTfjsVis : default to false, if true, the code shows additional information through tfjs-vis.</li>
 *   <li>consoleInfo : default to false, if true, the code shows addition information through the console log.</li>
 *   <li>optimizer : a {@link tf.Optimizer} instance for the training. If unspecified,
 *                   it will default to `tf.train.adam(1e-3, .5, .999, 1e-8)`</li>
 * </ul>
 * <br>
 * Extend this class to create your own model logic, all you have to do is
 * implement the method {@link initModel} that creates and returns a new
 * Sequential model, not compiled. The method {@link finalizeModel} adds
 * the last dense layer to the number of classes and compiles the model,
 * you can also override this one to modify the behavior of the last layers
 * or how to compile the model.
 */
export abstract class TensorflowModel extends Model {
  protected readonly trainingRatio: number;
  protected readonly testingRatio: number;
  private readonly optimizer: tf.Optimizer;

  protected model: null | tf.LayersModel = null;

  protected constructor(args: TensorflowModelProperties | DataTraining) {
    if (args instanceof DataTraining) args = {dataTraining: args};
    super(args);

    this.trainingRatio = args.trainingRatio == null ? 1 : args.trainingRatio;
    this.testingRatio = args.testingRatio == null ? .9 : args.testingRatio;

    this.optimizer = args.optimizer == null ? tf.train.adam(1e-3, .5, .999, 1e-8) : args.optimizer;
  }

  /**
   * Create a new sequential model with {@link tf.sequential()}, add
   * all hidden layers and returns the result. Notice that the input
   * shape will always be `[this.data.data.getXSize(), this.data.data.getYSize()]`
   * or `[null, this.data.data.getYSize()]` if we want to allow different
   * time steps for recurrent models after training (however, for training, a
   * recurrent model such as LSTM must have a fixed input shape for gradient
   * descent).
   * @return The new sequential model.
   */
  abstract initModel(): tf.Sequential;

  /**
   * Add the last layer that decides the class of the input
   * for the classifier and compile the model.
   * <br>
   * The last layer is dense with 'varianceScaling' kernel initializer,
   * and we look at accuracy metric.
   * <br>
   * When there is only one class in {@link dataTraining!DataTraining} ([0] for human,
   * [1] for bot), we use a binaryCrossentropy loss function and the
   * activation of the dense layer is the sigmoid function. If we have
   * two classes ([1,0] for human, [0,1] for bot), it is the categoricalCrossentropy
   * loss function and softmax activation.
   * @param model The model returned by {@link initModel}.
   * @return The same model as input but compiled.
   */
  finalizeModel(model: tf.Sequential): tf.Sequential {
    model.add(tf.layers.dense({
      units: this.data.data.numClasses,
      kernelInitializer: 'varianceScaling',
      activation: this.data.data.numClasses === 1 ? 'sigmoid' : 'softmax'
    }));

    model.compile({
      optimizer: this.optimizer,
      loss: this.data.data.numClasses === 1 ? 'binaryCrossentropy' : 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });

    return model;
  }

  /**
   * Load an existing model with {@link tf.loadLayersModel} and return it.
   */
  async loadExistingModel() {
    this.model = await tf.loadLayersModel(this.nameToLoad);
  }

  async saveModel() {
    const saveResult = await this.model.save(this.nameToSave);
    if (this.consoleInfo) console.log("Saved !", saveResult);
  }

  /**
   * Run the entire training process, from data loading to model training, then
   * model testing with accuracy and confusion matrix. If field {@link TensorflowModelProperties.nameToLoad}
   * is specified, training is ignored because we directly call {@link loadExistingModel}.
   * If field {@link TensorflowModelProperties.nameToSave} is specified, it save the model by calling {@link tf.LayersModel.save}.
   */
  async run() {
    if (!this.data.isLoaded()) {
      await this.data.load(this.consoleInfo);
    }

    if (this.nameToLoad == null) {

      this.model = this.finalizeModel(this.initModel());

      if (this.useTfjsVis) await tfvis.show.modelSummary({name: 'Model Architecture', tab: 'Model'}, this.model);
      if (this.consoleInfo) console.log("Model constructed !");

      let history = await this.train();
      if (this.consoleInfo) console.log("Train finished ! history :", history);

      if (this.nameToSave != null) await this.saveModel();

    } else {

      await this.loadExistingModel();
      if (this.useTfjsVis) await tfvis.show.modelSummary({name: 'Model Architecture', tab: 'Model'}, this.model);

    }

    if (this.useTfjsVis || this.consoleInfo) {
      const [predicts, labels] = this.doPredictionBatch();
      await this.testingModel(predicts, labels);
    }
    if (this.consoleInfo) console.log("-- end --");
  }

  /**
   * Train the model with {@link tf.LayersModel.fit}. We first get a fragment
   * of training and testing datas according to {@link TensorflowModelProperties.trainingRatio} and
   * {@link TensorflowModelProperties.testingRatio}, default set to 1 and 0.9 respectively.
   * Then we fit the model with the corresponding batchSize and epochs.
   */
  async train() {
    const TRAIN_DATA_SIZE = Math.round(this.data.nbrTrainElements*this.trainingRatio);
    const TEST_DATA_SIZE = Math.round(this.data.nbrTestElements*this.testingRatio);

    const [trainXs, trainYs] = tf.tidy(() => {
      const d = this.data.nextTrainBatch(TRAIN_DATA_SIZE);
      return [
        d.xs.reshape(this.model.inputs[0].shape.length === 4
          ? [TRAIN_DATA_SIZE, this.data.data.getXSize(), this.data.data.getYSize(), 1]
          : [TRAIN_DATA_SIZE, this.data.data.getXSize(), this.data.data.getYSize()]),
        d.ys
      ];
    });

    const [testXs, testYs] = tf.tidy(() => {
      const d = this.data.nextTestBatch(TEST_DATA_SIZE);
      return [
        d.xs.reshape(this.model.inputs[0].shape.length === 4
          ? [TEST_DATA_SIZE, this.data.data.getXSize(), this.data.data.getYSize(), 1]
          : [TEST_DATA_SIZE, this.data.data.getXSize(), this.data.data.getYSize()]),
        d.ys
      ];
    });

    if (this.consoleInfo) console.log("Test data", testXs.arraySync(), testYs.arraySync());

    return this.model.fit(trainXs, trainYs, {
      batchSize: this.batchSize,
      validationData: [testXs, testYs],
      epochs: this.epoch,
      shuffle: true,
      callbacks: this.useTfjsVis
        ? tfvis.show.fitCallbacks({name: 'Model Training', tab: 'Model', styles: {height: '1000px'}}, ['loss', 'val_loss', 'acc', 'val_acc'])
        : null,
    });
  }

  doPredictionBatch(size = 1000): tf.Tensor1D[] {
    const testData = this.data.nextTestBatch(size);
    const testXs = testData.xs.reshape(this.model.inputs[0].shape.length === 4
      ? [size, this.data.data.getXSize(), this.data.data.getYSize(), 1]
      : [size, this.data.data.getXSize(), this.data.data.getYSize()]);
    let labels = testData.ys;
    let predicts = this.model.predict(testXs) as tf.Tensor;
    if (!predicts.isFinite().all()) {
      console.warn("Warning, model predicted at least one NaN value.");
    }

    if (predicts.shape[1] === 1) {
      labels = labels.reshape([labels.size]).round();
      predicts = predicts.reshape([predicts.size]).round();
    } else {
      labels = labels.argMax(-1);
      predicts = predicts.argMax(-1);
    }

    testXs.dispose();
    return [predicts as tf.Tensor1D, labels as tf.Tensor1D];
  }
}

export class ModelConvolutional extends TensorflowModel {

  initModel(): tf.Sequential {
    // Model from TFJS tutorial https://www.tensorflow.org/js/tutorials/training/handwritten_digit_cnn?hl=fr
    const model = tf.sequential();

    model.add(tf.layers.conv2d({
      inputShape: [this.data.data.getXSize(), this.data.data.getYSize(), 1],
      kernelSize: 5,
      filters: 8,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    model.add(tf.layers.conv2d({
      kernelSize: 5,
      filters: 16,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    model.add(tf.layers.flatten());

    return model;
  }
}

export class ModelDense extends TensorflowModel {

  initModel(): tf.Sequential {
    const model = tf.sequential();

    model.add(tf.layers.flatten({inputShape: [this.data.data.getXSize(), this.data.data.getYSize()]}));
    model.add(tf.layers.dense({
      units: 128,
      activation: 'relu',
    }));
    model.add(tf.layers.dropout({rate:0.4}));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.dense({
      units: 64,
      activation: 'relu',
    }));
    model.add(tf.layers.dropout({rate:0.3}));

    return model;
  }
}

export class ModelRNN extends TensorflowModel {

  initModel(): tf.Sequential {
    const model = tf.sequential();

    model.add(tf.layers.lstm({
      inputShape: [null, this.data.data.getYSize()],
      units: 128,
      returnSequences: true,
      recurrentDropout : .5,
    }));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.leakyReLU({alpha: .1}));
    model.add(tf.layers.dropout({rate:0.3}));

    model.add(tf.layers.lstm({
      units: 64,
      returnSequences: false,
      recurrentDropout : .8,
    }));
    model.add(tf.layers.leakyReLU({alpha: .1}));
    model.add(tf.layers.dropout({rate:0.1}));

    return model;
  }
}

export class ModelRNN2 extends TensorflowModel {
  initModel(): tf.Sequential  {
    const model = tf.sequential();

    model.add(tf.layers.lstm({
      inputShape: [null, this.data.data.getYSize()],
      units: 128,
      returnSequences: false,
      activation: 'relu'
    }));
    model.add(tf.layers.dropout({rate:0.4}));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.dense({
      units: 64,
      //activation: 'relu',
      kernelRegularizer: tf.regularizers.l1l2({l1: 1e-5, l2: 1e-4}),
      biasRegularizer: tf.regularizers.l2({l2: 1e-4}),
    }));
    model.add(tf.layers.leakyReLU({alpha: .1}));
    model.add(tf.layers.dropout({rate:0.2}));

    return model;
  }
}

export class ModelRNN3 extends TensorflowModel {
  initModel(): tf.Sequential {
    const model = tf.sequential();

    model.add(tf.layers.lstm({
      inputShape: [null, this.data.data.getYSize()],
      units: 64,
      returnSequences: true,
      recurrentDropout : .3,
    }));
    model.add(tf.layers.leakyReLU({alpha: .1}));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.dropout({rate:0.3}));
    model.add(tf.layers.lstm({
      units: 32,
      returnSequences: false,
      recurrentDropout : .2,
    }));
    model.add(tf.layers.leakyReLU({alpha: .1}));
    model.add(tf.layers.dropout({rate:0.1}));
    model.add(tf.layers.dense({
      units: 16,
      activation: 'relu',
      kernelRegularizer: tf.regularizers.l1l2({l1: 1e-5, l2: 1e-4}),
      biasRegularizer: tf.regularizers.l2({l2: 1e-4}),
    }));

    return model;
  }
}
