import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import {DataTraining} from "./dataTraining";

export interface ModelProperties {
  dataTraining: DataTraining;
  epoch?: number;
  batchSize?: number;
  trainingRatio?: number;
  testingRatio?: number;
  nameToSave?: string;
  nameToLoad?: string;
  useTfjsVis?: boolean;
  consoleInfo?: boolean;
  optimizer?: tf.Optimizer;
}

/**
 * Abstract class used to create a new TensorFlow.js model and train it
 * from a dataset {@link DataTraining}. Call
 * <br>
 * The constructor takes an object with parameters :
 * <ul>
 *   <li>dataTraining : a {@link DataTraining} instance to load datas.</li>
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
export abstract class Model {
  private readonly nameToSave: string;
  private readonly nameToLoad: string;
  private readonly epoch: number;
  private readonly batchSize: number;
  private readonly trainingRatio: number;
  private readonly testingRatio: number;
  private readonly useTfjsVis: boolean;
  private readonly consoleInfo: boolean;
  private readonly optimizer: tf.Optimizer;

  protected readonly data: DataTraining;
  protected model: null | tf.LayersModel = null;

  constructor(args: ModelProperties | DataTraining) {
    if (args instanceof DataTraining) args = {dataTraining: args};

    this.data = args.dataTraining;
    this.epoch = args.epoch == null ? 10 : args.epoch;
    this.batchSize = args.batchSize == null ? 128 : args.batchSize;
    this.useTfjsVis = args.useTfjsVis == null ? false : args.useTfjsVis;
    this.consoleInfo = args.consoleInfo == null ? false : args.consoleInfo;
    this.nameToSave = args.nameToSave;
    this.nameToLoad = args.nameToLoad;
    this.optimizer = args.optimizer == null ? tf.train.adam(1e-3, .5, .999, 1e-8) : args.optimizer;

    this.trainingRatio = args.trainingRatio == null ? 1 : args.trainingRatio;
    this.testingRatio = args.testingRatio == null ? .9 : args.testingRatio;
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
   * When there is only one class in {@link DataTraining} ([0] for human,
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
  async loadExistingModel(): Promise<tf.LayersModel> {
    return await tf.loadLayersModel(this.nameToLoad);
  }

  /**
   * Run the entire training process, from data loading to model training, then
   * model testing with accuracy and confusion matrix. If field {@link nameToLoad}
   * is specified, training is ignored because we directly call {@link loadExistingModel}.
   * If field {@link nameToSave} is specified, it save the model by calling {@link tf.LayersModel.save}.
   */
  async run() {
    await this.data.load();

    if (this.nameToLoad == null) {

      this.model = this.finalizeModel(this.initModel());

      if (this.useTfjsVis) await tfvis.show.modelSummary({name: 'Model Architecture', tab: 'Model'}, this.model);
      if (this.consoleInfo) console.log("Model constructed !");

      let history = await this.train();
      if (this.consoleInfo) console.log("Train finished ! history :", history);

      if (this.nameToSave != null) {
        const saveResult = await this.model.save(this.nameToSave);
        if (this.consoleInfo) console.log("Saved !", saveResult);
      }

    } else {

      this.model = await this.loadExistingModel();
      if (this.useTfjsVis) await tfvis.show.modelSummary({name: 'Model Architecture', tab: 'Model'}, this.model);

    }

    if (this.useTfjsVis || this.consoleInfo) {
      const classAccuracy = await this.getClassAccuracy();
      const confusionMatrix = await this.getConfusionMatrix();
      if (this.useTfjsVis) {
        await tfvis.show.perClassAccuracy({name: 'Accuracy', tab: 'Evaluation'}, classAccuracy, ['human', 'bot']);
        await tfvis.render.confusionMatrix({name: 'Confusion Matrix', tab: 'Evaluation'}, {values: confusionMatrix, tickLabels: ['human', 'bot']});
      }
      if (this.consoleInfo) {
        console.log("Class Accuracy", confusionMatrix);
        console.log("Confusion Matrix", confusionMatrix);
      }
    }

    if (this.consoleInfo) console.log("-- end --");
  }

  /**
   * Train the model with {@link tf.LayersModel.fit}. We first get a fragment
   * of training and testing datas according to {@link trainingRatio} and
   * {@link testingRatio}, default set to 1 and 0.9 respectively.
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

  /**
   * Gets a sample of test datas and returns the model output, with some basic operations
   * to reshape the output as 1d tensor.
   * @param size Default to 1000, the batch size for the sample.
   * @return A 1d tensor corresponding to a list [b1, b2, ...] of size {@link size} of
   *         booleans 0 or 1 for each batch element to be a bot trajectory (1 means bot).
   */
  doPredictionBatch(size = 1000): tf.Tensor1D[] {
    const testData = this.data.nextTestBatch(size);
    const testxs = testData.xs.reshape(this.model.inputs[0].shape.length === 4
      ? [size, this.data.data.getXSize(), this.data.data.getYSize(), 1]
      : [size, this.data.data.getXSize(), this.data.data.getYSize()]);
    let labels = testData.ys;
    let preds = this.model.predict(testxs) as tf.Tensor;
    if (!preds.isFinite().all()) {
      console.warn("Warning, model predicted at least one NaN value.");
    }

    if (preds.shape[1] === 1) {
      labels = labels.reshape([labels.size]).round();
      preds = preds.reshape([preds.size]).round();
    } else {
      labels = labels.argMax(-1);
      preds = preds.argMax(-1);
    }

    testxs.dispose();
    return [preds as tf.Tensor1D, labels as tf.Tensor1D];
  }

  /**
   * Call {@link doPredictionBatch} and compare the model output with the
   * real data labels, then show the accuracy with tfjs-vis or directly to the console.
   * @param size Default to 1000, the batch size for the sample.
   */
  async getClassAccuracy(size = 1000): Promise<{ accuracy: number, count: number }[]> {
    const [preds, labels] = this.doPredictionBatch();
    const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
    labels.dispose();
    return classAccuracy;
  }

  /**
   * Call {@link doPredictionBatch} and compare the model output with the
   * real data labels, then show the confusion matrix with tfjs-vis or directly to the console.
   * @param size Default to 1000, the batch size for the sample.
   */
  async getConfusionMatrix(size = 1000): Promise<number[][]> {
    const [preds, labels] = this.doPredictionBatch();
    const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
    labels.dispose();
    return confusionMatrix;
  }
}

export class ModelConvolutional extends Model {

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

export class ModelDense extends Model {

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

export class ModelRNN extends Model {

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

export class ModelRNN2 extends Model {
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

export class ModelRNN3 extends Model {
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
