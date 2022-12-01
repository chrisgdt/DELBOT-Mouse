import * as tf from "@tensorflow/tfjs";
import * as delbot from "@chrisgdt/delbot-mouse";

export interface Session {
  human: string[];
  bot: string[];
}

export interface DataTrainingProperties {
  /**
   * The data object to parse raw data with {@link delbot.data.Data.loadDataSet}.
   */
  data: delbot.data.Data;

  /**
   * The path to a JSON file containing an object {@link Session}. Default to "../../python/sessions.json".
   */
  filePath?: string;

  /**
   * A number between 0 and 1, the %/100 of the training set compared to the whole dataset, default to 0.8 for 80%.
   */
  trainingRatio?: number;
}

/**
 *  Utility class to load raw data from your computer, parse them with a Data object and get final
 *  tensors for a model. You first need a JSON file that represents a {@link Session} object, with
 *  a list of files containing human or bot trajectories. Each file must be something like:
 *  ```
 *  resolution:1536,864
 *  9131.1,Pressed,717,361
 *  9134.8,Move,717,361
 *  9151.8,Move,717,361
 *  [...]
 *  10402.3,Move,722,360
 *  10419.1,Move,718,358
 *  10425.8,Released,717,360
 * ```
 * Were, as first line, the resolution to normalize X and Y and
 * the following lines are `timestamp,actionType,x,y`.
 * <br>
 * A part of the dataset if used as training data and the other part for testing.
 * <br>
 * To use this class, just call {@link DataTraining.load}, then you can call
 * {@link DataTraining.nextTrainBatch} and {@link DataTraining.nextTestBatch}.
 */
export class DataTraining {
  /**
   * The data object to parse raw data with {@link delbot.data.Data.loadDataSet}.
   */
  public readonly data: delbot.data.Data;

  /**
   * The path to a JSON file containing an object {@link Session}. Default to "../../python/sessions.json".
   */
  public readonly filePath: string;

  /**
   * A number between 0 and 1, the %/100 of the training set compared to the whole dataset, default to 0.8 for 80%.
   */
  public readonly trainingRatio: number;

  public nbrTrainElements: number = -1;
  public nbrTestElements: number = -1;

  private shuffledTestIndex: number = 0;
  private shuffledTrainIndex: number = 0;

  private trainData: number[][][] = [];
  private trainLabel: number[][] = [];
  private testData: number[][][] = [];
  private testLabel: number[][] = [];

  /**
   * Constructor of the DataTraining object, it takes parameters :
   * <ul>
   *   <li>data : The data object to parse raw data with {@link delbot.data.Data.loadDataSet}.</li>
   *   <li>filePath : The path to a JSON file containing an object {@link Session},
   *                  default to "../../python/sessions.json"</li>
   *   <li>trainingRatio : A number between 0 and 1, the %/100 of the training set
   *                       compared to the whole dataset, default to 0.8 for 80%.</li>
   * </ul>
   */
  constructor(args: delbot.data.Data | DataTrainingProperties) {
    if (args instanceof delbot.data.Data) args = {data: args};

    this.data = args.data;
    this.filePath = args.filePath == null ? "../../python/sessions.json" : args.filePath;
    this.trainingRatio = args.trainingRatio == null ? .8 : args.trainingRatio;
  }

  /**
   * Return whether data have been loaded.
   */
  isLoaded(): boolean {
    return this.nbrTrainElements > 0;
  }

  /**
   * Loads the JSON file {@link Session} containing every mouse trajectories and
   * their labels according to the {@link data} object and separate the dataset into
   * training data set and testing dataset, with a shuffle.
   */
  async load(consoleInfo: boolean=false) {
    let datasetData: number[][][] = [];
    let datasetLabels: number[][] = [];

    const sessions: Session = JSON.parse(await delbot.utils.loadFile(this.filePath));
    if (sessions == null) {
      throw Error(`The JSON file ${this.filePath} does not exist !`);
    }
    const recorder = new delbot.Recorder();

    let userIndex = -1;
    for (let user in sessions) {
      userIndex++;
      //if (userIndex >= this.numClasses) continue;
      for (let sess of sessions[user]) {
        recorder.loadRecordsFromString(await delbot.utils.loadFile(sess), this.data.mayNormalize() ? -1 : 1, this.data.mayNormalize() ? -1 : 1);
        const data = this.data.loadDataSet(recorder, userIndex);
        datasetData = datasetData.concat(data.datasetData);
        datasetLabels = datasetLabels.concat(data.datasetLabels);
      }
      if (consoleInfo) console.log(`debug: total length of dataset with ${user}=${datasetData.length}`);
    }

    const nbrDatasetElements = datasetData.length;
    this.nbrTrainElements = Math.round(nbrDatasetElements * this.trainingRatio);
    this.nbrTestElements = nbrDatasetElements - this.nbrTrainElements;

    let shuffledIndices = tf.util.createShuffledIndices(nbrDatasetElements);
    const trainIndices = shuffledIndices.slice(0, this.nbrTrainElements).values();
    const testIndices = shuffledIndices.slice(this.nbrTrainElements).values();

    let index = trainIndices.next();
    while (!index.done) {
      this.trainData.push(datasetData[index.value]);
      this.trainLabel.push(datasetLabels[index.value]);
      index = trainIndices.next()
    }

    index = testIndices.next();
    while (!index.done) {
      this.testData.push(datasetData[index.value]);
      this.testLabel.push(datasetLabels[index.value]);
      index = testIndices.next()
    }

    if (consoleInfo) console.log(`debug: trainSize=${this.trainData.length}, testSize=${this.testData.length}`);
  }

  /**
   * Get a batch of data and labels as number array from the training set.
   * @param batchSize
   */
  nextTrainBatchRaw(batchSize: number): { xs: number[][][]; ys: number[][]; } {
    const result = this.nextBatchRaw(batchSize, this.trainData, this.trainLabel, this.shuffledTrainIndex);
    this.shuffledTrainIndex = (this.shuffledTrainIndex + batchSize) % this.trainData.length;
    return result;
  }

  /**
   * Get a batch of tensor data and labels from the training set.
   * @param batchSize The batch size.
   */
  nextTrainBatch(batchSize: number): { xs: tf.Tensor<tf.Rank>; ys: tf.Tensor<tf.Rank>; } {
    const {xs, ys} = this.nextTrainBatchRaw(batchSize);
    return {
      xs: tf.tensor3d(xs),
      ys: tf.tensor2d(ys)
    }
  }

  /**
   * Get a batch of data and labels as number array from the testing set.
   * @param batchSize The batch size.
   */
  nextTestBatchRaw(batchSize: number): { xs: number[][][]; ys: number[][]; } {
    const result = this.nextBatchRaw(batchSize, this.testData, this.testLabel, this.shuffledTestIndex);
    this.shuffledTestIndex = (this.shuffledTestIndex + batchSize) % this.testData.length;
    return result;
  }

  /**
   * Get a batch of tensor data and labels from the testing set.
   * @param batchSize The batch size.
   */
  nextTestBatch(batchSize: number): { xs: tf.Tensor<tf.Rank>; ys: tf.Tensor<tf.Rank>; } {
    const {xs, ys} = this.nextTestBatchRaw(batchSize);
    return {
      xs: tf.tensor3d(xs),
      ys: tf.tensor2d(ys)
    }
  }

  private nextBatchRaw(batchSize: number, images: number[][][], labels: number[][], shuffledIndex: number): { xs: number[][][]; ys: number[][]; } {
    if (!this.isLoaded()) {
      throw new Error("Cannot get a train batch when the training data are not loaded !");
    }
    const xs = [];
    const ys = [];
    for (let i = 0; i < batchSize; i++) {
      xs.push(images[shuffledIndex]);
      ys.push(labels[shuffledIndex]);
      shuffledIndex = (shuffledIndex + 1) % images.length;
    }
    return { xs, ys };
  }
}
