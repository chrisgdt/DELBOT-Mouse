import * as tf from "@tensorflow/tfjs";
import * as delbot from "@chrisgdt/delbot-mouse";

export interface Session {
  human: string[];
  bot: string[];
}

export interface DataTrainingProperties {
  /**
   * The data objet to parse raw datas with {@link delbot.data.Data.loadDataSet}.
   */
  data: delbot.data.Data;

  /**
   * The path to a json file containing an object {@link Session}. Default to "../../python/sessions.json".
   */
  filePath?: string;

  /**
   * A number between 0 and 1, the %/100 of the training set compared to the whole dataset, default to 0.8 for 80%.
   */
  trainingRatio?: number;
}

/**
 *  Utility class to load raw datas from your computer, parse them with a Data objet and get final
 *  tensors for a model. You first need a json file that represents a {@link Session} object, with
 *  a list of files containing human or bot trajectories. Each files must be something like:
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
 * were, as first line, the resolution to normalize X and Y and
 * the following lines are `timestamp,actionType,x,y`.
 * <br>
 * A part of the dataset if used as training datas and the other part for testing.
 * <br>
 * To use this class, just call {@link DataTraining.load} then you can call
 * {@link DataTraining.nextTrainBatch} and {@link DataTraining.nextTestBatch}.
 */
export class DataTraining {
  public readonly data: delbot.data.Data;
  public readonly filePath: string;
  public readonly trainingRatio: number;

  public nbrTrainElements: number = -1;
  public nbrTestElements: number = -1;

  private shuffledTestIndex: number = 0;
  private shuffledTrainIndex: number = 0;

  private trainImages: number[][][] = [];
  private trainLabels: number[][] = [];
  private testImages: number[][][] = [];
  private testLabels: number[][] = [];

  /**
   * Constructor of the DataTraining object, it takes parameters :
   * <ul>
   *   <li>data : The data objet to parse raw datas with {@link delbot.data.Data.loadDataSet}.</li>
   *   <li>filePath : The path to a json file containing an object {@link Session},
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
   * Return whether datas have been loaded.
   */
  isLoaded(): boolean {
    return this.nbrTrainElements > 0;
  }

  /**
   * Loads the json file {@link Session} containing every mouse trajectories and
   * their labels according to the {@link data} objet and separate the dataset into
   * training data set and testing dataset, with a shuffle.
   */
  async load(consoleInfo: boolean=false) {
    let datasetData: number[][][] = [];
    let datasetLabels: number[][] = [];

    const sessions: Session = JSON.parse(await delbot.utils.loadFile(this.filePath));
    if (sessions == null) {
      throw Error(`The json file ${this.filePath} does not exist !`);
    }
    const recorder = new delbot.Recorder();

    let userIndex = -1;
    for (let user in sessions) {
      userIndex++;
      //if (userIndex >= this.numClasses) continue;
      for (let sess of sessions[user]) {
        recorder.loadRecordsFromString(await delbot.utils.loadFile(sess), this.data.mayNormalize() ? -1 : 1, this.data.mayNormalize() ? -1 : 1);
        const datas = this.data.loadDataSet(recorder, userIndex);
        datasetData = datasetData.concat(datas.datasetData);
        datasetLabels = datasetLabels.concat(datas.datasetLabels);
      }
      if (consoleInfo) console.log(`debug: total length of data set with ${user}=${datasetData.length}`);
    }

    const nbrDatasetElements = datasetData.length;
    this.nbrTrainElements = Math.round(nbrDatasetElements * this.trainingRatio);
    this.nbrTestElements = nbrDatasetElements - this.nbrTrainElements;

    let shuffledIndices = tf.util.createShuffledIndices(nbrDatasetElements);
    const trainIndices = shuffledIndices.slice(0, this.nbrTrainElements).values();
    const testIndices = shuffledIndices.slice(this.nbrTrainElements).values();

    let index = trainIndices.next();
    while (!index.done) {
      this.trainImages.push(datasetData[index.value]);
      this.trainLabels.push(datasetLabels[index.value]);
      index = trainIndices.next()
    }

    index = testIndices.next();
    while (!index.done) {
      this.testImages.push(datasetData[index.value]);
      this.testLabels.push(datasetLabels[index.value]);
      index = testIndices.next()
    }

    if (consoleInfo) console.log(`debug: trainSize=${this.trainImages.length}, testSize=${this.testImages.length}`);
  }

  /**
   * Get a batch of datas and labels as number array from the training set.
   * @param batchSize
   */
  nextTrainBatchRaw(batchSize: number): { xs: number[][][]; ys: number[][]; } {
    const result = this.nextBatchRaw(batchSize, this.trainImages, this.trainLabels, this.shuffledTrainIndex);
    this.shuffledTrainIndex = (this.shuffledTrainIndex + batchSize) % this.trainImages.length;
    return result;
  }

  /**
   * Get a batch of tensor datas and labels from the training set.
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
   * Get a batch of datas and labels as number array from the testing set.
   * @param batchSize The batch size.
   */
  nextTestBatchRaw(batchSize: number): { xs: number[][][]; ys: number[][]; } {
    const result = this.nextBatchRaw(batchSize, this.testImages, this.testLabels, this.shuffledTestIndex);
    this.shuffledTestIndex = (this.shuffledTestIndex + batchSize) % this.testImages.length;
    return result;
  }

  /**
   * Get a batch of tensor datas and labels from the testing set.
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
      throw new Error("Cannot get a train batch when the training datas are not loaded !");
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
