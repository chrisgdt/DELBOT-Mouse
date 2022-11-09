import * as tf from "@tensorflow/tfjs";
import * as delbot from "@chrisgdt/delbot-mouse";
import {loadFile} from "./util";


export interface Session {
  human: string[];
  bot: string[];
}

export interface TensorData {
  xs: tf.Tensor<tf.Rank>;
  ys: tf.Tensor<tf.Rank>;
}

export interface DataTrainingProperties {
  /**
   * The data objet to parse raw datas with {@link delbot.data.Data.loadDataSet}.
   */
  data: delbot.data.Data;

  /**
   * A boolean, be default to true, to know if we let the default
   * normalization of the datas. If false, set the normalization to 1.
   */
  normalize?: boolean;

  /**
   * The path to a json file containing an object {@link Session}. Default to "../../python/sessions.json".
   */
  filePath?: string;

  /**
   * A number between 0 and 1, the %/100 of the training set compared to the whole dataset, default to 0.85 for 85%.
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
  public readonly normalize: boolean;
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
   *   <li>normalize A boolean, be default to true, to know if we let the default normalization of
   *                 the datas. If false, set the normalization to 1 so no data is normalized.</li>
   *   <li>filePath : The path to a json file containing an object {@link Session},
   *                  default to "../../python/sessions.json"</li>
   *   <li>trainingRatio : A number between 0 and 1, the %/100 of the training set
   *                       compared to the whole dataset, default to 0.85 for 85%.</li>
   * </ul>
   */
  constructor(args: delbot.data.Data | DataTrainingProperties) {
    if (args instanceof delbot.data.Data) args = {data: args};

    this.data = args.data;
    this.filePath = args.filePath == null ? "../../python/sessions.json" : args.filePath;
    this.normalize = args.normalize == null ? true : args.normalize;
    this.trainingRatio = args.trainingRatio == null ? .85 : args.trainingRatio;
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
   * training data set and testing dataset, with a shuffle. The parameter gives
   * the ratio, by default you will have 85% of your dataset as training and 15% as
   * testing.
   */
  async load(consoleInfo: boolean=false) {
    let datasetData: number[][][] = [];
    let datasetLabels: number[][] = [];

    const sessions: Session = JSON.parse(await loadFile(this.filePath));
    if (sessions == null) {
      throw Error(`The json file ${this.filePath} does not exist !`);
    }
    const recorder = new delbot.Recorder();

    let userIndex = -1;
    for (let user in sessions) {
      userIndex++;
      //if (userIndex >= this.numClasses) continue;
      for (let sess of sessions[user]) {
        recorder.loadRecordsFromString(await loadFile(sess), this.normalize ? -1 : 1, this.normalize ? -1 : 1);
        const datas = this.data.loadDataSet(recorder, userIndex);
        datasetData = datasetData.concat(datas.datasetData);
        datasetLabels = datasetLabels.concat(datas.datasetLabels);
      }
      if (consoleInfo) console.log(`debug: total length of data set with ${user}=${datasetData.length}`);
    }

    const nbrDatasetElements = datasetData.length;
    this.nbrTrainElements = Math.round(nbrDatasetElements * this.trainingRatio); // default to 85%
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
   * Get a batch of tensor datas and labels from the training set.
   * @param batchSize The batch size.
   */
  nextTrainBatch(batchSize: number): TensorData {
    if (!this.isLoaded()) {
      throw new Error("Cannot get a train batch when the training datas are not loaded !");
    }
    const img = [];
    const lab = [];
    for (let i=0; i<batchSize; i++) {
      img.push(this.trainImages[this.shuffledTrainIndex]);
      lab.push(this.trainLabels[this.shuffledTrainIndex]);
      this.shuffledTrainIndex = (this.shuffledTrainIndex + 1) % this.nbrTrainElements;
    }
    return {
      xs: tf.tensor3d(img),
      ys: tf.tensor2d(lab)
    };
  }

  /**
   * Get a batch of tensor datas and labels from the testing set.
   * @param batchSize The batch size.
   */
  nextTestBatch(batchSize: number): TensorData {
    if (!this.isLoaded()) {
      throw new Error("Cannot get a train batch when the training datas are not loaded !");
    }
    const img = [];
    const lab = [];
    for (let i=0; i<batchSize; i++) {
      img.push(this.testImages[this.shuffledTestIndex]);
      lab.push(this.testLabels[this.shuffledTestIndex]);
      this.shuffledTestIndex = (this.shuffledTestIndex + 1) % this.nbrTestElements;
    }
    return {
      xs: tf.tensor3d(img),
      ys: tf.tensor2d(lab)
    };
  }
}
