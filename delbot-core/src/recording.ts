import * as tf from '@tensorflow/tfjs';
import { RandomForestClassifier } from 'ml-random-forest';
import { Data } from "./data";
import { loadFile } from "./utils";

/**
 * Represent a generic instance of a single recorded mouse action. It must contain at least
 * three data : time, x and y. Other features are calculated from these three data
 * in the Recorder class.
 */
export interface SingleRecord {
  /**
   * The time stamp when this movement was recorded, often given by `event.timeStamp`.
   */
  time: number;

  /**
   * The pixel of the x position of the movement. If used with a normalization where the scale is the
   * screen resolution, it is a number between 0 and 1, where 0 represents the left side and 1 the right side.
   */
  x: number;

  /**
   * The pixel of the y position of the movement. If used with a normalization where the scale is the
   * screen resolution, it is a number between 0 and 1, where 0 represents the left side and 1 the right side.
   */
  y: number;

  /**
   * The type of the recorded movement, it can be "Pressed", "Released" or "Move" (ending with "Touch" if from
   * a mobile device). It isn't actually used, so it can be undefined.
   */
  type?: string;

  /**
   * The difference `dt` between two consecutive times.
   */
  timeDiff?: number;

  /**
   * The x offset between the current point and the previous one.
   */
  dx?: number;

  /**
   * The y offset between the current point and the previous one.
   */
  dy?: number;

  /**
   * The speed in X axis between the current point and the previous one, calculated with `dx/dt`.
   */
  speedX?: number;

  /**
   * The speed in Y axis between the current point and the previous one, calculated with `dy/dt`.
   */
  speedY?: number;

  /**
   * The acceleration in X axis between the current point and the previous one, calculated with
   * the difference of two consecutive speed x divided by `dt`.
   */
  accelX?: number;

  /**
   * The acceleration in Y axis between the current point and the previous one, calculated with
   * the difference of two consecutive speed y divided by `dt`.
   */
  accelY?: number;

  /**
   * The total Euclidean distance between the current point and the previous one.
   */
  distance?: number;

  /**
   * The total speed between the current point and the previous one, calculated with `distance/dt`.
   */
  speed?: number;

  /**
   * The total acceleration between the current point and the previous one,
   * calculated the difference of two consecutive speed divided by `dt`.
   */
  accel?: number;

  /**
   * The difference of acceleration between the current point and the previous one divided by `dt`.
   */
  jerk?: number;

  /**
   * The speed x divided by distance.
   */
  speedXAgainstDistance?: number;

  /**
   * The speed y divided by distance.
   */
  speedYAgainstDistance?: number;

  /**
   * The acceleration x divided by distance.
   */
  accelXAgainstDistance?: number;

  /**
   * The acceleration y divided by distance.
   */
  accelYAgainstDistance?: number;

  /**
   * The average speed from the first recorded point divided by the distance.
   */
  averageSpeedAgainstDistance?: number;

  /**
   * The average acceleration from the first recorded point divided by the distance.
   */
  averageAccelAgainstDistance?: number;

  /**
   * The angle between the current point and the previous one, calculated by `atan(dy/dx)`.
   */
  angle?: number;
}

/**
 * Represents the result of the model to decide whether the record was human or not.
 * It's simply a boolean with a reason, i.e. small description of the result.
 */
export interface Result {

  /**
   * The simple boolean, true if the recording was considered as human. Otherwise,
   * the {@link Result.reason} gives more details.
   */
  result: boolean;

  /**
   * One of the three static field of Recorder class :
   * <ul>
   *   <li>{@link Recorder.success} if the record was considered as human.</li>
   *   <li>{@link Recorder.fail} if the record was considered as bot.</li>
   *   <li>{@link Recorder.notEnoughProvidedData} if there were not enough provided data.</li>
   * </ul>
   */
  reason: Symbol;
}

/**
 * An abstract class that represents a model with its associated data used during the
 * training. Concretely, you can give to the constructor a way to load a
 * model with {@link getModel} and describe how to use this model to in method
 * {@link predict} with parsed data from the given record and {@link data}.
 * <br>
 * The model never loads if you never call {@link getModel}, so you can instantiate
 * this class multiple times without troubles.
 */
export abstract class Model<G> {
  /**
   * The model field or null if {@link getModel} has never been called.
   * @protected
   */
  protected model: null | G;

  /**
   * The way to load the model with {@link getModel}, once loaded this field
   * is never used again.
   * @protected
   */
  protected readonly loadingPath: string;

  /**
   * The data object to format input.
   * @protected
   */
  protected readonly data: Data;

  /**
   * @param loadingPath The loadingPath (url, localstorage, indexeddb, ...) to load the model.
   * @param data The {@link data!Data} instance related to this model.
   */
  constructor(loadingPath: string, data: Data) {
    this.model = null;
    this.loadingPath = loadingPath;
    this.data = data;
  }

  /**
   * @Return The private field {@link data}.
   */
  getData(): Data {
    return this.data;
  }

  /**
   * @Return The loaded model instance. If there was no call of {@link getModel}, throw an error instead.
   */
  getLoadedModel(): G {
    if (this.model == null) {
      throw new Error("You have to load the model with getModel() first before trying to get the model synchronously.")
    }
    return this.model
  }

  /**
   * Loads the model (if it has never been done before) and returns it.
   * @return A promise of the loaded model.
   */
  abstract getModel(): Promise<G>;

  /**
   * Given a record of mouse features, use both {@link data} and {@link model} fields to
   * format the record and predict a list of values, each element is a probability corresponding to
   * an element of the dataset to be a bot trajectory.
   * @param record The record object with computed mouse features.
   * @param uniqueDataset An optional boolean, if true then we reshape the dataset to have a single
   *                      element with all our data, so the returned list should have a single element
   *                      (a single prediction), otherwise the model predict element by element and returns
   *                      the prediction array. All models do not support modified input shape.
   */
  abstract predict(record: Recorder, uniqueDataset: boolean): Promise<number[]>;
}

/**
 * An implementation of {@link Model} for TensorFlow layers models.
 * The loadingPath is directly sent to {@link tf.loadLayersModel} and leads to
 * the json file of the model.
 * @see Model
 */
export class TensorFlowModel extends Model<tf.LayersModel> {
  async getModel(): Promise<tf.LayersModel> {
    if (this.model === null) {
      this.model = await tf.loadLayersModel(this.loadingPath);
    }
    return this.model;
  }

  async predict(record: Recorder, uniqueDataset: boolean = false): Promise<number[]> {
    const dataset = this.getData().loadDataSet(record).datasetData;

    if (dataset.length === 0) {
      return [];
    }

    const tfjsModel = await this.getModel();
    let datasetTensor = tf.tensor3d(dataset);

    if (uniqueDataset) {
      // Put a single element with large chunk, we may lose some accuracy since the model trained on a fixed time-step
      datasetTensor = datasetTensor.reshape([1, dataset.length*this.getData().getXSize(), this.getData().getYSize()]);
    } else {
      datasetTensor = datasetTensor.reshape([dataset.length, this.getData().getXSize(), this.getData().getYSize()]);
    }

    // For convolutional models, input needs to be in 4 dimensions, replace every number n by 1d tensor [n]
    if (tfjsModel.inputs[0].shape.length === 4) {
      datasetTensor = datasetTensor.expandDims(3);
    }

    let predictions = tfjsModel.predict(datasetTensor) as tf.Tensor;
    if (predictions.shape[1] === 1) {
      predictions = tf.reshape(predictions, [predictions.size]);//.round();
    } else {
      // Get the bot probability p from prediction [1-p, p] for respectively human and bot,
      // get [p] if p > 1/2 and [1-p] otherwise. If we want to output 0 or 1 and not the
      // proba p, just use 'predictions = tf.argMax(predictions, -1)'.
      const {values, indices} = tf.topk(predictions, 1);
      // abs(1 - idx_max - p) = what we need
      predictions = tf.abs(tf.scalar(1).sub(indices).sub(values));
      predictions = tf.reshape(predictions, [predictions.size]);
    }
    const output = predictions.arraySync() as number[];

    datasetTensor.dispose();
    predictions.dispose();

    return output;
  }
}

/**
 * An implementation of {@link Model} for Random Forest classifiers.
 * The loadingPath is directly sent to {@link loadFile} then parsed to
 * JSON format and loaded with {@link RandomForestClassifier.load}.
 * @see Model
 */
export class RandomForestModel extends Model<RandomForestClassifier> {
  async getModel(): Promise<RandomForestClassifier> {
    if (this.model === null) {
      if (this.data.numClasses !== 1) {
        throw Error("Random Forest should have one class as 'numClasses' !");
      }
      this.model = new RandomForestClassifier({});
      this.model = RandomForestClassifier.load(JSON.parse(await loadFile(this.loadingPath)));
    }
    return this.model;
  }

  async predict(record: Recorder, uniqueDataset: boolean): Promise<number[]> {
    const dataset = this.getData().loadDataSet(record).datasetData;

    if (dataset.length === 0) {
      return [];
    }

    const randomForest = await this.getModel();
    const reshapedDataset = await tf.reshape(dataset, uniqueDataset
      ? [1, dataset.length*dataset[0].length*dataset[0][0].length]
      : [dataset.length, dataset[0].length*dataset[0][0].length]
    ).array() as number[][];

    // randomForest.predict(reshapedDataset) -> returns 0 or 1 for the label, we want the probability for average
    return randomForest.predictProbability(reshapedDataset, 1);
  }
}

/**
 * Recorder class to keep track of previously recorded mouse actions, mainly for move events.
 * It automatically computes all mouse features needed for {@link data!Data} format.
 * <br>
 * A simple usage would be to
 * <ol>
 *     <li>add an event listener for mousemove or touchmove event;</li>
 *     <li>call {@link addRecord} with the timestamp and (x,y) coordinates every time the event fires;</li>
 *     <li>call {@link isHuman} with a preloaded model to know if the trajectory is from human or not.</li>
 * </ol>
 * Which gives something like :
 * ```
 * recorder = new Recorder(window.screen.width, window.screen.height);
 * document.addEventListener("mousemove", event => {
 *   recorder.addRecord({
 *     time: event.timeStamp,
 *     x: event.clientX,
 *     y: event.clientY,
 *     type: "Move" // optional, not used in practice
 *   });
 *
 *   if (recorder.getRecords().length > 100) {
 *       const isHuman = recorder.isHuman(delbot.Models.rnn3);
 *       recorder.clearRecord();
 *       // ...
 *   }
 * });
 * ```
 * Be careful not to call `getPrediction` or `isHuman` too often, these may be heavy for smaller
 * configurations.
 */
export class Recorder {

  /**
   * Static field describing a success for {@link isHuman}.
   */
  public static readonly success: Symbol = Symbol(1);

  /**
   * Static field describing a fail for {@link isHuman}.
   */
  public static readonly fail: Symbol = Symbol(0);

  /**
   * Static field describing an error for {@link isHuman}.
   */
  public static readonly notEnoughProvidedData: Symbol = Symbol(0);

  /**
   * A 2-array with two numbers (a,b). When a new line with (x,y) is added
   * to the current record, we compute features with (x/a, y/b).
   */
  public normalizer: number[];

  /**
   * The list of calculated mouse features from the beginning of this record.
   * @private
   */
  private currentRecord: SingleRecord[];

  /**
   * The accumulated distance from the beginning of this record, used to compute average values of mouse features.
   * @private
   */
  private totalDistance: number;

  /**
   * The accumulated acceleration from the beginning of this record, used to compute average values of mouse features.
   * @private
   */
  private totalAccel: number;

  /**
   * The accumulated speed from the beginning of this record, used to compute average values of mouse features.
   * @private
   */
  private totalSpeed: number;

  /**
   * The length of the trajectory, it will always be equals to `currentRecord.length`
   * unless maxSize is defined and the current record is already reached.
   * @private
   */
  private totalLength: number;

  /**
   * The previous line used to compute the next mouse features, e.g. time diff = current time - previous time.
   * @private
   */
  private previousLine: SingleRecord;

  /**
   * The max size of the record, default to -1 (unlimited size), to prevent high memory usage.
   * If the maxSize is reached, new elements shift the entire array and are added to the end.
   * @private
   */
  private maxSize: number;

  /**
   * Create an empty recorder.
   * @param scaleX The x resolution of the screen to keep x value between 0 and 1. If unspecified, set to 1.
   * @param scaleY The y resolution of the screen to keep y value between 0 and 1. If unspecified, set to 1.
   */
  constructor(scaleX: number=1, scaleY: number=1) {
    this.clearRecord();
    if (scaleX === 0 || scaleY === 0) {
      throw Error("Cannot use a normalizer with value 0.")
    }
    this.normalizer = [scaleX, scaleY];
    this.maxSize = -1;
  }

  /**
   * Set the max size of record. When reached, it shifts all elements.
   * @param maxSize The new max size value.
   */
  setMaxSize(maxSize) {
    this.maxSize = maxSize;
  }

  /**
   * Clear the current record without creating a new object.
   * @return The same recorder instance for chain calls.
   */
  clearRecord(): Recorder {
    this.currentRecord = [];

    this.previousLine = null;
    this.totalDistance = 0;
    this.totalSpeed = 0;
    this.totalAccel = 0;
    this.totalLength = 0;

    return this;
  }

  /**
   * Get the mouse trajectory as a list of points and action
   * type with their calculated features.
   */
  getRecords(): SingleRecord[] {
    return this.currentRecord;
  }

  /**
   * Add a line to the current record and calculate all mouse features.
   * You don't have to normalize the x and y components, it is done automatically
   * with the normalizer from the constructor.
   * @param line The recorded action that must contain at least timestamp, x and y positions.
   * @return The same recorder instance for chain calls.
   */
  addRecord(line: SingleRecord): Recorder {
    // normalize input (assuming the user doesn't have to do it)
    line.x = line.x / this.normalizer[0];
    line.y = line.y / this.normalizer[1];

    if (this.previousLine == null) {
      this.previousLine = line;
      this.previousLine.speedX = 0;
      this.previousLine.speedY = 0;
      this.previousLine.speed = 0;
      this.previousLine.accel = 0;
    }

    line.timeDiff = line.time - this.previousLine.time;

    // Compute features for all action, even pressed or released clicks,
    // to not get high values if not pressed (but let the user choose)
    line.dx = line.x - this.previousLine.x;
    line.dy = line.y - this.previousLine.y;

    line.speedX = line.dx / line.timeDiff;
    line.speedY = line.dy / line.timeDiff;
    line.accelX = (line.speedX - this.previousLine.speedX) / line.timeDiff;
    line.accelY = (line.speedY - this.previousLine.speedY) / line.timeDiff;

    line.distance = Math.sqrt(line.dx * line.dx + line.dy * line.dy);
    this.totalDistance += !Number.isFinite(line.distance) || Number.isNaN(line.distance) ? 0 : line.distance;

    line.speed = line.distance / line.timeDiff;
    line.accel = (line.speed - this.previousLine.speed) / line.timeDiff;
    this.totalSpeed += !Number.isFinite(line.speed) || Number.isNaN(line.speed) ? 0 : line.speed;
    this.totalAccel += !Number.isFinite(line.accel) || Number.isNaN(line.accel) ? 0 : line.accel;

    line.speedXAgainstDistance = line.speed / line.dx;
    line.accelXAgainstDistance = line.accel / line.dx;
    line.speedYAgainstDistance = line.speed / line.dy;
    line.accelYAgainstDistance = line.accel / line.dy;

    line.averageSpeedAgainstDistance = (this.totalSpeed / (this.totalLength+1)) / this.totalDistance;
    line.averageAccelAgainstDistance = (this.totalAccel / (this.totalLength+1)) / this.totalDistance;

    line.angle = Math.atan2(line.dy, line.dx); // atan(dy/dx)
    line.jerk = line.accel - this.previousLine.accel / line.timeDiff;

    // Remove all NaN and +-Infinite of datas, replaced by 0
    for (const [key, value] of Object.entries(line)) {
      if (typeof value === "number" && (!Number.isFinite(value) || Number.isNaN(value))) {
        line[key] = 0;
      }
    }

    this.previousLine = line;

    if (this.currentRecord.length == this.maxSize) {
      this.currentRecord.shift();
    }
    this.currentRecord.push(line);
    this.totalLength++;

    return this;
  }

  /**
   * Load an entire trajectory as string or string array into this {@link Recorder}
   * instance. If the given input is a string, we split it with the separator `\n`.
   * The string array must be of the following format :
   *  ```
   *  ["resolution:1536,864",
   *  "9131.1,Pressed,717,361",
   *  "9134.8,Move,717,361",
   *  "9151.8,Move,717,361",
   *  ...
   *  "10402.3,Move,722,360",
   *  "10419.1,Move,718,358",
   *  "10425.8,Released,717,360]"
   * ```
   * As first line we have the screen resolution to normalize
   * X and Y and all other lines are `timestamp,actionType,x,y`.
   * We then add each line with {@link addRecord}.
   * <br>
   * Notice that the instance is cleared with {@link clearRecord} when calling
   * this method and the first line with resolution is optional, if absent,
   * we keep the previous normalizers (1 if unspecified in constructor).
   * @param recordString The string describing the trajectory.
   * @param xScale If specified and > 0, the x normalization will be this value.
   * @param yScale If specified and > 0, the y normalization will be this value.
   * @returns The same recorder instance for chain calls.
   */
  loadRecordsFromString(recordString: string | string[], xScale: number=-1, yScale: number=-1): Recorder {
    this.clearRecord();

    if (!Array.isArray(recordString)) {
      recordString = recordString.split("\n");
    }

    if (recordString[0].includes("resolution")) {
      // first line e.g. resolution:1536,864
      this.normalizer = recordString[0].split(":")[1].split(",").map(n => Number.parseInt(n));
      recordString = recordString.slice(1);
    }

    if (xScale > 0) this.normalizer[0] = xScale;
    if (yScale > 0) this.normalizer[1] = yScale;

    const rawRecords = recordString
      .map(line => line.split(","))
      .filter(line => line.length === 4); // ignore last empty line

    for (let line of rawRecords) {
      this.addRecord({
        time: Number.parseFloat(line[0]),
        type: line[1], // it is something like : (Pressed|Released|Move)(Touch)?
        x: Number.parseFloat(line[2]),
        y: Number.parseFloat(line[3]),
      })
    }
    return this;
  }

  /**
   * This function takes a model {@link Model} containing a TensorFlow.js layer model
   * and a {@link data!Data} instance and returns the list of probabilities for each batch
   * element to be a bot trajectory. The batch is obtained from {@link currentRecord}.
   * <br>
   * This function may be heavy for smaller configurations, be careful not to call it too often.
   * @param model A {@link Model} instance, with at least `predict()`.
   * @param uniqueDataset Default to false, if true then all datasets are merged into one unique,
   *                      so there is one prediction and the average is its unique value.
   *                      Might throw error if the classifier doesn't support it.
   * @return A list of probabilities, empty list if is not enough datas.
   * @see isHuman
   */
  async getPrediction(model: Model<any>, uniqueDataset: boolean = false): Promise<number[]> {
    return model.predict(this, uniqueDataset);
  }

  /**
   * This function takes a model {@link Model} containing a TensorFlow.js layer model
   * and a {@link data!Data} instance and returns whether the trajectory stored in
   * {@link currentRecord} is considered as human or bot for the model.
   * <br>
   * There may be more than one input batch for the prediction, leading to more than
   * one probability `p` for the sample to be a bot. It consequently takes the average
   * of those probabilities and returns `true` if the average is less than a given threshold.
   * <br>
   * If there is not enough input data, so we have a batch size of 0, it returns
   * `false` and a reason {@link Recorder.notEnoughProvidedData}.
   * <br>
   * This function may be heavy for smaller configurations, be careful not to call it too often.
   * @param model A {@link Model} instance, with at least `getData()` and `getModel()`.
   * @param threshold A number between 0 and 1, if the average probability to be a bot is less
   *                  than this value, we consider the trajectory as human.
   * @param uniqueDataset Default to false, if true then all datasets are merged into one unique,
   *                      so there one prediction and the average is its unique value.
   *                      Might throw error if the classifier doesn't support it.
   * @param consoleInfo Default to false, if true then it prints the prediction to console.
   * @return A promise of an instance of {@link Result} with two fields:
   * <ul>
   *     <li>result, the boolean `true` iff it is a human trajectory</li>
   *     <li>reason of the return, set to {@link Recorder.notEnoughProvidedData} if there is not enough data.</li>
   * </ul>
   * @see getPrediction
   */
  async isHuman(model: Model<any>, threshold: number = 0.2, uniqueDataset: boolean = false, consoleInfo: boolean = false): Promise<Result> {
    const predictions = await this.getPrediction(model, uniqueDataset);
    if (predictions.length === 0) {
      return { result: false, reason: Recorder.notEnoughProvidedData };
    }
    const average = tf.mean(predictions).arraySync();

    if (consoleInfo) console.log("pred=", predictions, "average=", average);

    if (average <= threshold) {
      return { result: true, reason: Recorder.success };
    } else {
      return { result: false, reason: Recorder.fail };
    }
  }
}
