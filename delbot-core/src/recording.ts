import * as tf from '@tensorflow/tfjs';
import {Data} from "./data";

/**
 * Represent a generic instance of a single recorded mouse action. It must contain at least
 * three datas : time, x and y. Other features are calculated from these three datas
 * in the Recorder class.
 */
export interface SingleRecord {
  /**
   * The time stamp when this movement was recorded, often given by the field event.timeStamp.
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

  timeDiff?: number;
  dx?: number;
  dy?: number;
  speedX?: number;
  speedY?: number;
  accelX?: number;
  accelY?: number;
  speed?: number;
  accel?: number;
  distance?: number;
  speedXAgainstDistance?: number;
  speedYAgainstDistance?: number;
  accelXAgainstDistance?: number;
  accelYAgainstDistance?: number;
  averageSpeedAgainstDistance?: number;
  averageAccelAgainstDistance?: number;
  jerk?: number;
  angle?: number;
}

export interface Result {
  result: boolean;
  reason: Symbol;
}

/**
 * A class that represents a model with its associated data used during the
 * training. Concretely, you can give to the constructor a way to load a
 * Tensorflow.js model from {@link tf.loadLayersModel}, like a URL to the
 * model.json file. Then the data object is only used to format the raw datas
 * from {@link Recorder} and reshape the input tensors.
 * <br>
 * The model never loads if you never call {@link getModel}.
 */
export class Model {
  private model: null | tf.LayersModel;

  private readonly loadingPath: string;
  private readonly data: Data;

  /**
   * @param loadingPath The loadingPath (url, localstorage, indexeddb, ...) to load with {@link tf.loadLayersModel}.
   * @param data The {@link data!Data} instance related to this model.
   */
  constructor(loadingPath: string, data: Data) {
    this.model = null;
    this.loadingPath = loadingPath;
    this.data = data;
  }

  /**
   * Loads the model if it has never been done before, then returns it.
   * @return A promise of the loaded layer model.
   */
  async getModel(): Promise<tf.LayersModel> {
    if (this.model === null) {
      this.model = await tf.loadLayersModel(this.loadingPath);
    }
    return this.model;
  }

  /**
   * Get the model instance after at least one call of {@link getModel} that loads it once.
   */
  getLoadedModel(): tf.LayersModel {
    if (this.model == null) {
      throw new Error("You have to load the model with getModel() first before trying to get the model synchronously.")
    }
    return this.model
  }

  getData(): Data {
    return this.data;
  }
}

/**
 * Recorder class to keep track of previously recorded mouse actions, mainly for move events.
 * It automatically computes all mouse features needed for {@link data!Data} format.
 * <br>
 * A simple usage would be to
 * <ol>
 *     <li>add an event listener for mousemove or touchmove event;</li>
 *     <li>call {@link addRecord} with the timestamp and (x,y) coordinates everytime the event fires;</li>
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
  public static readonly success: Symbol = Symbol(1);
  public static readonly fail: Symbol = Symbol(0);
  public static readonly notEnoughProvidedDatas: Symbol = Symbol(0);

  public normalizer: number[];
  public currentRecord: SingleRecord[];

  private totalDistance: number;
  private totalAccel: number;
  private totalSpeed: number;
  private previousLine: SingleRecord;

  /**
   * Create an empty recorder.
   * @param scaleX The x resolution of the screen to keep x value between 0 and 1. If unspecified, set to 1.
   * @param scaleY The y resolution of the screen to keep y value between 0 and 1. If unspecified, set to 1.
   */
  constructor(scaleX: number=1, scaleY: number=1) {
    this.clearRecord();
    this.normalizer = [scaleX, scaleY];
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

    // TODO: do we need to keep the type of action to filter move only ?
    //if (line.type == null || line.type.includes("Move")) {
    line.dx = line.x - this.previousLine.x;
    line.dy = line.y - this.previousLine.y;

    line.speedX = line.dx / line.timeDiff;
    line.speedY = line.dy / line.timeDiff;
    line.accelX = (line.speedX - this.previousLine.speedX) / line.timeDiff;
    line.accelY = (line.speedY - this.previousLine.speedY) / line.timeDiff;

    line.distance = Math.sqrt(line.dx * line.dx + line.dy * line.dy);
    this.totalDistance += line.distance;

    line.speed = line.distance / line.timeDiff;
    line.accel = (line.speed - this.previousLine.speed) / line.timeDiff;
    this.totalSpeed += line.speed;
    this.totalAccel += line.accel;

    line.speedXAgainstDistance = line.speed / line.dx;
    line.accelXAgainstDistance = line.accel / line.dx;
    line.speedYAgainstDistance = line.speed / line.dy;
    line.accelYAgainstDistance = line.accel / line.dy;

    line.averageSpeedAgainstDistance = (this.totalSpeed / (this.getRecords().length+1)) / this.totalDistance;
    line.averageAccelAgainstDistance = (this.totalAccel / (this.getRecords().length+1)) / this.totalDistance;

    line.angle = Math.atan2(line.dy, line.dx); // atan(dy/dx)
    line.jerk = line.accel - this.previousLine.accel / line.timeDiff;

    // Remove all NaN and +-Infinite of datas, replaced by 0
    for (const [key, value] of Object.entries(line)) {
      if (typeof value === "number" && (!Number.isFinite(value) || Number.isNaN(value))) {
        // @ts-ignore - we checked that is it number
        line[key] = 0;
      }
    }
    //}
    this.previousLine = line;

    this.currentRecord.push(line);
    return this;
  }

  /**
   * Load an entire trajectory as string or string array into this {@link Recorder}
   * instance. If the given input is a string, we split it with the separator `\n`.
   * The string array must be of the form
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
   * were, as first line we have the screen resolution to normalize
   * X and Y and the following lines are `timestamp,actionType,x,y`.
   * We then add each line with {@link addRecord}.
   * <br>
   * Notice that the instance is cleared when calling this method and
   * the first line with resolution is optional, if absent, we keep the
   * previous normalizers (1 if unspecified in constructor).
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
   * This function takes a model {@link Model} containing a Tensorflow.js layer model
   * and a {@link data!Data} instance and returns the list of probabilities for each batch
   * element to be a bot trajectory. The batch is obtained from {@link currentRecord}.
   * <br>
   * This function may be heavy for smaller configurations, be careful not to call it too often.
   * @param model A {@link Model} instance, with at least `getData()` and `getModel()`.
   * @return A list of probabilities, empty list if is not enough datas.
   * @see isHuman
   */
  async getPrediction(model: Model): Promise<number[]> {
    const dataset = model.getData().loadDataSet(this).datasetData;

    if (dataset.length === 0) {
      return [];
    }

    const tfjsModel = await model.getModel();

    // We could reshape to have one batch and more time-steps if the model accepts a variable number of
    // time steps, but we may lose some accuracy since the model trained on a fix time-step.
    const datasetTensor = tf.tensor3d(dataset).reshape(tfjsModel.inputs[0].shape.length === 4
      ? [dataset.length, model.getData().getXSize(), model.getData().getYSize(), 1]
      : [dataset.length, model.getData().getXSize(), model.getData().getYSize()]);
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

  /**
   * This function takes a model {@link Model} containing a Tensorflow.js layer model
   * and a {@link data!Data} instance and returns whether the trajectory stored in
   * {@link currentRecord} is considered as human or bot for the model.
   * <br>
   * There may be more than one input batch for the prediction, leading to more than
   * one probability `p` for the sample to be a bot. It consequently takes the average
   * of those probabilities and returns `true` if the average is less than a given threshold.
   * <br>
   * If there is not enough input datas, so we have a batch size of 0, it returns
   * `false` and a reason {@link Recorder.notEnoughProvidedDatas}.
   * <br>
   * This function may be heavy for smaller configurations, be careful not to call it too often.
   * @param model A {@link Model} instance, with at least `getData()` and `getModel()`.
   * @param threshold A number between 0 and 1, if the average probability to be a bot is less
   *                  than this value, we consider the trajectory as human.
   * @return A promise of an instance of {@link Result} with two fields:
   * <ul>
   *     <li>result, the boolean `true` iff it is a human trajectory</li>
   *     <li>reason of the return, set to {@link Recorder.notEnoughProvidedDatas} if there is not enough datas.</li>
   * </ul>
   * @see getPrediction
   */
  async isHuman(model: Model, threshold: number = 0.2): Promise<Result> {
    const predictions = await this.getPrediction(model);
    if (predictions.length === 0) {
      return { result: false, reason: Recorder.notEnoughProvidedDatas };
    }
    const average = tf.mean(predictions).arraySync();

    // TODO: debug
    console.log("pred=", predictions, "average=", average);

    if (average <= threshold) {
      return { result: true, reason: Recorder.success };
    } else {
      return { result: false, reason: Recorder.fail };
    }
  }
}
