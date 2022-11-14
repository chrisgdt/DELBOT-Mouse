import * as tf from '@tensorflow/tfjs';
import {Recorder, SingleRecord} from "./recording";


export interface DataProperties {
  numClasses?: 1 | 2;
}

export interface Dataset {
  datasetData: number[][][];
  datasetLabels: number[][];
}

/**
 * An abstract class that extracts some mouse features from {@link recording!Recorder}
 * and format them with {@link loadDataSet} to create an input of a tf.js model.
 * This input is an array, but you can call {@link tf.tensor3d} and {@link tf.reshape}
 * to get the tensor. It is always a 3D tensor, each element of the dataset being a matrix
 * {@link xSize} x {@link ySize}.
 * <br>
 * You can also specify an integer `userIndex` to work with labels for training.
 * While dataset is a 3D array with first dimension representing the size of the dataset,
 * labels are represented by a 2D array with the same first dimension and the second one
 * is the label of the corresponding sample.
 * <br>
 * The field {@link numClasses} can be 1 or 2. You can ignore it and let the default value to 2.
 * If you want more details: as 1, if a model is constructed with `delbot-training`, it
 * will have a single neuron as output and `binaryCrossentropy` loss function, each label
 * during the training will be a singleton [0] or [1] for respectively human and bot, the
 * model then outputs the probability [p] for the sample to be a bot. If there are two classes,
 * the loss function is `categoricalCrossentropy` for two neurons, the labels are [1,0] or
 * [0,1] for respectively human and bot samples and outputs from the model are [1-p,p] where p
 * is the probability to be a bot.
 * <br>
 * Extend this class and implement the method {@link loadDataSet} to create your own logic and
 * data format. Your implementation has to look something like 1. empty data and label
 * arrays, 2. iterate through {@link recording!Recorder.getRecords} and 3. add labels userIndex is positive.
 * You can also check if {@link recording!SingleRecord.type} is null or includes "Move" to filter only
 * move actions.
 */
export abstract class Data {
  readonly numClasses: number;
  protected xSize: number;
  protected ySize: number;
  public mayNormalizeData;

  protected constructor(args: DataProperties={}) {
    // if 2 classes: ['human', 'bot'], if 1 class ['bot'] (proba p to be a bot according to the model)
    this.numClasses = args.numClasses == null ? 2 : args.numClasses;
    this.mayNormalizeData = true;
  }

  /**
   * Gets the current array of labels from a dataset and append a new
   * label according to the number of classes {@link numClasses}.
   * If there is one class, [0] means human and [1] bot while with
   * two classes, [1,0] means human and [0,1] bot.
   * @param datasetLabels The current dataset label array, we add an element to it.
   * @param userIndex The index of the class, 0 for human and 1 for bot.
   * @protected
   */
  protected addLabel(datasetLabels: number[][], userIndex: number) {
    if (userIndex >= 0) {
      if (this.numClasses === 1) {
        datasetLabels.push([userIndex]);
      } else {
        let label = new Array(this.numClasses).fill(0);
        label[userIndex] = 1;
        datasetLabels.push(label)
      }
    }
  }

  /**
   * This method gets a recorder object and loads it as a {@link Dataset} instance
   * with the right format. The return value might contain empty arrays if the
   * recorder as too few elements. The userIndex is an integer that says what is the
   * class index for the label, in our case of binary classifier human-bot, 0 means
   * human and 1 means bot.
   * @param recorder The recorder containing loaded records and features.
   * @param userIndex The index of the class from record, if unspecified or negative,
   *                  the label array of the return value {@link Dataset} is empty.
   */
  abstract loadDataSet(recorder: Recorder, userIndex?: number): Dataset;

  public getXSize(): number {
    return this.xSize;
  }

  public getYSize(): number {
    return this.ySize;
  }

  public mayNormalize(): boolean {
    return this.mayNormalizeData;
  }
}


export interface DataMovementMatrixProperties extends DataProperties{
  xMinMov?: number;
  xMaxMov?: number;
  yMinMov?: number;
  yMaxMov?: number;
  neighbourRange?: number;
  steps?: number[];
}

/**
 * A Data class that represents a movement matrix. For each step N of {@link steps}
 * we create a 0-matrix {@link xMaxMov}-{@link xMinMov} x {@link yMaxMov}-{@link yMinMov}
 * and add a small number at position (dx,dy) for at most N trajectories, where
 * dx = x(n) - x(n-1) is the pixel offset in x coordinate between two recorded movements
 * in the trajectory. We repeat that until there are no sample left, then for the next N.
 * <br>
 * If a value should be outside the matrix, we put it on the edge. The default size
 * is 50x50 (range -25 to 25) and default steps are [25, 50, 100, 150, 200, 250].
 * @see Data
 */
export class DataMovementMatrix extends Data {
  protected readonly xMinMov: number;
  protected readonly xMaxMov: number;
  protected readonly yMinMov: number;
  protected readonly yMaxMov: number;
  protected readonly neighbourRange: number;
  protected readonly steps: number[];

  constructor(args: DataMovementMatrixProperties={}) {
    super(args);

    this.xMinMov = args.xMinMov == null ? -25 : args.xMinMov;
    this.xMaxMov = args.xMaxMov == null ? 25 : args.xMaxMov;
    this.yMinMov = args.yMinMov == null ? -25 : args.yMinMov;
    this.yMaxMov = args.yMaxMov == null ? 25 : args.yMaxMov;

    if (this.xMinMov >= this.xMaxMov) {
      throw Error(`Cannot create DataMovementMatrix instance because xMin >= xMax (${this.xMinMov} >= ${this.xMaxMov}.`);
    }
    if (this.yMinMov >= this.yMaxMov) {
      throw Error(`Cannot create DataMovementMatrix instance because yMin >= yMax (${this.yMinMov} >= ${this.yMaxMov}.`);
    }

    this.neighbourRange = args.neighbourRange == null ? 3 : args.neighbourRange;
    this.steps = args.steps == null ? [25, 50, 100, 150, 200, 250] : args.steps;

    this.xSize = this.xMaxMov - this.xMinMov;
    this.ySize = this.yMaxMov - this.yMinMov;

    this.mayNormalizeData = false;
  }

  /**
   * Create a new empty 0-matrix of size xSize times ySize.
   * @private
   */
  private newEmptyData(): number[][] {
    return tf.zeros([this.xSize, this.ySize]).arraySync() as number[][];
  }

  /**
   * Force the value to stand in `[0, max-min[` for index of matrix, so
   * min is the first index 0 and max the last index. If the value is
   * greater than max or less than min, we consider the bound.
   * @param min An integer, lower bound of initial domain.
   * @param max An integer, upper bound of initial domain.
   * @param value An integer, the value to constraint.
   * @return The value in the new integer domain starting from 0.
   * @private
   */
  private getInRangeForArray(min: number, max: number, value: number): number {
    if (value < min) {
      return 0;
    } else if (value >= max) {
      return max-min-1;
    } else {
      return value-min;
    }
  }

  /**
   * Get integer neighbours of the given value within a distance of range and
   * return a list of unique elements of this neighbourhood (in case the value
   * is on an edge, there will be less than range squared values).
   * @private
   */
  private getNeighbour(min: number, max: number, value: number): number[] {
    value = Math.round(value);
    const neighbour = [];
    for (let i = -this.neighbourRange; i <= this.neighbourRange; i++) {
      const element = this.getInRangeForArray(min, max, value+i);
      if (!neighbour.includes(element)) {
        neighbour.push(element);
      }
    }
    return neighbour;
  }

  loadDataSet(recorder: Recorder, userIndex: number=-1): Dataset {
    const datasetData = [];
    const datasetLabels = [];

    let data = this.newEmptyData();
    for (let step of this.steps) {
      const inc = 1/step;
      let stepModulo = 0;
      for (let line of recorder.getRecords()) {
        if (line.type == null || line.type.includes("Move")) {
          for (let dx of this.getNeighbour(this.xMinMov, this.xMaxMov, line.dx)) {
            for (let dy of this.getNeighbour(this.yMinMov, this.yMaxMov, line.dy)) {
              data[dx][dy] += inc;
            }
          }
          if (++stepModulo >= step) {
            datasetData.push(data);
            this.addLabel(datasetLabels, userIndex);
            data = this.newEmptyData();
            stepModulo = 0;
          }
        }
      }
    }
    return {datasetData, datasetLabels};
  }
}


export interface DataFeaturesProperties extends DataProperties {
  shouldCompleteXSize?: boolean;
  xSize?: number;
}

/**
 * A Data class that represents chunks of mouse features selected from
 * the {@link recording!Recorder} objet. The field {@link xSize} is the chunk size,
 * default to 24, and {@link ySize} the number of extracted features.
 * <br>
 * The dataset there is a list of chunks of mouse features.
 * <br>
 * It can be the case that a chunk is not fully complete after reading the
 * record, if the field {@link shouldCompleteXSize} is set to true, we complete
 * it with trailing zeros.
 * <br>
 * This class takes 8 features :
 * <ol>
 *   <li>{@link recording!SingleRecord.dx}</li>
 *   <li>{@link recording!SingleRecord.dy}</li>
 *   <li>{@link recording!SingleRecord.speedX}</li>
 *   <li>{@link recording!SingleRecord.speedY}</li>
 *   <li>{@link recording!SingleRecord.speed}</li>
 *   <li>{@link recording!SingleRecord.accel}</li>
 *   <li>{@link recording!SingleRecord.distance}</li>
 *   <li>{@link recording!SingleRecord.timeDiff}</li>
 * </ol>
 * @see Data
 */
export class DataFeatures extends Data {
  protected readonly shouldCompleteXSize: boolean;

  constructor(args: DataFeaturesProperties={}) {
    super(args);
    this.xSize = args.xSize == null ? 24 : args.xSize; // chunk size
    this.ySize = 8; // nbr features
    this.shouldCompleteXSize = args.shouldCompleteXSize == null ? false : args.shouldCompleteXSize;
  }

  /**
   * Return a new chunk element with arbitrary chosen mouse features from
   * the record.
   * @param line The line of the record with calculated mouse features.
   * @protected
   */
  protected getChunkElement(line: SingleRecord): number[] {
    return [
      line.dx, line.dy, // 0, 1
      line.speedX, line.speedY, // 2, 3
      line.speed, line.accel, // 4, 5
      line.distance, line.timeDiff, // 6, 7
    ];
  }

  public loadDataSet(recorder: Recorder, userIndex: number=-1): Dataset {
    const datasetData = [];
    const datasetLabels = [];

    let chunk = [];
    for (let line of recorder.getRecords()) {
      if (line.type == null || line.type.includes("Move")) {
        chunk.push(this.getChunkElement(line));
        if (chunk[0].length !== this.ySize) {
          throw Error(`The number of features (${this.ySize}) is not matching the actual number (${chunk[0].length}).`)
        }
        if (chunk.length === this.xSize) {
          datasetData.push(chunk);
          this.addLabel(datasetLabels, userIndex);
          chunk = [];
        }
      }
    }

    if (this.shouldCompleteXSize && chunk.length > 0) {
      while (chunk.length < this.xSize) {
        chunk.push(new Array(this.ySize).fill(0));
      }
      datasetData.push(chunk);
      this.addLabel(datasetLabels, userIndex);
    }

    return {datasetData, datasetLabels};
  }
}

/**
 * This class extends {@link DataFeatures} and takes 10 features :
 * <ol>
 *   <li>{@link recording!SingleRecord.dx}</li>
 *   <li>{@link recording!SingleRecord.dy}</li>
 *   <li>{@link recording!SingleRecord.speedX}</li>
 *   <li>{@link recording!SingleRecord.speedY}</li>
 *   <li>{@link recording!SingleRecord.speed}</li>
 *   <li>{@link recording!SingleRecord.accel}</li>
 *   <li>{@link recording!SingleRecord.distance}</li>
 *   <li>{@link recording!SingleRecord.timeDiff}</li>
 *   <li>{@link recording!SingleRecord.jerk}</li>
 *   <li>{@link recording!SingleRecord.angle}</li>
 * </ol>
 * @see DataFeatures
 */
export class DataFeatures2 extends DataFeatures {
  constructor(args: DataFeaturesProperties={}) {
    super(args);
    this.ySize = 10; // nbr features
  }

  protected getChunkElement(line: SingleRecord): number[] {
    return [
      line.dx, line.dy, // 0, 1
      line.speedX, line.speedY, // 2, 3
      line.speed, line.accel, // 4, 5
      line.distance, line.timeDiff, // 6, 7
      line.angle, line.jerk, // 8, 9
    ]
  }
}

/**
 * This class extends {@link DataFeatures} and takes 13 features :
 * <ol>
 *   <li>{@link recording!SingleRecord.dx}</li>
 *   <li>{@link recording!SingleRecord.dy}</li>
 *   <li>{@link recording!SingleRecord.speedX}</li>
 *   <li>{@link recording!SingleRecord.speedY}</li>
 *   <li>{@link recording!SingleRecord.accelX}</li>
 *   <li>{@link recording!SingleRecord.accelY}</li>
 *   <li>{@link recording!SingleRecord.speedXAgainstDistance}</li>
 *   <li>{@link recording!SingleRecord.speedYAgainstDistance}</li>
 *   <li>{@link recording!SingleRecord.accelXAgainstDistance}</li>
 *   <li>{@link recording!SingleRecord.accelYAgainstDistance}</li>
 *   <li>{@link recording!SingleRecord.distance}</li>
 *   <li>{@link recording!SingleRecord.averageSpeedAgainstDistance}</li>
 *   <li>{@link recording!SingleRecord.averageAccelAgainstDistance}</li>
 * </ol>
 * @see DataFeatures
 */
export class DataMoreFeatures extends DataFeatures {
  constructor(args: DataFeaturesProperties={}) {
    super(args);
    this.ySize = 13; // nbr features
  }

  protected getChunkElement(line: SingleRecord): number[] {
    return [
      line.dx, line.dy, // 0, 1
      line.speedX, line.speedY, // 2, 3
      line.accelX, line.accelY, // 4, 5
      line.speedXAgainstDistance, line.speedYAgainstDistance, // 6, 7
      line.accelXAgainstDistance, line.accelYAgainstDistance, // 8, 9
      line.distance, // 10
      line.averageSpeedAgainstDistance, // 11
      line.averageAccelAgainstDistance // 12
    ]
  }
}

/**
 * This class extends {@link DataFeatures} and takes 2 features :
 * <ol>
 *   <li>{@link recording!SingleRecord.x}</li>
 *   <li>{@link recording!SingleRecord.y}</li>
 * </ol>
 * @see DataFeatures
 */
export class DataSimplePosition extends DataFeatures {

  constructor(args: DataFeaturesProperties={}) {
    super(args);
    this.ySize = 2; // nbr features
  }

  protected getChunkElement(line: SingleRecord): number[] {
    return [line.x, line.y];
  }
}

/**
 * This class extends {@link DataFeatures} and takes 2 features :
 * <ol>
 *   <li>{@link recording!SingleRecord.dx}</li>
 *   <li>{@link recording!SingleRecord.dy}</li>
 * </ol>
 * @see DataFeatures
 */
export class DataOffsetPosition extends DataFeatures {
  constructor(args: DataFeaturesProperties={}) {
    super(args);
    this.ySize = 2; // nbr features
  }

  protected getChunkElement(line: SingleRecord): number[] {
    return [line.dx, line.dy];
  }
}
