import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import {DataTraining} from './dataTraining';
import * as delbot from '@chrisgdt/delbot-mouse'

/**
 * Apply an array of layers to a symbolic input tensor and output a symbolic tensor.
 * @param layers
 * @param inputs
 */
export function applyLayers(layers: tf.layers.Layer[], inputs: tf.SymbolicTensor | tf.SymbolicTensor[]): tf.SymbolicTensor | tf.SymbolicTensor[] {
  let outputs = inputs;
  for (let layer of layers) {
    outputs = layer.apply(outputs) as tf.SymbolicTensor | tf.SymbolicTensor[];
  }
  return outputs;
}

/**
 * Returns a random number between 0 and 1 (exclusive) from a normal distribution
 * from 0 and 1 (inclusive) with the Box-Muller transform.
 */
export function randomNormalBoxMuller(): number {
  let u = Math.random(), v = Math.random();
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  let num = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  num = num / 10.0 + 0.5; // Translate to 0 -> 1
  if (num > 1 || num < 0) return randomNormalBoxMuller(); // resample between 0 and 1
  return num;
}

export interface GANProperties {
  /**
   * Data training instance to know how to format input data and give
   * them to the models. Once loaded, it should contain human data, real bot
   * data are not used here.
   */
  dataTraining: DataTraining;

  /**
   * The number of epochs to train our models. Default to 500.
   */
  epoch?: number;

  /**
   * The size of the batch for each epoch. Default to 64.
   */
  batchSize?: number;

  /**
   * The seed size for the generator, the number of random numbers
   * normally distributed between 0 and 1 for generator input.
   */
  generatorSeedSize?: number;

  /**
   * Default to false, if true and used by a web browser, tfjs-vis shows information
   * of models and during the training.
   */
  useTfjsVis?: boolean;

  /**
   * Default to false, if true then some information are printed to the console.
   */
  consoleInfo?: boolean;

  /**
   * Default to -1. If set to a positive number `n`, Both models are downloaded
   * every `downloadModels` epochs and after the training.
   */
  downloadModels?: number;

  /**
   * Default to false. If true, then after each epoch we download a sample of generator
   * and the output of the step (loss and accuracy) as json file.
   */
  downloadSample?: boolean;

  /**
   *  A {@link tf.Optimizer} instance for the training of both discriminator
   *  and generator. If unspecified, it will default to `tf.train.adam(1e-4, .5, .999, 1e-8)`
   */
  optimizer?: tf.Optimizer;
}

/**
 * Class to run Generic Adversarial Networks logic, with a {@link discriminator} and a {@link generator}.
 * This is a basic implementation with RRN layers to generate synthetic mouse trajectories.
 * <br>
 * The constructor takes an object with parameters :
 * <ul>
 *   <li>dataTraining : a {@link dataTraining!DataTraining} instance to load data.</li>
 *   <li>epoch : integer number of times to iterate over a random batch. Default to 500</li>
 *   <li>batchSize : number of samples per gradient update. If unspecified, it will default to 64.</li>
 *   <li>generatorSeedSize : the input size of the generator for random numbers.</li>
 *   <li>testingRatio : a number between 0 and 1, default to 0.9, the %/100 of the testing set used for validation.</li>
 *   <li>downloadModels : if specified and positive, it will download both models every 'downloadModels' epoch
 *                        and after the training.</li>
 *   <li>downloadSample : default to false, if true, after each epoch, we download a sample of generator
 *                        and the output of the step (loss and accuracy) as json file.</li>
 *   <li>useTfjsVis : default to false, if true, the code shows additional information through tfjs-vis.</li>
 *   <li>consoleInfo : default to false, if true, the code shows addition information through the console log.</li>
 *   <li>optimizer : a {@link tf.Optimizer} instance for the training of both discriminator
 *                   and generator. If unspecified, it will default to `tf.train.adam(1e-4, .5, .999, 1e-8)`</li>
 * </ul>
 * <br>
 * Notice that the generator output size (number of points in the generated trajectory) is
 * the same as the xSize of the Data instance.
 * <br>
 * Extend this class to create your own discriminator or generator. You can override methods
 * {@link getNewGeneratorLayers} and {@link getNewDiscriminatorLayers} that just return a list
 * of layers to get new models. You can also modify the way to train models by overriding
 * {@link trainDiscriminator} and {@link trainGenerator}.
 * @see https://en.wikipedia.org/wiki/Generative_adversarial_network
 */
export class GAN {
  protected readonly generator: tf.LayersModel;
  protected readonly discriminator: tf.LayersModel;
  protected readonly data: DataTraining;

  protected readonly adversarialModel: tf.LayersModel;

  protected readonly epoch: number;
  protected readonly batchSize: number;
  protected readonly generatorOutputSize: number;
  protected readonly generatorNodes: number;
  protected readonly useTfjsVis: boolean;
  protected readonly consoleInfo: boolean;
  protected readonly downloadModels: number;
  protected readonly downloadSample: boolean;

  /**
   * The 2d tensor that represents an entire batch (of size batchSize) of human labels.
   * For a single batch, it is [0] if there is one binary class (data.data.numClasses),
   * otherwise [1,0].
   */
  public readonly humanLabel: tf.Tensor;

  /**
   * The 2d tensor that represents an entire batch (of size batchSize) of human labels.
   * For a single batch, it is [1] if there is one binary class (data.data.numClasses),
   * otherwise [0,1].
   */
  public readonly botLabel: tf.Tensor;

  constructor(args: GANProperties | DataTraining) {
    if (args instanceof DataTraining) args = {dataTraining: args};

    this.data = args.dataTraining;
    this.epoch = args.epoch == null ? 500 : args.epoch;
    this.batchSize = args.batchSize == null ? 64 : args.batchSize;
    this.useTfjsVis = args.useTfjsVis == null ? false : args.useTfjsVis;
    this.consoleInfo = args.consoleInfo == null ? false : args.consoleInfo;
    this.downloadModels = args.downloadModels == null ? -1 : args.downloadModels;
    this.downloadSample = args.downloadSample == null ? false : args.downloadSample;
    this.generatorOutputSize = this.data.data.getXSize();


    if (this.data.data.numClasses === 1) {
      // label -> 0 means human, 1 means bot or invalid
      this.humanLabel = tf.zeros([this.batchSize, 1]);
      this.botLabel = tf.ones([this.batchSize, 1]);
    } else {
      const human = [];
      const bot = [];
      for (let i=0; i<this.batchSize; i++) {
        human.push([1,0]);
        bot.push([0,1])
      }
      this.humanLabel = tf.tensor2d(human);
      this.botLabel = tf.tensor2d(bot);
    }

    const generatorSeedSize = args.generatorSeedSize == null ? 100 : args.generatorSeedSize;

    // keep approx 100 random value
    this.generatorNodes = Math.max(1, Math.round(generatorSeedSize / this.generatorOutputSize));

    const optimizer = args.optimizer == null ? tf.train.adam(1e-4, .5, .999, 1e-8) : args.optimizer;
    const {discriminator, discriminatorInputs, discriminatorOutputs} = this.initDiscriminator();
    const {generator, generatorInputs, generatorOutputs} = this.initGenerator();

    this.discriminator = discriminator;
    this.generator = generator;

    this.discriminator.compile({
      optimizer: optimizer,
      loss: this.data.data.numClasses === 1 ? 'binaryCrossentropy' : "categoricalCrossentropy",
      metrics: ['accuracy'],
    });

    // Create a frozen discriminator with a disabled training (need not compiled model) for the
    // adversarialModel (generator -> discriminator), so only generator's layers are trained and
    // the first discriminator can be trained alone, more details : https://github.com/keras-team/keras/issues/8585
    // Alternatively, we could have created a custom layer with custom loss function to simulate the discriminator.
    // However, with custom layers we might get uncommon errors according to the Tensorflow version.
    const discriminatorFrozen = tf.model({
      inputs:discriminatorInputs,
      outputs:discriminatorOutputs,
      name:"frozen_discriminator"
    });
    discriminatorFrozen.trainable = false;

    const outputAdversarial = discriminatorFrozen.apply(generatorOutputs) as tf.SymbolicTensor | tf.SymbolicTensor[];

    this.adversarialModel = tf.model({
      inputs:generatorInputs,
      outputs:outputAdversarial,
      name:"adversarial_model"
    });

    this.adversarialModel.compile({
      optimizer: optimizer,
      loss: this.data.data.numClasses === 1 ? 'binaryCrossentropy' : "categoricalCrossentropy",
      metrics: ['accuracy'],
    });

    if (this.consoleInfo) {
      this.generator.summary();
      console.log("\n----\n");
      this.discriminator.summary();
      console.log("\n----\n");
      this.adversarialModel.summary();
    }
    if (this.useTfjsVis) {
      tfvis.show.modelSummary({name: 'Model Generator', tab: 'Models'}, this.generator);
      tfvis.show.modelSummary({name: 'Model Discriminator', tab: 'Models'}, this.discriminator);
      tfvis.show.modelSummary({name: 'Adversarial Model', tab: 'Models'}, this.adversarialModel);
    }
  }

  getDiscriminator(): tf.LayersModel {
    return this.discriminator;
  }

  getGenerator(): tf.LayersModel {
    return this.generator;
  }

  getData(): DataTraining {
    return this.data;
  }

  /**
   * Asynchronously sample a batch of trajectories from the generator.
   * @param batch The number of trajectories to return.
   */
  async sampleTrajectory(batch: number=1): Promise<number[][]> {
    return await this.getTrajectory(batch).array() as number[][]; // batch size of 1, so get the single batch sample
  }

  /**
   * Synchronously sample a batch of trajectories from the generator.
   * @param batch The number of trajectories to return.
   */
  sampleTrajectorySync(batch: number=1): number[][] {
    return this.getTrajectory(batch).arraySync() as number[][];
  }

  /**
   * Return a list of all discriminator layers, no need to specify the input shape. The last layer must be
   *   ```
   *   tf.layers.dense({
   *     units: this.data.data.numClasses,
   *     activation: this.data.data.numClasses === 1 ? 'sigmoid' : 'softmax'
   *   })
   *   ```
   * or something similar to keep a classifier structure.
   * <br>
   * Override this method to create a new generator structure.
   */
  getNewDiscriminatorLayers(): tf.layers.Layer[] {
    // LeakyRelu and dropout are used to prevent over fitting
    return [
      tf.layers.lstm({
        units: 64,
        returnSequences: true,
        recurrentDropout : .3
      }),
      tf.layers.leakyReLU({alpha: .2}),
      tf.layers.batchNormalization(),
      tf.layers.dropout({rate:0.5}),
      tf.layers.lstm({
        units: 32,
        returnSequences: false,
        recurrentDropout : .2,
      }),
      tf.layers.leakyReLU({alpha: .1}),
      tf.layers.dropout({rate:0.3}),
      tf.layers.dense({
        units: 16,
        activation: 'relu',
        kernelRegularizer: tf.regularizers.l1l2({l1: 1e-5, l2: 1e-4}),
        biasRegularizer: tf.regularizers.l2({l2: 1e-4}),
      }),
      tf.layers.dense({
        units: this.data.data.numClasses,
        activation: this.data.data.numClasses === 1 ? 'sigmoid' : 'softmax'
      })
    ];
  }

  /**
   * Return a list of all generator layers, no need to specify the input shape. It must contain
   * pairs of 2 units to output the trajectory (or 3 if you want to include the time, but
   * adapt discriminator if so).
   * <br>
   * Override this method to create a new generator structure.
   */
  getNewGeneratorLayers(): tf.layers.Layer[] {
    return [
      tf.layers.lstm({
        units: 128,
        returnSequences: true,
        activation: 'relu',
        recurrentDropout : .2
      }),
      tf.layers.leakyReLU({alpha: .1}),
      tf.layers.batchNormalization(),
      tf.layers.lstm({
        units: 64,
        returnSequences: true,
        activation: 'relu',
        recurrentDropout : .1
      }),
      tf.layers.timeDistributed({
        // without activation : output can become huge, we also want to avoid tanh for negative values, so sigmoid
        layer: tf.layers.dense({units: 2, activation: 'sigmoid'}),
      })
    ]
  }

  /**
   * Train the generator and discriminator and save samples or model
   * according to {@link downloadModels} and {@link downloadSample} values.
   * @param ganTest A number to enumerate downloaded models so your n+1-th train does not override your n-th one.
   */
  async train(ganTest: number = 35) {
    if (!this.data.isLoaded()) {
      await this.data.load(this.consoleInfo);
    }
    if (this.consoleInfo) console.log("TRAINING GAN");

    const generatorHistory = [];
    const discriminatorHistory = [];

    for (let i=0; i < this.epoch; i++) {

      generatorHistory.push(await this.trainGenerator());
      discriminatorHistory.push(await this.trainDiscriminator());

      if (this.consoleInfo) console.log(`epoch=${i} [Discriminator : ${Object.values(discriminatorHistory[discriminatorHistory.length-1])}] [Generator : ${Object.values(generatorHistory[generatorHistory.length-1])}]`);
      if (this.useTfjsVis) {
        await tfvis.show.history({name: 'Discriminator Training', tab: 'Training'}, discriminatorHistory, ['loss', 'acc']);
        await tfvis.show.history({name: 'Generator Training', tab: 'Training'}, generatorHistory, ['loss', 'acc']);
      }

      if (this.downloadSample) {
        delbot.utils.download(JSON.stringify({
          discriminatorLoss: Object.values(discriminatorHistory[discriminatorHistory.length - 1]),
          generatorLoss: Object.values(generatorHistory[generatorHistory.length - 1]),
          sample: await this.sampleTrajectory()
        }), `GAN${ganTest}_${i}.txt`);
      }

      if (this.downloadModels > 0 && i > 0 && i % this.downloadModels === 0) {
        await this.generator.save(`downloads://GAN${ganTest}_Generator_epoch${i}`);
        await this.discriminator.save(`downloads://GAN${ganTest}_Discriminator_epoch${i}`);
      }
    }

    if (this.downloadModels > 0) {
      await this.generator.save(`downloads://GAN${ganTest}_Generator_epoch${this.epoch}`);
      await this.discriminator.save(`downloads://GAN${ganTest}_Discriminator_epoch${this.epoch}`);
    }
  }

  /**
   * Train the generator to fool the discriminator for one epoch and returns the average loss and accuracy.
   * Call {@link parseHistory} with a 2d array `[[loss1,acc1], ..., [loss_n,acc_n]]` to compute the return
   * value, it will simply get the column average.
   */
  protected async trainGenerator(): Promise<{ loss: number, acc: number }> {
    // test some more training for generator
    const genOtpt = [];
    for (let j=0; j<5; j++) {
      // Train the generator (generator good when discriminator says his fake trajectory is valid)
      genOtpt.push(await this.adversarialModel.trainOnBatch(this.getNoise(this.batchSize), this.humanLabel) as number[]);
    }
    return this.parseHistory(genOtpt);
  }

  /**
   * Train the discriminator for one epoch, on generator outputs and real human trajectories, and
   * returns the average loss and accuracy. Call {@link parseHistory} with a 2d array
   * `[[loss1,acc1], ..., [loss_n,acc_n]]` to compute the return value, it will simply get the column average.
   */
  protected async trainDiscriminator(): Promise<{ loss: number, acc: number }> {
    // Select a random batch (we could use half size to prevent over fitting)
    const {xs, ys} = this.data.nextTrainBatch(this.batchSize);
    const syntheticTrajectory = this.generator.predict(this.getNoise(this.batchSize));

    // If we always have human samples, then we could replace 'ys' by 'valid'
    const discriminatorLossFake = await this.discriminator.trainOnBatch(syntheticTrajectory, this.botLabel) as number[];
    const discriminatorLossReal = await this.discriminator.trainOnBatch(xs, ys) as number[];
    return this.parseHistory([discriminatorLossReal, discriminatorLossFake]);
  }

  /**
   * Returns an input random tensor to run the generator.
   * @param batchSize The batchSize of the input, number of times we want generator random values.
   * @protected
   */
  protected getNoise(batchSize: number): tf.Tensor<tf.Rank> {
    return tf.rand([batchSize, this.generatorOutputSize, this.generatorNodes], randomNormalBoxMuller);
  }

  /**
   * Initializes the discriminator according to layers of {@link getNewDiscriminatorLayers} and returns it.
   * @protected
   */
  protected initDiscriminator(): { discriminator: tf.LayersModel, discriminatorInputs: tf.SymbolicTensor, discriminatorOutputs: tf.SymbolicTensor } {
    const inputs = tf.input({shape: [this.data.data.getXSize(), this.data.data.getYSize()]});
    const outputs = applyLayers(this.getNewDiscriminatorLayers(), inputs) as tf.SymbolicTensor;
    return {
      discriminator:tf.model({inputs, outputs, name:"discriminator"}),
      discriminatorInputs:inputs,
      discriminatorOutputs:outputs
    };
  }

  /**
   * Initializes the generator according to layers of {@link getNewGeneratorLayers} and returns it.
   * @protected
   */
  protected initGenerator(): { generator: tf.LayersModel, generatorInputs: tf.SymbolicTensor, generatorOutputs: tf.SymbolicTensor } {
    const inputs = tf.input({shape: [this.generatorOutputSize, this.generatorNodes]});
    const outputs = applyLayers(this.getNewGeneratorLayers(), inputs) as tf.SymbolicTensor;
    return {
      generator:tf.model({inputs, outputs, name:"generator"}),
      generatorInputs:inputs,
      generatorOutputs:outputs
    };
  }

  /**
   * Gets a 2d array of loss and accuracy values of shape [[loss1,...,loss_n], [acc1,...,acc_n]] and
   * returns the average values for both losses and accuracies.
   * @param history The 2d array with loss and accuracy values.
   * @protected
   */
  protected parseHistory(history: number[][]): {loss: number, acc: number} {
    // mean on the first dimension, so by columns to get [loss, acc] for all history
    const average = tf.mean(history, 0).arraySync();
    return {
      "loss": average[0],
      "acc": average[1]
    };
  }

  /**
   * Sample a generator trajectory as tensor.
   * @param batch The number of trajectories to sample.
   * @protected
   */
  protected getTrajectory(batch: number): tf.Tensor<tf.Rank> {
    return this.generator.predict(this.getNoise(batch)) as tf.Tensor;
  }
}
