# DEep Learning for BOT detection by Mouse movements (DELBOT Mouse)

This repository is a small TypeScript library to distinguish humans and bots from their mouse movements with
the usage of [TensorFlow.js](https://www.tensorflow.org/js). It was initiated by an internship with the
[Bureau404](https://www.bureau404.fr) company as part of the Master's in Mathematics:
Specialist Focus on Careers in Computer Science and Artificial Intelligence ([University of Mons](https://web.umons.ac.be/fr/))

- Author : Christophe Grandmont.
- Internship supervisor (Bureau404) : Loïc Jean-Fulcrand ([website](https://www.malt.ch/profile/loicjeanfulcrand))
- Internship supervisor (UMONS) : Christophe Troestler ([website](https://staff.umons.ac.be/christophe.troestler/))

## About

Delbot Mouse is a prototype open-source tool using [TensorFlow.js](https://www.tensorflow.org/js) and some more
other libraries like RandomForest to develop a generic way to distinguish humans and bots from their mouse movements on
a web page. The repository contains three distinct JavaScript or Typescript modules:

1. delbot-core: the core of the library that contains everything to load an existing model and use it.
2. delbot-training: the part of the code using delbot-core to train new models from scratch. It also contains a 
   Generative Adversarial Network (GAN) implementation to generate bot trajectories.
3. delbot-example: example of usage of delbot-core and delbot-training to train new models or use one.

We also have a `python/` folder being the script location to manipulate data.

Delbot-core:
[![npm version](https://img.shields.io/npm/v/@chrisgdt/delbot-mouse?color=33cd56&logo=npm)](https://www.npmjs.com/package/@chrisgdt/delbot-mouse)

Delbot-training:
[![npm version](https://img.shields.io/npm/v/@chrisgdt/delbot-training?color=33cd56&logo=npm)](https://www.npmjs.com/package/@chrisgdt/delbot-training)


## How to install

### For Node
You can install delbot-core or delbot-training from NPM for Node with:
```
npm install @chrisgdt/delbot-mouse
```
```
npm install @chrisgdt/delbot-training
```

You will also need to install TensorFlow, we let you chose the version.
```
npm install @tensorflow/tfjs
```

__Notice that Delbot Mouse is initially meant to be used in the web browser__. Using Node.js or other
JavaScript engines might lead to some unknown errors that you can report here as issue.

### For the browser
For a browser usage, you can simply add some script tags. To load delbot-mouse, you need TensorFlow.js.
```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.0.0/dist/tf.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@chrisgdt/delbot-mouse@1.1.2/dist/delbot.js"></script>
```

To load  delbot-training, you need both Tensorflow.js and delbot-training, and it is highly encouraged to load tfjs-vis
for training visualization.
```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.0.0/dist/tf.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-vis@1.5.1/dist/tfjs-vis.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@chrisgdt/delbot-mouse@1.1.2/dist/delbot.js"></script>
<script src=" https://cdn.jsdelivr.net/npm/@chrisgdt/delbot-training@1.1.2/dist/delbot-training.js"></script>
```

## How to use

With Node, you can import delbot with `import * as delbot from "@chrisgdt/delbot-mouse";`.

If needed, you can also import tensorFlow with `import * as tf from '@tensorflow/tfjs';`.

To import delbot-training, use `import * as delbotrain from "@chrisgdt/delbot-training";`.

---

For browser, once you have loaded delbot mouse with the script tag, everything is packed in the single variable
`delbot`, same for the variable `delbotrain` with delbot-training. Everything is exported from `src/index.ts`, check the
[doc of delbot-core](https://chrisgdt.github.io/DELBOT-Mouse/delbot-core/docs/modules/index.html) or the
[doc of delbot-training](https://chrisgdt.github.io/DELBOT-Mouse/delbot-training/docs/modules/index.html) for more information.

The folder `trained-models/` contains some pre-trained models that you can use easily from delbot-core with, for example,
`delbot.Models.rnn3`.

Otherwise, you have an entire example of usage without the training in `/delbot-example/` [here](https://chrisgdt.github.io/DELBOT-Mouse/delbot-example/src/index.html).


## Basic usage - Example

```ts
import * as delbot from "@chrisgdt/delbot-mouse"; // not needed in browser

// Create an empty recorder to store and compute mouse features
// Set the screen size to normalize (x,y) positions in [0,1]^2
recorder = new Recorder(window.screen.width, window.screen.height);

// Set max size to prevent high memory usage
recorder.setMaxSize(1e6);

// For each small movement, store it in the recorder
document.addEventListener("mousemove", event => {
  recorder.addRecord({
    time: event.timeStamp,
    x: event.clientX,
    y: event.clientY,
    type: "Move" // optional
  });
});

myVerifyElement.addEventListener("click", event => {
  if (recorder.getRecords().length > 100) {
    // Models are obtained from delbot.Models
    const isHuman = recorder.isHuman(delbot.Models.rnn1);
    recorder.clearRecord();
    // ...
  }
});
```

## Training

Here some example and explanations about the training of new models.

### Models

Some pre-trained models are available through `delbot.Models` :

- rnn1 : a neural network with two LSTM layers, the most efficient so far.
- rrn1faster : same, but with less LSTM cells, slightly less efficient.
- rnn2 : same, but the second LSTM layer is replaced by a dense layer.
- rnn3 : a model between rnn1 and rnn3, two LSTM layers then one dense.
- denseMatrix : a neural network with only dense layers and that takes a movement matrix.
- convolutional : a convolutional model that takes a movement matrix.
- randomForest : a random forest machine learning model.

To train your own model with custom tfjs layers, you can look at `delbot-training/src/models.ts` and read
the documentation. You only have to extend a class and define a single method.

You also have a GAN architecture in `delbot-training/src/GANtrain.ts`.

### Dataset

To manipulate the dataset, you can use `python/` folder, here is a short description of each python script.
- parseSample.py : takes some folders with text files of either bot or human trajectories and parse them to a single
                session.json file containing all relative paths to those text files. This JSON file is the entry
                of an instance of dataTraining.
- redraw.py : verifies that text trajectories in a folder is what we except by drawing them with matplotlib.
- botDrawing.py : starts python selenium and draw same shapes from arbitrary heuristics and multiple librairies.
- ganParser.py : takes some GAN folder of generator ouputs and parse them to something readable by `parseSample.py`.

For the dataset usage in the training itself, unzip the file `dataset.tar.gz` to obtain multiple data folder inside
`python/`, then run `parseSample.py` to get `sessions.json` (for normal training) and `sessions_human_only.json` (for GAN).

### Example

The code of a basic training looks like this :
```ts
const data = new delbot.data.DataFeatures2({xSize: 24, numClasses: 2});
const filePath = "../../python/sessions.json"; // path from code to session.json

const datatraining = new delbotrain.DataTraining({filePath: filePath, data: data});

const rnn1Features2 = new delbotrain.ModelRNN({
  dataTraining: datatraining,
  nameToSave: "downloads://model-rnn1-features2", // directly use layerModel.save(nameToSave)
  epoch: 25,
  batchSize: 256,
  useTfjsVis: true,
  consoleInfo: true
});
// or simply 'rfFeature2.run()'
document.addEventListener('DOMContentLoaded', () => {rfFeature2.run()});
```

And for a GAN training :
```ts
// DataSimplePosition: output (x,y) for each movement
const data = new delbot.data.DataSimplePosition({xSize: 35, numClasses: 1});
const filePath = "../../python/sessions_human_only.json"; // path from code to sessions_human_only.json

const datatraining = new delbotrain.DataTraining({
  filePath: filePath,
  data: data,
  trainingRatio: .9 // use 90% of the dataset as training set and 10% as validation
});

// Delay for tfjs-vis used in constructor
document.addEventListener('DOMContentLoaded', () => {
  const gan = new delbotrain.GAN({
    dataTraining: datatraining, // xSize is 35 so the generator outputs 35 movements per trajectories
    epoch: 1000,
    batchSize: 64,
    generatorSeedSize: 100, // generator input is 100 random numbers
    useTfjsVis: true,
    consoleInfo: true,
    downloadModels: 50, // save models every 50 epochs
    downloadSample: true
  });

  gan.train(1); // training number 1 to have unique ID per GAN train
});
```
