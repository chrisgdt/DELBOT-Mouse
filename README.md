# DEep Learning for BOT detection by Mouse movements (DELBOT Mouse)

This repository is a small TypeScript library to distinguish humans and bots from their mouse movements with
the usage of TensorFlow.js. It was initiated by an internship with the [Bureau404](https://www.bureau404.fr) company
as part of the Master's in Mathematics: Specialist Focus on Careers in Computer Science and Artificial Intelligence
([University of Mons](https://web.umons.ac.be/fr/))

Author : Christophe Grandmont, University of Mons.

## About

Delbot Mouse is a prototype open-source tool using TensorFlow.js and its DeepLearning models to develop a generic way
to distinguish humans and bots from their mouse movements on internet. The repository contains three distinct
Javascript or Typescript modules:

1. delbot-core: the core of the library that contains everything to load an existing model and use it.
2. delbot-training: the part of the code using delbot-core to train new models from scratch. It also contains a 
   Generative Adversarial Network (GAN) implementation.
3. delbot-example: example of usage of delbot-core and delbot-training to train new models or use one.

We also have a `python/` folder being the script location to manipulate datas.

Delbot-core: [![npm version](https://img.shields.io/npm/v/@chrisgdt/delbot-mouse?color=33cd56&logo=npm)](https://www.npmjs.com/package/@chrisgdt/delbot-mouse)

Delbot-training: [![npm version](https://img.shields.io/npm/v/@chrisgdt/delbot-training?color=33cd56&logo=npm)](https://www.npmjs.com/package/@chrisgdt/delbot-training)


## How to install

### For Node
You can install delbot-core or delbot-training from NPM for Node with:
```
npm install @chrisgdt/delbot-mouse
```
```
npm install @chrisgdt/delbot-training
```

You will also need to install Tensorflow, we let you chose the version.
```
npm install @tensorflow/tfjs
```

__Notice that Delbot Mouse is initially meant to be used in the web browser__. Using Node.js or other
JavaScript engines might lead to some unknown errors that you can report as issue.

### For the browser
For a browser usage, you can simply add some script tags. To load delbot-mouse, you need Tensorflow.js.
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
<script src="https://cdn.jsdelivr.net/npm/@chrisgdt/delbot-training@1.1.2/dist/delbot-training.js"></script>
```

## How to use with Node

Import delbot with `import * as delbot from "@chrisgdt/delbot-mouse";`.

If needed, you can also import tensorflow with `import * as tf from '@tensorflow/tfjs';`.

To import delbot-training, use `import * as delbotrain from "@chrisgdt/delbot-training";`.


## How to use from Browser

Once you have loaded delbot mouse in your browser, everything is packed in the single variable `delbot`, same for
the variable `delbotrain` with delbot-training. Everything is exported from `src/index.ts`, check the
[doc of delbot-core](chrisgdt.github.io/DELBOT-Mouse/delbot-core/docs/modules/index.html) or the
[doc of delbot-training](chrisgdt.github.io/DELBOT-Mouse/delbot-training/docs/modules/index.html) for more information.

The folder `trained-models/` contains some pre-trained models that you can use easily from delbot-core with, for example,
`delbot.Models.rnn3`.

Otherwise, you have an entire example of usage in `/delbot-example/` [here](chrisgdt.github.io/DELBOT-Mouse/delbot-example/src/index.html).
