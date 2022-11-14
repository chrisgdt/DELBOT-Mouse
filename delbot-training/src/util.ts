import {readFile, writeFile} from "fs";
import * as tf from "@tensorflow/tfjs";

/**
 * Check if the code is run through the browser or with Node.js. Used to
 * know how to load a local file with {@link loadFile} or download text.
 */
function isBrowser(): boolean {
  // Check if the environment is Node.js
  if (typeof process === "object" &&
    typeof require === "function") {
    return false;
  }

  // Check if the environment is a Service worker
  if (typeof importScripts === "function") {
    return false;
  }

  // Check if the environment is a Browser
  if (typeof window === "object") {
    return true;
  }
}

/**
 * Write a string, like json stringify, to a local file. If used
 * in browser, download it instead.
 * @param content
 * @param fileName
 */
export function download(content: string, fileName: string) {
  if (isBrowser()) {
    const a = document.createElement("a");
    const file = new Blob([content], {type: "text/plain"});
    a.href = URL.createObjectURL(file);
    a.download = fileName;
    a.click();
  } else {
    writeFile(fileName, content, 'utf8', function (err) {
      if (err) {
        console.log("An error occurred while writing JSON Object to File.");
        return console.log(err);
      }
    });
  }
}

/**
 * Read a local text file and returns its content as string.
 * @param filePath The path from the script to the file, e.g.
 *        `../../python/circles_human_pc1/circleHide_1664742992568.txt`
 */
export async function loadFile(filePath: string): Promise<string> {
  return new Promise((resolve, reject) => {
    if (isBrowser()) {
      let xhr = new XMLHttpRequest();
      xhr.open("GET", filePath, true);
      xhr.onload = function () {
        if (this.status >= 200 && this.status < 300) {
          resolve(xhr.response);
        } else {
          reject({
            status: this.status,
            statusText: xhr.statusText
          });
        }
      };
      xhr.onerror = function () {
        reject({
          status: this.status,
          statusText: xhr.statusText
        });
      };
      xhr.send();
    } else {
      readFile(filePath, 'utf8', function (err, data) {
        if (err) {
          reject(err);
        }
        resolve(data);
      });
    }
  });
}


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
