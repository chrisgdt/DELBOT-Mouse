const path = require("path");

module.exports = {
  mode: "production",
  entry: "./src/index.ts",
  output: {
    path: path.resolve(__dirname, "dist"),
    filename: "delbot-training.js",
    library: "delbotrain",
    libraryTarget: "umd"
  },
  optimization: {
    minimize: true,
  },
  module: {
    rules: [
      {
        test: /\.ts(x*)?$/,
        exclude: /node_modules/,
        use: {
          loader: "ts-loader",
          options: {
            configFile: "tsconfig.umd.json"
          }
        }
      }
    ]
  },
  resolve: {
    extensions: [".ts", ".js"]
  },
  externals: {
    '@chrisgdt/delbot-mouse' : 'delbot',
    '@tensorflow/tfjs': 'tf',
    '@tensorflow/tfjs-vis': 'tfvis'
  }
};
