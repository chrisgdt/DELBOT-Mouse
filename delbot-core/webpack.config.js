const path = require("path");

module.exports = {
  mode: "production",
  entry: "./src/index.ts",
  output: {
    path: path.resolve(__dirname, "dist"),
    filename: "delbot.js",
    library: "delbot",
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
    extensions: [".ts", ".js"],
    fallback: {
      "path": false,
      "crypto": false,
      "fs": false
    }
  },
  externals: {
    '@tensorflow/tfjs': 'tf'
  }
};
