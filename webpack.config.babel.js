import webpack from "webpack";

const ENTRY_DIR = './src/main.js'
const BUILD_DIR = './static'

module.exports = {
    entry: ENTRY_DIR,
    output: {
        path: BUILD_DIR,
        publicPath: 'http://127.0.0.1:5000/templates/',
        filename: 'bundle.js'
    },
    module: {
        loaders: [{
            test: /.jsx?$/,
            loader: 'babel-loader',
            exclude: /node_modules/,
            query: {
                presets: ['es2015', 'react']
            }
        }]
    },
    stats: {
        colors: true
    },
    devtool: 'source-map'
};
