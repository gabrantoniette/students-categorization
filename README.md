# Students Categorization

![Node.js](https://img.shields.io/badge/Node.js-18.11%2B-339933?logo=nodedotjs&logoColor=white)
![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-4.22-FF6F00?logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/license-ISC-blue)

A JavaScript neural network that categorizes students into **premium**, **medium** and **basic** profiles based on age, favorite color and location, built with [TensorFlow.js](https://www.tensorflow.org/js) on Node.js.

## About the project

**Students Categorization** is a study project on Machine Learning with JavaScript. The goal is to walk, step by step, through the full path of a classification problem: turning real-world data into numbers a neural network can understand, training a model with that data and using it to predict which category a new student fits into.

Everything runs locally with [`@tensorflow/tfjs-node`](https://www.npmjs.com/package/@tensorflow/tfjs-node), which runs tensor operations on TensorFlow's native library, directly in Node.js.

> **Status:** in development. The current stage covers data preparation and the creation of the model's input and output tensors. See the [roadmap](#roadmap).

## How it works

### Sample data

| Name   | Age | Favorite color | Location  | Category |
| ------ | --: | -------------- | --------- | -------- |
| Erick  |  30 | blue           | São Paulo | premium  |
| Ana    |  25 | red            | Rio       | medium   |
| Carlos |  40 | green          | Curitiba  | basic    |

### Preprocessing

Neural networks only work with numbers, so each student is converted into a numeric vector:

- **Age:** normalized between 0 and 1 with min-max scaling, `(age - min_age) / (max_age - min_age)`. With ages between 25 and 40, Erick's age becomes `(30 - 25) / (40 - 25) ≈ 0.33`.
- **Favorite color and location:** converted with one-hot encoding. Each possible value gets a position in the vector, which is set to `1` when the student has that value and `0` otherwise.
- **Category (label):** also one-hot encoded, in the order `[premium, medium, basic]`.

Each input vector follows the order `[normalized_age, blue, red, green, São Paulo, Rio, Curitiba]`:

| Name   | Input vector               | Label       |
| ------ | -------------------------- | ----------- |
| Erick  | `[0.33, 1, 0, 0, 1, 0, 0]` | `[1, 0, 0]` |
| Ana    | `[0, 0, 1, 0, 0, 1, 0]`    | `[0, 1, 0]` |
| Carlos | `[1, 0, 0, 1, 0, 0, 1]`    | `[0, 0, 1]` |

These vectors become 2D tensors with `tf.tensor2d`: `xs` (input, shape `[3, 7]`) and `ys` (output, shape `[3, 3]`), which will be used to train the model.

## Technologies

- [Node.js](https://nodejs.org)
- [TensorFlow.js](https://www.tensorflow.org/js) with [`@tensorflow/tfjs-node`](https://www.npmjs.com/package/@tensorflow/tfjs-node) 4.22

## Prerequisites

- Node.js 18.11 or later (the `start` script uses `node --watch`). The project is developed with Node.js 24.
- npm

During installation, `@tensorflow/tfjs-node` runs a script that downloads the native TensorFlow binary for your operating system. This script is already approved in the `allowScripts` field of `package.json`, which recent versions of npm use to control which dependencies may run install scripts. If something goes wrong at this step, see the [tfjs-node documentation](https://github.com/tensorflow/tfjs/tree/master/tfjs-node).

## Installation and usage

```bash
git clone https://github.com/gabrantoniette/students-categorization.git
cd students-categorization
npm install
```

To run in watch mode (the script runs again every time a file is saved):

```bash
npm start
```

To run it once:

```bash
node index.js
```

### Expected output

```text
Tensor
    [[0.33, 1, 0, 0, 1, 0, 0],
     [0   , 0, 1, 0, 0, 1, 0],
     [1   , 0, 0, 1, 0, 0, 1]]
Tensor
    [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]
```

Before the tensors, TensorFlow may print informational messages about CPU optimizations. They are normal and do not indicate an error.

## Project structure

```text
.
├── index.js           # Sample data, preprocessing and tensor creation
├── package.json       # Metadata, scripts and dependencies
├── package-lock.json  # Exact dependency versions
├── LICENSE
└── README.md
```

## Roadmap

- [x] Define the sample data
- [x] Normalize the age and one-hot encode favorite color, location and category
- [x] Create the input (`xs`) and output (`ys`) tensors
- [ ] Define the neural network architecture
- [ ] Train the model
- [ ] Predict the category of new students

## Contributing

This is a study project, but suggestions and improvements are welcome. Open an [issue](https://github.com/gabrantoniette/students-categorization/issues) to report a problem or discuss an idea, or send a pull request:

1. Fork the repository
2. Create a branch for your change: `git checkout -b feat/my-improvement`
3. Commit your changes: `git commit -m "feat: describe the improvement"`
4. Push the branch: `git push origin feat/my-improvement`
5. Open a pull request

## License

Distributed under the ISC License. See the [LICENSE](LICENSE) file for details.

## Author

Developed by [Gabriel Antoniette](https://github.com/gabrantoniette).
