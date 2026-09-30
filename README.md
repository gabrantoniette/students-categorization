# Students Categorization

[![CI](https://github.com/gabrantoniette/students-categorization/actions/workflows/ci.yml/badge.svg)](https://github.com/gabrantoniette/students-categorization/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-3.0-150458?logo=pandas&logoColor=white)
![Node.js](https://img.shields.io/badge/Node.js-18.11%2B-339933?logo=nodedotjs&logoColor=white)
![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-4.22-FF6F00?logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/license-ISC-blue)

A study project that walks the full path of a classification problem, from a dirty CSV to a neural network: a Python medallion pipeline (bronze, silver, gold) cleans and encodes 10,000 fictional students, and a [TensorFlow.js](https://www.tensorflow.org/js) network on Node.js will learn to categorize them as **premium**, **medium** or **basic** from their age, favorite color and location.

## About the project

**Students Categorization** is a study project on data engineering and Machine Learning. The goal is to walk, step by step, through the full path of a classification problem: taking messy data, validating it against a contract, turning it into numbers a neural network can understand, training a model with that data and using it to predict which category a new student fits into.

The dataset is fictional: 10,000 made-up students in a CSV with the problems real exports have, such as ages written as `60 Anos`, colors in Portuguese (`azul`) or misspelled (`gren`), the same city written in several ways (`SP`, `são paulo`, `São Paulo, SP`), placeholders such as `N/A` and `Desconhecido`, empty lines and duplicates.

The pipeline runs in Python with [pandas](https://pandas.pydata.org) and stores each layer as Parquet. The neural network runs locally with [`@tensorflow/tfjs-node`](https://www.npmjs.com/package/@tensorflow/tfjs-node), which runs tensor operations on TensorFlow's native library, directly in Node.js.

> **Status:** in development. The data pipeline is done, and its gold layer holds the input and output vectors for 6,574 students. `index.js` still builds its tensors from three sample students; loading the gold vectors, training the model and predicting are next. See the [roadmap](#roadmap).

## How it works

### The data contract

[`src/contract.json`](src/contract.json) describes what a valid student looks like, and every step of the pipeline reads from it instead of hardcoding the rules:

- **Format:** CSV, UTF-8, comma-delimited.
- **Columns:** `name`, `age`, `color`, `location` and `category`, with no extra columns allowed (`strict_columns`).
- **Per column:** the type (`text`, `int` or `category`), whether it is required, the valid range (age from 18 to 65), the allowed values and their synonyms (`azul` is `blue`, `sp` and `sao paulo - sp` are `São Paulo`, `cwb` is `Curitiba`, `medio` is `medium`).
- **Dataset rules:** drop empty rows (`drop_empty_rows`) and drop duplicates (`deduplicate`).

The order of each `allowed` list is also the order of the one-hot encoding in the gold layer.

### The medallion pipeline

| Layer  | Output | What it does | Rows |
| ------ | ------ | ------------ | ---- |
| Bronze | `src/bronze/students.parquet` | Stores the source as it arrived, only checking its columns | 10,000 |
| Silver | `src/silver/students.parquet` and `students_rejected.parquet` | Applies the contract to every value | 6,574 valid, 3,248 rejected, 178 empty or duplicate |
| Gold   | `src/gold/xs.json` and `ys.json` | Encodes each valid student as numbers | `xs` (6574, 7), `ys` (6574, 3) |

**Bronze** ([`to_bronze`](src/ingest.py)) reads every value as text, so nothing is converted or lost before the contract decides. It normalizes the headers (`Name`, ` Age`, `Color `, `CATEGORY` become `name`, `age`, `color`, `category`), raises a `SchemaError` if a column is missing or unexpected, and adds three lineage columns: `_source_file`, `_source_line` and `_ingested_at`.

**Silver** ([`to_silver`](src/ingest.py), [`cleaning.py`](src/cleaning.py)) drops the empty rows and cleans each value according to its column in the contract:

- **Missing values:** placeholders such as `--`, `?`, `N/A`, `null`, `Não informado` and `Desconhecido` count as missing, and a missing required value rejects the row.
- **Integers:** `53`, ` 34`, `60 Anos` and `53.0` become whole numbers; `18.5`, `-25` and `2S` are rejected.
- **Text:** extra spaces are removed and names are title-cased, so `henrique cardoso` becomes `Henrique Cardoso`.
- **Categories:** compared in lowercase and without accents, then mapped through the synonyms, so `AZUL`, `azul` and `Blue` all become `blue`.
- **Ranges and allowed values:** an age outside 18 to 65, or a color, city or category that is not in the contract, rejects the row.

Valid rows keep their lineage columns and are deduplicated after cleaning, so a row with `sp` and the same row with `São Paulo` count as duplicates. Rejected rows keep their raw values and get one `<column>_error` column per field saying why:

| Field      | Rejections | Reasons |
| ---------- | ---------: | ------- |
| `age`      | 1,273 | 572 not a whole number, 492 missing, 209 out of range |
| `location` |   799 | 410 missing, 389 city not in the contract |
| `color`    |   786 | 406 missing, 380 color not in the contract |
| `category` |   605 | 311 category not in the contract, 294 missing |
| `name`     |   308 | 308 missing |

A row can fail more than one field: 2,759 rejected rows fail one, 455 fail two and 34 fail three.

**Gold** ([`to_gold`](src/ingest.py)) turns each valid student into numbers:

- **Age:** min-max scaled between 0 and 1 with the contract bounds, `(age - 18) / (65 - 18)`. It uses the contract and not the min and max of the data, so a new student is scaled the same way at prediction time.
- **Favorite color and location:** one-hot encoded. Each possible value gets a position in the vector, which is set to `1` when the student has that value and `0` otherwise.
- **Category (label):** also one-hot encoded, in the order `[premium, medium, basic]`.

Each input vector follows the order `[normalized_age, blue, red, green, São Paulo, Rio, Curitiba]`. Four lines of the source file, through the whole pipeline:

| Line | Raw row | Result |
| ---: | ------- | ------ |
| 3  | `henrique cardoso,53,verde,Curitiba,basic` | `xs` `[0.7447, 0, 0, 1, 0, 0, 1]`, `ys` `[0, 0, 1]` |
| 5  | `Letícia Ribeiro,60 Anos,red,são paulo,premium` | `xs` `[0.8936, 0, 1, 0, 1, 0, 0]`, `ys` `[1, 0, 0]` |
| 4  | `Larissa Cardoso,64,--,Manaus,Basic` | Rejected: `color` missing, `location` `'manaus'` not allowed |
| 16 | `eduardo marques,-25,blue,Rio,medium` | Rejected: `age` cannot be parsed as an integer |

The gold layer has 3,245 medium students (49.4%), 1,879 basic (28.6%) and 1,450 premium (22.1%). A model that always answers medium is right 49.4% of the time, so that is the accuracy the trained network has to beat.

### The model

[`index.js`](index.js) builds the model's input (`xs`) and output (`ys`) tensors with `tf.tensor2d`. For now it uses three sample students, encoded by hand in the same vector order as the gold layer:

| Name   | Age | Favorite color | Location  | Category | Input vector               | Label       |
| ------ | --: | -------------- | --------- | -------- | -------------------------- | ----------- |
| Erick  |  30 | blue           | São Paulo | premium  | `[0.33, 1, 0, 0, 1, 0, 0]` | `[1, 0, 0]` |
| Ana    |  25 | red            | Rio       | medium   | `[0, 0, 1, 0, 0, 1, 0]`    | `[0, 1, 0]` |
| Carlos |  40 | green          | Curitiba  | basic    | `[1, 0, 0, 1, 0, 0, 1]`    | `[0, 0, 1]` |

These sample vectors predate the pipeline, so their age is scaled with the min and max of the three students (25 to 40) instead of the contract bounds. The next step replaces them with `src/gold/xs.json` and `src/gold/ys.json`.

## Technologies

- [Python](https://www.python.org) with [pandas](https://pandas.pydata.org) 3.0 and [PyArrow](https://arrow.apache.org/docs/python/) (Parquet)
- [Node.js](https://nodejs.org)
- [TensorFlow.js](https://www.tensorflow.org/js) with [`@tensorflow/tfjs-node`](https://www.npmjs.com/package/@tensorflow/tfjs-node) 4.22

## Prerequisites

- Python 3.12 or later, the minimum for the pinned NumPy. The project is developed with Python 3.14.
- Node.js 18.11 or later (the `start` script uses `node --watch`). The project is developed and tested with Node.js 24, the version set in [`.nvmrc`](.nvmrc) (with [nvm](https://github.com/nvm-sh/nvm), run `nvm use`).
- npm

During installation, `@tensorflow/tfjs-node` runs a script that downloads the native TensorFlow binary for your operating system. For version 4.22 that prebuilt binary is only published for Linux x64; on Windows and macOS npm falls back to compiling it from source, which needs a C++ build toolchain and can fail. On Windows, running the project inside [WSL](https://learn.microsoft.com/windows/wsl/) avoids this. This script is already approved in the `allowScripts` field of `package.json`, which recent versions of npm use to control which dependencies may run install scripts. If something goes wrong at this step, see the [tfjs-node documentation](https://github.com/tensorflow/tfjs/tree/master/tfjs-node).

The approval is pinned to the installed version, and CI fails when a dependency has an install script that is not approved. After upgrading `@tensorflow/tfjs-node` (for example, in a Dependabot pull request), review the new version and run `npm install-scripts approve @tensorflow/tfjs-node` to update `package.json`.

## Installation and usage

```bash
git clone https://github.com/gabrantoniette/students-categorization.git
cd students-categorization
```

### Data pipeline

```bash
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

`main.py` runs the three layers in order and writes them to `src/bronze/`, `src/silver/` and `src/gold/`. These folders are not versioned, so run the pipeline once after cloning. Expected output:

```text
bronze  10000 rows
silver  6574 valid | 3248 rejected | 178 empty or duplicate
gold    xs (6574, 7) | ys (6574, 3)
```

### Model

```bash
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

Expected output:

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

### Tests

```bash
node --test src/labels.js
```

The smoke test uses the [Node.js test runner](https://nodejs.org/api/test.html): it runs `index.js` and checks that the input and output tensors above are printed. The Python pipeline has no tests yet.

## Continuous integration

Every pull request and every push to `main` runs the [CI workflow](.github/workflows/ci.yml) on GitHub Actions:

- **Test:** installs the dependencies from `package-lock.json` (failing if a dependency has an install script that is not approved in `allowScripts`), verifies the npm registry signatures of the installed packages and runs `npm test`.
- **Dependency review:** fails the pull request if it adds or updates a dependency with a known vulnerability.

The workflow covers the Node.js side only; the Python pipeline does not run in CI yet.

[Dependabot](.github/dependabot.yml) opens weekly pull requests to update npm packages and GitHub Actions, and GitHub's CodeQL code scanning looks for security issues in the code.

## Project structure

```text
.
├── .github/
│   ├── workflows/
│   │   └── ci.yml          # CI pipeline: install, signature check, tests and dependency review
│   └── dependabot.yml      # Weekly dependency updates
├── src/
│   ├── docs/
│   │   └── students_raw.csv  # Source: 10,000 fictional students, raw and messy
│   ├── contract.json       # Data contract: columns, types, ranges, allowed values and synonyms
│   ├── cleaning.py         # Value cleaning: missing values, integers, text and synonyms
│   ├── ingest.py           # Bronze, silver and gold layers
│   ├── labels.js           # Smoke test: checks the tensors printed by index.js
│   ├── bronze/             # Generated by main.py, not versioned
│   ├── silver/             # Generated by main.py, not versioned
│   └── gold/               # Generated by main.py, not versioned: xs.json and ys.json
├── main.py                 # Runs the pipeline and prints the rows in each layer
├── requirements.txt        # Pinned Python dependencies
├── index.js                # Sample data, preprocessing and tensor creation
├── package.json            # Metadata, scripts and dependencies
├── package-lock.json       # Exact dependency versions
├── .nvmrc                  # Node.js version used in development and CI
├── LICENSE
├── SECURITY.md             # How to report a vulnerability
└── README.md
```

## Roadmap

- [x] Define the sample data
- [x] Normalize the age and one-hot encode favorite color, location and category
- [x] Create the input (`xs`) and output (`ys`) tensors
- [x] Write a data contract for the students dataset
- [x] Build the bronze, silver and gold layers for 10,000 students
- [ ] Load the gold vectors in `index.js`
- [ ] Define the neural network architecture
- [ ] Train the model and beat the 49.4% baseline
- [ ] Predict the category of new students

## Contributing

This is a study project, but suggestions and improvements are welcome. Open an [issue](https://github.com/gabrantoniette/students-categorization/issues) to report a problem or discuss an idea, or send a pull request:

1. Fork the repository
2. Create a branch for your change: `git checkout -b feat/my-improvement`
3. Make sure the pipeline runs (`python main.py`) and the tests pass (`node --test src/labels.js`)
4. Commit your changes: `git commit -m "feat: describe the improvement"`
5. Push the branch: `git push origin feat/my-improvement`
6. Open a pull request

The `main` branch is protected: changes only reach it through pull requests, after the CI checks pass.

## Security

Please do not report security problems in public issues. See the [security policy](SECURITY.md) to report a vulnerability privately.

## License

Distributed under the ISC License. See the [LICENSE](LICENSE) file for details.

## Author

Developed by [Gabriel Antoniette](https://github.com/gabrantoniette).
