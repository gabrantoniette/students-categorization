# Students Categorization

[![CI](https://github.com/gabrantoniette/students-categorization/actions/workflows/ci.yml/badge.svg)](https://github.com/gabrantoniette/students-categorization/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-3.0-150458?logo=pandas&logoColor=white)
![Node.js](https://img.shields.io/badge/Node.js-18.20%2B-339933?logo=nodedotjs&logoColor=white)
![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-4.22-FF6F00?logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/license-ISC-blue)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/gabrantoniette/gabrantoniette/main/assets/generated/languages/students-categorization-dark.svg">
  <img src="https://raw.githubusercontent.com/gabrantoniette/gabrantoniette/main/assets/generated/languages/students-categorization-light.svg" alt="Languages in students-categorization, by share of code">
</picture>

A study project that walks the full path of a classification problem, from a dirty CSV to a neural network: a Python medallion pipeline (bronze, silver, gold) cleans 10,000 rows of fictional students and encodes the 6,574 valid ones, and a [TensorFlow.js](https://www.tensorflow.org/js) network on Node.js learns to categorize them as **premium**, **medium** or **basic** from their age, favorite color and location. Students who sign up later go through the same contract, only the ones who signed up after the last student already in are added, and the network suggests a category for each of them.

## About the project

**Students Categorization** is a study project on data engineering and Machine Learning. The goal is to walk, step by step, through the full path of a classification problem: taking messy data, validating it against a contract, turning it into numbers a neural network can understand, training a model with that data and using it to predict which category a new student fits into.

The dataset is fictional: a CSV with 10,000 rows of made-up students and the problems real exports have, such as ages written as `60 Anos`, colors in Portuguese (`azul`) or misspelled (`gren`), the same city written in several ways (`SP`, `são paulo`, `São Paulo, SP`), placeholders such as `N/A` and `Desconhecido`, rows with every field empty, and duplicates.

The pipeline runs in Python with [pandas](https://pandas.pydata.org) and stores each layer as Parquet. The neural network runs locally with [`@tensorflow/tfjs-node`](https://www.npmjs.com/package/@tensorflow/tfjs-node), which runs tensor operations on TensorFlow's native library, directly in Node.js.

> **Status:** complete as a study project. The data pipeline writes the input and output vectors for 6,574 students, `src/upsert.py` adds the students who sign up later, and `src/index.js` trains a network and categorizes each new student. What it does not do is evaluate the model on students it has not seen. See the [roadmap](#roadmap) and the [known issues](#known-issues).

## How it works

### The data contract

[`src/contract.json`](src/contract.json) describes what a valid student looks like. The pipeline reads the columns, types, ranges, allowed values and synonyms from it instead of hardcoding them:

- **Format:** CSV, UTF-8, comma-delimited.
- **Columns:** `name`, `age`, `color`, `location` and `category`, with no extra columns allowed (`strict_columns`).
- **Per column:** the type (`text`, `int`, `category` or `datetime`), whether it is required, the valid range (age from 18 to 65), the allowed values and their synonyms (`azul` is `blue`, `sp` and `sao paulo - sp` are `São Paulo`, `cwb` is `Curitiba`, `medio` is `medium`).
- **Dataset rules:** drop empty rows (`drop_empty_rows`) and drop duplicates (`deduplicate`).
- **New students:** `label` names the column the model predicts (`category`), and `new_students` describes the students who sign up later: every column except the label, plus `created_at`, the time they signed up, in ISO 8601. It is stored in UTC, and a time without an offset is read as UTC.

The order of each `allowed` list is also the order of the one-hot encoding in the gold layer.

### The medallion pipeline

| Layer  | Output | What it does | Rows |
| ------ | ------ | ------------ | ---- |
| Bronze | `src/bronze/students.parquet` | Stores every value as it arrived, checking the columns and adding lineage | 10,000 |
| Silver | `src/silver/students.parquet` and `students_rejected.parquet` | Applies the contract to every value | 6,574 valid, 3,248 rejected, 178 empty or duplicate |
| Gold   | `src/gold/xs.json` and `ys.json` | Encodes each valid student as numbers | `xs` (6574, 7), `ys` (6574, 3) |

**Bronze** ([`to_bronze`](src/process.py)) reads every value as text, so nothing is converted or lost before the contract decides. It normalizes the headers (`Name`, ` Age`, `Color `, `CATEGORY` become `name`, `age`, `color`, `category`), raises a `SchemaError` if a column is missing (or unexpected, since the contract sets `strict_columns`), and adds three lineage columns: `_source_file`, `_source_line` and `_ingested_at`.

**Silver** ([`to_silver`](src/process.py), [`cleaning.py`](src/cleaning.py)) drops the empty rows and cleans each value according to its column in the contract:

- **Missing values:** an empty field and placeholders such as `--`, `?`, `N/A`, `null`, `Não informado` and `Desconhecido` count as missing, and a missing required value rejects the row. This list lives in `cleaning.py` (`NULL_TOKENS`), not in the contract; a column can add its own placeholders with a `nulls` key in the contract.
- **Integers:** `53`, ` 34`, `60 Anos`, `56.0` and `60,0` become whole numbers; `18.5`, `2S` and `-25` are rejected. The parser only reads digits, so a negative age fails here, as unparseable, before the range check.
- **Text:** extra spaces are removed and names are title-cased, so `henrique cardoso` becomes `Henrique Cardoso`.
- **Categories:** compared in lowercase and without accents, then mapped through the synonyms, so `AZUL`, `azul` and `Blue` all become `blue`.
- **Ranges and allowed values:** an age outside 18 to 65, or a color, city or category that is not in the contract, rejects the row.

Valid rows keep their lineage columns and are deduplicated after cleaning, so a row with `sp` and the same row with `São Paulo` count as duplicates. Rejected rows keep their raw values and lineage, so `_source_line` points back to the line in the CSV, and get one `<column>_error` column per field, filled where that field failed. They are not deduplicated: the same bad row can appear more than once.

| Field      | Rejections | Reasons |
| ---------- | ---------: | ------- |
| `age`      | 1,273 | 572 unparseable (69 of them negative), 492 missing, 209 out of range |
| `location` |   799 | 410 missing, 389 city not in the contract |
| `color`    |   786 | 406 missing, 380 color not in the contract |
| `category` |   605 | 311 category not in the contract, 294 missing |
| `name`     |   308 | 308 missing |

A row can fail more than one field: 2,759 rejected rows fail one, 455 fail two and 34 fail three.

**Gold** ([`to_gold`](src/process.py)) turns each valid student into numbers:

- **Age:** min-max scaled between 0 and 1 with the contract bounds, `(age - 18) / (65 - 18)`, rounded to four decimals. It uses the contract and not the min and max of the data, so a new student is scaled the same way at prediction time.
- **Favorite color and location:** one-hot encoded. Each possible value gets a position in the vector, which is set to `1` when the student has that value and `0` otherwise.
- **Category (label):** also one-hot encoded, in the order `[premium, medium, basic]`.

Each input vector follows the order `[normalized_age, blue, red, green, São Paulo, Rio, Curitiba]`. Four lines of the source file, through the whole pipeline:

| Line | Raw row | Result |
| ---: | ------- | ------ |
| 3  | `henrique cardoso,53,verde,Curitiba,basic` | `xs` `[0.7447, 0, 0, 1, 0, 0, 1]`, `ys` `[0, 0, 1]` |
| 4  | `Larissa Cardoso,64,--,Manaus,Basic` | Rejected: `color` missing, `location` `'manaus'` not allowed |
| 5  | `Letícia Ribeiro,60 Anos,red,são paulo,premium` | `xs` `[0.8936, 0, 1, 0, 1, 0, 0]`, `ys` `[1, 0, 0]` |
| 16 | `eduardo marques,-25,blue,Rio,medium` | Rejected: `age` cannot be parsed as an integer |

The gold layer has 3,245 medium students (49.4%), 1,879 basic (28.6%) and 1,450 premium (22.1%). A model that always answers medium is right 49.4% of the time, so that is the accuracy the trained network has to beat.

### New students: the upsert

Students who sign up after the training set was built arrive in [`src/docs/students_new.csv`](src/docs/students_new.csv), with the time they signed up and without a category, which is what the model predicts. [`upsert`](src/process.py) takes them through the same contract and the same layers, in files of their own, so the training set never changes:

| Layer | Output | What it does |
| ----- | ------ | ------------ |
| Bronze | `src/bronze/new_students.parquet` | Keeps every line of the file once, as it arrived the first time, with the same lineage columns |
| Silver | `src/silver/new_students.parquet` | Applies the contract and adds only the students who signed up after the last one already in, each with an `id` |
| Quarantine | `src/quarantine/new_students.csv` | Records every row that did not go in, with the reason |
| Gold | `src/gold/new_students.json` | Encodes every new student like the training set: `id`, `name`, `created_at` and the 7 input numbers |

The timestamp works as a watermark. A run reads the whole file, but only the students whose `created_at` is later than the most recent one already in silver are added, so running again on the same file adds nothing, and a file that keeps growing only contributes its new lines. Names repeat in the data (the 6,574 valid students share 1,996 names), so there is no key to update a student who is already in: a line identical to a student in silver is the same student sent again, and it does not go in.

Every line that does not go in is recorded in the quarantine file once, with the raw values it arrived with, its lineage and a `reason`. A line that already went in on an earlier run is not recorded: it is in silver.

| Reason | When |
| ------ | ---- |
| `invalid` | A field failed the contract. One `<column>_error` column per field says why, as in `students_rejected.parquet` |
| `empty` | Every field is empty |
| `duplicate` | The same student is already in silver from another line, or appears more than once among the lines being added, where the first one goes in |
| `late` | A valid student who is not in silver yet but signed up before the most recent student already in, so the watermark keeps them out |

### The model

[`src/index.js`](src/index.js) imports `src/gold/xs.json` and `src/gold/ys.json`, builds the input (`xs`) and output (`ys`) tensors with `tf.tensor2d`, trains a network on them and categorizes the new students.

| Layer  | Units | Activation | What it does |
| ------ | ----: | ---------- | ------------ |
| Dense (input of 7 values) | 80 | `relu` | Combines the 7 values of a student into 80 patterns; negative results become 0 |
| Dense (output) | 3 | `softmax` | One probability per category (`premium`, `medium`, `basic`), summing to 100% |

The model is compiled with the `adam` optimizer, the `categoricalCrossentropy` loss and the `accuracy` metric, and `model.fit` trains it for 100 epochs, shuffling the students on each one. The per-epoch logging callback is left commented out in the file.

After training, the script categorizes every student in `src/gold/new_students.json` who is not in `src/gold/categorized.json` yet. `prediction()` runs `model.predict` on the student's `[1, 7]` tensor and returns each category with its probability, and [`bestCategory`](src/categorize.js) keeps the most likely one, its probability and the probability of each category. The results are appended to `src/gold/categorized.json`, so each student is categorized once. When there is no new student to categorize, the script says so and does not train.

## Technologies

- [Python](https://www.python.org) with [pandas](https://pandas.pydata.org) 3.0 and [PyArrow](https://arrow.apache.org/docs/python/) (Parquet)
- [Node.js](https://nodejs.org)
- [TensorFlow.js](https://www.tensorflow.org/js) with [`@tensorflow/tfjs-node`](https://www.npmjs.com/package/@tensorflow/tfjs-node) 4.22

## Prerequisites

- Python 3.12 or later, the minimum for the pinned NumPy. The project is developed with Python 3.14.
- Node.js 18.20 or later (the code imports JSON with `with { type: 'json' }` and the `start` script uses `node --watch`). Training runs on Node.js 22 or earlier; on Node.js 23 and later it needs a workaround (see [Known issues](#known-issues)). [`.nvmrc`](.nvmrc) still says 24 (with [nvm](https://github.com/nvm-sh/nvm), run `nvm use 22` to train).
- npm

During installation, `@tensorflow/tfjs-node` runs a script that downloads the native TensorFlow binary for your operating system. This script is already approved in the `allowScripts` field of `package.json`, which recent versions of npm use to control which dependencies may run install scripts. If something goes wrong at this step, see the [tfjs-node documentation](https://github.com/tensorflow/tfjs/tree/master/tfjs-node).

For version 4.22 the prebuilt binary is only published for Linux x64. On Windows and macOS the script finds no binary and falls back to compiling it from source, which needs a C++ build toolchain and can fail. On Windows, running the project inside [WSL](https://learn.microsoft.com/windows/wsl/) avoids this. The Python pipeline is not affected.

The approval is pinned to the installed version, and CI fails when a dependency has an install script that is not approved. After upgrading `@tensorflow/tfjs-node` (for example, in a Dependabot pull request), review the new version and run `npm install-scripts approve @tensorflow/tfjs-node` to update `package.json`.

## Installation and usage

```bash
git clone https://github.com/gabrantoniette/students-categorization.git
cd students-categorization
```

### Data pipeline

```bash
python3 -m venv .venv       # on Windows: py -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.ingest
```

Run it from the project root. [`src/ingest.py`](src/ingest.py) runs the three layers of [`src/process.py`](src/process.py) in order and writes them to `src/bronze/`, `src/silver/` and `src/gold/`. These folders are not versioned, so run the pipeline once after cloning. Expected output:

```text
bronze  10000 rows
silver  6574 valid | 3248 rejected | 178 empty or duplicate
gold    xs (6574, 7) | ys (6574, 3)
```

### New students

After the pipeline, from the project root:

```bash
python -m src.upsert
```

[`src/upsert.py`](src/upsert.py) reads `src/docs/students_new.csv`, adds the students who signed up after the last one already in, and records the rest in `src/quarantine/new_students.csv`. On the sample file, the first run prints:

```text
bronze      10 rows in students_new.csv | 10 new
silver      5 new students (first load) | 0 already in
quarantine  3 invalid, 1 empty, 1 duplicate -> src/quarantine/new_students.csv
gold        5 new students encoded for the model
```

The three invalid lines are a color that is not in the contract (`roxo`), a date in another format (`06/10/2026 09:15`) and an age out of range (`72`). Running it again on the same file adds nothing:

```text
bronze      10 rows in students_new.csv | 0 new
silver      0 new students (signed up after 2026-10-06 11:02 UTC) | 5 already in
quarantine  nothing new -> src/quarantine/new_students.csv
gold        5 new students encoded for the model
```

With two more lines appended to the file, one student who signed up after the last student in and one who signed up before:

```text
bronze      12 rows in students_new.csv | 2 new
silver      1 new student (signed up after 2026-10-06 11:02 UTC) | 5 already in
quarantine  1 late -> src/quarantine/new_students.csv
gold        6 new students encoded for the model
```

### Model

```bash
npm install
```

Run the pipeline and the upsert first, so `src/gold/` exists. To train the model and categorize the new students:

```bash
node src/index.js
```

To run in watch mode (the script runs again every time a file is saved, and keeps waiting after it finishes; press `Ctrl+C` to leave):

```bash
npm start
```

Training takes a few minutes: 100 epochs over 6,574 students took about 3 to 4 minutes on WSL. Then it prints the category of each new student, with its probability. After the first upsert of the sample file:

```text
Mariana Souza: medium (78.81%)
Rafael Lima: premium (67.93%)
Beatriz Rocha: medium (76.01%)
Thiago Alves: medium (78.78%)
Camila Ferreira: basic (88.19%)
5 new students categorized -> src/gold/categorized.json
```

After the second upsert, a new run categorizes only the student added since:

```text
Ana Ribeiro: medium (73.68%)
1 new student categorized -> src/gold/categorized.json
```

`src/gold/categorized.json` keeps each student's `id`, `name`, `created_at`, `category`, `probability` and the probability of every category. The weights start random, so the percentages, and sometimes the category, change from one training to the next, and the model is not evaluated yet (see [Known issues](#known-issues)).

Before the result, TensorFlow may print an informational message about CPU optimizations. It is normal and does not indicate an error. Setting `TF_CPP_MIN_LOG_LEVEL=2` hides it.

### Tests

The upsert and the date parsing have tests with Python's `unittest`, and the functions that pick which students to categorize have tests with Node.js's built-in test runner. Run them from the project root, with the virtual environment active:

```bash
python -m unittest discover -s tests
npm test
```

The rest of the pipeline and the model have no tests yet.

## Continuous integration

Every pull request and every push to `main` runs the [CI workflow](.github/workflows/ci.yml) on GitHub Actions:

- **Test:** installs the dependencies from `package-lock.json` (failing if a dependency has an install script that is not approved in `allowScripts`), verifies the npm registry signatures of the installed packages and runs `npm test`, the Node.js tests in [`src/categorize.test.js`](src/categorize.test.js).
- **Dependency review:** fails the pull request if it adds or updates a dependency with a known vulnerability.

The workflow covers the Node.js side only; the Python tests run locally, and the Python pipeline does not run in CI yet.

[Dependabot](.github/dependabot.yml) opens weekly pull requests to update npm packages and GitHub Actions; the Python dependencies in `requirements.txt` are not covered yet. GitHub's CodeQL code scanning looks for security issues in the code.

## Known issues

- **The model is not evaluated.** It is trained and asked to predict on the same students, and the script prints no accuracy. The probabilities say nothing about how it handles students it has not seen, and the 49.4% baseline has not been compared. Holding out part of the data (for example, `validationSplit: 0.2` in `model.fit`) is the next step.
- **A late student stays out.** The watermark keeps out any student who signed up before the most recent one already in, even one who was never added: a row that arrived after newer ones, or one that failed the contract, was fixed and was sent again. Those rows are recorded in the quarantine file with the reason `late`, and adding them is a manual decision.
- **The model is trained again on every run.** `src/index.js` does not save the model, so categorizing new students takes a full training, and a new student can get a different category than they would have in an earlier run. A student is categorized only once, so earlier results do not change.
- **Training fails on Node.js 23 and later.** `@tensorflow/tfjs-node` 4.22 still calls `util.isNullOrUndefined`, which Node.js removed in version 23. `model.fit` throws `util_1.isNullOrUndefined is not a function` on Node.js 24, the version in `.nvmrc`, so `npm start` fails there; use Node.js 22 (`nvm use 22`). Defining the function again also works on Node.js 24: put this in its own module and import it before `@tensorflow/tfjs-node`.

  ```js
  import util from 'node:util';

  if (typeof util.isNullOrUndefined !== 'function') {
      util.isNullOrUndefined = (value) => value === null || value === undefined;
  }
  ```

## Project structure

```text
.
├── .github/
│   ├── workflows/
│   │   └── ci.yml          # CI pipeline: install, signature check, tests and dependency review
│   └── dependabot.yml      # Weekly dependency updates
├── src/
│   ├── docs/
│   │   ├── students_raw.csv  # Source: 10,000 rows of fictional students, raw and messy
│   │   └── students_new.csv  # Students who signed up later, without a category, also messy
│   ├── __init__.py         # Makes src a package, so python -m src.ingest can import src.process
│   ├── contract.json       # Data contract: columns, types, ranges, allowed values, synonyms and new students
│   ├── cleaning.py         # Value cleaning: missing values, integers, dates, text and synonyms
│   ├── process.py          # Bronze, silver and gold layers, and the upsert of new students
│   ├── ingest.py           # Entry point: runs the three layers and prints the rows in each
│   ├── upsert.py           # Entry point: adds the new students and prints what went where
│   ├── bronze/             # Generated by ingest.py and upsert.py, not versioned
│   ├── silver/             # Generated by ingest.py and upsert.py, not versioned
│   ├── quarantine/         # Generated by upsert.py, not versioned: the new students who did not go in
│   ├── gold/               # Generated, not versioned: xs.json, ys.json, new_students.json, categorized.json
│   ├── categorize.js       # Which new students still need a category, and the most likely one
│   ├── categorize.test.js  # Tests for categorize.js (npm test)
│   └── index.js            # Loads the gold vectors, trains the model and categorizes the new students
├── tests/
│   └── test_upsert.py      # Tests for the upsert and the date parsing (python -m unittest discover -s tests)
├── requirements.txt        # Pinned Python dependencies
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
- [x] Build the bronze, silver and gold layers for the 10,000-row dataset
- [x] Load the gold vectors in `src/index.js`
- [x] Define the neural network architecture
- [x] Train the model
- [x] Predict the category of a student from the dataset
- [x] Add new students incrementally, by the time they signed up, and record the ones that do not go in
- [x] Predict the category of new students
- [ ] Evaluate the model on students it was not trained on and beat the 49.4% baseline

## Contributing

This is a study project, but suggestions and improvements are welcome. Open an [issue](https://github.com/gabrantoniette/students-categorization/issues) to report a problem or discuss an idea, or send a pull request:

1. Fork the repository
2. Create a branch for your change: `git checkout -b feat/my-improvement`
3. Make sure the tests pass (`python -m unittest discover -s tests` and `npm test`), the pipeline runs (`python -m src.ingest`, then `python -m src.upsert`) and the model trains (`node src/index.js`)
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
