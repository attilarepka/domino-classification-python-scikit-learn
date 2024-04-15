# Domino Classification with Python and Scikit-Learn

This repository contains a Python implementation for classifying dominos using Scikit-Learn.

## Installation

1. Clone this repository:

``` bash
$ git clone https://github.com/attilarepka/domino-classification-python-scikit-learn.git
```

2. Navigate to the project directory:

``` bash
$ cd domino-classification-python-scikit-learn
```

3. Install the required Python packages:

``` bash
$ pip install -r requirements.txt
```

## Usage

1. **Dataset Preparation**: Run `dataset.py` to split and create datasets from a large domino set image.

2. **Contour Validation and Display**: Utilize `dots.py` for contour validation and display, which is used by `dataset.py`.

3. **Training and Validation**: Use `main.py` for training and validation of the classifier.

4. **Inferencing**: Perform inferencing using `predict.py` based on the created dataset.

## License

This project is licensed under the [MIT License](LICENSE).
