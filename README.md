# Domino Classification with Python and Scikit-Learn
------------
This repository contains a Python implementation for classifying dominos using Scikit-Learn.
------------
<h3 align="center">Dataset validation / Inference result</h3>
<p align="center" width="100%">
    <img width="13%" alt="Dataset validation" src="docs/dataset.png" title="Dataset validation" hspace="10"> 
    <img width="10%" alt="Inference result" src="docs/prediction.png" title="Inference result" hspace="10"/>
</p>

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

``` bash
$ python dataset.py -i <input_image_path>
```

2. **Contour Validation and Display**: Utilize `dots.py` for contour validation and display, which is used by `dataset.py`.

3. **Training and Validation**: Use `main.py` for training and validation of the classifier.

``` bash
$ python main.py [-o OUTPUT] [-d DATASET]
```

4. **Inferencing**: Perform inferencing using `predict.py` based on the created dataset.

## License

This project is licensed under the [MIT License](LICENSE).
