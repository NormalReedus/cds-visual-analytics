# Week 9 - Convolutional neural network

## How to run
- Clone this repo and open the directory in bash
- Create the venv and install dependencies:
```bash
bash create_visual_venv.sh # for Mac
bash create_visual_win_venv.sh # for Windows
```
- Manually download the dataset from [here](https://www.kaggle.com/delayedkarma/impressionist-classifier-data "kaggle")

- Navigate into `week9/`:
```bash
cd week9
```
- Run the script
```bash
python cnn_artists.py # for Windows
python3 cnn_artists.py # for Mac
```

## Output
A classification report and a graph of the model performance can be found in `week9/output/`.

## Sample run
Due to the size of the dataset, this code will only use 1000 images to train on (and 320 to validate).
The model performs very poorly, which might partly be because of this.

## Memory overflow
You should use worker2 if you are having issues with keeping all of the images in memory when running the script. Running out of memory can crash your computer.

## Data
You will need a dataset in the following directory structure inside this week9/directory. Every artist's folder should be filled with image files:
├── cnn-artists.py
├── output
│   ├── loss_and_accuracy.png
│   └── report.txt
└── data
    ├── training
    │   ├── Cezanne
    │   ├── Degas
    │   ├── Gauguin
    │   ├── Hassam
    │   ├── Matisse
    │   ├── Monet
    │   ├── Pissarro
    │   ├── Renoir
    │   ├── Sargent
    │   └── VanGogh
    └── validation
        ├── Cezanne
        ├── Degas
        ├── Gauguin
        ├── Hassam
        ├── Matisse
        ├── Monet
        ├── Pissarro
        ├── Renoir
        ├── Sargent
        └── VanGogh