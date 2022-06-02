# Unsupervised Acoustic Keylogger

## Restoration example

Red characters are errors.

![restoration_example](https://github.com/AlexN1ght/Acoustic-Keylogger/blob/main/doc/imgs/restoration_example.png?raw=true)

## Installation
Just download this repo.
There's no requirements list yet. You just have to install all of requirements manually.

## Usage
This is not a library. This is a set of tools to perform unsupervised acoustic keyboard attack. File main.ipynb in src dir includes a demo of an attack. Feel free to change any step in the process.

### Overview of the attack
0. Data collection. Collect data for the attack. datasetMaker.py is designed to obtain labeled dataset for testing purposes.

1. Keystrokes detection. This step built-in KeySoundDataset class. For now, it can only work with labeled data (only for purpose of evaluation) but you can easily change it yourself for unlabeled data (sound only). Keystrokes detection IS NOT perfect, so you must test out the results of this step and tune some hyperparameters to reach certain level of keystrokes detection precision before next step.

2. Feature extraction. This step built-in KeySoundDataset class. You can choose between Mel-Frequency Spectrograms and MFCC ('mel_spec' and 'mfcc' for 'mode' parameter).

3. Dimensionality reduction of  extracted features. This step built-in KeySoundDataset class. Use 
reduse_dims parameter (None for no dimensionality reduction, positive integer N to reduce dimensionality to N). We use UMAP for this purpose.

4. Training CBoS for improving clustering. This is an optional step that can increase precision in the next step. Implemented in the main.ipynb.

5. HMM prediction.  Demo in main.ipynb. We cluster data and predict the original text based on thees clusters using Hidden Markov Model.

6. Language correction. It's not working great, but can improve the next step precision.

7. Classifier training. We build classifier on filtered labeled data obtained in previous steps. Then we predict text using classifier and repeat this process several times.


## Data
Dir various_datasets has several demo labeled audio recordings.

## Results
Precision after HMM step and Classifier cycle for one text on 3 different setups.


|                     |     tea_1     | tea_2_cleared | tea_3_cleared |
|---------------------|---------------|---------------|---------------|
|**HMM Step**         |    90.2%      |     80.9%     |      70%      |
|**Classifier Cycle** |    96.5%      |     93.6%     |     86.6%     |


## Theoretical background 

For more theoretical background and statistic you can read my bachelor's degree thesis "Clustering Contextual Data For Acoustic Audio Attack" in doc dir. It is in Russian, but you can easily translate it with Google translate or Yandex Translate(because it works better for Russian language).

