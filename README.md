# NeuralModels
'''
This project reads contract text files, cleans and lemmatizes them, converts them into sequences of tokens, and predicts structural section labels (S1–S6) for each token. It supports three architectures:
Conv1D baseline
UNet-style 1D CNN
PSPNet-style 1D CNN
The goal is to segment and classify text into meaningful contract sections automatically.

Features:
Russian text preprocessing with punctuation removal and pymorphy2 lemmatization.
Custom Word2Vec embeddings trained directly on your contract corpus.
Sliding window sequence generation for sequence labeling.
Multiple architectures for experimentation:
Conv1D
UNet (1D)
PSPNet (1D)
Per-section and overall accuracy evaluation.
'''
