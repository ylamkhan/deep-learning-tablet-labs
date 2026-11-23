# Lab: Handwritten Digit Recognition with Deep Learning

**Course:** Deep Learning  
**Topic:** CNN, image classification and model deployment  
**Duration:** 3 hours

## Learning objectives
Students will learn to visualize handwritten image data, train a simple convolutional neural network, evaluate classification results, export a trained model, and connect this model to an interactive web application.

## Part 1 — MNIST (20 min)
1. Load MNIST using Keras and print the shapes of the training/test arrays.
2. Display a grid of 20 images and their labels.
3. Explain why pixel normalization is necessary.

## Part 2 — CNN (35 min)
1. Build the CNN architecture provided in the notebook.
2. Identify the roles of convolution, max-pooling, flatten, dense and softmax.
3. Compile the CNN with Adam and sparse categorical cross-entropy.

## Part 3 — Training and evaluation (35 min)
1. Train the CNN and plot training/validation accuracy and loss.
2. Evaluate the model on the reserved test set.
3. Display a confusion matrix and describe common errors.
4. Export the model in `.keras` format.

## Part 4 — Streamlit (55 min)
1. Install the packages in `requirements.txt`.
2. Study the inputs/outputs of `prepare_digit`.
3. Run `streamlit run app.py`.
4. Use a mouse to draw a digit; then connect a tablet via the local network.
5. Explain how the app loads the model and generates probabilities.

## Part 5 — Tests and discussion (35 min)
1. Write each digit from 0 to 9 on the tablet at least three times.
2. Record the true digit, prediction and softmax confidence.
3. Identify two misclassifications, if any, and propose improvements.
4. Compare reported MNIST test accuracy with accuracy on new handwriting.

## Questions for discussion
- Why is centering the handwritten digit important?
- Does a 99% MNIST test accuracy imply 99% accuracy on the tablet?
- What is the difference between the predicted class and softmax confidence?
- How could augmentation or collecting real tablet drawings improve the model?

## Deliverables
1. Completed notebook with training/validation plots and test metrics.
2. Working Streamlit interface.
3. Brief table with handwriting tests and a two-paragraph conclusion.
