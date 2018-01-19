# Simple RNN
Creating a simple, parallel recurrent neural network for C++, as an extension of an already implemented BPNN.

I'm extending the Back Propagation Neural Network created in [my earlier project](https://github.com/setu4993/Fatality_Rate_Prediction_Navigation_Tool) as a recurrent neural network for another project. This isn't focused creating a network that performs fast, but only to understand the differences between a BPNN and a simple RNN, and of course to create one without relying on external libraries.

Update (January, 18): I completed most of the updated copy in March, 2017. Since then, I have been working on optimizing it further for the task at hand: predicting daily water demand for Central Indiana. This project was completed with help and valuable inputs from Citizens Energy Group. I authored a research paper for this project with my mentor, and it has been accepted for publication at the Thirtieth Conference on Innovative Applications of Artificial Intelligence (IAAI '18). (I will add the link once it is published online.)

The code attached creates 12 recurrent neural networks, one for each month, and predicts the daily water demand based on weather inputs, all of which is read from a CSV file.