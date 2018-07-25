# Simple RNN
Creating a simple, parallel recurrent neural network for C++, as an extension of an already implemented BPNN.

I'm extending the Back Propagation Neural Network created in [my earlier project](https://github.com/setu4993/Fatality_Rate_Prediction_Navigation_Tool) as a recurrent neural network for another project. This isn't focused creating an optimal implementation, but only to understand the differences between a BPNN and a simple RNN, and of course to create one without relying on external libraries.

I completed most of the updated copy in March, 2017. Since then, I have been working on optimizing it further for the task at hand: predicting daily water demand for Central Indiana. This project was completed with help and valuable inputs from Citizens Energy Group, Indianapolis.

My paper ['A water demand prediction model for Central Indiana'](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16511) describing this project was published as a part of the Thirtieth Conference on Innovative Applications of Artificial Intelligence (IAAI '18).

The code attached creates 12 recurrent neural networks, one for each month, and predicts the daily water demand based on weather inputs, all of which is read from a CSV file. More details about the implementation can be found in the paper. Cite as:
`
@paper{AAAI1816511,
	author = {Setu Shah and Mahmood Hosseini and Zina Ben Miled and Rebecca Shafer and Steve Berube},
	title = {A Water Demand Prediction Model for Central Indiana},
	conference = {AAAI Conference on Artificial Intelligence},
	year = {2018},
	keywords = {Prediction; Modeling; Neural Networks},
	abstract = {Due to the limited natural water resources and the increase in population, managing water consumption is becoming an increasingly important subject worldwide. In this paper, we present and compare different machine learning models that are able to predict water demand for Central Indiana. The models are developed for two different time scales: daily and monthly. The input features for the proposed model include weather conditions (temperature, rainfall, snow), social features (holiday, median income), date (day of the year, month), and operational features (number of customers, previous water demand levels). The importance of these input features as accurate predictors is investigated. The results show that daily and monthly models based on recurrent neural networks produced the best results with an average error in prediction of 1.69% and 2.29%, respectively for 2016. These models achieve a high accuracy with a limited set of input features.},
	url = {https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16511}
}
`