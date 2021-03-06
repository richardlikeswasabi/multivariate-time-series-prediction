# multivariate-time-series-prediction
Multivariate time series prediction, predicting load given certain inputs. This model was used to accurately predict electricity load on air conditioners given certain environmental inputs.

# Data
The data is contained in load.txt. Exploratory analysis shows some data types are correlated with time. Below is are inputs normalised over time.

![Normalised inputs over time](https://i.imgur.com/Gq5rn8H.png)

# Results
The RNN predicts load with over 91% accuracy on test data. 
![Image of predicted/actual](https://i.imgur.com/gDy8q6b.png)

Uses a SimpleRNN using the tensorflow keras API. Can be configured to use LSTMs/GRUs and so forth.

## Usage
```
pip install -r requirements.txt
python3 rnn.py
```

