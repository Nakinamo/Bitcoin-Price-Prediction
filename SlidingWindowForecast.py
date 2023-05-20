import numpy as np
from tqdm import tqdm


class SlidingWindowForecast:
    def __init__(self, model=None, forecast_interval=6, train_interval=24):
        self.setModel(model)
        self.forecast_interval = forecast_interval
        self.train_interval = train_interval
        self.prediction = None

    def setModel(self, model):
        self.model = model
        return self

    def forecast(self, data, target):
        assert (self.model is not None)

        prediction = np.array([])

        for _ in tqdm(range((data.shape[0] + self.forecast_interval - 1) // self.forecast_interval)):
            # while data.shape[0] > self.train_interval:
            X_train = data[:self.train_interval]
            y_train = target[:self.train_interval]

            X_test = data[self.train_interval:self.train_interval +
                          self.forecast_interval]

            data = data[self.forecast_interval:]
            target = target[self.forecast_interval:]

            self.model.fit(X_train, y_train, verbose=0)
            prediction = np.concatenate(
                (prediction, self.model.predict(X_test)))

        self.prediction = prediction

        return prediction

    def latestForecast(self):
        assert (self.prediction is not None)
        return self.prediction
