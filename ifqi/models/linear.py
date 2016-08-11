from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class Linear():
    def __init__(self,
                 degree=3):
        self.degree = degree
        self.poly = PolynomialFeatures(self.degree)
        self.X = None
        self.model = self.initModel()
        
    def fit(self, X, y, **kwargs):
        if self.X is None:
            self.X = self.poly.fit_transform(X)

        return self.model.fit(self.X, y, **kwargs)
      
    def predict(self, x, **kwargs):
        x = self.poly.transform(x)
        
        return self.model.predict(x, **kwargs)
        
    def adapt(self, iteration=1):
        pass

    def initModel(self):
        model = LinearRegression()

        return model