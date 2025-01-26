from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from file_provider import FileProvider


file_provider = FileProvider()


data = file_provider.get_train_data()
print(data.shape)
x = data.drop(['Serial No.', 'Chance of Admit '], axis=1).to_numpy()
y = data['Chance of Admit '].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)


def get_train():
    return data[0:int(len(data) * 0.7)]


def get_test():
    return data[int(len(data) * 0.7):len(data)]


class StudentSuccessPrediction:
    def __init__(self):
        self.model = LinearRegression()

        self.model.fit(x_train, y_train)


    def predict(self, x_data):
        return self.model.predict(x_data)

    def get_score(self):
        return self.model.score(x_test, y_test)
