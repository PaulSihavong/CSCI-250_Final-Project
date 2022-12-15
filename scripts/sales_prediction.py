import pandas as pd

import random
import sys

from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QPushButton, QLineEdit, QVBoxLayout, QComboBox

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
from scipy.sparse import hstack

from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT

from matplotlib.figure import Figure
import seaborn

model = None
vectorizer = None
encoder = None
scaler = None

class MainWindow(QWidget):
    def __init__(self, graph_data):
        super().__init__()
        self.setWindowTitle("Video Game Sales Predictor")
        self.setMinimumWidth(400)
        self.setMinimumHeight(400)

        # Create a layout for the window.
        self.layout = QVBoxLayout()

        # Create a label and line edit for the game title.
        self.title_label = QLabel("Enter the title of your game:")
        self.title_input = QLineEdit()

        # Create a label and combo box for the game year.
        self.year_label = QLabel("Enter the year the game was released:")
        self.year_input = QComboBox()
        for year in range(1980, 2023):
            self.year_input.addItem(str(year))

        # Create a label and combo box for the game platform.
        self.platform_label = QLabel("Enter the platform the game was released on:")
        self.platform_input = QComboBox()
        platforms = ["Wii", "WiiU", "NES", "SNES", "N64", "GB", "GC", "DS", "3DS", "XB", "X360", "XOne", "PS", "PSP", "PS2", "PS3", "PS4", "PC", "Atari"]
        for platform in platforms:
            self.platform_input.addItem(platform)

        # Create a label and combo box for the game genre.
        self.genre_label = QLabel("Enter the genre of your game:")
        self.genre_input = QComboBox()
        genres = ["Sports", "Platform", "Racing", "Role-Playing", "Shooter", "Simulation", "Action", "Puzzle", "Fighting", "Misc"]
        for genre in genres:
            self.genre_input.addItem(genre)

        # Create a label and line edit for the game publisher.
        self.publisher_label = QLabel("Enter the publisher of your game:")
        self.publisher_input = QLineEdit()

        # Create a button to submit the user input.
        self.submit_button = QPushButton("Predict Sales")
        self.submit_button.clicked.connect(self.predict)

        # Create a label and line edit to display the prediction.
        self.prediction_label = QLabel("Predicted Global Sales:")
        self.prediction_output = QLineEdit()
        self.prediction_output.setReadOnly(True)

        
        # Create a chart to display the prediction.
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)

        self.ax = self.fig.add_subplot(111)
        self.ax.clear()
        self.ax.plot('Global_Sales', 'Year', 'c.', data=graph_data)
        self.canvas.draw()

        self.layout.addWidget(self.title_label)
        self.layout.addWidget(self.title_input)
        self.layout.addWidget(self.year_label)
        self.layout.addWidget(self.year_input)
        self.layout.addWidget(self.platform_label)
        self.layout.addWidget(self.platform_input)
        self.layout.addWidget(self.genre_label)
        self.layout.addWidget(self.genre_input)
        self.layout.addWidget(self.publisher_label)
        self.layout.addWidget(self.publisher_input)
        self.layout.addWidget(self.submit_button)
        self.layout.addWidget(self.prediction_label)
        self.layout.addWidget(self.prediction_output)

        self.setLayout(self.layout)

    def predict(self):
        title = self.title_input.text()
        year = int(self.year_input.currentText())
        platform = self.platform_input.currentText()
        genre = self.genre_input.currentText()
        publisher = self.publisher_input.text()
    
        # Check if the input data contains any NaN values.
        if any(pd.isnull([title, year, platform, genre, publisher])):
            print("One or more of the input values is invalid. Please try again.")
            return

        game_title_vector = vectorizer.transform([title])

        game_other = pd.DataFrame(data={
            "Year": [year],
            "Platform": [platform],
            "Genre": [genre],
            "Publisher": [publisher]
        })        

        game_other_vectors = encoder.transform(game_other)
        game_vector = hstack([game_title_vector, game_other_vectors])
        game_vector_normalized = scaler.transform(game_vector)

        prediction = model.predict(game_vector_normalized)[0]

        prediction = prediction.round(4)

        self.prediction_output.setText(str(prediction))

        new_data = {'Year': [game_other["Year"]], 'Global_Sales': [prediction]}
        
        color_set = random.choice([ 'b', 'g', 'r', 'm', 'y' ])
        style = '{}.'.format(color_set)

        self.ax.plot('Global_Sales', 'Year', style, data=new_data)
        self.canvas.draw()

def main():
    global model, vectorizer, encoder, scaler

    source = "data/vgsales.csv"                     # We're going to set the path to our data here, just for testing purposes.
    print(f"Reading training data from {source}.")  # We'll print the path as well, just to provide some transparency.
    data = pd.read_csv(source)                      # Here, we actually get around to reading the data from the source file.

    if (data.empty):
        print("Unable to read data source file. Please make sure you have the dataset in the correct position and try again.")
        return

    # Next, we select the variables we want to train on. For now, these are the titles, years, platforms, genres, and publishers
    # of each game.
    X = data[["Name", "Year", "Platform", "Genre", "Publisher"]]
    print("Selecting the data we're going to train on from our dataset...")
    print(X)

    # Then we select the target variable. For now, we're focusing on global scales, but we can add individual countries' sales later.
    y = data["Global_Sales"]

    # Because we want the user to be able to input arbitrary game titles, we need to do some wizardry to the titles.
    game_titles = X["Name"].values.tolist()

    # We have to vectorize the names so that they can be used as features in our model.
    # Since they're not encodable (there aren't a set number of them) and they aren't ints (years), we need to alter them so that
    # we can pass them to our model.
    print("Vectorizing game titles...")
    vectorizer = CountVectorizer()
    vectorizer.fit(game_titles)
    name_vectors = vectorizer.transform(game_titles)

    # Then we can hand the rest of our data to OneHotEncoder.
    # We have to separate our other data from our title data and transform it.
    print("Encoding year, platform, genre, and publisher data...")
    encoder = OneHotEncoder(handle_unknown="ignore")
    x_other = X.drop("Name", axis=1)
    encoder.fit(x_other)
    other_vectors = encoder.transform(x_other)

    # Now we're going to concatenate both back together.
    print("Concatenating data...")
    x_encoded = hstack([name_vectors, other_vectors])

    # And now we're going to normalize the data so that our predictions can be more accurate.
    print("Normalizing data...")
    scaler = MaxAbsScaler()
    scaler.fit(x_encoded)
    x_normalized = scaler.transform(x_encoded)
    
    # We went with a Random Forest model because it provides more accurate predictions
    # for the kind of data we're dealing with.
    print("Training random forest regression model on our data...")
    model = RandomForestRegressor(n_estimators=10)
    x_normalized = x_normalized[:len(y)]
    model.fit(x_normalized, y)
    print("Model trained!")

    # We're also going to calculate the r-squared of the model, just for kicks.
    # I wanted to see how accurate the model was.
    r2 = model.score(x_normalized, y)
    r2 = r2.round(8)
    print(f"R-squared value of the model: {r2}.")

    graph_data = data[["Year", "Global_Sales"]]

    app = QApplication(sys.argv)
    window = MainWindow(graph_data)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()