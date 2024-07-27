# Titanic Dataset Survival Prediction Using Logistic Regression

## Overview

This project involves predicting the survival of passengers on the Titanic using Logistic Regression. The dataset used is from Kaggle and contains information about the passengers such as age, sex, ticket class, and more. The goal is to build a predictive model that can determine whether a passenger survived the Titanic disaster based on these features.

## Dataset

The dataset can be downloaded from [Kaggle's Titanic Dataset](https://www.kaggle.com/c/titanic/data). The dataset includes the following files:
- `train.csv`: The training dataset.
- `test.csv`: The test dataset.

### Data Fields

- `PassengerId`: Unique ID for each passenger.
- `Survived`: Survival (0 = No, 1 = Yes).
- `Pclass`: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd).
- `Name`: Name of the passenger.
- `Sex`: Sex of the passenger.
- `Age`: Age of the passenger.
- `SibSp`: Number of siblings/spouses aboard the Titanic.
- `Parch`: Number of parents/children aboard the Titanic.
- `Ticket`: Ticket number.
- `Fare`: Passenger fare.
- `Cabin`: Cabin number.
- `Embarked`: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

## Project Structure

The repository contains the following files:

- `README.md`: Project overview and setup instructions.
- `Titanic_Survival_Prediction.ipynb`: Jupyter Notebook with the data analysis, preprocessing, model training, and evaluation.
- `train.csv`: Training dataset (on repo or can be downloaded from Kaggle).
- `test.csv`: Test dataset (on repo or can be downloaded from Kaggle).

## Requirements

- Python 3.x
- Jupyter Notebook
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install the required libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Data Preprocessing

The following preprocessing steps were performed:

1. Handling missing values: Filling or dropping missing values in the dataset.
2. Feature engineering: Creating new features or modifying existing ones to improve the model's performance.
3. Encoding categorical variables: Converting categorical variables into numerical values using one-hot encoding.
4. Splitting the data: Dividing the dataset into training and validation sets.

## Model Training

The Logistic Regression model was used for training. The steps involved are:

1. Importing the necessary libraries and the dataset.
2. Preprocessing the data.
3. Splitting the data into training and testing sets.
4. Training the Logistic Regression model.
5. Evaluating the model's performance using metrics such as accuracy, precision, recall, and F1 score.

## Evaluation

The model's performance was evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## Results

The Logistic Regression model provided a baseline for predicting passenger survival. Further improvements can be made by trying different algorithms, feature selection methods, and hyperparameter tuning.

## Conclusion

This project demonstrates how to build a Logistic Regression model to predict the survival of passengers on the Titanic. The steps include data preprocessing, feature engineering, model training, and evaluation. The notebook serves as a starting point for further exploration and improvement.

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/Prathamesh-Palatshaha/Titanic-Dataset-Survival-Prediction-Using-Logistic-Regression.git
   ```

2. Navigate to the project directory:
   ```bash
   cd Titanic-Dataset-Survival-Prediction-Using-Logistic-Regression
   ```

3. Download the dataset from Kaggle and place `train.csv` and `test.csv` in the project directory.

4. Install the required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

5. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Titanic_Survival_Prediction.ipynb
   ```

6. Follow the steps in the notebook to preprocess the data, train the model, and evaluate its performance.

## Acknowledgments

- [Kaggle]([https://www.kaggle.com](https://www.kaggle.com/competitions/titanic/data)) for providing the Titanic dataset.
- [Scikit-learn](https://scikit-learn.org) for the machine learning tools.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to contribute to this project by submitting issues or pull requests. For major changes, please open an issue first to discuss what you would like to change.

Happy coding!
