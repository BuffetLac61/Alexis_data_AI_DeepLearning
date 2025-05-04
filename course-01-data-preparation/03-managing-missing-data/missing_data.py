#!/home/alexis/dev/bin/python3
import pandas as pd                                     # for dataset manipulation (DataFrames)
import numpy as np                                      # allows some mathematical operations
import matplotlib.pyplot as plt                         # library used to display graphs
import seaborn as sns                                   # more convenient visualisation library for dataframes
from sklearn.model_selection import train_test_split    # for classification
from sklearn.neighbors import KNeighborsClassifier      # for classification
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

print("This code is for educational purposes only. It is not intended to be used in a production environment. Use at your own risk.")
input("This code required to be lauched with a csv file named 'diabetes_data.csv' in the same directory. Press enter to continue.")

df = pd.read_csv("diabetes_data.csv")
data_columns = list(df.columns)

# Data Quick View

def data_quickview(df:pd.DataFrame) -> None:
    print("Data Quick View")
    print("--------------")
    print("data sample\n",df.head())
    print("data shape\n",df.shape)
    print("data info\n",df.info())
    print("data description\n",df.describe())
    print("data null values\n",df.isnull().sum())
    print("data duplicates extract\n",df[df.duplicated()])
    print("data columns\n",df.columns)
    print("data types\n",df.dtypes)
    print("data missing values \n", df.isnull().sum())
    return

data_quickview(df)

# Transformation of null values that are actually missing values (for future use because right now knn classifier will not work with NaN values)

def zero_to_nan(df:pd.DataFrame, col_list:list) -> None:
    df[col_list] = df[col_list].replace(0, np.nan)
    print("data missing values \n", df.isnull().sum())
    print("data missing values percentage\n", df.isnull().mean()*100)
    return



# THE FOLLOWING CODE WORKS ONLY IF YOUR TERGET COLUMN IS NAMED "Outcome"
input("warning: the target column must be named 'Outcome' for the following code to work")

# KNN-based classification to evaluate the impact of data preprocessing

def split_data(data):
    X = data.drop("Outcome", axis=1)
    y = data.Outcome
    return train_test_split(X, y,
                            test_size=0.10,   # 10% of the data will be used for testing
                            random_state=42,  # ensures reproducibility of the test
                            stratify=y        # ensures the proportion of ill people is the same in the train and test sets
                            )

def print_knn_score(scores, data_type=""):
    max_score = max(scores)
    k_values_max_score = [i + 1 for i, v in enumerate(scores) if v == max_score]
    print(f'Max {data_type} score {max_score * 100} % for k = {[i for i in k_values_max_score]}')

def diagnosis_knn(data):
    """ KNN-based classification for diabetes diagnosis. """
    X_train, X_test, y_train, y_test = split_data(data)
    test_scores = []
    train_scores = []

    for k in range(1, 15):
        knn = KNeighborsClassifier(k)
        knn.fit(X_train, y_train)
        train_scores.append(knn.score(X_train, y_train))  # "score" for KNN is the accuracy of the classification
        test_scores.append(knn.score(X_test, y_test))

    print_knn_score(train_scores, "train")
    print_knn_score(test_scores, "test")

#Test with no changes made
diagnosis_knn(df)



def create_std_copy(df:pd.DataFrame) -> pd.DataFrame:
    """ Create a copy of the dataframe with the target column excluded. """
    data_columns = list(df.columns)
    scaler = StandardScaler()                                   # create an instance of the scaler
    outcome = df["Outcome"].to_numpy()                          # we use to_numpy() to avoid problems with the index
    df_std = df.drop(["Outcome"], axis=1, inplace=False)        # create a copy excluding target
    df_std = scaler.fit_transform(df_std)                       # fit the scaler WARNING : df_std becomes a numpy array here !
    df_std = pd.DataFrame(df_std, columns=data_columns[:-1])    # transform it back to dataframe
    df_std["Outcome"] = outcome                                 # add the outcome column back
    # -------------------------------------------------------------------------------------------------

    return df_std

print("\nTest with standardization")
print("--------------------------------------------------")
df_std = create_std_copy(df)
diagnosis_knn(df_std)

def create_nrm_copy(df:pd.DataFrame) -> pd.DataFrame:
    """ Create a copy of the dataframe with the target column excluded. """
    data_columns = list(df.columns)
    scaler = Normalizer()                                       # create an instance of the scaler
    outcome = df["Outcome"].to_numpy()                          # we use to_numpy() to avoid problems with the index
    df_nrm = df.drop(["Outcome"], axis=1, inplace=False)        # create a copy excluding target
    df_nrm = scaler.fit_transform(df_nrm)                       # fit the scaler WARNING : df_std becomes a numpy array here !
    df_nrm = pd.DataFrame(df_nrm, columns=data_columns[:-1])    # transform it back to dataframe
    df_nrm["Outcome"] = outcome                                 # add the outcome column back
    # -------------------------------------------------------------------------------------------------

    return df_nrm

print("\nTest with normalization")
print("--------------------------------------------------")
df_nrm = create_nrm_copy(df)
diagnosis_knn(df_nrm)


# Some values are actually missing values (0) and not real values. We will replace them with NaN and replace them before trying knn again
zero_to_nan(df, ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction'])


# Let's try some basic imputation methods to fill the missing values
# 1. Fill with mean
def mean_filling(df:pd.DataFrame) -> pd.DataFrame:
    df_null_to_mean = df.fillna(df.mean())
    return df_null_to_mean

print("\nTest with mean filling")
print("--------------------------------------------------")
df_null_to_mean = mean_filling(df)
diagnosis_knn(df_null_to_mean)
# 2. Fill with median
def median_filling(df:pd.DataFrame) -> pd.DataFrame:
    df_null_to_median = df.fillna(df.median())
    return df_null_to_median

print("\nTest with median filling")
print("--------------------------------------------------")
df_null_to_median = median_filling(df)
diagnosis_knn(df_null_to_median)


# Offer option to the user to fill the missing values with random values selected from a distribution
from scipy.stats import gaussian_kde
import questionary

def random_filling(df):
    df_null_to_complete_random = df.copy()
    for col in df.columns:
        if df[col].isnull().sum()>0:
            response = str(input(f"Do you want to complete {col} missing values with random values? (y/n)"))
            if response == "y":
                print(f"Here is the partition of values for {col}")
                sns.displot(df[col], kind="kde")
                plt.title(f"Distribution of {col}")
                plt.show()
                # Give the user the choice of the numpy function to use with a multiple choice request
                choix_distribution = questionary.select(
                    "Quelle distribution veux-tu utiliser pour remplir les NaNs ?\n1 Distribution uniforme (min → max)\n2 Distribution normale (mean ± std)\n3 Échantillonnage KDE (forme du displot)\n4 Bootstrap (valeurs existantes)",
                    choices=[
                        "1",
                        "2",
                        "3",
                        "4",
                    ]
                ).ask()

                print(f"\nTu as choisi : {choix_distribution}")
                if choix_distribution == "1":
                    print(f"Distribution uniforme entre {df[col].min()} et {df[col].max()}")
                    # Randomly select values from the uniform distribution between min and max
                    max = df[col].max()
                    min = df[col].min()
                    n_null = df[col].isnull().sum()
                    random_values = np.random.uniform(min,max,n_null)
                    df_null_to_complete_random.loc[df[col].isnull(), col] = random_values
                elif choix_distribution == "2":
                    print(f"Distribution normale entre {df[col].mean()} et {df[col].std()}")
                    # Randomly select values from the normal distribution between mean and std
                    mean = df[col].mean()
                    std = df[col].std()
                    n_null = df[col].isnull().sum()
                    random_values = np.random.normal(mean, std, n_null)
                    df_null_to_complete_random.loc[df[col].isnull(), col] = random_values
                elif choix_distribution == "3": 
                    print(f"Échantillonnage KDE entre {df[col].min()} et {df[col].max()}")
                    # Randomly select values from the KDE distribution between min and max
                    # This is a bit tricky, as we need to sample from the KDE
                    # We can use the scipy library to do this
                    n_null = df[col].isnull().sum()
                    kde = gaussian_kde(df[col].dropna())
                    x = np.linspace(df[col].min(), df[col].max(), 100)
                    kde_values = kde(x)
                    # Normalize the KDE values
                    kde_values /= kde_values.sum()
                    # Randomly select values from the KDE distribution
                    random_values = np.random.choice(x, size=n_null, p=kde_values)
                    df_null_to_complete_random.loc[df[col].isnull(), col] = random_values
                elif choix_distribution == "4":
                    print(f"Bootstrap entre {df[col].min()} et {df[col].max()}")
                    # Randomly select values from the existing values
                    n_null = df[col].isnull().sum()
                    x = df[col].dropna().values
                    random_values = np.random.choice(x, size=n_null)
                    df_null_to_complete_random.loc[df[col].isnull(), col] = random_values
                else:
                    print("Invalid choice")
                    continue
            elif response == "n":
                print(f"Skipping {col}")
                continue
    return df_null_to_complete_random

print("\nTest with random filling")
print("--------------------------------------------------")
df_null_to_complete_random = random_filling(df)
diagnosis_knn(df_null_to_complete_random)


print("\nTest with random filling + standardization")
print("--------------------------------------------------")
df_null_to_complete_random_std = create_std_copy(df_null_to_complete_random)
diagnosis_knn(df_null_to_complete_random_std)

print("\nTest with mean filling + standardization")
print("--------------------------------------------------")
df_null_to_mean_std = create_std_copy(df_null_to_mean)
diagnosis_knn(df_null_to_mean_std)