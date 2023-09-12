#!/usr/bin/env python
# coding: utf-8

# # Python AI Project with Earthquake Dataset
# 
# 
# 
# ---
# 
# 
# dataset location: https://www.kaggle.com/datasets/serhatk/turkey-20-years-earthquakes-csv
# 
# 
# 



#import os
#print(os.getcwd())


# 

# ### I. Importing Libraries

# In the beginning in Anaconda Prompt(cmd) we install the libraries to our environment(myenv):
# conda create --name myenv
# conda activate myenv
# conda install pandas numpy matplotlib scikit-learn seaborn tensorflow keras xgboost folium geopandas
# conda install -c conda-forge geopy      # for coordinates
# conda install -c conda-forge basemap    # for maps
# conda install -c conda-forge tabulate   # for printing tables
# conda install -c conda-forge ffmpeg     # for writing to mp4 file
# conda install --channel conda-forge geopandas  # first tried this one after conda install geopandas gave error
# pip install geopandas  #  because 'conda install --channel conda-forge geopandas' wasn't successful, I installed with this code
# pip install scikeras      # for kerasregressor
# 



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, neighbors
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


#conda install pandas numpy matplotlib scikit-learn seaborn tensorflow
#conda install -c conda-forge geopy
#https://anaconda.org/conda-forge/geopy/
#$ conda install numpy scipy pandas matplotlib sympy cython seaborn spyder-kernels pyarrow


# 



# for google colab:
# from google.colab import drive
# drive.mount('/content/drive')


# ## II. Data Preprocessing

# ### Getting dataset into Pandas dataframe



# !pwd




dataset_path = 'TREartquake\eq20.csv' # https://medium.com/geekculture/how-to-load-a-dataset-from-the-google-drive-to-google-colab-67d0478bc634#:~:text=Loading%20a%20Dataset%20from%20the%20Google%20Drive%20to%20Google%20Colab&text=Click%20the%20link%20given.,keys%20for%20your%20google%20drive.&text=Now%20add%20it%20in%20the%20early%20appeared%20shell%20and%20press%20enter.&text=Now%20your%20google%20drive%20is%20mounted.
df = pd.read_csv(dataset_path, encoding="utf_16", on_bad_lines="skip")  #ref: fixing the unicode error; https://www.kaggle.com/code/fatihsen20/fixing-the-unicode-error/notebook
#dataset_path = 'https://www.kaggle.com/datasets/serhatk/turkey-20-years-earthquakes-csv'
df.head()




# Get the number of data points (rows) in the dataset
num_data_points = df.shape[0]

print(f"Number of data points: {num_data_points}")


# Number of data points: 284583






df.describe()




df.isnull().sum() 


# ### Data visualisations:

# Histograms



import matplotlib.pyplot as plt

# Create and save the histogram of earthquake magnitudes
plt.figure(figsize=(10, 6))
plt.hist(df['MAG'], bins=20, color='blue', alpha=0.7)
plt.xlabel('Magnitude')
plt.ylabel('Frequency')
plt.title('Histogram of Earthquake Magnitudes')
plt.grid(True)
plt.savefig('magnitude_histogram.png')
plt.close()

# Create and save the depth distribution visualization
plt.figure(figsize=(10, 6))
plt.hist(df['DEPTH'], bins=20, color='orange', alpha=0.7)
plt.xlabel('Depth')
plt.ylabel('Frequency')
plt.title('Depth Distribution of Earthquakes')
plt.grid(True)
plt.savefig('depth_distribution.png')
plt.close()

# Create and save the geographic distribution map
plt.figure(figsize=(10, 6))
plt.scatter(df['LNG'], df['LAT'], c=df['MAG'], cmap='coolwarm', alpha=0.5)
plt.colorbar(label='Magnitude')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Geographic Distribution of Earthquakes')
plt.grid(True)
plt.savefig('geographic_distribution.png')
plt.close()




import matplotlib.pyplot as plt

# Set the figure size
plt.figure(figsize=(10, 6))

# Plotting the time series graph
plt.plot(df['DATE_'], df['MAG'], marker='o', linestyle='-', color='b', label='Magnitude')

# Adding labels and title
plt.xlabel('Date')
plt.ylabel('Magnitude')
plt.title('Earthquake Magnitude Over Time')

# Adding legend
plt.legend()

# Rotating x-axis labels for better readability
plt.xticks(rotation=45)

# Display the plot
plt.tight_layout()
plt.savefig('MAG_over_time.png')
plt.show()




import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



# Set the Seaborn style
sns.set(style="whitegrid")

# Set the figure size
plt.figure(figsize=(10, 6))

# Create a line plot using Seaborn
sns.lineplot(x='DATE_', y='MAG', data=data, marker='o', label='Magnitude Line')

# Adding labels and title
plt.xlabel('Date')
plt.ylabel('Magnitude')
plt.title('Earthquake Magnitude Over Time (Line Plot)')

# Adding legend
plt.legend()

# Rotating x-axis labels for better readability
plt.xticks(rotation=45)

# Display the plot
plt.tight_layout()
plt.savefig('MAG_over_time_line.png')
plt.show()




import matplotlib.pyplot as plt

# Set the figure size
plt.figure(figsize=(10, 6))

# Creating a scatter plot
plt.scatter(data['DEPTH'], data['MAG'], color='b', alpha=0.5)

# Adding labels and title
plt.xlabel('Depth')
plt.ylabel('Magnitude')
plt.title('Relationship between Depth and Magnitude')

# Display the plot
plt.tight_layout()
plt.savefig('Depth_Magnitude_relationship.png')
plt.show()




import seaborn as sns
import numpy as np

data = df = pd.read_csv(dataset_path, encoding="utf_16", on_bad_lines="skip")  #ref: fixing the unicode error; https://www.kaggle.com/code/fatihsen20/fixing-the-unicode-error/notebook
# Convert the 'DATE_' column to datetime format
data['DATE_'] = pd.to_datetime(data['DATE_'])

# Reshape the data for heatmap
data['Year'] = data['DATE_'].dt.year
heatmap_data = data.pivot_table(values='MAG', index='Month', columns='Year', aggfunc='mean')

# Create a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, cmap='YlOrRd')
plt.title('Heatmap Calendar of Average Earthquake Magnitude')
plt.savefig('MAG_Heatmap_Calendar_of_Average_Earthquake.png')

plt.show()




fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# Distribution
axs[0].hist(data['MAG'], bins=20, color='blue', alpha=0.7)
axs[0].set_xlabel('Magnitude')
axs[0].set_ylabel('Frequency')
axs[0].set_title('Distribution of Earthquake Magnitudes')

# Trend
axs[1].plot(data['DATE_'], data['MAG'], marker='o', color='green')
axs[1].set_xlabel('Date')
axs[1].set_ylabel('Magnitude')
axs[1].set_title('Earthquake Magnitude Trend Over Time')

# Seasonality
monthly_avg = data.groupby(data['DATE_'].dt.month)['MAG'].mean()
axs[2].bar(monthly_avg.index, monthly_avg.values, color='orange')
axs[2].set_xlabel('Month')
axs[2].set_ylabel('Average Magnitude')
axs[2].set_title('Average Earthquake Magnitude by Month')

plt.tight_layout()
plt.show()




# (ref:https://github.com/ageron/handson-ml3/blob/main/02_end_to_end_machine_learning_project.ipynb)
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

df.hist(bins=50, figsize=(20, 9))
plt.show()




# ref:https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
from pandas.plotting import scatter_matrix

plt.figure(figsize=(23, 11)) 

# box and whisker plots
df.plot(kind='box', subplots=True, layout=(2,3), sharex=False, sharey=False)
plt.show()

# scatter plot matrix
scatter_matrix(df, figsize=(15, 7))
plt.show()




# (ref: https://gist.githubusercontent.com/amankharwal/b0064f6749d1ca82209840fdddccaa33/raw/93520b1131399072b939a9967ee84caf799ee16b/earthquake.py):

from mpl_toolkits.basemap import Basemap
import cartopy.crs as ccrs
import cartopy

#ax = plt.axes(projection=ccrs.PlateCarree())
#ax.set_extent([-180, 180, -80, 80], crs=ccrs.PlateCarree())



m = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')

longitudes = df["LNG"].tolist()
latitudes = df["LAT"].tolist()
#m = Basemap(width=12000000,height=9000000,projection='lcc',
            #resolution=None,lat_1=80.,lat_2=55,lat_0=80,lon_0=-107.)
x,y = m(longitudes,latitudes)

fig = plt.figure(figsize=(17,11))
plt.title("All affected areas")
#ax.coastlines()


m.plot(x, y, "o", markersize = 2, color = 'blue')
m.drawcoastlines()
m.fillcontinents(color='coral',lake_color='aqua')
m.drawmapboundary()
m.drawcountries()

# Plot the data on the map
#ax.plot(df["LNG"], df["LAT"], "o", markersize=2, color='blue', transform=ccrs.PlateCarree())

plt.show()




from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

# Create a Basemap object
m = Basemap(projection='mill', llcrnrlat=33, urcrnrlat=45, llcrnrlon=21, urcrnrlon=46, resolution='l')

# Convert coordinates to map projection
longitudes = df["LNG"].tolist()
latitudes = df["LAT"].tolist()
x, y = m(longitudes, latitudes)

# Plot the map
fig = plt.figure(figsize=(12, 10))
plt.title("Affected Areas in Turkey")
m.plot(x, y, "o", markersize=3, color='blue')
m.drawcoastlines()
m.fillcontinents(color='coral', lake_color='aqua')
m.drawmapboundary()
m.drawcountries()

plt.show()


# ### Simulating missing data



np.random.seed(42) #setting a random seed to simulate missing data randomly
percentage_missing = 0.05  #missing data percentage %5




#making a random mask for values to be chosen for missing:
random_mask = np.random.random(df.shape) < percentage_missing
#To set selected values as missing and replace them with NaN:
df[random_mask] = np.nan




print(df.isnull().sum())


# Now we have %5 percent null NaN values

# ### Handling Missing Data

# #### Handle missing MAG values



print(df.isnull().sum())




get_ipython().run_cell_magic('time', '', "df.dropna(subset=['MAG'], inplace=True)\n# df['MAG'] = df['MAG'].interpolate(method='polynomial', order=5)\n")


# #### Handle Missing DATE values

# Drop rows with missing DATE's:



get_ipython().run_cell_magic('time', '', "df.dropna(subset=['DATE_'], inplace=True)\nprint(df.isnull().sum())\ndf.info()\n")


# Convert the DATE_ values to datetime format:



# Convert DATE_ values to datetime format:
df['DATE_'] = pd.to_datetime(df['DATE_'])




df.info()


# #### Handle missing LOCATION, LAT, and LNG values:
# 

# With nan LOCATION, LAT and LNG column values are interdependent on each other. So it would be good to create a var with location names depending on LAT and LNG:



location_data = df[['LOCATION_', 'LAT', 'LNG']].dropna().drop_duplicates()




location_data_dict = {}
for _, row in location_data.iterrows():
    location_data_dict[(row['LAT'], row['LNG'])] = row['LOCATION_']


# Check any of three values are missing row by row, and if any look for it in location_data and fill the missing values accordingly:
# 



def fill_missing_values_LOC_LAT_LNG(row):
    if row[['LOCATION_', 'LAT', 'LNG']].isna().any():
        location = row['LOCATION_']
        lat, lng = row['LAT'], row['LNG']
                
        if pd.notna(lat) and pd.notna(lng):
            if pd.isna(row['LOCATION_']):
                # Set location based on lat and lng
                subset = location_data_dict.get((lat, lng))
                if subset:
                    location = subset
        
        if pd.isna(lat) or pd.isna(lng):
            # Retrieve lat, lng based on location if available
            if pd.notna(location):
                for key, value in location_data_dict.items():
                    if value == location:
                        lat, lng = key
                        break

        
        # Update the missing values in the row
        row['LOCATION_'] = location if pd.isna(row['LOCATION_']) else row['LOCATION_']
        row['LAT'] = lat if pd.isna(row['LAT']) else row['LAT']
        row['LNG'] = lng if pd.isna(row['LNG']) else row['LNG']
        
    return row




def fill_missing_values_LOCgeocode_LAT_LNG(row):
    if pd.isna(row['LOCATION_']) or pd.isna(row['LAT']) or pd.isna(row['LNG']):
        location = row['LOCATION_']
        lat, lng = row['LAT'], row['LNG']

        if pd.isna(row['LOCATION_']):
            if pd.notna(lat) and pd.notna(lng):
                # Reverse geocode to obtain location based on lat and lng
                location = reverse_geocode(lat, lng)

        # Update the missing values in the row
        row['LOCATION_'] = location if pd.isna(row['LOCATION_']) else row['LOCATION_']

        if pd.isna(lat) or pd.isna(lng):
            # Retrieve lat, lng based on location if available
            if pd.notna(location):
                subset = location_data[(location_data['LOCATION_'] == location) & pd.notna(location_data['LAT']) & pd.notna(location_data['LNG'])]
                if not subset.empty:
                    lat, lng = subset.iloc[0]['LAT'], subset.iloc[0]['LNG']
        
        # Update the missing values in the row
        row['LAT'] = lat if pd.isna(row['LAT']) else row['LAT']
        row['LNG'] = lng if pd.isna(row['LNG']) else row['LNG']
        
    return row




# !pip install geopy




# ref.; https://geopy.readthedocs.io/en/latest/:
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent='my-app') 

def reverse_geocode(lat, lng):
    location = ''
    try:
        # Use geolocator to reverse geocode the coordinates to location
        address = geolocator.reverse((lat, lng), exactly_one=True)
        if address:
            raw_address = address.raw['address']
            # print(raw_address)            
            # Determine the biggest component (village, county, suburb, neighbourhood)
            components = ['village', 'neighbourhood', 'suburb', 'county']
            biggest_component = None
            for component in components:
                if component in raw_address:
                    biggest_component = component
                    break
            
            city = raw_address.get('province', '')
            state = raw_address.get('state', '')
            town = raw_address.get('town', '')
            region = raw_address.get('region', '')
            # suburb = raw_address.get('suburb', '')
            # county = raw_address.get('county', '')
            # village = raw_address.get('village', '')
            # neighbourhood = raw_address.get('neighbourhood', '')
            # Define the hierarchy of location components
            location_parts = []
            if biggest_component:
                location_parts.append(raw_address[biggest_component])
            if town:
                location_parts.append(town)
            elif region:
                location_parts.append(region)
                      
            location = '-'.join(location_parts)
            if city != '':              
                location = f"{location} ({city})"
            elif state != '':
                location = f"{location} ({state})"
    except:
        pass
    return location.upper() #to be compatible with database and make all capital





def handle_still_missing_values_LOC_LAT_LNG():         
  # Handle case where location is still missing            
  df.dropna(subset=['LOCATION_'], inplace=True)                
  
  # Handle case where lat is still missing  
  df.dropna(subset=['LAT'], inplace=True)
  
  # Handle case where lng is still missing    
  df.dropna(subset=['LNG'], inplace=True)
            
  return df




print(df.isnull().sum())




get_ipython().run_cell_magic('time', '', 'df = df.apply(fill_missing_values_LOC_LAT_LNG, axis=1)\n')


# 

# CPU times: total: 3min 28s
# Wall time: 3min 28s



print(df.isnull().sum())




# %%time
df = df.apply(fill_missing_values_LOCgeocode_LAT_LNG, axis=1)




get_ipython().run_cell_magic('time', '', 'df = handle_still_missing_values_LOC_LAT_LNG()\n')




print(df.isnull().sum())




get_ipython().run_cell_magic('time', '', "# !nvidia-smi -L\n# !pip install tensorflow\nimport tensorflow as tf\nprint(tf.config.list_physical_devices('GPU'))\n")


# 

# #### Fill missing ID values:



df.head()




get_ipython().run_cell_magic('time', '', "def fill_missing_ID(df):\n    df = df.sort_values('ID')\n    df['prev_id'] = df['ID'].shift(1)\n    df['next_id'] = df['ID'].shift(-1)\n    \n    for index, row in df.iterrows():\n        current_id = row['ID']\n        if pd.isna(current_id):\n            prev_id = row['prev_id']\n            next_id = row['next_id']\n            \n            if pd.notna(prev_id):\n                df.at[index, 'ID'] = prev_id + 1\n            elif pd.notna(next_id):\n                df.at[index, 'ID'] = next_id - 1\n    \n    df = df.drop(['prev_id', 'next_id'], axis=1)\n    df['ID'] = df['ID'].fillna(method='ffill').fillna(method='bfill') \n    return df\n\ndf = fill_missing_ID(df)\n")




print(df.isnull().sum())



df.info()


df.head()


# #### Handle Missing RECORDDATE dates:



get_ipython().run_cell_magic('time', '', "df['DATE_numeric'] = df['RECORDDATE']\nprint(df.isnull().sum())\n\n# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html\ndf['RECORDDATE'] = df['RECORDDATE'].interpolate(method='pad', limit=2)\nprint(df.isnull().sum())\ndf.head()\n")




# Convert the interpolated RECORDDATE values to datetime format
df['RECORDDATE'] = pd.to_datetime(df['RECORDDATE'], format='mixed')

# Remove the temporary numeric column
df.drop('DATE_numeric', axis=1, inplace=True)


# Drop still missing RECORDDATE valued rows:



df.dropna(subset=['RECORDDATE'], inplace=True)




print(df.isnull().sum())





# #### Fill missing DEPTH values



df['DEPTH'] = df['DEPTH'].interpolate(method='polynomial', order=2)


print(df.isnull().sum())


# We handled all the missing values
# 

# ### Correlation Matrix:



import seaborn as sns

# Calculate the correlation matrix
correlation_matrix = df[['LAT', 'LNG', 'MAG', 'DEPTH']].corr()

# Plot the correlation matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


# ## III. Splitting the Data

# - Our problem here is earthquake magnitude prediction, so that we would define magnitude as our target variable.



# input variables - features:
X = df.drop('MAG', axis=1).copy()  #axis is 1 to drop target var column
# Output variable - Target variable:
y = df['MAG']

# https://towardsdatascience.com/how-to-split-data-into-three-sets-train-validation-and-test-and-why-e50d22d3e54c
#Splitting into train and test sets which is %89 train(to be further split) and %11 test:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.11, random_state=42)
#And then continue splitting Train set into train and validation sets which is (0.89*0.88~=)%78 train and (0.89*0.12~=)%11 validation:
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.12, random_state=42)
print(X_train.shape), print(y_train.shape)
print(X_val.shape), print(y_val.shape)
print(X_test.shape), print(y_test.shape)


# We have split our dataset into %78 train, %11 test and %11 validation sets


get_ipython().run_cell_magic('time', '', '\nfrom sklearn.model_selection import KFold, GridSearchCV\nfrom sklearn.neural_network import MLPRegressor\nfrom sklearn.metrics import mean_squared_error\nimport numpy as np\n\n# Define the number of folds\nnum_folds = 5\n\n# Create a KFold cross-validation splitter\nkf = KFold(n_splits=num_folds)\n\n# Define the hyperparameter search space\nparam_space = {\n    \'hidden_layer_sizes\': [(50,), (100,), (50, 25), (100, 50)],\n    \'activation\': [\'relu\', \'tanh\'],\n    \'max_iter\': [50, 100, 200]\n}\n\n# Initialize a list to store the cross-validation scores\ncv_scores = []\n\nnumerical_features = [\n    "LAT",\n    "LNG",\n    "DEPTH",\n]\nX_train_numerical = X_train[numerical_features]\nX_val_numerical = X_val[numerical_features]\nX_test_numerical = X_val[numerical_features]\n\n\n# Iterate through the cross-validation splits\nfor train_index, val_index in kf.split(X_train_numerical):\n    X_train_fold, X_val_fold = X_train_numerical.iloc[train_index], X_train_numerical.iloc[val_index]\n    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]\n    \n    # Initialize the MLPRegressor with a set of hyperparameters\n    model = MLPRegressor(hidden_layer_sizes=(50,), activation=\'relu\', max_iter=100)\n    \n    # Train the model on the training fold\n    model.fit(X_train_fold, y_train_fold)\n    \n    # Make predictions on the validation fold\n    y_pred_val = model.predict(X_val_fold)\n    \n    # Calculate the mean squared error for this fold\n    mse_fold = mean_squared_error(y_val_fold, y_pred_val)\n    \n    cv_scores.append(mse_fold)\n\n# Calculate the average cross-validation score\naverage_cv_score = np.mean(cv_scores)\nprint("Average Cross-Validation MSE:", average_cv_score)\n\n\n# Initialize the GridSearchCV with the MLPRegressor model and parameter space\ngrid_search = GridSearchCV(MLPRegressor(), param_grid=param_space, cv=kf)\n\n# Fit the GridSearchCV on the training data\ngrid_search.fit(X_train_numerical, y_train)\n\n# Get the best model from the GridSearchCV\nbest_model_grid = grid_search.best_estimator_\n\n# Make predictions on the validation set using the best model\ny_pred_val = best_model_grid.predict(X_val)\n\n# Calculate the mean squared error on the validation set\nval_mse = mean_squared_error(y_val, y_pred_val)\nprint("Validation MSE:", val_mse)\n\n# Evaluate the best model on the test set\ny_pred_test = best_model_grid.predict(X_test)\ntest_mse = mean_squared_error(y_test, y_pred_test)\nprint("Test MSE:", test_mse)\n\n')


# ## IV. Feature Engineering

# ### Handling Categorical Data = LOCATION_

# We have 'LOCATION_' column consisting of string object values, which we have to translate to numerical data.

# We will use target encoding to convert nominal data column LOCATION_ to numerical. But first we are going to compare target encoder with other encoders: 

# #### Comparing Target Encoder with Other encoders (ref: https://scikit-learn.org/stable/auto_examples/preprocessing/plot_target_encoder.html#comparing-target-encoder-with-other-encoders)

# The TargetEncoder uses the value of the target to encode each categorical feature. We will compare 3 different encoding methods; TargetEncoder, OrdinalEncoder, OneHotEncoder and dropping the category.


numerical_features = [
    "LAT",
    "LNG",
    "DEPTH",
]

categorical_features = ["LOCATION_"]
target_name = "MAG"

_ = y.hist()




df['DATE_'] = pd.to_datetime(df['DATE_'])
df['RECORDDATE'] = pd.to_datetime(df['DATE_'])
df.info()




import sklearn.preprocessing
print(sklearn.__version__)
print(dir(sklearn.preprocessing))




from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder

categorical_preprocessors = [
    ("drop", "drop"),
    ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    (
        "one_hot",
        OneHotEncoder(handle_unknown="ignore", max_categories=20, sparse_output=False),
    ),
    ("target", TargetEncoder(target_type="continuous")),
]


# Now we evaluate the models using cross validation and record the results:


from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline

n_cv_folds = 3
max_iter = 20
results = []


def evaluate_model_and_store(name, pipe):
    result = cross_validate(
        pipe,
        X,
        y,
        scoring="neg_root_mean_squared_error",
        cv=n_cv_folds,
        return_train_score=True,
    )
    rmse_test_score = -result["test_score"]
    rmse_train_score = -result["train_score"]
    results.append(
        {
            "preprocessor": name,
            "rmse_test_mean": rmse_test_score.mean(),
            "rmse_test_std": rmse_train_score.std(),
            "rmse_train_mean": rmse_train_score.mean(),
            "rmse_train_std": rmse_train_score.std(),
        }
    )


for name, categorical_preprocessor in categorical_preprocessors:
    preprocessor = ColumnTransformer(
        [
            ("numerical", "passthrough", numerical_features),
            ("categorical", categorical_preprocessor, categorical_features),
        ]
    )
    pipe = make_pipeline(
        preprocessor, HistGradientBoostingRegressor(random_state=0, max_iter=max_iter)
    )
    evaluate_model_and_store(name, pipe)


# Root Mean Square Error (RMSE) is the standard deviation of the residuals (prediction errors). Residuals are a measure of how far from the regression line data points are; RMSE is a measure of how spread out these residuals are. In other words, it tells you how concentrated the data is around the line of best fit. Root mean square error is commonly used in climatology, forecasting, and regression analysis to verify experimental results.

# We build and evaluate a pipeline that uses native categorical feature support in HistGradientBoostingRegressor, which only supports up to 255 unique categories. In our dataset, the categorical feature which is LOCATION_ has more than 255 unique categories:



n_unique_categories = df[categorical_features].nunique().sort_values(ascending=False)
n_unique_categories


# To workaround the limitation above, The high cardinality feature which is LOCATION_ will be target encoded.



high_cardinality_features = n_unique_categories[n_unique_categories > 255].index
low_cardinality_features = n_unique_categories[n_unique_categories <= 255].index
mixed_encoded_preprocessor = ColumnTransformer(
    [
        ("numerical", "passthrough", numerical_features),
        (
            "high_cardinality",
            TargetEncoder(target_type="continuous"),
            high_cardinality_features,
        ),
        (
            "low_cardinality",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            low_cardinality_features,
        ),
    ],
    verbose_feature_names_out=False,
)

# The output of the preprocessor must be set to pandas so the gradient boosting model can detect the low cardinality features.
mixed_encoded_preprocessor.set_output(transform="pandas")
mixed_pipe = make_pipeline(
    mixed_encoded_preprocessor,
    HistGradientBoostingRegressor(
        random_state=0, max_iter=max_iter, categorical_features=low_cardinality_features
    ),
)
mixed_pipe


# Finally, we evaluate the pipeline using cross validation and record the results:



evaluate_model_and_store("mixed_target", mixed_pipe)


# ##### Plotting the Results:

# We display the results by plotting the test and train scores:




results_df = (
    pd.DataFrame(results).set_index("preprocessor").sort_values("rmse_test_mean")
)

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(12, 8), sharey=True, constrained_layout=True
)
xticks = range(len(results_df))
name_to_color = dict(
    zip((r["preprocessor"] for r in results), ["C0", "C1", "C2", "C3", "C4"])
)

for subset, ax in zip(["test", "train"], [ax1, ax2]):
    mean, std = f"rmse_{subset}_mean", f"rmse_{subset}_std"
    data = results_df[[mean, std]].sort_values(mean)
    ax.bar(
        x=xticks,
        height=data[mean],
        yerr=data[std],
        width=0.9,
        color=[name_to_color[name] for name in data.index],
    )
    ax.set(
        title=f"RMSE ({subset.title()})",
        xlabel="Encoding Scheme",
        xticks=xticks,
        xticklabels=data.index,
    )


# - RMSE is the least on Target encodings, so that we choose Target encoding to encode LOCATION_ data column.

# 
# #### Using Target Encoding to convert nominal data column LOCATION_ to numerical



X_test




from sklearn.preprocessing import TargetEncoder

encoder = TargetEncoder()
encoder.fit(X_train[['LOCATION_']], y_train)  # Reshape to a 2d array because scikit-learn's TargetEncoder expects the input to be a 2D array


X_train['LOCATION_'] = encoder.transform(X_train[['LOCATION_']]) # Reshape to a 2D array
X_val['LOCATION_'] = encoder.transform(X_val[['LOCATION_']]) # Reshape to a 2D array

# Get the unique encoded values and their corresponding means (to be later used for real LOCATION_ transformation from encoded LOCATION_ values)
location_encoded_values = X_test['LOCATION_']
print("location values: ", location_encoded_values)
X_test['LOCATION_'] = encoder.transform(X_test[['LOCATION_']]) # Reshape to a 2D array
mean_encoded_values = X_test['LOCATION_']  #(to be later used for real LOCATION_ transformation from encoded LOCATION_ values)


# Create the location_mapping dictionary  (to be later used for real LOCATION_ transformation from encoded LOCATION_ values)
location_mapping = dict(zip(location_encoded_values, mean_encoded_values))

# Print the location_mapping dictionary (to be later used for real LOCATION_ transformation from encoded LOCATION_ values)
print(location_mapping)



location_encoded_values



X_test



X_test["LOCATION_"]


# ### 5-fold cross-validation


X_train




get_ipython().run_cell_magic('time', '', '# Execute cross validation KFold again after handling categorical LOCATION_ column:\n\nfrom sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV\nfrom sklearn.neural_network import MLPRegressor\nfrom sklearn.metrics import mean_squared_error\nimport numpy as np\nimport time\n\n# Define the number of folds\nnum_folds = 5\n\n# Create a KFold cross-validation splitter\nkf = KFold(n_splits=num_folds)\n\n# Define the hyperparameter search space\nparam_space = {\n    \'hidden_layer_sizes\': [(50,), (100,), (50, 25), (100, 50)],\n    \'activation\': [\'relu\', \'tanh\'],\n    \'max_iter\': [50, 100, 200]\n}\n\n# Initialize a list to store the cross-validation scores\ncv_scores = []\n\nnumerical_features_locationadded = [\'LAT\', \'LNG\', \'LOCATION_\', \'DEPTH\']\n\nX_train_numerical = X_train[numerical_features_locationadded]\nX_val_numerical = X_val[numerical_features_locationadded]\nX_test_numerical = X_val[numerical_features_locationadded]\n\nt1 = time.time()\n# Iterate through the cross-validation splits\nfor train_index, val_index in kf.split(X_train_numerical):\n    X_train_fold, X_val_fold = X_train_numerical.iloc[train_index], X_train_numerical.iloc[val_index]\n    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]\n    \n    # Initialize the MLPRegressor with a set of hyperparameters\n    model = MLPRegressor(hidden_layer_sizes=(50,), activation=\'relu\', max_iter=100)\n    \n    # Train the model on the training fold\n    model.fit(X_train_fold, y_train_fold)\n    \n    # Make predictions on the validation fold\n    y_pred_val = model.predict(X_val_fold)\n    \n    # Calculate the mean squared error for this fold\n    mse_fold = mean_squared_error(y_val_fold, y_pred_val)\n    \n    cv_scores.append(mse_fold)\n\nt2 = time.time()\nprint(f"MLPRegressor regression Iterating through the cross-validation splits took {t2 - t1} seconds")\n# Calculate the average cross-validation score\naverage_cv_score = np.mean(cv_scores)\nprint("Average Cross-Validation MSE:", average_cv_score)\n\n\n\n# Initialize the GridSearchCV with the MLPRegressor model and parameter space\nrandom_search = RandomizedSearchCV(MLPRegressor(), param_distributions=param_space, cv=kf, verbose=2)\n\nt1 = time.time()\n# Fit the GridSearchCV on the training data\nrandom_search.fit(X_train_numerical, y_train)\nt2 = time.time()\nprint(f"RandomizedSearchCV on MLPRegressor took {t2 - t1} seconds")\n\n# Get the best model from the GridSearchCV\nbest_model_random = random_search.best_estimator_\n\n# Make predictions on the validation set using the best model\ny_pred_val = best_model_random.predict(X_val_numerical)\n\n# Calculate the mean squared error on the validation set\nval_mse = mean_squared_error(y_val, y_pred_val)\nprint("Validation MSE after hyperparameter tuning:", val_mse)\n\n# Evaluate the best model on the test set\ny_pred_test = best_model_random.predict(X_test_numerical)\ntest_mse = mean_squared_error(y_test, y_pred_test)\nprint("Test MSE after hyperparameter tuning:", test_mse)\n')


# Average Cross-Validation MSE: 0.20850262658856758


print(X_train_numerical.shape, y_train.shape)
print(X_val_numerical.shape, y_val.shape)
print(X_test_numerical.shape, y_test.shape)


# ### Scale numerical features (Normalization and Scaling)



X_train[numerical_features_locationadded]


# #### with normalizing:



import sklearn.preprocessing as preprocessing
from sklearn.preprocessing import StandardScaler

#scaler = MinMaxScaler()
#numerical_features = ['LAT', 'LNG', 'DEPTH']

#numerical_features = ['DEPTH']

#X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
#X_val[numerical_features] = scaler.transform(X_val[numerical_features])
#X_test[numerical_features] = scaler.transform(X_test[numerical_features])


# Step1: Extract the non-numerical features and create df.s
non_numerical_features = ['ID', 'DATE_', 'RECORDDATE']
X_train_non_numerical = X_train[non_numerical_features]
X_train_non_numerical_df = pd.DataFrame(X_train_non_numerical, columns=non_numerical_features)
X_val_non_numerical = X_val[non_numerical_features]
X_val_non_numerical_df = pd.DataFrame(X_val_non_numerical, columns=non_numerical_features)
X_test_non_numerical = X_test[non_numerical_features]
X_test_non_numerical_df =  pd.DataFrame(X_test_non_numerical, columns=non_numerical_features)

# Step2a: Normalize the numerical features in X_train, X_val, and X_test
# X_train_normalized_experiment = preprocessing.normalize(X_train[numerical_features_locationadded], norm='l2')
scaler = StandardScaler()
numerical_features = ['LAT', 'LNG', 'DEPTH']
numerical_features_locationadded = ['LAT', 'LNG', 'LOCATION_', 'DEPTH']

X_train_normalized = preprocessing.normalize(X_train[numerical_features_locationadded], norm='l2')
X_val_normalized = preprocessing.normalize(X_val[numerical_features_locationadded], norm='l2')
X_test_normalized = preprocessing.normalize(X_test[numerical_features_locationadded], norm='l2')

# Step2b: Standard scale the numerical features in X_train, X_val, and X_test
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_normalized)
X_val_scaled = scaler.transform(X_val_normalized)
X_test_scaled = scaler.transform(X_test_normalized)

#X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
#X_val[numerical_features] = scaler.transform(X_val[numerical_features])
#X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# Step3: Create dataframes for scaled features with the original indices:
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=numerical_features_locationadded, index=X_train.index)
X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=numerical_features_locationadded, index=X_val.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=numerical_features_locationadded, index=X_test.index)

# We have equaled indices of X_train_scaled_df, X_val_scaled_df, and X_test_scaled_df with the nonnumerical df indexes and they are equal to original so that they will be correctly alligned

# Step4: Concatenate non-numerical features with scaled and original-indiced numerical features
X_train_final = pd.concat([X_train_non_numerical_df, X_train_scaled_df], axis=1)
X_val_final = pd.concat([X_val_non_numerical_df, X_val_scaled_df], axis=1)
X_test_final = pd.concat([X_test_non_numerical_df, X_test_scaled_df], axis=1)




#Print the shape of X_test_final and X_test_scaled
print("Shape of X_test:", X_test.shape)
print("Shape of X_test_final:", X_test_final.shape)
print("Shape of X_test_scaled:", X_test_scaled.shape)
print("Shape of X_test_non_numerical_df:", X_test_non_numerical_df.shape)
print("Shape of X_test_scaled_df:", X_test_scaled_df.shape)




# Check if the indices match between X_test_non_numerical_df and X_test_scaled_df
print("Indices match:", X_test_non_numerical_df.index.equals(X_test_scaled_df.index))





# ### Apply PCA for Dimensionality Reduction


get_ipython().run_cell_magic('time', '', "import sklearn.preprocessing as preprocessing\n\nnumerical_features_locationadded = ['LAT', 'LNG', 'LOCATION_', 'DEPTH']\n\nscaler = preprocessing.MinMaxScaler()\n\nX_train_norm = scaler.fit_transform(X_train[numerical_features_locationadded])\nX_val_norm = scaler.transform(X_val[numerical_features_locationadded])\nX_test_norm = scaler.transform(X_test[numerical_features_locationadded])\n\n# Create dataframes for normalized features with the original indices:  (for l2 normalization we can use: X_train_normalized = preprocessing.normalize(X_train,norm='l2)') )\n\nX_train_norm_df = pd.DataFrame(X_train_norm, columns=numerical_features_locationadded, index=X_train.index)\nX_val_norm_df = pd.DataFrame(X_val_norm, columns=numerical_features_locationadded, index=X_val.index)\nX_test_norm_df = pd.DataFrame(X_test_norm, columns=numerical_features_locationadded, index=X_test.index)\n")




# #### Choosing the number of the components to retain:


from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

numerical_features = ['LAT', 'LNG', 'DEPTH']
numerical_features_locationadded = ['LAT', 'LNG', 'LOCATION_', 'DEPTH']

pca = PCA(n_components=None)
pca.fit(X_train_norm_df)
explained_variance = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance)


plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance')
plt.title('Scree Plot')
plt.show()

# Visualize the cumulative explained variance
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance Plot')
plt.show()



print(explained_variance)


# on 3 components, cumulative explained variance reaches to almost full-%99. But on 2 components, it is also around %90. So that we are continuing with n_components=3

# #### Applying PCA:



pca = PCA(n_components=3)  # Number of components we want to retain

X_train_pca = pca.fit_transform(X_train_norm_df)
X_val_pca = pca.transform(X_val_norm_df)
X_test_pca = pca.transform(X_test_norm_df)

# Create dataframes for scaled features with the original indices:
X_train_pca_df = pd.DataFrame(X_train_pca, columns=['Principal Component 1', 'Principal Component 2', 'Principal Component 3'], index=X_train.index)
X_val_pca_df = pd.DataFrame(X_val_pca, columns=['Principal Component 1', 'Principal Component 2', 'Principal Component 3'], index=X_val.index)
X_test_pca_df = pd.DataFrame(X_test_pca, columns=['Principal Component 1', 'Principal Component 2', 'Principal Component 3'], index=X_test.index)

# Visualize the transformed data:


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_train_pca_df['Principal Component 1'], X_train_pca_df['Principal Component 2'], X_train_pca_df['Principal Component 3'], label='Train')
ax.scatter(X_val_pca_df['Principal Component 1'], X_val_pca_df['Principal Component 2'], X_val_pca_df['Principal Component 3'], label='Validation')
ax.scatter(X_test_pca_df['Principal Component 1'], X_test_pca_df['Principal Component 2'], X_test_pca_df['Principal Component 3'], label='Test')

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('PCA Scatter Plot (3D)')
ax.legend()
plt.show()



import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.scatter(X_train_pca_df['Principal Component 1'], X_train_pca_df['Principal Component 2'], label='Train')
plt.scatter(X_val_pca_df['Principal Component 1'], X_val_pca_df['Principal Component 2'], label='Validation')
plt.scatter(X_test_pca_df['Principal Component 1'], X_test_pca_df['Principal Component 2'], label='Test')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Scatter Plot')
plt.legend()
plt.show()


# ### 5-fold cross-validation



from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
import numpy as np

model = MLPRegressor(hidden_layer_sizes=(50,), activation='relu', max_iter=100)

# Create a 5-fold cross-validation splitter
kf = KFold(n_splits=5)

# Initialize a list to store evaluation metrics (e.g., RMSE)
metrics = []
metrics_mse = []

# Iterate through the cross-validation splits
for train_index, val_index in kf.split(X_train_scaled_df):
    X_train_fold, X_val_fold = X_train_scaled_df.iloc[train_index], X_train_scaled_df.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    
    # Train the model
    model.fit(X_train_fold, y_train_fold)
    
    # Make predictions on validation data
    y_pred_fold = model.predict(X_val_fold)
    
    # Calculate evaluation metric (e.g., RMSE)
    mse_fold = mean_squared_error(y_val_fold, y_pred_fold)
    rmse_fold = np.sqrt(mse_fold)
    
    
    # Store the metric for this fold
    metrics.append(rmse_fold)
    metrics_mse.append(mse_fold)

# Calculate the average RMSE across all folds
average_rmse = sum(metrics) / len(metrics)
average_mse = sum(metrics_mse) / len(metrics_mse)
print("Average RMSE:", average_rmse)
print("Average MSE:", average_mse)


# Average RMSE: 0.4562060318860066
# Average MSE: 0.2081284418853123
# 

#  ## V. Probability Distribution, Correlation Matrix

# Prabability distributions of numerical features:


import matplotlib.pyplot as plt

# Plot histograms of numerical features
plt.figure(figsize=(12, 6))
plt.subplot(2, 3, 1)
plt.hist(X_train['LAT'], bins=20)
plt.title('LAT Probability Distribution')

plt.subplot(2, 3, 2)
plt.hist(X_train['LNG'], bins=20)
plt.title('LNG Probability Distribution')

plt.subplot(2, 3, 3)
plt.hist(X_train['DEPTH'], bins=20)
plt.title('DEPTH Probability Distribution')

plt.subplot(2, 3, 4)
plt.hist(X_train['LOCATION_'], bins=20)
plt.title('LOCATION_ Probability Distribution')

plt.subplot(2, 3, 5)
plt.hist(y, bins=20)
plt.title('MAG Probability Distribution')

plt.tight_layout()
plt.show()


#  Negative depth values most probably indicates events above the reference point or sea level.

# Correlation Matrix of numerical features:


import seaborn as sns

# Calculate the correlation matrix
correlation_matrix = X_train[numerical_features_locationadded].corr()

# Plot the correlation matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


import seaborn as sns

# Combine the numerical features and the target variable (MAG) into a single DataFrame
data_with_mag = X_train[numerical_features_locationadded].copy()
data_with_mag['MAG'] = y_train

# Calculate the correlation between the features and the target variable
correlation_with_mag = data_with_mag.corr()

# Print the correlation values
print(correlation_with_mag['MAG'])


# Scatter plots for correlations against target variable MAG:
# Create scatter plots for each numerical feature against the target variable (MAG)
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
sns.scatterplot(data=data_with_mag, x='LAT', y='MAG')
plt.title('Scatter Plot: LAT vs MAG')

plt.subplot(2, 2, 2)
sns.scatterplot(data=data_with_mag, x='LNG', y='MAG')
plt.title('Scatter Plot: LNG vs MAG')

plt.subplot(2, 2, 3)
sns.scatterplot(data=data_with_mag, x='DEPTH', y='MAG')
plt.title('Scatter Plot: DEPTH vs MAG')

plt.subplot(2, 2, 4)
sns.scatterplot(data=data_with_mag, x='LOCATION_', y='MAG')
plt.title('Scatter Plot: LOCATION_ vs MAG')

plt.tight_layout()
plt.show()


# Correlation Heatmap: correlation with the target variable


# Correlation Heatmap: correlation with the target variable

# Calculate the correlation matrix with the target variable
correlation_matrix_with_target = data_with_mag.corr()

# Plot the correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_with_target, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap with Target Variable (MAG)')
plt.show()


# Pairplot


get_ipython().run_cell_magic('time', '', "# Create a pairplot\nsns.pairplot(data=data_with_mag, vars=numerical_features_locationadded + ['MAG'], diag_kind='kde')\nplt.show()\n")


# Joint Distribution Plot




get_ipython().run_cell_magic('time', '', "import matplotlib.pyplot as plt\n\n# Create jointplot for LOCATION_ vs MAG\nplt.figure(figsize=(6, 6))\njointplot_location = sns.jointplot(data=data_with_mag, x='LOCATION_', y='MAG', kind='scatter', color='b')\nplt.title('Joint Distribution Plot: LOCATION_ vs MAG')\nplt.show()\n\n# Create jointplot for DEPTH vs MAG\nplt.figure(figsize=(6, 6))\njointplot_depth = sns.jointplot(data=data_with_mag, x='DEPTH', y='MAG', kind='scatter', color='g')\nplt.title('Joint Distribution Plot: DEPTH vs MAG')\nplt.show()\n")



get_ipython().run_cell_magic('time', '', "# sns.regplot for correlations btw MAG:\nplt.figure(figsize=(12, 8))\n\nplt.subplot(2, 2, 1)\nsns.regplot(data=data_with_mag, x='LAT', y='MAG')\nplt.title('Regression Plot: LAT vs MAG')\n\nplt.subplot(2, 2, 2)\nsns.regplot(data=data_with_mag, x='LNG', y='MAG')\nplt.title('Regression Plot: LNG vs MAG')\n\nplt.subplot(2, 2, 3)\nsns.regplot(data=data_with_mag, x='DEPTH', y='MAG')\nplt.title('Regression Plot: DEPTH vs MAG')\n\nplt.subplot(2, 2, 4)\nsns.regplot(data=data_with_mag, x='LOCATION_', y='MAG')\nplt.title('Regression Plot: LOCATION_ vs MAG')\n\nplt.tight_layout(rect=[0, 0, 1, 1])\nplt.show()\n")


# <Figure size 1200x800 with 4 Axes>
# CPU times: total: 5min 33s
# Wall time: 44.6 s
# 

# ### Finding key inputs to our model:


get_ipython().run_cell_magic('time', '', "from sklearn.ensemble import RandomForestRegressor\n\n# Initialize the model\nmodel = RandomForestRegressor()\n\n# Fit the model on the training data\nmodel.fit(X_train[numerical_features_locationadded], y_train)\n\n# Get feature importances\nfeature_importances = model.feature_importances_\n\n# Plot feature importances\nplt.figure(figsize=(8, 4))\nplt.barh(numerical_features_locationadded, feature_importances)\nplt.title('Feature Importances')\nplt.xlabel('Importance Score')\nplt.show()\n")


# - LOCATION_ is the most important feature as it is seen from the plot (~0.6 importance score)

# 2m 45s

# ## VI. Experiment with Multiple Regression Models

# ### 1. Random Forest Regression

# 

# Random Forest is an ensemble learning method that combines multiple decision trees to make predictions. It can handle a large number of features.



numerical_features_locationadded




X_train



get_ipython().run_cell_magic('time', '', '\n#check if a saved version of the model exists before training:\nimport os\nimport pickle\nfrom sklearn.ensemble import RandomForestRegressor\n\n# Check if the saved model file exists for each model\nif os.path.exists(\'rforest_model.pkl\'):\n    # Load the model from the file if it exists\n    with open(\'rforest_model.pkl\', \'rb\') as file:\n        rforest_model = pickle.load(file)\n        print("rforest_model loaded from file rforest_model.pkl")\nelse:\n    # Train the model if the file does not exist\n    rforest_model = RandomForestRegressor(random_state=42)\n    rforest_model.fit(X_train_scaled_df, y_train)\n    # Save the models to files\n    with open(\'rforest_model.pkl\', \'wb\') as file:\n        pickle.dump(rforest_model, file)\n        print("rforest_model saved to the file rforest_model.pkl")\n    \n    \n    \n    \nif os.path.exists(\'rforest_model2.pkl\'):\n    # Load the model from the file if it exists\n    with open(\'rforest_model2.pkl\', \'rb\') as file:\n        rforest_model2 = pickle.load(file)\n        print("rforest_model2 loaded from file rforest_model2.pkl")\nelse:\n    # Train the model if the file does not exist\n    rforest_model2 = RandomForestRegressor(random_state=42)\n    rforest_model2.fit(X_train_pca_df, y_train)\n    #save the model to file:\n    with open(\'rforest_model2.pkl\', \'wb\') as file:\n        pickle.dump(rforest_model2, file)\n        print("rforest_model2 saved to the file rforest_model2.pkl")\n')


# for training the models:
# CPU times: total: 5min 35s
# Wall time: 5min 38s
# 
# rforest_model2 saved to the file rforest_model2.pkl
# CPU times: total: 3min 31s
# Wall time: 3min 36s
# 



rforest_model.score(X_test_scaled_df, y_test)




print(rforest_model2.feature_importances_)
X_train_pca_df



print(rforest_model.feature_importances_)


# ### 2. Gradient Boosting Regression
# 

# Gradient Boosting is also an ensemble method that builds multiple decision trees sequentially, with each tree correcting the errors of the previous one. It often provides better predictive performance compared to Random Forest.



get_ipython().run_cell_magic('time', '', 'from sklearn.ensemble import GradientBoostingRegressor\n\n# Check if the saved model file exists for each model\nif os.path.exists(\'gboosting_model.pkl\'):\n    # Load the model from the file if it exists\n    with open(\'gboosting_model.pkl\', \'rb\') as file:\n        gboosting_model = pickle.load(file)\n        print("gboosting_model loaded from file gboosting_model.pkl")\nelse:\n    # Train the model if the file does not exist\n    gboosting_model = GradientBoostingRegressor(random_state=42)\n    gboosting_model.fit(X_train_scaled_df, y_train)\n    with open(\'gboosting_model.pkl\', \'wb\') as file:\n        pickle.dump(gboosting_model, file)\n        print("gboosting_model saved to file, gboosting_model.pkl")\n    \n    \n    \n    \n    \nif os.path.exists(\'gboosting_model2.pkl\'):\n    # Load the model from the file if it exists\n    with open(\'gboosting_model2.pkl\', \'rb\') as file:\n        gboosting_model2 = pickle.load(file)\n        print("gboosting_model2 loaded from file gboosting_model2.pkl")\nelse:\n    # Train the model if the file does not exist\n    gboosting_model2 = GradientBoostingRegressor(random_state=42)\n    gboosting_model2.fit(X_train_pca_df, y_train)\n    with open(\'gboosting_model2.pkl\', \'wb\') as file:\n        pickle.dump(gboosting_model2, file)\n        print("gboosting_model2 saved to file, gboosting_model2.pkl")\n')


# gboosting_model saved to file, gboosting_model.pkl
# gboosting_model2 saved to file, gboosting_model2.pkl
# 
# for training gboosting-model:
# CPU times: total: 23.3 s
# Wall time: 23.4 s
# 
# gboosting_model2 saved to file, gboosting_model2.pkl
# CPU times: total: 42.9 s
# Wall time: 43.5 s
# 


gboosting_model.feature_importances_




print(X_train_scaled_df)




gboosting_model2.feature_importances_



print(X_train_pca_df)


# ### 3. Support Vector Regression (SVR)

# SVR is a regression version of Support Vector Machine (SVM). It works well for smaller datasets and captures complex relationships between features and target.



get_ipython().run_cell_magic('time', '', 'import os\nimport pickle\nfrom sklearn.svm import SVR\n\n# Check if the saved model file exists for each model\nif os.path.exists(\'svr_model.pkl\'):\n    # Load the model from the file if it exists\n    with open(\'svr_model.pkl\', \'rb\') as file:\n        svr_model = pickle.load(file)\n        print("svr_model loaded from file svr_model.pkl")\nelse:\n    # Train the model if the file does not exist\n    #Training Support Vector Regression (SVR) model is computationally expensive, because the dataset is large. So that we use reduced dimensionality with PCA.\n    svr_model = SVR(kernel=\'linear\', cache_size=2000)\n    # Set n_jobs to utilize multiple CPU cores\n    svr_model.n_jobs = -1\n    svr_model.fit(X_train_scaled_df, y_train)\n    # Save the model to files\n    with open(\'svr_model.pkl\', \'wb\') as file:\n        pickle.dump(svr_model, file)\n        print("svr_model saved to file, svr_model.pkl")\n\n\n\n\n# Check if the saved model file exists\nif os.path.exists(\'svr_model2.pkl\'):\n    # Load the model from the file if it exists\n    with open(\'svr_model2.pkl\', \'rb\') as file:\n        svr_model2 = pickle.load(file)\n        print("svr_model2 loaded from file svr_model2.pkl")\nelse:\n    # Train the model if the file does not exist\n    #Training Support Vector Regression (SVR) model is computationally expensive, because the dataset is large. So that we use reduced dimensionality with PCA.\n    svr_model2 = SVR(kernel=\'linear\', cache_size=2000)\n    # Set n_jobs to utilize multiple CPU cores\n    svr_model2.n_jobs = -1\n    svr_model2.fit(X_train_pca_df, y_train)\n    # Save the model to files\n    with open(\'svr_model2.pkl\', \'wb\') as file:\n        pickle.dump(svr_model2, file)\n        print("svr_model2 saved to file, svr_model2.pkl")\n')


# svr_model saved to file, svr_model.pkl
# svr_model2 saved to file, svr_model2.pkl
# for training both models:
# CPU times: total: 1h 5min 5s
# Wall time: 1h 5min 11s
# 
# for training the svr with X_train_pca_df model:
# CPU times: total: 26min 31s
# Wall time: 26min 34s

# ### 4. Neural Networks using TensorFlow/Keras



get_ipython().run_cell_magic('time', '', 'from keras.models import Sequential\nimport tensorflow as tf\nimport os\nfrom keras.layers import Dense, BatchNormalization, Dropout\nfrom keras.optimizers import SGD, Adam, Optimizer\nfrom scikeras.wrappers import KerasRegressor\nfrom tensorflow import keras\nfrom keras.callbacks import EarlyStopping\nfrom keras.models import load_model\nfrom keras.callbacks import History\n\n\n\nimport pickle\n\ndef create_neural_network(learning_rate, momentum, dropout_rate, epochs, input_dim):\n    model = keras.models.Sequential()\n    model.add(Dense(64, activation=\'relu\', kernel_initializer=\'uniform\',  input_dim=input_dim.shape[1]))\n    model.add(BatchNormalization())\n    model.add(Dropout(dropout_rate)) # The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting. Inputs not set to 0 are scaled up by 1/(1 - rate) such that the sum over all inputs is unchanged.\n    model.add(Dense(32, activation=\'relu\'))\n    model.add(Dense(1, activation=\'linear\'))\n\n    """if optimizer == \'adam\':\n        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)\n    elif optimizer == \'sgd\':\n        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)"""\n    # Compile the model with the optimizer and loss function\n    \n    decay_rate = learning_rate / epochs\n    optimizer = SGD(learning_rate=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)\n    \n    model.compile(optimizer=optimizer, loss=\'mean_squared_error\', metrics=[\'mse\'])\n    #model.compile(loss=\'mean_squared_error\', metrics=[\'mse\'])\n    return model\n\n# Check if the saved model file exists for each model\nif os.path.exists(\'nn_history.pkl\'):\n    # Load the model from the file if it exists\n    with (open(\'nn_history.pkl\', \'rb\') as file):\n        nn_history = History()\n        nn_history.history = pickle.load(file)\n        print("nn_history loaded from file nn_history.pkl")\n    \n    # Check if the model file exists\n    if os.path.exists(\'nn_model.h5\'):\n        # Load the saved model\n        nn_model = load_model(\'nn_model.h5\')\n        print("nn_model loaded from file, nn_model.h5")\n        \nelse:\n    # Train the model if the file does not exist\n    # Set early stopping to prevent overfitting\n    early_stopping = EarlyStopping(patience=7, restore_best_weights=True)  #Stop training when a monitored metric has stopped improving.\n    # Create the NN model:\n    nn_model = create_neural_network(learning_rate=0.04, momentum=0.9, dropout_rate=0.5, epochs=67, input_dim=X_train_scaled_df)\n    #nn_model = KerasRegressor(model=create_neural_network(), epochs=100, batch_size=32, verbose=0, callbacks=[early_stopping])\n    \n    # Train the model on the training data and validate on the validation data\n    nn_history = nn_model.fit(X_train_scaled_df, y_train, validation_data=(X_val_scaled_df, y_val), epochs=65, batch_size=32, verbose=1, callbacks=[early_stopping])\n    \n    # Save the trained model to the file\n    nn_model.save(\'nn_model.h5\')\n    print("nn_model saved to file, nn_model.h5")\n\n    # Convert the history object to a dictionary\n    history_dict = nn_history.history\n\n    # Save the history dictionary to a file using pickle\n    with open(\'nn_history.pkl\', \'wb\') as file:\n        pickle.dump(history_dict, file)\n        print("nn_history saved to file, nn_history.pkl")\n\n    \n\n# nn_history2 = nn_model.fit(X_train_pca, y_train, validation_data=(X_val_pca, y_val), epochs=100, batch_size=32, callbacks=[early_stopping])\n# Check if the saved model file exists for model\nif os.path.exists(\'nn_history2.pkl\'):\n    # Load the model from the file if it exists\n    with (open(\'nn_history2.pkl\', \'rb\') as file):\n        nn_history2 = History()\n        nn_history2.history = pickle.load(file)\n        print("nn_history2 loaded from file nn_history2.pkl")\n        \n    # Check if the model file exists\n    if os.path.exists(\'nn_model2.h5\'):\n        # Load the saved model\n        nn_model2 = load_model(\'nn_model2.h5\')\n        print("nn_model2 loaded from file, nn_model2.h5")\n        \nelse:\n    # Train the model if the file does not exist\n    # Set early stopping to prevent overfitting\n    early_stopping = EarlyStopping(patience=7, restore_best_weights=True)  #Stop training when a monitored metric has stopped improving.\n    # Create the NN model:\n    nn_model2 = create_neural_network(learning_rate=0.009, momentum=0.9, dropout_rate=0.5, epochs=67, input_dim=X_train_pca_df)\n    #nn_model = KerasRegressor(model=create_neural_network(), epochs=100, batch_size=32, verbose=0, callbacks=[early_stopping])\n    \n    # Train the model on the training data and validate on the validation data\n    nn_history2 = nn_model2.fit(X_train_pca_df, y_train, validation_data=(X_val_pca_df, y_val), epochs=65, batch_size=16, verbose=1, callbacks=[early_stopping])\n    \n    # Save the trained model to the file:\n    nn_model2.save(\'nn_model2.h5\')\n    print("nn_model2 saved to file, nn_model2.h5")\n    \n    # Convert the history object to a dictionary\n    history_dict2 = nn_history2.history\n\n    # Save the history dictionary to a file using pickle\n    with open(\'nn_history2.pkl\', \'wb\') as file:\n        pickle.dump(history_dict2, file)\n        print("nn_history2 saved to file, nn_history2.pkl")\n    \n')


# nn_model2 saved to file, nn_model2.h5
# nn_history2 saved to file, nn_history2.pkl
# 
# CPU times: total: 11min 13s
# Wall time: 7min 30s




# Plot the loss function
fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(np.sqrt(nn_history.history['loss']), 'r', label='train')
ax.plot(np.sqrt(nn_history.history['val_loss']), 'b' ,label='val')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Loss', fontsize=20)
ax.legend()
ax.tick_params(labelsize=20)

# Plot the accuracy
fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(np.sqrt(nn_history.history['mse']), 'r', label='train')
ax.plot(np.sqrt(nn_history.history['val_mse']), 'b' ,label='val')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'MSE', fontsize=20)
ax.legend()
ax.tick_params(labelsize=20)





# Plot the loss function for nn_history2
fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(np.sqrt(nn_history2.history['loss']), 'r', label='train')
ax.plot(np.sqrt(nn_history2.history['val_loss']), 'b' ,label='val')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Loss', fontsize=20)
ax.legend()
ax.tick_params(labelsize=20)

# Plot the accuracy
fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(np.sqrt(nn_history2.history['mse']), 'r', label='train')
ax.plot(np.sqrt(nn_history2.history['val_mse']), 'b' ,label='val')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'MSE', fontsize=20)
ax.legend()
ax.tick_params(labelsize=20)


# Epoch 1/100
# 5905/5905 [==============================] - 7s 987us/step - loss: 0.2600 - mse: 0.2600 - val_loss: 0.3165 - val_mse: 0.3165
# Epoch 2/100
# 5905/5905 [==============================] - 6s 942us/step - loss: 0.2414 - mse: 0.2414 - val_loss: 0.3214 - val_mse: 0.3214
# Epoch 3/100
# 5905/5905 [==============================] - 6s 959us/step - loss: 0.2372 - mse: 0.2372 - val_loss: 0.4763 - val_mse: 0.4763
# Epoch 4/100
# 5905/5905 [==============================] - 6s 954us/step - loss: 0.2365 - mse: 0.2365 - val_loss: 0.4977 - val_mse: 0.4977
# Epoch 5/100
# 5905/5905 [==============================] - 6s 953us/step - loss: 0.2370 - mse: 0.2370 - val_loss: 0.4703 - val_mse: 0.4703
# Epoch 6/100
# 5905/5905 [==============================] - 6s 972us/step - loss: 0.2372 - mse: 0.2372 - val_loss: 0.4100 - val_mse: 0.4100
# Epoch 7/100
# 5905/5905 [==============================] - 6s 958us/step - loss: 0.2360 - mse: 0.2360 - val_loss: 0.5205 - val_mse: 0.5205
# Epoch 8/100
# 5905/5905 [==============================] - 6s 955us/step - loss: 0.2367 - mse: 0.2367 - val_loss: 0.4923 - val_mse: 0.4923
# Epoch 9/100
# 5905/5905 [==============================] - 6s 963us/step - loss: 0.2368 - mse: 0.2368 - val_loss: 0.5182 - val_mse: 0.5182
# Epoch 10/100
# 5905/5905 [==============================] - 6s 952us/step - loss: 0.2367 - mse: 0.2367 - val_loss: 0.4361 - val_mse: 0.4361
# Epoch 11/100
# 5905/5905 [==============================] - 6s 956us/step - loss: 0.2362 - mse: 0.2362 - val_loss: 0.5496 - val_mse: 0.5496
# INFO:tensorflow:Assets written to: ram://c0a5df86-4c82-4080-8cb1-e0c555b69a07/assets
# CPU times: total: 1min 49s
# Wall time: 1min 5s
# 

# ### 5. XGBoost Regression

# XGBoost and LightGBM are gradient boosting frameworks that are widely used for regression tasks. They are highly efficient and perform well on a wide range of datasets. They are also good for performance.




get_ipython().run_cell_magic('time', '', 'from xgboost import XGBRegressor\n\n# Check if the saved model file exists for each model\nif os.path.exists(\'xgb_model.pkl\'):\n    # Load the model from the file if it exists\n    with open(\'xgb_model.pkl\', \'rb\') as file:\n        xgb_model = pickle.load(file)\n        print(f"xgb_model loaded from xgb_model.pkl")\nelse:\n    # Train the model if the file does not exist\n    xgb_model = XGBRegressor(random_state=42)\n    xgb_model.fit(X_train_scaled_df, y_train)\n    with open(\'xgb_model.pkl\', \'wb\') as file:\n        pickle.dump(xgb_model, file)\n    \nif os.path.exists(\'xgb_model2.pkl\'):\n    # Load the model from the file if it exists\n    with open(\'xgb_model2.pkl\', \'rb\') as file:\n        xgb_model2 = pickle.load(file)\n        print(f"xgb_model2 loaded from xgb_model2.pkl")\nelse:\n    # Train the model if the file does not exist\n    xgb_model2 = XGBRegressor(random_state=42)\n    xgb_model2.fit(X_train_pca_df, y_train)\n    with open(\'xgb_model2.pkl\', \'wb\') as file:   \n        pickle.dump(xgb_model2, file)\n')


# CPU times: total: 2min 7s
# Wall time: 9.06 s
# 
# xgb_model2
# CPU times: total: 1min 16s
# Wall time: 6.67 s
# 

# ### - Saving the trained models to a file

# By saving the trained models to a file, we can avoid retraining them every time we run your project and we can simply load the pre-trained models from the file.




# %%time
# import pickle
# 
# # Assuming you have already trained your models and stored them in variables:
# # rforest_model, svr_model, nn_model, etc.
# 
# # Save the models to files
# with open('rforest_model.pkl', 'wb') as file:
#     pickle.dump(rforest_model, file)
#     print("rforest_model saved to the file rforest_model.pkl")
# 
# with open('rforest_model2.pkl', 'wb') as file:
#     pickle.dump(rforest_model2, file)
#     print("rforest_model2 saved to the file rforest_model2.pkl")
# 
# with open('gboosting_model.pkl', 'wb') as file:
#     pickle.dump(gboosting_model, file)
# 
# with open('gboosting_model2.pkl', 'wb') as file:
#     pickle.dump(gboosting_model2, file)
# 
# with open('svr_model.pkl', 'wb') as file:
#     pickle.dump(svr_model, file)
# 
# with open('nn_history.pkl', 'wb') as file:
#     pickle.dump(nn_history, file)
# 
# with open('xgb_model.pkl', 'wb') as file:
#     pickle.dump(xgb_model, file)
# 
# with open('xgb_model2.pkl', 'wb') as file:   
#     pickle.dump(xgb_model2, file)

# ctrl + / for commenting


# This code will save each trained model to a separate file with the specified names ('rforest_model.pkl', 'svr_model.pkl', etc.).

# ### - Evaluate Trained models on the test set:




get_ipython().run_cell_magic('time', '', 'from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n\nmodels_trained = [rforest_model, rforest_model2, gboosting_model, gboosting_model2, svr_model, svr_model2, nn_model, nn_model2, xgb_model,  xgb_model2]\nmodel_names_trained = [\'Random Forest\', \'Random Forest 2\', \'Gradient Boosting\', \'Gradient Boosting 2\', \'SVR\', \'SVR 2\', \'Neural Networks\', \'Neural Networks 2\', \'XGBoost\',  \'XGBoost 2\']\n\npickle_file_path = \'results_trained.pkl\'\nif os.path.exists(pickle_file_path):\n    results_trained_df = pd.read_pickle(pickle_file_path)\nelse:\n    results_trained = []\n    for model, name in zip(models_trained, model_names_trained):\n        if model == rforest_model or model == gboosting_model or model == svr_model or model == nn_model or model == xgb_model:\n            y_predict_val = model.predict(X_val_scaled_df)\n            test_score = model.predict(X_test_scaled_df)\n        else:\n            y_predict_val = model.predict(X_val_pca_df)\n            test_score = model.predict(X_test_pca_df)\n        mse = mean_squared_error(y_val, y_predict_val)\n        mae = mean_absolute_error(y_val, y_predict_val)\n        r2 = r2_score(y_val, y_predict_val)\n        test_mae = mean_absolute_error(y_test, test_score)\n\n        results_trained.append([name, mse, mae, r2, test_mae])\n        print(f"Model: {name}")\n        print(f"Mean Squared Error: {mse:.3f}")\n        print(f"Mean Absolute Error: {mae:.3f}")\n        print(f"R-squared: {r2:.2f}\\n\\n")\n    \n    #    print(f"Test Score on test set (MAE): {mean_squared_error(y_test, test_score)}\\n\\n")\n\n    # Create a DataFrame to store the results\n    columns = [\'Model\', \'Mean Squared Error (Validation)\', \'Mean Absolute Error (Validation)\', \'R-squared (Validation)\', \'Mean Absolute Error (Test)\']\n    results_trained_df = pd.DataFrame(results_trained, columns=columns)\n    results_trained_df.to_pickle(pickle_file_path)\n\n\nprint(results_trained_df)\n')





print(results_trained_df)





results_trained_df


# #### Choose the best performing model and evaluate on the test set
# 




# Choose the best performing model and evaluate on the test set
#best_model = models_trained[np.argmax([model.score() for model in models_trained])]
# Find the index of the best performing model based on validation MSE
best_model_idx = results_trained_df['Mean Squared Error (Validation)'].idxmin()
best_model_name = results_trained_df.loc[best_model_idx, 'Model']
best_model = models_trained[best_model_idx]

# Evaluate the best model on the test set
if best_model in [rforest_model, gboosting_model, svr_model, nn_model, xgb_model]:
    y_predict_test = best_model.predict(X_test_scaled_df)
else:
    y_predict_test = best_model.predict(X_test_pca_df)

mse_test = mean_squared_error(y_test, y_predict_test)
mae_test = mean_absolute_error(y_test, y_predict_test)
r2_test = r2_score(y_test, y_predict_test)
print(f"Best Performing Model: {best_model}")
print(f"Mean Squared Error on Test Set: {mse_test:.2f}")
print(f"Mean Absolute Error on Test Set: {mae_test:.2f}")
print(f"R-squared on Test Set: {r2_test:.2f}")


# Best Performing Model: XGBRegressor(base_score=None, booster=None, callbacks=None,
#              colsample_bylevel=None, colsample_bynode=None,
#              colsample_bytree=None, early_stopping_rounds=None,
#              enable_categorical=False, eval_metric=None, feature_types=None,
#              gamma=None, gpu_id=None, grow_policy=None, importance_type=None,
#              interaction_constraints=None, learning_rate=None, max_bin=None,
#              max_cat_threshold=None, max_cat_to_onehot=None,
#              max_delta_step=None, max_depth=None, max_leaves=None,
#              min_child_weight=None, missing=nan, monotone_constraints=None,
#              n_estimators=100, n_jobs=None, num_parallel_tree=None,
#              predictor=None, random_state=42, ...)
# 
# Mean Squared Error on Test Set: 0.27
# Mean Absolute Error on Test Set: 0.39
# R-squared on Test Set: 0.39
# 

# ## VII. Hyperparameter Tuning

# ### a. Random Forest Regression




get_ipython().run_cell_magic('time', '', 'from sklearn.model_selection import RandomizedSearchCV #, GridSearchCV\n\nimport joblib\n\ndef tune_model_rf(estimator, param_distributions, train, n_iter=10, cv=kf, scoring=\'neg_mean_squared_error\', random_state=42, model_name=\'\'):\n    model_file = f\'{model_name}_random_search.pkl\'\n    \n    if os.path.exists(model_file):\n        # Load the tuned model from the file if it exists\n        print(f"Loading {model_name} model from {model_file}")\n        tuned_model = joblib.load(model_file)\n    else:\n        # Perform hyperparameter tuning with RandomizedSearchCV if the file does not exist\n        print(f"Tuning {model_name} model...")\n        rf_random_search = RandomizedSearchCV(\n            estimator=estimator,\n            param_distributions=param_distributions,\n            n_iter=n_iter,\n            cv=cv,\n            scoring=scoring,\n            random_state=random_state,\n            n_jobs=-1,  # Parallelize the tuning process\n            verbose=2\n        )\n        \n        rf_random_search.fit(train, y_train)\n        tuned_model = rf_random_search\n        joblib.dump(tuned_model, model_file)\n        print(f"{model_name} model tuned and saved to {model_file}")\n    \n    return tuned_model\n\n# Define the hyperparameter distributions for Random Forest\nparam_random_rf = {\n    \'n_estimators\': [100, 200, 300],\n    \'max_depth\': [None, 5, 10],\n    \'min_samples_split\': [2, 5, 10]\n}\n\n# Tune the Random Forest model\ntuned_rf_model = tune_model_rf(estimator=rforest_model, param_distributions=param_random_rf, n_iter=5, model_name=\'rf\', train=X_train_scaled_df)\n\n# We can use the tuned_rf_model for prediction and evaluation\n#y_predict_val_rf = tuned_rf_model.best_estimator_.predict(X_val)\n')


# Tuning rf model...(with X_train_scaled_df)
# Fitting 5 folds for each of 5 candidates, totalling 25 fits
# rf model tuned and saved to rf_random_search.pkl
# CPU times: total: 1min 53s
# Wall time: 12min 29s
# 
# 




print(tuned_rf_model.best_params_)





print(tuned_rf_model.best_estimator_)


# rf_random_search model tuned and saved to rf_random_search.pkl(before scaling data)
# 
# CPU times: total: 36min 19s
# Wall time: 36min 22s




get_ipython().run_cell_magic('time', '', "# Tune the Random Forest model 2\ntuned_rf_model2 = tune_model_rf(estimator=rforest_model2, param_distributions=param_random_rf, n_iter=10, model_name='rf2', train=X_train_pca_df)\n")


# Tuning rf2 model...
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# rf2 model tuned and saved to rf2_random_search.pkl
# CPU times: total: 1min 25s
# Wall time: 10min 34s
# 




print(tuned_rf_model2.best_params_)


# Tuning rf2 model...(before pca applied)
# rf2 model tuned and saved to rf2_random_search.pkl
# CPU times: total: 11min 27s
# Wall time: 16min 9s
# 




print(tuned_rf_model.n_features_in_)
X_val_pca_df.shape


# ### b. Gradient Boosting Regression




get_ipython().run_cell_magic('time', '', 'import os\nfrom sklearn.model_selection import RandomizedSearchCV\nimport joblib\n\ndef tune_model_gb(estimator, param_distributions, train, n_iter=10, cv=kf, scoring=\'neg_mean_squared_error\', random_state=42, model_name=\'\'):\n    model_file = f\'{model_name}_random_search.pkl\'\n    \n    if os.path.exists(model_file):\n        # Load the tuned model from the file if it exists\n        print(f"Loading {model_name} model from {model_file}")\n        tuned_model = joblib.load(model_file)\n    else:\n        # Perform hyperparameter tuning with RandomizedSearchCV if the file does not exist\n        print(f"Tuning {model_name} model...")\n        gb_random_search = RandomizedSearchCV(\n            estimator=estimator,\n            param_distributions=param_distributions,\n            n_iter=n_iter,\n            cv=cv,\n            scoring=scoring,\n            random_state=random_state,\n            verbose=3,\n            n_jobs=-1  # Parallelize the tuning process\n        )\n        gb_random_search.fit(train, y_train)\n        tuned_model = gb_random_search\n        joblib.dump(tuned_model, model_file)\n        print(f"{model_name} model tuned and saved to {model_file}")\n    \n    return tuned_model\n\n# Define the hyperparameter distributions for Gradient Boosting\nparam_random_gb = {\n    \'n_estimators\': [50, 100, 200, 300],\n    \'learning_rate\': [0.001, 0.01, 0.1, 0.5]\n}\n\n# Tune the Gradient Boosting model\ntuned_gb_model = tune_model_gb(estimator=gboosting_model, param_distributions=param_random_gb, n_iter=10, model_name=\'gb\', train=X_train_scaled_df)\n\n# We can use the tuned_gb_model for prediction and evaluation\n#y_predict_val_gb = tuned_gb_model.best_estimator_.predict(X_val)\n')


# Tuning gb model...
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# gb model tuned and saved to gb_random_search.pkl
# CPU times: total: 2min 56s
# Wall time: 8min 52s
# 




print(tuned_gb_model.best_params_)





get_ipython().run_cell_magic('time', '', "# Tune the Gradient Boosting model2\ntuned_gb_model2 = tune_model_gb(estimator=gboosting_model2, param_distributions=param_random_gb, n_iter=5, model_name='gb2', train=X_train_pca_df)\n")


# Tuning gb2 model...
# Fitting 5 folds for each of 5 candidates, totalling 25 fits
# gb2 model tuned and saved to gb2_random_search.pkl
# CPU times: total: 43.1 s
# Wall time: 2min 29s
# 




print(tuned_gb_model2.best_params_)


# ### c. Support Vector Regression (SVR)
# 

# #### RandomizedSearchCV  (computationally expensive - +5 hours..)




# %%time
# def tune_model_svr(estimator, param_distributions, train, y_train, n_iter=10, cv=kf, scoring='neg_mean_squared_error', random_state=42, model_name=''):
#     model_file = f'{model_name}_random_search.pkl'
#     
#     if os.path.exists(model_file):
#         # Load the tuned model from the file if it exists
#         print(f"Loading {model_name} model from {model_file}")
#         tuned_model = joblib.load(model_file)
#     else:
#         # Perform hyperparameter tuning with RandomizedSearchCV if the file does not exist
#         print(f"Tuning {model_name} model...")
#         svr_random_search = RandomizedSearchCV(
#             estimator=estimator,
#             param_distributions=param_distributions,
#             n_iter=n_iter,
#             cv=cv,
#             scoring=scoring,
#             random_state=random_state,
#             verbose=3,
#             n_jobs=-1  # Parallelize the tuning process
#         )
#         svr_random_search.fit(train, y_train)
#         tuned_model = svr_random_search
#         joblib.dump(tuned_model, model_file)
#         print(f"{model_name} model tuned and saved to {model_file}")
#     
#     return tuned_model
# 
# # Define the hyperparameter distributions for SVR
# param_random_svr = {
#     'C': [0.1, 1, 10],
#     'kernel': ['linear', 'rbf', 'poly'],
#     'gamma': ['scale', 'auto', 0.1, 1],
# }
# 
# # Choose a smaller subset of the data for tuning
# subset_size = 3000
# X_train_subset = X_train_scaled_df[:subset_size]
# y_train_subset = y_train[:subset_size]
# 
# # Tune the SVR model
# tuned_svr_model = tune_model_svr(estimator=svr_model, param_distributions=param_random_svr, n_iter=3, model_name='svr', train=X_train_subset, y_train=y_train_subset)
# # Print the best hyperparameters found
# print("Best Hyperparameters:")
# print(tuned_svr_model.best_params_)
# 
# # We can use the tuned_svr_model for prediction and evaluation
# #y_predict_val_svr = tuned_svr_model.best_estimator_.predict(X_val_pca)


# svr model tuned and saved to svr_random_search.pkl
# CPU times: total: 1h 14min 46s
# Wall time: 1h 15min 11s




# %%time
# 
# def tune_model_svr(estimator, param_distributions, train, y_train, n_iter=10, cv=3, scoring='neg_mean_squared_error', random_state=42, model_name=''):
#     model_file = f'{model_name}_random_search.pkl'
#     
#     if os.path.exists(model_file):
#         # Load the tuned model from the file if it exists
#         print(f"Loading {model_name} model from {model_file}")
#         tuned_model = joblib.load(model_file)
#     else:
#         # Perform hyperparameter tuning with RandomizedSearchCV if the file does not exist
#         print(f"Tuning {model_name} model...")
#         svr_random_search = RandomizedSearchCV(
#             estimator=estimator,
#             param_distributions=param_distributions,
#             n_iter=n_iter,
#             cv=cv,
#             scoring=scoring,
#             random_state=random_state,
#             verbose=3,
#             n_jobs=-1  # Parallelize the tuning process
#         )
#         svr_random_search.fit(train, y_train)
#         tuned_model = svr_random_search
#         joblib.dump(tuned_model, model_file)
#         print(f"{model_name} model tuned and saved to {model_file}")
#     
#     return tuned_model
# 
# # Define the hyperparameter distributions for SVR
# param_random_svr = {
#     'C': [0.1, 1, 10],
#     'kernel': ['linear', 'rbf', 'poly'],
#     'gamma': ['scale', 'auto', 0.1, 1],
# }
# 
# 
# 
# 
# subset_size = 500
# X_train_pca_subset = X_train_pca_df[:subset_size]
# y_train_subset = y_train[:subset_size]
# # Tune the SVR model 2
# tuned_svr_model2 = tune_model_svr(estimator=svr_model2, param_distributions=param_random_svr, n_iter=4, model_name='svr2', train=X_train_pca_subset, y_train=y_train_subset)
# print("Best Hyperparameters:")
# print(tuned_svr_model2.best_params_)


# 3h 2m 32s (full train data)

# #### BayesSearchCV




get_ipython().run_cell_magic('time', '', 'from skopt import BayesSearchCV\nfrom skopt.space import Real, Categorical, Integer\nfrom sklearn.metrics import mean_squared_error\nfrom sklearn.svm import SVR\n\n\ndef tune_model_svr_bayesian(estimator, param_distributions, train, y_train, n_iter=10, cv=kf, scoring=\'neg_mean_squared_error\', random_state=42, model_name=\'\'):\n    model_file = f\'{model_name}_bayes_search.pkl\'\n    \n    if os.path.exists(model_file):\n        # Load the tuned model from the file if it exists\n        print(f"Loading {model_name} model from {model_file}")\n        tuned_model = joblib.load(model_file)\n    else:\n        # Perform hyperparameter tuning with RandomizedSearchCV if the file does not exist\n        print(f"Tuning {model_name} model...")\n        svr_bayes_search = BayesSearchCV(\n            estimator,\n            param_distributions,\n            n_iter=n_iter,\n            cv=cv,\n            scoring=scoring,\n            random_state=random_state,\n            verbose=3,\n            n_jobs=-1  # Parallelize the tuning process\n        )\n        svr_bayes_search.fit(train, y_train)\n        tuned_model = svr_bayes_search\n        joblib.dump(tuned_model, model_file)\n        print(f"{model_name} model tuned and saved to {model_file}")\n    \n    return tuned_model\n\n# Define the hyperparameter distributions for SVR\nparam_bayes_svr = {\n    \'C\': Real(0.1, 10.0, prior=\'log-uniform\'),\n    \'kernel\': Categorical([\'linear\', \'rbf\', \'poly\']),\n    \'gamma\': Categorical([\'scale\', \'auto\'])\n}\n\n# Choose a smaller subset of the data for tuning\nsubset_size = 10000\nX_train_subset = X_train_scaled_df[:subset_size]\ny_train_subset = y_train[:subset_size]\n\n# Tune the SVR model\ntuned_svr_model_bayesian = tune_model_svr_bayesian(estimator=svr_model, param_distributions=param_bayes_svr, n_iter=10, model_name=\'svr\', train=X_train_subset, y_train=y_train_subset)\n# Print the best hyperparameters found\nprint("Best Hyperparameters:")\nprint(tuned_svr_model_bayesian.best_params_)\n\n# We can use the tuned_svr_model for prediction and evaluation\n#y_predict_val_svr = tuned_svr_model.best_estimator_.predict(X_val_pca)\n')


# Tuning svr model...
# Fitting 5 folds for each of 1 candidates, totalling 5 fits
# Fitting 5 folds for each of 1 candidates, totalling 5 fits
# Fitting 5 folds for each of 1 candidates, totalling 5 fits
# Fitting 5 folds for each of 1 candidates, totalling 5 fits
# Fitting 5 folds for each of 1 candidates, totalling 5 fits
# Fitting 5 folds for each of 1 candidates, totalling 5 fits
# Fitting 5 folds for each of 1 candidates, totalling 5 fits
# Fitting 5 folds for each of 1 candidates, totalling 5 fits
# Fitting 5 folds for each of 1 candidates, totalling 5 fits
# Fitting 5 folds for each of 1 candidates, totalling 5 fits
# svr model tuned and saved to svr_bayes_search.pkl
# 
# Best Hyperparameters:
# OrderedDict([('C', 0.661009829541915), ('gamma', 'scale'), ('kernel', 'rbf')])
# CPU times: total: 7.02 s
# Wall time: 1min 49s
# 

# had an error with bayessearch in the beginning:
# AttributeError: module 'numpy' has no attribute 'int'.
# Replaced all np.int with int in the file 'anaconda3\envs\myenv\Lib\site-packages\skopt\space\transformers.py'
# restarted kernel,
# and error fixed.




X_train_pca_df





get_ipython().run_cell_magic('time', '', '# Choose a smaller subset of the data for tuning\nsubset_size = 25000\nX_train_pca_subset = X_train_pca_df[:subset_size]\ny_train_subset = y_train[:subset_size]\n\n# Tune the SVR model 2 with bayesian\ntuned_svr_model_bayesian2 = tune_model_svr_bayesian(estimator=svr_model2, param_distributions=param_bayes_svr, n_iter=10, model_name=\'svr2\', train=X_train_pca_subset, y_train=y_train_subset)\n# Print the best hyperparameters found\nprint("Best Hyperparameters:")\nprint(tuned_svr_model_bayesian2.best_params_)\n')


# Tuning svr2 model...
# Fitting 5 folds for each of 1 candidates, totalling 5 fits
# Fitting 5 folds for each of 1 candidates, totalling 5 fits
# Fitting 5 folds for each of 1 candidates, totalling 5 fits
# Fitting 5 folds for each of 1 candidates, totalling 5 fits
# Fitting 5 folds for each of 1 candidates, totalling 5 fits
# Fitting 5 folds for each of 1 candidates, totalling 5 fits
# Fitting 5 folds for each of 1 candidates, totalling 5 fits
# Fitting 5 folds for each of 1 candidates, totalling 5 fits
# Fitting 5 folds for each of 1 candidates, totalling 5 fits
# Fitting 5 folds for each of 1 candidates, totalling 5 fits
# svr2 model tuned and saved to svr2_bayes_search.pkl
# Best Hyperparameters:
# OrderedDict([('C', 8.146216961026964), ('gamma', 'scale'), ('kernel', 'rbf')])
# CPU times: total: 26.2 s
# Wall time: 4min 15s
# 

# ### d. Neural Networks - hyperopt
# 




get_ipython().run_cell_magic('time', '', 'import os\nimport numpy as np\nimport tensorflow as tf\nfrom tensorflow import keras\nfrom hyperopt import fmin, tpe, hp, Trials\nfrom sklearn.metrics import mean_squared_error\n\n# Define the objective function\ndef objective(params):\n    model = create_neural_network(\n        learning_rate=params[\'learning_rate\'],\n        momentum=params[\'momentum\'],\n        dropout_rate=params[\'dropout_rate\'],\n        epochs=int(params[\'epochs\']),\n        input_dim=X_train_scaled_df\n    )\n    \n    model.compile(optimizer=\'adam\', loss=\'mse\')  # Adjust optimizer and loss if needed\n    \n    model.fit(X_train_scaled_df, y_train, validation_data=(X_val_scaled_df, y_val), verbose=0)\n    \n    y_pred = model.predict(X_val_scaled_df)\n    mse = mean_squared_error(y_val, y_pred)\n    \n    return mse\n\n# Check if the tuned model file exists\ntuned_model_file = \'nn_tuned_tuner_hyperopt.joblib\'\n\nif os.path.exists(tuned_model_file):\n    # Load the tuned model from the file\n    nn_tuned_tuner_hyperopt = joblib.load(tuned_model_file)\nelse:\n    # Define the hyperparameter search space\n    space = {\n        \'learning_rate\': hp.loguniform(\'learning_rate\', np.log(0.0001), np.log(0.01)),\n        \'momentum\': hp.uniform(\'momentum\', 0.1, 0.9),\n        \'dropout_rate\': hp.uniform(\'dropout_rate\', 0.0, 0.5),\n        \'epochs\': hp.quniform(\'epochs\', 10, 110, 1)\n    }\n\n    # Perform hyperparameter search\n    nn_tuned_tuner_hyperopt = fmin(\n        fn=objective,\n        space=space,\n        algo=tpe.suggest,\n        max_evals=10,  # Number of iterations\n        trials=Trials(),  # Required for Hyperopt to keep track of results\n    )\n\n    # Save the tuned tuner to a file\n    joblib.dump(nn_tuned_tuner_hyperopt, tuned_model_file)\n\nprint("Best Hyperparameters:")\nprint(nn_tuned_tuner_hyperopt)\n\n# Get the best hyperparameters\nbest_learning_rate_hyperopt = nn_tuned_tuner_hyperopt[\'learning_rate\']\nbest_momentum_hyperopt = nn_tuned_tuner_hyperopt[\'momentum\']\nbest_dropout_rate_hyperopt = nn_tuned_tuner_hyperopt[\'dropout_rate\']\nbest_epochs_hyperopt = int(nn_tuned_tuner_hyperopt[\'epochs\'])\n\n# Create the best model\nbest_nn_model = create_neural_network(\n    learning_rate=best_learning_rate_hyperopt,\n    momentum=best_momentum_hyperopt,\n    dropout_rate=best_dropout_rate_hyperopt,\n    epochs=best_epochs_hyperopt,\n    input_dim=X_train_scaled_df\n)\n\n# Compile the model\nbest_nn_model.compile(optimizer=\'adam\', loss=\'mse\')  # Adjust optimizer and loss if needed\n\n# Train the best model on the entire training dataset\nbest_nn_model.fit(X_train_scaled_df, y_train, verbose=1)\n\n# Save the best model to disk\nbest_model_file = \'best_nn_model_hyperopt.h5\'\nbest_nn_model.save(best_model_file)\nprint(f"Best model saved to file {best_model_file}")\n\n# Later, we can load the best model and use it for predictions\nloaded_best_nn_model = tf.keras.models.load_model(best_model_file)\ny_pred_val = loaded_best_nn_model.predict(X_val_scaled_df)\n# ... (rest of your prediction code)\n\n')


# 
# 100%|██████████| 10/10 [01:06<00:00,  6.68s/trial, best loss: 0.28228757065384763]
# Best Hyperparameters:
# {'dropout_rate': 0.3337388960850514, 'epochs': 104.0, 'learning_rate': 0.002509538499018026, 'momentum': 0.5668255997767192}
# 5905/5905 [==============================] - 6s 870us/step - loss: 0.2625
# Best model saved to file best_nn_model_hyperopt.h5
# 806/806 [==============================] - 1s 579us/step
# 
# CPU times: total: 1min 57s
# Wall time: 1min 13s
# 

# do the tuning for X_train_pca_df:




get_ipython().run_cell_magic('time', '', 'import os\nimport numpy as np\nimport tensorflow as tf\nfrom tensorflow import keras\nfrom hyperopt import fmin, tpe, hp\nfrom sklearn.metrics import mean_squared_error\n\n# Define the objective function\ndef objective2(params):\n    model = create_neural_network(\n        learning_rate=params[\'learning_rate\'],\n        momentum=params[\'momentum\'],\n        dropout_rate=params[\'dropout_rate\'],\n        epochs=int(params[\'epochs\']),\n        input_dim=X_train_pca_df\n    )\n    \n    model.compile(optimizer=\'adam\', loss=\'mse\')  # Adjust optimizer and loss if needed\n    \n    model.fit(X_train_pca_df, y_train, validation_data=(X_val_pca_df, y_val))\n    \n    y_pred = model.predict(X_val_pca_df)\n    mse = mean_squared_error(y_val, y_pred)\n    \n    return mse\n\n# Check if the tuned model file exists\ntuned_model_file2 = \'nn_tuned_tuner_hyperopt2.joblib\'\n\nif os.path.exists(tuned_model_file2):\n    # Load the tuned model from the file\n    nn_tuned_tuner_hyperopt2 = joblib.load(tuned_model_file2)\nelse:\n    # Define the hyperparameter search space\n    space = {\n        \'learning_rate\': hp.loguniform(\'learning_rate\', np.log(0.0001), np.log(0.01)),\n        \'momentum\': hp.uniform(\'momentum\', 0.1, 0.9),\n        \'dropout_rate\': hp.uniform(\'dropout_rate\', 0.0, 0.5),\n        \'epochs\': hp.quniform(\'epochs\', 10, 110, 1)\n    }\n    # Perform hyperparameter search\n    nn_tuned_tuner_hyperopt2 = fmin(\n        fn=objective2,\n        space=space,\n        algo=tpe.suggest,\n        max_evals=10  # Number of iterations\n    )\n    # Save the tuned tuner to a file\n    joblib.dump(nn_tuned_tuner_hyperopt2, tuned_model_file2)\n\nprint("Best Hyperparameters:")\nprint(nn_tuned_tuner_hyperopt2)\n\n\n# Get the best hyperparameters\nbest_learning_rate_hyperopt2 = nn_tuned_tuner_hyperopt2[\'learning_rate\']\nbest_momentum_hyperopt2 = nn_tuned_tuner_hyperopt2[\'momentum\']\nbest_dropout_rate_hyperopt2 = nn_tuned_tuner_hyperopt2[\'dropout_rate\']\nbest_epochs_hyperopt2 = int(nn_tuned_tuner_hyperopt2[\'epochs\'])\n\n# Create the best model\nbest_nn_model2 = create_neural_network(\n    learning_rate=best_learning_rate_hyperopt2,\n    momentum=best_momentum_hyperopt2,\n    dropout_rate=best_dropout_rate_hyperopt2,\n    epochs=best_epochs_hyperopt2,\n    input_dim=X_train_pca_df\n)\n\n# Compile the model\nbest_nn_model2.compile(optimizer=\'adam\', loss=\'mse\')  # Adjust optimizer and loss if needed\n\n# Train the best model on the entire training dataset\nbest_nn_model2.fit(X_train_pca_df, y_train, verbose=1)\n\n# Save the best model to disk\nbest_model_file2 = \'best_nn_model_hyperopt2.h5\'\nbest_nn_model2.save(best_model_file2)\nprint(f"Best model saved to file {best_model_file2}")\n\n# Later, we can load the best model and use it for predictions\n#loaded_best_nn_model2 = tf.keras.models.load_model(best_model_file2)\n#y_pred_val = loaded_best_nn_model2.predict(X_val_pca_df)\n# ... (rest of your prediction code)\n\n')


#  1/806 [..............................] - ETA: 48s                              
#  83/806 [==>...........................] - ETA: 0s 
# 168/806 [=====>........................] - ETA: 0s
# 256/806 [========>.....................] - ETA: 0s
# 344/806 [===========>..................] - ETA: 0s
# 430/806 [===============>..............] - ETA: 0s
# 517/806 [==================>...........] - ETA: 0s
# 600/806 [=====================>........] - ETA: 0s
# 688/806 [========================>.....] - ETA: 0s
# 776/806 [===========================>..] - ETA: 0s
# 806/806 [==============================] - 1s 582us/step
# 
# 100%|██████████| 10/10 [01:14<00:00,  7.45s/trial, best loss: 0.28653451183189993]
# Best Hyperparameters:
# {'dropout_rate': 0.1506556154557538, 'epochs': 53.0, 'learning_rate': 0.0007482544731455019, 'momentum': 0.2632593069069997}
# 5905/5905 [==============================] - 6s 856us/step - loss: 0.2597
# Best model saved to file best_nn_model_hyperopt2.h5
# CPU times: total: 2min 6s
# Wall time: 1min 20s

# ### d0. Neural Networks - Keras Tuner skopt  (expensive and errors)

# ### d1. Neural Networks - optuna (too computationally expensive - 3h)




# %%time
# import optuna
# import os
# import json
# import tensorflow as tf
# from tensorflow import keras
# 
# # Define the objective function for Optuna
# def objective(trial):
#     # Sample hyperparameters
#     dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
#     learning_rate = trial.suggest_float('learning_rate', 0.001, 0.1)
#     momentum = trial.suggest_float('momentum', 0.1, 0.9)
#     batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
#     epochs = trial.suggest_categorical('epochs', [50, 100, 150, 200])
# 
#     # Create the Keras model
#     model = create_neural_network(learning_rate=learning_rate, momentum=momentum, dropout_rate=dropout_rate, epochs=epochs)
# 
#     # Set early stopping to prevent overfitting
#     early_stopping = EarlyStopping(patience=7, restore_best_weights=True)
# 
#     # Train the model on the training data and validate on the validation data
#     model.fit(
#         X_train[numerical_features_locationadded],
#         y_train,
#         epochs=epochs,
#         batch_size=batch_size,
#         validation_data=(X_val[numerical_features_locationadded], y_val),
#         callbacks=[early_stopping],
#         verbose=0
#     )
# 
#     # Evaluate the model on the validation data and return the mean squared error
#     val_score = model.evaluate(X_val[numerical_features_locationadded], y_val, verbose=0)[1]
#     return val_score
# 
# def hyperparameter_tuning_optuna_nn(n_trials=100, study_name='my_study_nn'):
#     study_file = study_name + '.pkl'
#     best_hyperparameters_file = 'best_nn_hyperparameters.json'
#     best_model_file = 'best_nn_model.h5'
# 
#     # Check if the study file exists
#     if os.path.exists(study_file):
#         # Load the existing study from the file
#         #study = optuna.create_study(
#         #   study_name=study_name,
#         #  storage=f"sqlite:///{study_file}",
#         #    load_if_exists=True,
#         #)
#         study = optuna.load_study(study_name=study_name, storage=f'sqlite:///{study_file}')
#         print(f"Existing study loaded from {study_file} file")
#         
#         #print(f"Existing study loaded from {study_file} file")
#     else:
#         # Create a new study and perform the hyperparameter search using Optuna
#         study = optuna.create_study(direction='minimize', study_name=study_name, storage=f'sqlite:///{study_file}')
#         study.optimize(objective, n_trials=n_trials)
#         
#         # Save the study to the file
#         study.trials_dataframe().to_csv(study_file)
# 
# 
#     # Check if the best hyperparameters file exists
#     if os.path.exists(best_hyperparameters_file):
#         # Load the best hyperparameters from the file
#         with open(best_hyperparameters_file, 'r') as f:
#             best_params = json.load(f)
#     else:
#         # Get the best hyperparameters from the study
#         best_params = study.best_params
#         # Save the best hyperparameters to a file (JSON)
#         with open(best_hyperparameters_file, 'w') as f:
#             json.dump(best_params, f)
#         print("Best parameters saved to the file, best_nn_hyperparameters.json")
# 
# 
#     # Check if the best model file exists
#     if os.path.exists(best_model_file):
#         # Load the best model from the file
#         best_model = keras.models.load_model(best_model_file)
#     else:
#         # Create and train the best model using the best hyperparameters
#         best_model = create_neural_network(dropout_rate=best_params['dropout_rate'],
#                                            learning_rate=best_params['learning_rate'],
#                                            momentum=best_params['momentum'],
#                                            epochs=best_params['epochs'])
#         # Set early stopping to prevent overfitting
#         early_stopping = EarlyStopping(patience=10, restore_best_weights=True)  #Stop training when a monitored metric has stopped improving.
#         # Train the best model on the full training data and validate on the validation data
#         best_model.fit(X_train[numerical_features_locationadded], y_train,
#                        epochs=best_params['epochs'],
#                        batch_size=best_params['batch_size'],
#                        validation_data=(X_val[numerical_features_locationadded], y_val),
#                        verbose=0, callbacks=[early_stopping])
#         
#         # Save the best model
#         best_model.save(best_model_file)
#         print(f"Best model trained with X_train[numerical_features_locationadded], validated with X_val[numerical_features_locationadded], and saved to the file, {best_model_file}")
#     
# 
# # Call the function to perform hyperparameter tuning
# hyperparameter_tuning_optuna_nn(n_trials=100)
# 
# best_hyperparameters_file = 'best_nn_hyperparameters.json'
# best_model_file = 'best_nn_model.h5'
# if os.path.exists(best_hyperparameters_file):
#     # Load the best hyperparameters from the file
#     with open(best_hyperparameters_file, 'r') as f:
#         best_params_nn = json.load(f)
#     print(f"Best nn hyperparameters loaded from the file, {best_hyperparameters_file}")
# if os.path.exists(best_model_file):
#     # Load the best model from the file
#     best_model_nn = keras.models.load_model(best_model_file)
#     print(f"Best nn model loaded from the file, {best_model_file}")
# 
# """
# # Get the best hyperparameters and model
# best_params = study.best_params
# best_model = create_neural_network(dropout_rate=best_params['dropout_rate'], learning_rate=best_params['learning_rate'], momentum=best_params['momentum'])
# 
# # Train the best model on the full training data
# best_model.fit(X_train, y_train, epochs=100, batch_size=best_params['batch_size'], verbose=0)
# 
# # Evaluate on the test set
# test_score = best_model.evaluate(X_test, y_test, verbose=0)[1]
# print(f"Test Score (MSE): {test_score}")
# """


# Executed in 3h 2m 32s




#best_params_nn


# [I 2023-08-04 16:56:28,899] Trial 99 finished with value: 0.28738343715667725 and parameters: {'dropout_rate': 0.13114790675230317, 'learning_rate': 0.08032125978988923, 'momentum': 0.17225150193359284, 'batch_size': 16, 'epochs': 200}. Best is trial 43 with value: 0.2830849885940552.
# 
# Executed in 3h 2m 32s
# 
# [I 2023-08-04 15:07:06,717] Trial 43 finished with value: 0.2830849885940552 and parameters: {'dropout_rate': 0.1409261291340914, 'learning_rate': 0.09551075408699548, 'momentum': 0.1494320819116141, 'batch_size': 16, 'epochs': 200}. Best is trial 43 with value: 0.2830849885940552.
# 

# best_params:  {'dropout_rate': 0.1409261291340914, 'learning_rate': 0.09551075408699548, 'momentum': 0.1494320819116141, 'batch_size': 16, 'epochs': 200}
# best_trial:  FrozenTrial(number=43, state=TrialState.COMPLETE, values=[0.2830849885940552], datetime_start=datetime.datetime(2023, 8, 4, 15, 3, 31, 538026), datetime_complete=datetime.datetime(2023, 8, 4, 15, 7, 6, 716643), params={'dropout_rate': 0.1409261291340914, 'learning_rate': 0.09551075408699548, 'momentum': 0.1494320819116141, 'batch_size': 16, 'epochs': 200}, user_attrs={}, system_attrs={}, intermediate_values={}, distributions={'dropout_rate': FloatDistribution(high=0.5, log=False, low=0.1, step=None), 'learning_rate': FloatDistribution(high=0.1, log=False, low=0.001, step=None), 'momentum': FloatDistribution(high=0.9, log=False, low=0.1, step=None), 'batch_size': CategoricalDistribution(choices=(16, 32, 64, 128)), 'epochs': CategoricalDistribution(choices=(50, 100, 150, 200))}, trial_id=43, value=None)
# 

# ### e. XGBoost Regression




get_ipython().run_cell_magic('time', '', 'from scipy.stats import randint\n\ndef tune_model_xgb(estimator, param_distributions, train, n_iter=10, cv=kf, scoring=\'neg_mean_squared_error\', random_state=42, model_name=\'\'):\n    model_file = f\'{model_name}_random_search.pkl\'\n    \n    if os.path.exists(model_file):\n        # Load the tuned model from the file if it exists\n        print(f"Loading {model_name} model from {model_file}")\n        tuned_model = joblib.load(model_file)\n    else:\n        # Perform hyperparameter tuning with RandomizedSearchCV if the file does not exist\n        print(f"Tuning {model_name} model...")\n        xgb_random_search = RandomizedSearchCV(\n            estimator=estimator,\n            param_distributions=param_distributions,\n            n_iter=n_iter,\n            cv=cv,\n            scoring=scoring,\n            random_state=random_state,\n            verbose=3,\n            #n_jobs=-1  # Parallelize the tuning process\n        )\n        xgb_random_search.fit(train, y_train)\n        tuned_model = xgb_random_search\n        joblib.dump(tuned_model, model_file)\n        print(f"{model_name} model tuned and saved to {model_file}")\n    \n    return tuned_model\n\nparam_random_xgb = {\n    \'n_estimators\': randint(50, 300),\n    \'learning_rate\': [0.01, 0.1, 0.5],\n    \'max_depth\': [None, 5, 10],\n    \'min_child_weight\': randint(1, 21)\n}\n\n# Tune the XGB model\ntuned_xgb_model = tune_model_xgb(estimator=xgb_model, param_distributions=param_random_xgb, n_iter=10, model_name=\'xgb\', train=X_train_scaled_df)\n\n# We can use the tuned_svr_model for prediction and evaluation\n#y_predict_val_xgb = tuned_xgb_model.best_estimator_.predict(X_val_scaled_df)\n')


# Tuning xgb model...
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# xgb model tuned and saved to xgb_random_search.pkl
# CPU times: total: 1h 57min 32s
# Wall time: 7min 31s
# 




print(X_train_scaled_df)





print(tuned_xgb_model.best_params_)





get_ipython().run_cell_magic('time', '', "# Tune the XGB model\ntuned_xgb_model2 = tune_model_xgb(estimator=xgb_model2, param_distributions=param_random_xgb, n_iter=10, model_name='xgb2', train=X_train_pca_df)\n")


# Tuning xgb2 model...
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# xgb2 model tuned and saved to xgb2_random_search.pkl
# CPU times: total: 1h 53min 30s
# Wall time: 7min 16s




print(tuned_xgb_model.best_params_)


# {'learning_rate': 0.1,
#  'max_depth': None,
#  'min_child_weight': 12,
#  'n_estimators': 138}




print(tuned_xgb_model2.best_params_)


# {'learning_rate': 0.1,
#  'max_depth': None,
#  'min_child_weight': 12,
#  'n_estimators': 138}

# Here we tried to use Randomized Search: Because the hyperparameter grid is still too large, we decided using RandomizedSearchCV instead of GridSearchCV. RandomizedSearchCV samples a fixed number of hyperparameter combinations randomly, which can be more efficient for large grids.

# ## VIII. Compare Model Performance on Validation and Test Set




get_ipython().run_cell_magic('time', '', 'from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n\nmodels = [tuned_rf_model, tuned_rf_model2, tuned_gb_model, tuned_gb_model2, tuned_svr_model_bayesian, tuned_svr_model_bayesian2, best_nn_model, best_nn_model2, tuned_xgb_model,  tuned_xgb_model2]\nmodel_names = [\'Random Forest\', \'Random Forest 2\', \'Gradient Boosting\', \'Gradient Boosting 2\', \'SVR\', \'SVR 2\', \'Neural Networks\', \'Neural Networks 2\', \'XGBoost\',  \'XGBoost 2\']\ntraining_time_model = [\'2min 53s\', \'3min 36s\', \'23.4 s\', \'43.5 s\', \'38min 37s\', \'26min 34s\', \'4min 47s\', \'2min 43 s\', \'9.06 s\', \'6.67 s\']\ntraining_time_tuning = [\'12min 29s\', \'10min 34s\', \'8min 52s\', \'2min 29s\', \'1min 49s\', \'4min 15s\', \'1min 13s\', \'1min 20s\', \'7min 31s\', \'7min 16s\']\n\n# Lists to store the results\nmethods = []\npreprocessing = []\ndata_splits = []\nfeature_extraction = []\nmae_test_set = []\nmae_independent_test_set = []\ntime_model = []\ntime_hyperparameters = []\ntuned_parameters = []\ncross_validation = []\n\nfor model, name, time_m, time_t in zip(models, model_names, training_time_model, training_time_tuning):\n    if model == best_nn_model:\n        y_predict_val = model.predict(X_val_scaled_df)\n        preprocessing.append(\'Norm\')\n    elif model == best_nn_model2:\n        y_predict_val = model.predict(X_val_pca_df)\n        preprocessing.append(\'Norm, \\nPCA\')\n    elif model == tuned_rf_model2 or model == tuned_gb_model2 or model == tuned_svr_model_bayesian2 or model == tuned_xgb_model2:\n        y_predict_val = model.best_estimator_.predict(X_val_pca_df)\n        preprocessing.append(\'Norm, \\nPCA\')\n        cross_validation.append(\'5-fold CV\')\n    else:\n        y_predict_val = model.best_estimator_.predict(X_val_scaled_df)\n        preprocessing.append(\'Norm\')\n        cross_validation.append(\'5-fold CV\')\n    mse = mean_squared_error(y_val, y_predict_val)\n    mae = mean_absolute_error(y_val, y_predict_val)\n    r2 = r2_score(y_val, y_predict_val)\n    print(f"Model: {name}")\n    print(f"Mean Squared Error: {mse:.3f}")\n    print(f"Mean Absolute Error: {mae:.3f}")\n    print(f"R-squared: {r2:.2f}\\n\\n")\n    \n    # Append results to the lists\n    methods.append(name)\n    #preprocessing.append(\'PCA\' if \'with X_val_pca\' in name else \'Norm\')\n    data_splits.append(\'random\')\n    \n    feature_extraction.append(\'None\' if \'with X_val_pca\' in name else \'Helper Function\')\n    mae_test_set.append(mae)\n    time_model.append(time_m)  # model training time\n    time_hyperparameters.append(time_t)  # training time for hyperparameter tuning\n    if model == best_nn_model:\n        tuned_parameters.append(nn_tuned_tuner_hyperopt)\n    elif model == best_nn_model2:\n        tuned_parameters.append(nn_tuned_tuner_hyperopt2)\n    else:\n        tuned_parameters.append(model.best_params_)\n        \n# Evaluate best models on test sets:\nfor model, name in zip(models, model_names):\n    if model == best_nn_model:\n        y_predict_test = model.predict(X_test_scaled_df)\n    elif model == best_nn_model2:\n        y_predict_test = model.predict(X_test_pca_df)\n    elif model == tuned_rf_model2 or model == tuned_gb_model2 or model == tuned_svr_model_bayesian2 or model == tuned_xgb_model2:\n        y_predict_test = model.best_estimator_.predict(X_test_pca_df)\n    else:\n        y_predict_test = model.best_estimator_.predict(X_test_scaled_df)\n    mae_test = mean_absolute_error(y_test, y_predict_test)\n    mae_independent_test_set.append(mae_test)\n\n# Create a DataFrame to store the results\nfinal_results_df = pd.DataFrame({\n    \'Method\': methods,\n    \'Preprocessing\': preprocessing,\n    \'Data Splits\': data_splits,\n    \'Feature Extraction\': feature_extraction,\n    \'MAE on Test Set (mm)\': mae_test_set,\n    \'MAE on Independent Test Set\': mae_independent_test_set,\n    \'Training Time for Model training\': time_model,\n    \'Training Time for Hyperparameter Tuning\': time_hyperparameters,\n    \'Tuned Parameters (Best Hyperparameters)\': tuned_parameters,\n    \'CV (Hyperparameter tuning)\': cross_validation\n})\n\n\n\n# Display the table\nprint(final_results_df)\nfinal_results_df\n')


# CPU times: total: 1min 58s
# Wall time: 1min 57s




from tabulate import tabulate

table_str = tabulate(final_results_df, headers='keys', tablefmt='grid')
print(table_str)





final_results_df


# Lower MSE and MAE values indicate better performance, where 0 is the best possible value.
# Higher R-squared values (closer to 1) indicate a better fit of the model to the data.

# ### Visualize Model Performances




import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.violinplot(data=final_results_df, x='Method', y='MAE on Test Set (mm)')
plt.xlabel('Method')
plt.ylabel('MAE on Test Set (mm)')
plt.title('MAE Comparison on Test Sets')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()






import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.pointplot(data=final_results_df, x='Method', y='MAE on Independent Test Set', dodge=True)
plt.xlabel('Method')
plt.ylabel('MAE on Independent Test Set')
plt.title('MAE Comparison on Independent Test Sets')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()






import matplotlib.pyplot as plt
import re

# Extract minutes and seconds from the training time strings
def extract_time_in_seconds(time_str):
    minutes_match = re.search(r'(\d+)min', time_str)
    seconds_match = re.search(r'(\d+)s', time_str)
    
    minutes = int(minutes_match.group(1)) if minutes_match else 0
    seconds = int(seconds_match.group(1)) if seconds_match else 0
    
    return minutes * 60 + seconds

# Convert training times to seconds and add them as new columns
final_results_df['Model Training Time (sec)'] = final_results_df['Training Time for Model training'].apply(extract_time_in_seconds)
final_results_df['Hyperparameter Tuning Time (sec)'] = final_results_df['Training Time for Hyperparameter Tuning'].apply(extract_time_in_seconds)

# Plotting
plt.figure(figsize=(10, 6))

# Plot Model Training Time in seconds
plt.barh(final_results_df['Method'], final_results_df['Model Training Time (sec)'], label='Model Training', alpha=0.7)

# Plot Hyperparameter Tuning Time in seconds on top of Model Training Time
plt.barh(final_results_df['Method'], final_results_df['Hyperparameter Tuning Time (sec)'], left=final_results_df['Model Training Time (sec)'], label='Hyperparameter Tuning', alpha=0.7)

# Adding labels and title
plt.xlabel('Time (seconds)')
plt.ylabel('Method')
plt.title('Training Time Comparison (in Seconds)')

# Create a legend
plt.legend()

# Add tight layout and show the plot
plt.tight_layout()
plt.show()





def extract_time_in_minutes(time_str):
    minutes_match = re.search(r'(\d+)min', time_str)
    seconds_match = re.search(r'(\d+)s', time_str)
    
    minutes = int(minutes_match.group(1)) if minutes_match else 0
    seconds = int(seconds_match.group(1)) if seconds_match else 0
    
    total_minutes = minutes + seconds / 60  # Convert seconds to minutes
    return total_minutes

# Convert training times to minutes and add them as new columns
final_results_df['Model Training Time (min)'] = final_results_df['Training Time for Model training'].apply(extract_time_in_minutes)
final_results_df['Hyperparameter Tuning Time (min)'] = final_results_df['Training Time for Hyperparameter Tuning'].apply(extract_time_in_minutes)

# Plotting
plt.figure(figsize=(10, 6))

# Plot Model Training Time in minutes
plt.bar(final_results_df['Method'], final_results_df['Model Training Time (min)'], label='Model Training', alpha=0.7)

# Plot Hyperparameter Tuning Time in minutes on top of Model Training Time
plt.bar(final_results_df['Method'], final_results_df['Hyperparameter Tuning Time (min)'], bottom=final_results_df['Model Training Time (min)'], label='Hyperparameter Tuning', alpha=0.7)

# Adding labels and title
plt.xlabel('Method')
plt.ylabel('Time (minutes)')
plt.title('Training Time Comparison (in Minutes)')
plt.xticks(rotation=45, ha='right')

# Create a legend outside of plt.bar() using plt.legend()
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Add tight layout and show the plot
plt.tight_layout()
plt.show()


# ## IX. Choose the best performing model and evaluate on the test set




# Find the best performing model based on the lowest MAE on the test set
best_model_index = final_results_df['MAE on Test Set (mm)'].idxmin()
best_model_final = models[best_model_index]

# Save the best_model_final using joblib
joblib.dump(best_model_final, 'best_model_final.pkl')


# Predict on the test set using the best model
if best_model_final == best_nn_model:
    y_predict_test = best_model_final.predict(X_test_scaled_df)
    print("Test set used: X_test_scaled_df ", X_test_scaled_df)
elif best_model_final == best_nn_model2:
    y_predict_test = best_model_final.predict(X_test_pca_df)
    print("Test set used: X_test_pca_df ", X_test_pca_df)
elif best_model_final in (tuned_rf_model2, tuned_gb_model2, tuned_svr_model_bayesian2, tuned_xgb_model2):
    y_predict_test = best_model_final.best_estimator_.predict(X_test_pca_df)
    best_model_final = best_model_final.best_estimator_
    print("Test set used: X_test_pca_df ", X_test_pca_df)
else:
    y_predict_test = best_model_final.best_estimator_.predict(X_test_scaled_df)
    best_model_final = best_model_final.best_estimator_
    print("Test set used: X_test_scaled_df ", X_test_scaled_df)


# Get the best hyperparameters for the best model
best_hyperparameters_final = final_results_df.loc[best_model_index, 'Tuned Parameters (Best Hyperparameters)']

# Calculate evaluation metrics
mse_test = mean_squared_error(y_test, y_predict_test)
mae_test = mean_absolute_error(y_test, y_predict_test)
r2_test = r2_score(y_test, y_predict_test)
print(f"Best Performing Model: {best_model_final}")
print(f"Best Hyperparameters for Best Model: {best_hyperparameters_final}")
print(f"Mean Squared Error on Test Set: {mse_test:.2f}")
print(f"Mean Absolute Error on Test Set: {mae_test:.2f}")
print(f"R-squared on Test Set: {r2_test:.2f}")


# Best Performing Model: 
# - XGBRegressor(base_score=None, booster=None, callbacks=None,
#              colsample_bylevel=None, colsample_bynode=None,
#              colsample_bytree=None, early_stopping_rounds=None,
#              enable_categorical=False, eval_metric=None, feature_types=None,
#              gamma=None, gpu_id=None, grow_policy=None, importance_type=None,
#              interaction_constraints=None, learning_rate=0.1, max_bin=None,
#              max_cat_threshold=None, max_cat_to_onehot=None,
#              max_delta_step=None, max_depth=None, max_leaves=None,
#              min_child_weight=12, missing=nan, monotone_constraints=None,
#              n_estimators=138, n_jobs=None, num_parallel_tree=None,
#              predictor=None, random_state=42, ...)
#              
# Test set used: X_test_scaled_df 
# Best Hyperparameters for Best Model: {'learning_rate': 0.1, 'max_depth': None, 'min_child_weight': 12, 'n_estimators': 138}
# Mean Squared Error on Test Set: 0.28
# Mean Absolute Error on Test Set: 0.39
# R-squared on Test Set: 0.37

# We can use this best model to make predictions on new data. Lower values of MSE indicate better performance.
# 

# 

# ## X. Using best model XGBRegressor with the best Hyperparameters to make predictions on new data.




X_test_scaled_df


# We can use this best model (XGBRegressor) to predict outcomes on unseen data - Assuming we have new data in X_test_scaled_df:




print(X_test_scaled_df.shape)
print(y_test.shape)





get_ipython().run_cell_magic('time', '', 'import time\nt1 = time.time()\n# Retrain model on combined training and validation sets\nbest_model_final.fit(X_train_scaled_df, y_train)\nt2 = time.time()\nprint(f"Model training took {t2 - t1} seconds")\n\n\n# evaluate performance on test set\nfinal_test_predictions = best_model_final.predict(X_test_scaled_df)\n\nprint("Test predictions using the Best Hyperparameters with the best model XGBRegressor:\\n", final_test_predictions)\n\n# evaluate the performance of the predictions using metrics from sklearn\nfrom sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\nmse_final = mean_squared_error(y_test, final_test_predictions)\nmae_final = mean_absolute_error(y_test, final_test_predictions)\nr2_final = r2_score(y_test, final_test_predictions)\nprint(f"Test predictions results of the best model {best_model_final}:")\nprint("Mean Squared Error:", mse_final)\nprint("Mean Absolute Error:", mae_final)\nprint("R-squared:", r2_final)\n')


# Test predictions results of the best model XGBRegressor(base_score=None, booster=None, callbacks=None,
#              colsample_bylevel=None, colsample_bynode=None,
#              colsample_bytree=None, early_stopping_rounds=None,
#              enable_categorical=False, eval_metric=None, feature_types=None,
#              gamma=None, gpu_id=None, grow_policy=None, importance_type=None,
#              interaction_constraints=None, learning_rate=0.1, max_bin=None,
#              max_cat_threshold=None, max_cat_to_onehot=None,
#              max_delta_step=None, max_depth=None, max_leaves=None,
#              min_child_weight=12, missing=nan, monotone_constraints=None,
#              n_estimators=138, n_jobs=None, num_parallel_tree=None,
#              predictor=None, random_state=42, ...):
# Mean Squared Error: 0.28157587010381213
# Mean Absolute Error: 0.39358361894732236
# R-squared: 0.36845470106621103
# CPU times: total: 1min 47s
# Wall time: 9.2 s

# #### Listing predictions




print(X_test)





# Create a new DataFrame to store the decoded results
decoded_df = X_test.copy()

# Iterate through the indices and replace 'location_' in decoded_df with the original location names 
# we will use indices to cross-check with original df:
for index, row in decoded_df.iterrows():
    original_location = df.loc[index, 'LOCATION_']
    decoded_df.at[index, 'LOCATION_'] = original_location

# Print or view the updated decoded_df
print(decoded_df)
decoded_df.to_csv('decoded_df.csv', index=False)


# # Use the 'location_mapping' dictionary to inverse transform the encoded locations
# decoded_df['LOCATION_'] = decoded_df['LOCATION_'].map(location_mapping)


# Now we can see the actual location names along with the predicted MAG values
decoded_df['PREDICTED_MAG'] = final_test_predictions
decoded_df['MAG'] = y_test

# Print the DataFrame to see the results
decoded_df.info()
decoded_df


# #### Biggest Predicted MAG vs. Biggest Actual MAG




# Find the index of the row with the biggest predicted MAG
index_max_predicted = decoded_df['PREDICTED_MAG'].idxmax()

# Find the index of the row with the biggest actual MAG
index_max_actual = decoded_df['MAG'].idxmax()

# Get the info for the biggest predicted MAG:
location_predicted_max = decoded_df.loc[index_max_predicted, 'LOCATION_']
date_predicted_max = decoded_df.loc[index_max_predicted, 'DATE_']
predicted_mag_value_for_predicted_max = decoded_df.loc[index_max_predicted, 'PREDICTED_MAG']  # Get predicted MAG value
actual_mag_value_for_predicted_max = decoded_df.loc[index_max_predicted, 'MAG']  # Get actual MAG value for the predicted MAG 


# Get the info for the biggest actual MAG:
location_actual_max = decoded_df.loc[index_max_actual, 'LOCATION_']
date_actual_max = decoded_df.loc[index_max_actual, 'DATE_']
actual_mag_value_for_actual_max = decoded_df.loc[index_max_actual, 'MAG']  # Get actual MAG value
predicted_mag_value_for_actual_max = decoded_df.loc[index_max_actual, 'PREDICTED_MAG']  # Get predicted MAG value for the actual MAG


# Print the results
print("Biggest Predicted MAG:")
print("Date:", date_predicted_max)
print("Location:", location_predicted_max)
print("Predicted MAG Value:", predicted_mag_value_for_predicted_max)
print("Actual MAG Value for this Location:", actual_mag_value_for_predicted_max)  # Print actual MAG value


print("\nBiggest Actual MAG:")
print("Date:", date_actual_max)
print("Location:", location_actual_max)
print("Actual MAG Value:", actual_mag_value_for_actual_max)
print("Predicted MAG Value for this Location:", predicted_mag_value_for_actual_max)  # Print predicted MAG value


# - Biggest Predicted MAG:
# Date: 2017-11-13 06:14:35
# Location: IRAN                                              
# Predicted MAG Value: 4.737505
# Actual MAG Value for this Location: 4.2
# 
# 
# - Biggest Actual MAG:
# Date: 2020-05-02 15:51:06
# Location: GIRIT ADASI ACIKLARI (AKDENIZ)                    İlksel
# Actual MAG Value: 6.6
# Predicted MAG Value for this Location: 3.7084682
# 




import matplotlib.pyplot as plt

# Create a scatter plot to visualize the biggest predicted MAG
plt.figure(figsize=(10, 6))
plt.scatter(date_predicted_max, predicted_mag_value_for_predicted_max, color='red', label='Predicted MAG')
plt.scatter(date_predicted_max, actual_mag_value_for_predicted_max, color='blue', label='Actual MAG')
plt.annotate(f'Predicted: {predicted_mag_value_for_predicted_max:.2f}', 
             (date_predicted_max, predicted_mag_value_for_predicted_max), 
             textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, color='red')
plt.annotate(f'Actual: {actual_mag_value_for_predicted_max:.2f}', 
             (date_predicted_max, actual_mag_value_for_predicted_max), 
             textcoords="offset points", xytext=(0,-20), ha='center', fontsize=10, color='blue')
plt.xlabel('Date')
plt.ylabel('Magnitude')
plt.title('Biggest Predicted MAG')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()

# Create a scatter plot to visualize the biggest actual MAG
plt.figure(figsize=(10, 6))
plt.scatter(date_actual_max, actual_mag_value_for_actual_max, color='blue', label='Actual MAG')
plt.scatter(date_actual_max, predicted_mag_value_for_actual_max, color='red', label='Predicted MAG')
plt.annotate(f'Actual: {actual_mag_value_for_actual_max:.2f}', 
             (date_actual_max, actual_mag_value_for_actual_max), 
             textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, color='blue')
plt.annotate(f'Predicted: {predicted_mag_value_for_actual_max:.2f}', 
             (date_actual_max, predicted_mag_value_for_actual_max), 
             textcoords="offset points", xytext=(0,-20), ha='center', fontsize=10, color='red')
plt.xlabel('Date')
plt.ylabel('Magnitude')
plt.title('Biggest Actual MAG')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()





import matplotlib.pyplot as plt

# Create a bar plot for actual and predicted MAG values
locations = ['Biggest Predicted', 'Biggest Actual']
actual_values = [actual_mag_value_for_actual_max, actual_mag_value_for_actual_max]
predicted_values = [predicted_mag_value_for_predicted_max, predicted_mag_value_for_predicted_max]

plt.figure(figsize=(10, 6))
plt.bar(locations, actual_values, width=0.4, align='center', label='Actual MAG')
plt.bar(locations, predicted_values, width=0.4, align='edge', label='Predicted MAG')
plt.xlabel('Location')
plt.ylabel('MAG Value')
plt.title('Comparison of Actual and Predicted MAG Values')
plt.legend()
plt.show()

# Create a line plot for MAG values over time (date)
plt.figure(figsize=(10, 6))
plt.plot([date_predicted_max, date_actual_max], [predicted_mag_value_for_predicted_max, actual_mag_value_for_actual_max], marker='o')
plt.xlabel('Date')
plt.ylabel('MAG Value')
plt.title('Trend of MAG Values over Time')
plt.xticks(rotation=45)
plt.grid()
plt.show()





decoded_df.info()


# #### Visualising Predictions




import matplotlib.pyplot as plt
import numpy as np

# Create an array of indices for the data points
indices = np.arange(len(y_test))

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the actual test data (depth vs. magnitude)
ax.scatter(indices, y_test, label='Actual Test Data', color='blue', alpha=0.5, s=10)

# Plot the predicted test data (depth vs. predicted magnitude)
ax.plot(indices, decoded_df['PREDICTED_MAG'], label='Predicted Test Data', color='red')

# Add labels and title
ax.set_xlabel('Data Point Index')
ax.set_ylabel('Magnitude')
ax.set_title('Actual vs. Predicted Test Data')

# Add legend
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()





plt.figure(figsize=(12, 6))
plt.plot(decoded_df['DATE_'], decoded_df['MAG'], label='Actual Magnitude', color='blue')
plt.plot(decoded_df['DATE_'], decoded_df['PREDICTED_MAG'], label='Predicted Magnitude', color='red')
plt.xlabel('Date')
plt.ylabel('Magnitude')
plt.title('Actual vs. Predicted Magnitude Over Time')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()





from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(decoded_df['MAG'], decoded_df['PREDICTED_MAG'], decoded_df['DEPTH'], c=decoded_df['DEPTH'], cmap='plasma')
ax.set_xlabel('Actual Magnitude')
ax.set_ylabel('Predicted Magnitude')
ax.set_zlabel('Depth')
ax.set_title('3D Scatter Plot of Magnitude and Depth')
plt.colorbar(scatter, label='Depth')
plt.tight_layout()
plt.savefig('3D Scatter Plot of Magnitude and Depth.png')
plt.show()





decoded_df





import matplotlib.pyplot as plt
import seaborn as sns


# Set the style for the plot
sns.set(style="whitegrid")

# Create a figure and axis
plt.figure(figsize=(10, 6))

# Plot the error distribution using a violin plot
ax = sns.violinplot(x=decoded_df['DATE_'], y=decoded_df['MAE'], color='r')

# Adding labels and title
plt.xlabel('Date')
plt.ylabel('Magnitude Error')
plt.title('Error Plot: Actual vs. Predicted Magnitude')

# Rotating x-axis labels for better readability
plt.xticks(rotation=45)

# Display the plot
plt.tight_layout()
plt.savefig('Error_plot_violin.png')
plt.show()





import matplotlib.pyplot as plt

# Create a figure and axis
plt.figure(figsize=(10, 6))

# Create a contour plot
contour = plt.tricontourf(decoded_df['MAG'], decoded_df['DEPTH'], decoded_df['PREDICTED_MAG'], cmap='viridis')

# Adding colorbar
cbar = plt.colorbar(contour)
cbar.set_label('Predicted Magnitude')

# Adding labels and title
plt.xlabel('Actual Magnitude')
plt.ylabel('Depth')
plt.title('Contour Plot: Predicted Magnitude based on Actual Magnitude and Depth')

# Display the plot
plt.tight_layout()
plt.savefig('Contour_plot.png')
plt.show()





import matplotlib.pyplot as plt

# Assuming you have a DataFrame named 'decoded_df' with 'DATE' and 'PREDICTED_MAG' columns

# Sort the DataFrame by date
decoded_df_sorted = decoded_df.sort_values('DATE_')

# Plotting MAG values over time for predicted values
plt.figure(figsize=(10, 6))
plt.plot(decoded_df_sorted['DATE_'], decoded_df_sorted['PREDICTED_MAG'], marker='o', label='Predicted MAG')
plt.xlabel('Date')
plt.ylabel('MAG Value')
plt.title('MAG Values over Time (Predicted)')
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.show()

# Assuming you also have a 'MAG' column for real MAG values

# Sort the DataFrame by date
decoded_df_sorted = decoded_df.sort_values('DATE_')

# Plotting MAG values over time for real values
plt.figure(figsize=(10, 6))
plt.plot(decoded_df_sorted['DATE_'], decoded_df_sorted['MAG'], marker='o', color='orange', label='Real MAG')
plt.xlabel('Date')
plt.ylabel('MAG Value')
plt.title('MAG Values over Time (Real)')
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.show()





from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

# Assuming you have the latitude and longitude values for the test data
test_latitudes = X_test["LAT"].tolist()
test_longitudes = X_test["LNG"].tolist()

# Create a Basemap object
#m = Basemap(projection='mill', llcrnrlat=-80, urcrnrlat=80, llcrnrlon=-180, urcrnrlon=180, lat_ts=20, resolution='c')
m = Basemap(projection='mill', llcrnrlat=33, urcrnrlat=45, llcrnrlon=21, urcrnrlon=46, resolution='l')
#m = Basemap(projection='mill', llcrnrlat=25, urcrnrlat=45, llcrnrlon=35, urcrnrlon=45, resolution='l')

# Convert latitudes and longitudes to Basemap coordinates
x, y = m(test_longitudes, test_latitudes)

# Plot the locations with colors representing the predicted MAG values
fig = plt.figure(figsize=(17, 11))
plt.title("Predicted MAG values at locations")
m.scatter(x, y, s=100, c=final_test_predictions, cmap='coolwarm', edgecolors='k', alpha=0.7)
#m.plot(x, y, "o", markersize=5, color='blue')
m.drawcoastlines()
m.fillcontinents(color='coral', lake_color='aqua')
m.drawmapboundary()
m.drawcountries()
plt.colorbar(label='Predicted MAG')

# Save the plot to a file (e.g., PNG format)
plt.savefig('line_plot_Predicted MAG values at locations.png')
plt.savefig('line_plot_Predicted MAG values at locations.pdf')  

plt.show()





from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

# Assuming you have the latitude and longitude values for the test data
test_latitudes = X_test["LAT"].tolist()
test_longitudes = X_test["LNG"].tolist()

# Create a Basemap object
m = Basemap(projection='mill', llcrnrlat=33, urcrnrlat=45, llcrnrlon=21, urcrnrlon=46, resolution='l')

# Convert latitudes and longitudes to Basemap coordinates
x, y = m(test_longitudes, test_latitudes)

# Create a figure and axis
fig = plt.figure(figsize=(17, 11))
ax = fig.add_subplot(1, 1, 1)

# Plot the map
m.drawcoastlines()
m.fillcontinents(color='coral', lake_color='aqua')
m.drawmapboundary()
m.drawcountries()

# Plot the locations with colors representing the predicted MAG values on top of the map
scatter = ax.scatter(x, y, s=100, c=final_test_predictions, cmap='coolwarm', edgecolors='k', alpha=0.7, zorder=10)

# Add a colorbar
cbar = plt.colorbar(scatter, label='Predicted MAG')

# Save the plot to a file (e.g., PNG format)
plt.savefig('scatter_plot_Predicted_MAG_values_at_locations.png')
plt.savefig('scatter_plot_Predicted_MAG_values_at_locations.pdf')  

plt.show()


decoded_df_sorted





plt.figure(figsize=(12, 6))
plt.plot(decoded_df['DATE_'], decoded_df['PREDICTED_MAG'], label='Predicted MAG')
plt.plot(decoded_df['DATE_'], decoded_df['MAG'], label='Actual MAG')
plt.xlabel('Date')
plt.ylabel('MAG Value')
plt.title('Predicted and Actual MAG Values Over Time')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()



decoded_df.columns




decoded_df




# Sort by mean MAE in descending order
sorted_df = grouped_df.sort_values(by='MAE', ascending=False)

# Select the top N locations to visualize
top_n = 10  # You can adjust this number as needed
top_locations = sorted_df.head(top_n)

# Set the style for seaborn plots
sns.set(style="whitegrid")

# Create a bar plot of mean MAE for the top locations
plt.figure(figsize=(12, 6))
sns.barplot(x='LOCATION_', y='MAE', data=top_locations)
plt.xlabel('Location')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title(f'Top {top_n} Locations by Mean Absolute Error')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()





plt.figure(figsize=(10, 6))
prediction_errors = decoded_df['PREDICTED_MAG'] - decoded_df['MAG']
plt.hist(prediction_errors, bins=20, edgecolor='black')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Histogram of Prediction Errors')
plt.tight_layout()
plt.show()





plt.figure(figsize=(10, 6))
decoded_df['Prediction Error'] = prediction_errors
plt.boxplot(decoded_df.groupby('LOCATION_')['Prediction Error'].apply(list).values)
plt.xlabel('Location')
plt.ylabel('Prediction Error')
plt.title('Box Plot of Prediction Errors by Location')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# create a map and add markers for each data point, where the marker color represents the predicted MAG value and the marker icon represents the actual MAG value:




get_ipython().run_cell_magic('time', '', 'import folium\nfrom folium.plugins import MarkerCluster\n\n# Create a folium map centered at the mean coordinates\nmean_lat, mean_lng = decoded_df[\'LAT\'].mean(), decoded_df[\'LNG\'].mean()\nmap_object = folium.Map(location=[mean_lat, mean_lng], zoom_start=5)\n\n# Create a marker cluster for both predicted and actual MAG values\nmarker_cluster = MarkerCluster().add_to(map_object)\n\n# Add markers for predicted and actual MAG values\nfor index, row in decoded_df.iterrows():\n    folium.Marker(location=[row[\'LAT\'], row[\'LNG\']],\n                  popup=f\'Predicted MAG: {row["PREDICTED_MAG"]}\',\n                  icon=folium.Icon(color=\'blue\')).add_to(marker_cluster)\n\n    folium.Marker(location=[row[\'LAT\'], row[\'LNG\']],\n                  popup=f\'Actual MAG: {row["MAG"]}\',\n                  icon=folium.Icon(color=\'red\')).add_to(marker_cluster)\n\n# Display the map\nmap_object\n')




map_object.save('map_MAG_PredictedMAG.html')
import webbrowser
webbrowser.open('map_MAG_PredictedMAG.html')




import folium
from folium.plugins import MarkerCluster

# Create a folium map centered at the mean coordinates
mean_lat, mean_lng = decoded_df['LAT'].mean(), decoded_df['LNG'].mean()
map_object = folium.Map(location=[mean_lat, mean_lng], zoom_start=5)

# Create a marker cluster for both predicted and actual MAG values
marker_cluster = MarkerCluster().add_to(map_object)

# Define how many markers you want to show (for example, every 100th row)
marker_step = 100

# Create a list to store markers
marker_list = []

# Loop through the DataFrame
for index, row in decoded_df.iterrows():
    if index % marker_step == 0:  # Only add markers for every Nth row
        predicted_marker = folium.Marker(
            location=[row['LAT'], row['LNG']],
            popup=f'Predicted MAG: {row["PREDICTED_MAG"]}',
            icon=folium.Icon(color='blue')
        )
        
        actual_marker = folium.Marker(
            location=[row['LAT'], row['LNG']],
            popup=f'Actual MAG: {row["MAG"]}',
            icon=folium.Icon(color='red')
        )
        
        marker_list.extend([predicted_marker, actual_marker])

# Add all markers to the marker cluster
for marker in marker_list:
    marker.add_to(marker_cluster)

# Display the map
map_object


map_object.save('map_MAG_PredictedMAG.html')
import webbrowser
webbrowser.open('map_MAG_PredictedMAG.html')


# on the map on html page, it shows predicted mag and actual mag values on locations

# plot the predicted MAG and MAG values on the Turkey map:




import geopandas as gpd
import matplotlib.pyplot as plt

# Load the shapefile of Turkey
turkey_map = gpd.read_file("D:\ThomasMoreStudy\Fase-5-(F2-Semester-2)\PythonforA.I.(YT6408-2023)\Project/gadm41_TUR_1.shp")

# Create a GeoDataFrame from the DataFrame with latitude and longitude information
gdf = gpd.GeoDataFrame(decoded_df, geometry=gpd.points_from_xy(decoded_df['LNG'], decoded_df['LAT']))

# Set up the plot
fig, ax = plt.subplots(figsize=(17, 12))
turkey_map.boundary.plot(ax=ax, linewidth=0.8, color='black')

# Plot the predicted MAG values with larger markers and higher transparency
predicted_plot = gdf.plot(ax=ax, markersize=decoded_df['PREDICTED_MAG'] * 20, cmap='Reds', alpha=0.7, label='Predicted MAG')

# Plot the actual MAG values with larger markers and higher transparency
actual_plot = gdf.plot(ax=ax, markersize=decoded_df['MAG'] * 20, cmap='Blues', alpha=0.7, label='MAG')

# Set the title
plt.title("Predicted MAG and MAG on Turkey Map")

# Add a colorbar and legend
cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])  # [x, y, width, height]
plt.colorbar(predicted_plot.get_children()[0], cax=cax)
cax.set_ylabel('MAG Density')

# Show the plot
plt.show()



print(decoded_df)


# # end of the project notes:

# For the AI project ProjectAIEartquake.ipynb:
# - PyCharm used with anaconda environment(conda) as IDE.
# - Jupyter Notebook was used.
# - As a result, I would choose XGBRegressor model from XGBoost(eXtreme Gradient Boosting library) because it is very well fitted to large dataset of our project, and for performance matter, it is quite well and fast. Being its MAE the least on test set I would choose this model.

