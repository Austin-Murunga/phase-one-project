
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

aviation_df= pd.read_csv(r'data/AviationData.csv',encoding='Windows-1252')

aviation_df.head()

aviation_df.shape

aviation_df.columns
aviation_df.info()
aviation_df.describe()
aviation_df.tail()
#checking for null values 
aviation_df.isna().sum() #returns the sum of null values
#i will use Event.Id as my primary key
duplicate_count = aviation_df.duplicated(subset='Event.Id').sum()

#drop duplicates in event id 
event_id_cleaned = aviation_df.drop_duplicates(subset='Event.Id')
columns_to_drop=['Accident.Number', 'Event.Date', 'Location', 'Country',
'Latitude', 'Longitude', 'Airport.Code', 'Airport.Name', 'Registration.Number', 'FAR.Description',
'Schedule', ]
df_cleaned = aviation_df.drop(columns= columns_to_drop)
df_cleaned
#inspection of the cleaned data 
df_cleaned.columns
df_cleaned.head()
def fill_nan_aircraft_category(df_cleaned):
    def fill_mode_or_unknown(series):
        mode = series.mode()
        if not mode.empty:
            return series.fillna(mode[0])
        else:
            return series.fillna('Unknown')

    df_cleaned['Aircraft.Category'] = df_cleaned.groupby('Make')['Aircraft.Category'].apply(fill_mode_or_unknown).reset_index(level=0, drop=True)
    return df_cleaned
#check for null values
df_cleaned.isna().sum()

# Fill numerical columns with mean or median
df_cleaned['Total.Fatal.Injuries'].fillna(df_cleaned['Total.Fatal.Injuries'].median(), inplace=True)
df_cleaned['Total.Serious.Injuries'].fillna(df_cleaned['Total.Serious.Injuries'].median(), inplace=True)
df_cleaned['Total.Minor.Injuries'].fillna(df_cleaned['Total.Minor.Injuries'].median(), inplace=True)
df_cleaned['Total.Uninjured'].fillna(df_cleaned['Total.Uninjured'].median(), inplace=True)
df_cleaned['Number.of.Engines'].fillna(df_cleaned['Number.of.Engines'].median(), inplace=True)


# Fill categorical columns with mode
df_cleaned['Make'].fillna(df_cleaned['Make'].mode()[0], inplace=True)
df_cleaned['Model'].fillna(df_cleaned['Model'].mode()[0], inplace=True)
df_cleaned['Investigation.Type'].fillna(df_cleaned['Investigation.Type'].mode()[0], inplace=True)
df_cleaned['Injury.Severity'].fillna(df_cleaned['Injury.Severity'].mode()[0], inplace=True)
df_cleaned['Aircraft.damage'].fillna(df_cleaned['Aircraft.damage'].mode()[0], inplace=True)
df_cleaned['Amateur.Built'].fillna(df_cleaned['Amateur.Built'].mode()[0], inplace=True)
df_cleaned['Engine.Type'].fillna(df_cleaned['Engine.Type'].mode()[0], inplace=True)
df_cleaned['Purpose.of.flight'].fillna(df_cleaned['Purpose.of.flight'].mode()[0], inplace=True)
df_cleaned['Weather.Condition'].fillna(df_cleaned['Weather.Condition'].mode()[0], inplace=True)
df_cleaned['Broad.phase.of.flight'].fillna(df_cleaned['Broad.phase.of.flight'].mode()[0], inplace=True)
df_cleaned['Report.Status'].fillna(df_cleaned['Report.Status'].mode()[0], inplace=True)

# Drop columns with high missing values or not relevant
df_cleaned.drop(columns=['Air.carrier', 'Publication.Date'], inplace=True)

# Fill 'Aircraft.Category' using the custom function
df_cleaned = fill_nan_aircraft_category(df_cleaned)

# Check remaining missing values
print(df_cleaned.isna().sum())
df_cleaned.shape
df_cleaned.head()
#dealing with duplicates in categories, they are all the same thing
df_cleaned['Make'] = df_cleaned['Make'].str.upper()
# Grouping by 'Model' and counting accidents
accidents_by_model = df_cleaned['Make'].value_counts()

# Selecting the top 10 models with the most accidents
top_10_models = accidents_by_model.head(10)

# Plotting
plt.figure(figsize=(14, 7))
top_10_models.plot(kind='bar', color='black', width=0.4)
plt.title('Top 10 Aircraft Models with Most Accidents')
plt.xlabel('Aircraft Model')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45)
plt.show()
#Fatality Rates by Aircraft Model{Cessna}

# Filter data for 'Cessna' models
cessna_data = df_cleaned[df_cleaned['Make'] == 'CESSNA']

# Calculate the number of accidents for each Cessna model
accidents_by_model = cessna_data['Model'].value_counts().head(10)

# Calculate the total number of fatal injuries for each Cessna model
fatalities_by_model = cessna_data.groupby('Model')['Total.Fatal.Injuries'].sum()

# Calculate the fatality rate (total fatal injuries / number of accidents)
fatality_rate = (fatalities_by_model / accidents_by_model).dropna().sort_values(ascending=False)

# Plot the fatality rates for Cessna models
plt.figure(figsize=(14, 7))
plt.title('Fatality Rates by Aircraft Model{Cessna}')
fatality_rate.plot(kind='bar', color='black', width= 0.4)
plt.xlabel('Cessna Model')
plt.ylabel('Fatality Rate %')
plt.xticks(rotation=45)
plt.tight_layout()  # Adjust layout to make room for labels
plt.show()
plt.figure(figsize=(12, 6))
sns.countplot(data=df_cleaned, y='Broad.phase.of.flight', order=df_cleaned['Broad.phase.of.flight'].value_counts().index, color='black') 
plt.title('Number of Accidents by Broad Phase of Flight')
plt.xlabel('Number of accidents')
plt.ylabel('Broad phase of flight')
plt.show()
#covert back to csv 
df_cleaned.to_csv("df_cleaned.csv", na_rep='NA', index=False)
pwd
