import wbdata
import pycountry
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class DataManipulation:

    def __init__(self) -> None:
        pass

    def get_wb_data(self, indicators, date_range, countries='all'):
        
        print(f'Downloading data from {date_range[0]} to {date_range[1]}')
        
        # Retrieve the data from WB API
        df = wbdata.get_dataframe(indicators, country=countries, date=date_range)

        # Reset index to move 'country' and 'date' into columns
        df.reset_index(inplace=True)

        return df
    
    def eliminate_non_country_data(self, df, oecd_countries=True):

        if oecd_countries:
            valid_countries = [
                        "Australia", "Austria", "Belgium", "Canada", "Chile", "Colombia", 
                        "Costa Rica", "Czechia", "Denmark", "Estonia", "Finland", "France", 
                        "Germany", "Greece", "Hungary", "Iceland", "Ireland", "Israel", 
                        "Italy", "Japan", "Korea, Rep.", "Latvia", "Lithuania", "Luxembourg", 
                        "Mexico", "Netherlands", "New Zealand", "Norway", "Poland", 
                        "Portugal", "Slovak Republic", "Slovenia", "Spain", "Sweden", 
                        "Switzerland", "Turkiye", "United Kingdom", "United States"]
        else:
            valid_countries = [country.name for country in pycountry.countries]
        
        new_df = df[df['country'].isin(valid_countries)].reset_index(drop=True)
        
        return new_df
    
    def missing_data_percentage(self, df):
        
        missing_data = df.isnull().sum()
        missing_percentage = (missing_data / len(df)) * 100
        print(missing_percentage)
    
    def get_missing_value_groupby(self, df, groupby_column, sort_by, cols_to_drop=['country', 'date', 'Population', 'Urbanization_rate']):
        
        missing_values_groupby = df.groupby(groupby_column).apply(lambda x: x.isnull().sum()).drop(columns=cols_to_drop).reset_index()
        missing_values_groupby = missing_values_groupby.sort_values(by=sort_by, ascending=False)
        
        return missing_values_groupby
    
    def get_items_to_remove(self, df, missing_values_df, target_col_name, groupby_col):

        num_of_unique_items = len(df[target_col_name].unique())
        item_threshold  =  num_of_unique_items/ 2
        items_to_remove = missing_values_df[(missing_values_df.CO2_emissions > item_threshold) | 
                                                (missing_values_df.GDP > item_threshold) |
                                                (missing_values_df.Energy_use > item_threshold) |
                                                (missing_values_df.Renewable_elec_output > item_threshold)][groupby_col]
        return items_to_remove
    
    def countries_with_missing_values(self, df):

        df_missing_vals  = df[(df.CO2_emissions.isna()) |
                              (df.GDP.isna()) |
                              (df.Energy_use.isna())]
        return df_missing_vals
    
    def impute_data(self, df, countries_with_missing_vals):
        
        df_imputed = df.copy()

        
        df_imputed = df_imputed.infer_objects(copy=False)
        
        
        numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns

        for country_name in countries_with_missing_vals:
        
            country_mask = df_imputed['country'] == country_name
            
            df_imputed.loc[country_mask, numeric_cols] = df_imputed.loc[country_mask, numeric_cols]\
                                                            .interpolate(method='linear', axis=0)\
                                                            .ffill().bfill()

        return df_imputed
    
    def q4_create_target_variable(self, cotwo_df, start='1997', end='2006', threshold = -10):
        
        cotwo_df_copy = cotwo_df.copy()
        
        cotwo_df_copy['Percentage Change'] = ((cotwo_df_copy[end] - cotwo_df_copy[start]) / cotwo_df_copy[start]) * 100
        
       
        cotwo_df_copy['Target'] = cotwo_df_copy['Percentage Change'].apply(lambda x: 1 if x <= threshold else 0)

        return cotwo_df_copy
    
    def log_transformation(self, df, cols_to_transform):

        df_log = df.copy()

        for col_name in cols_to_transform:
            new_col_name = 'log_' + col_name
            df_log[new_col_name] = np.log(df_log[col_name] + 1)
        return df_log
                
class ModelEvaluation:

    def __init__(self) -> None:
        pass

    def run_q2_regression_experiment(self, model, X, y, poly_degree=None, scaling=True, cv_folds=5):
        
        X = np.array(X)
        y = np.array(y)

        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        rmse_scores = []
        r2_scores = []
        
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            

            if poly_degree:
                poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
                X_train = poly.fit_transform(X_train)
                X_test = poly.transform(X_test)
            

            if scaling:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            
        
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            rmse = root_mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            rmse_scores.append(rmse)
            r2_scores.append(r2)
        
        return {
            "Model": model.__class__.__name__,
            "Poly Degree": poly_degree,
            "Scaling": scaling,
            "Average RMSE": np.mean(rmse_scores),
            "Average R²": np.mean(r2_scores)
        }


class MultiplePlotMaker:

    def __init__(self, df):
        self.df = df

    def get_column_names(self):
        column_names = list(self.df.columns)
        return column_names
    
    def plot_multiple_side_by_side_boxplots(self, target_variable_name):
        column_names = self.get_column_names()
        column_names.remove(target_variable_name)

        # Create boxplots to identify outliers
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))

        # Flatten the axes array for easier iteration
        axes = axes.flatten()

        # Plot each boxplot
        for i, col in enumerate(column_names):
            sns.boxplot(data=self.df, x=target_variable_name, y=col, ax=axes[i])
            axes[i].set_title(f'Side-by-side boxplot of {col}')

        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Adjust layout
        plt.tight_layout()
        plt.show()

    def plot_multiple_overlapping_hist(self, target_variable_name):
        column_names = self.get_column_names()
        column_names.remove(target_variable_name)

        # Create subplots
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))

        # Flatten the axes array for easier iteration
        axes = axes.flatten()

        # Loop through each column to create overlapping histograms
        for i, col in enumerate(column_names):
            subset1 = self.df[self.df[target_variable_name] == 0][col]
            subset2 = self.df[self.df[target_variable_name] == 1][col]

            axes[i].hist(subset1, color="blue", label="0", density=True, alpha=0.5)
            axes[i].hist(subset2, color="red", label="1", density=True, alpha=0.5)

            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Density')
            axes[i].set_title(f'Overlapping Histogram of {col}')
            axes[i].legend()

        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Adjust layout
        plt.tight_layout()
        plt.show()
