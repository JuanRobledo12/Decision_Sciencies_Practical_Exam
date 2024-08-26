import wbdata
import pycountry

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
    
    def eliminate_non_country_data(self, df):

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