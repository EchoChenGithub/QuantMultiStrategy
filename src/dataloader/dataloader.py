import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Optional, List, Union
from datetime import datetime
import warnings

class DataLoader:
    """
    A class for loading and preprocessing quantitative factor data.
    
    This class handles the preprocessing pipeline for factor data including:
    - Data preparation and cleaning
    - Outlier treatment using MAD
    - Missing value handling
    - Industry and market cap neutralization
    - Standardization
    - Optional PCA transformation
    
    Attributes:
        raw_data (pd.DataFrame): The original factor data
        processed_data (pd.DataFrame): The preprocessed data
    """
    
    def __init__(self, raw_data: pd.DataFrame):
        """
        Initialize the DataLoader with raw data.
        
        Args:
            raw_data (pd.DataFrame): Raw factor data containing datetime, code, factors, and target
        """
        if not isinstance(raw_data, pd.DataFrame):
            raise TypeError("raw_data must be a pandas DataFrame")
            
        self.raw_data = raw_data.copy()
        self.processed_data = None
        
    def load_and_prepare_data(self) -> pd.DataFrame:
        """
        Load and prepare the initial data by setting multi-index and handling negative ratios.
        
        Returns:
            pd.DataFrame: Prepared data with multi-index and processed ratios
        """
        # Set multi-index
        data = self.raw_data.copy()

        # 设置多索引
        data = data.set_index(['datetime', 'code'])
        
        # Handle negative PE ratio
        if 'pe_ratio' in data.columns:
            data['ep_ratio'] = 1 / data['pe_ratio']
            data = data.drop('pe_ratio', axis=1)
            
        # Handle negative PB ratio
        if 'pb_ratio' in data.columns:
            data.loc[data['pb_ratio'] < 0, 'pb_ratio'] = np.nan
            
        return data
    
    def _mad_winsorization(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply MAD winsorization to factor values.
        
        Args:
            data (pd.DataFrame): Data to be winsorized
            
        Returns:
            pd.DataFrame: Winsorized data
        """
        # Get numeric columns excluding industry_code, market_cap, and next_ret
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        exclude_cols = ['industry_code', 'market_cap', 'next_ret']
        factor_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        def winsorize_group(group):
            for col in factor_cols:
                if col in group.columns:
                    median = group[col].median()
                    mad = np.median(np.abs(group[col] - median))
                    lower_bound = median - 3 * mad
                    upper_bound = median + 3 * mad
                    group[col] = group[col].clip(lower=lower_bound, upper=upper_bound)
            return group
            
        # 先重置索引，执行 groupby，然后再设置回多索引
        data = data.reset_index()
        data = data.groupby('datetime').apply(winsorize_group)
        data = data.set_index(['datetime', 'code'])
        
        return data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values by industry mean imputation.
        
        Args:
            data (pd.DataFrame): Data with missing values
            
        Returns:
            pd.DataFrame: Data with imputed missing values
        """
        # Drop rows where industry_code is missing
        data = data.dropna(subset=['industry_code'])
        
        # Get numeric columns excluding industry_code, market_cap, and next_ret
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        exclude_cols = ['industry_code', 'market_cap', 'next_ret']
        factor_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Fill missing values with industry mean
        for col in factor_cols:
            if col in data.columns:
                industry_means = data.groupby(['datetime', 'industry_code'])[col].transform('mean')
                data[col] = data[col].fillna(industry_means)
                # Fill remaining NaN with 0
                data[col] = data[col].fillna(0)
                
        return data
    
    def _neutralize_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Neutralize factors by industry and market cap.
        
        Args:
            data (pd.DataFrame): Data to be neutralized
            
        Returns:
            pd.DataFrame: Neutralized data
        """
        # Get numeric columns excluding industry_code, market_cap, and next_ret
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        exclude_cols = ['industry_code', 'market_cap', 'next_ret']
        factor_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        def neutralize_group(group):
            # Prepare independent variables
            industry_dummies = pd.get_dummies(group['industry_code'])
            log_market_cap = np.log(group['market_cap'].clip(lower=1e-6))  # Handle non-positive values
            
            # Ensure all data is numeric
            X = pd.concat([industry_dummies, log_market_cap], axis=1)
            X = sm.add_constant(X)
            
            # Convert to numpy arrays and ensure numeric types
            X = X.astype(float)
            
            # Neutralize each factor
            for col in factor_cols:
                if col in group.columns:
                    y = group[col].astype(float)  # Ensure y is numeric
                    try:
                        model = sm.OLS(y, X, missing='drop')
                        results = model.fit()
                        group[col] = results.resid
                    except Exception as e:
                        print(f"Warning: Failed to neutralize {col} for group {group.name}: {str(e)}")
                        # If neutralization fails, keep original values
                        continue
                    
            return group
            
        # 先重置索引，执行 groupby，然后再设置回多索引
        data = data.reset_index()
        data = data.groupby('datetime').apply(neutralize_group)
        data = data.set_index(['datetime', 'code'])
        
        return data
    
    def _standardize_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize factors using Z-score method.
        
        Args:
            data (pd.DataFrame): Data to be standardized
            
        Returns:
            pd.DataFrame: Standardized data
        """
        # Get numeric columns excluding industry_code, market_cap, and next_ret
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        exclude_cols = ['industry_code', 'market_cap', 'next_ret']
        factor_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        def standardize_group(group):
            for col in factor_cols:
                if col in group.columns:
                    mean = group[col].mean()
                    std = group[col].std()
                    if std != 0:
                        group[col] = (group[col] - mean) / std
            return group
            
        # 先重置索引，执行 groupby，然后再设置回多索引
        data = data.reset_index()
        data = data.groupby('datetime').apply(standardize_group)
        data = data.set_index(['datetime', 'code'])
        
        return data
    
    def preprocess_data(self, pca_components: Optional[int] = None) -> pd.DataFrame:
        """
        Main preprocessing pipeline for the factor data.
        
        Args:
            pca_components (Optional[int]): Number of PCA components to use. If None, PCA is skipped.
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        # Load and prepare data
        data = self.load_and_prepare_data()
        
        # Apply preprocessing steps
        data = self._mad_winsorization(data)
        data = self._handle_missing_values(data)
        data = self._neutralize_factors(data)
        data = self._standardize_factors(data)
        
        # Optional PCA transformation
        if pca_components is not None:
            from sklearn.decomposition import PCA
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            exclude_cols = ['industry_code', 'market_cap', 'next_ret']
            factor_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            pca = PCA(n_components=pca_components)
            pca_features = pd.DataFrame(
                pca.fit_transform(data[factor_cols]),
                index=data.index,
                columns=[f'pca_{i}' for i in range(pca_components)]
            )
            data = pd.concat([data, pca_features], axis=1)
        
        # Reorder columns to ensure next_ret is last
        cols = [col for col in data.columns if col != 'next_ret']
        cols.append('next_ret')
        data = data[cols]
        
        # 重置索引，确保只有 datetime 和 code
        data = data.reset_index()
        data = data.set_index(['datetime', 'code'])
        
        self.processed_data = data
        return data
    
    def get_processed_data(
        self,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None
    ) -> pd.DataFrame:
        """
        Get processed data for a specific time period.
        
        Args:
            start_date (Optional[Union[str, datetime]]): Start date of the period
            end_date (Optional[Union[str, datetime]]): End date of the period
            
        Returns:
            pd.DataFrame: Processed data for the specified period
        """
        if self.processed_data is None:
            raise ValueError("Data has not been processed yet. Call preprocess_data() first.")
            
        data = self.processed_data.copy()
        
        if start_date is not None:
            data = data[data.index.get_level_values('datetime') >= pd.to_datetime(start_date)]
        if end_date is not None:
            data = data[data.index.get_level_values('datetime') <= pd.to_datetime(end_date)]
            
        return data


