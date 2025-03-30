import asyncio
import concurrent.futures
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score, roc_auc_score, r2_score, 
    accuracy_score, mean_squared_error
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from category_encoders import CatBoostEncoder
from lightgbm import LGBMClassifier, LGBMRegressor
import numpy as np

# Import your AP_Feature_Selector implementation here
from ap_feature_selector import AP_Feature_Selector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

import pdb
import pandas as pd
import requests
from urllib.error import HTTPError
import os

# Cache for loaded datasets to avoid redundant downloads
_DATASET_CACHE = {}

def get_dataset(dataset_name, verbose=1, use_cache=True):
    """
    Fetch dataset from PyCaret's GitHub repository
    
    Parameters:
    dataset_name (str): Name of dataset to load
    verbose (bool): Whether to print status messages
    use_cache (bool): Whether to cache loaded datasets
    
    Returns:
    pd.DataFrame or None: Loaded dataset or None if error occurs
    """
    # Validate dataset name first
    #valid_names = load_dataset_names()
    valid_names = ['bank', 'blood', 'cancer', 'credit', 'diabetes', 'electrical_grid', 'employee',
 'heart', 'heart_disease', 'hepatitis', 'income', 'juice', 'nba', 'wine', 'telescope', 'titanic', 'us_presidential_election_results']
    
    if dataset_name not in valid_names:
        raise ValueError(f"Invalid dataset name: {dataset_name}. Valid names are: {valid_names}")

    # Check cache first
    if use_cache and dataset_name in _DATASET_CACHE:
        if verbose:
            print(f"Loading cached dataset: {dataset_name}")
        return _DATASET_CACHE[dataset_name].copy()

    # Construct GitHub raw content URL
    base_url = "https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/"
    
    # Handle special characters in filenames
    formatted_name = dataset_name.replace(" ", "%20") + ".csv"
    url = f"{base_url}{formatted_name}"

    try:
        # Check if file exists
        response = requests.head(url)
        response.raise_for_status()

        # Load dataset
        if verbose:
            print(f"Downloading dataset: {dataset_name}")
            
        df = pd.read_csv(url)
        
        # Add to cache
        if use_cache:
            _DATASET_CACHE[dataset_name] = df.copy()
            
        return df

    except HTTPError as e:
        raise ValueError(f"Dataset '{dataset_name}' not found in repository. Status code: {e.response.status_code}")
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to connect to GitHub: {str(e)}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"Dataset '{dataset_name}' appears to be empty or corrupt")
    except Exception as e:
        raise RuntimeError(f"Error loading dataset '{dataset_name}': {str(e)}")

def load_dataset_names():
    """Return list of valid dataset names with error handling"""
    try:
        url = "https://api.github.com/repos/pycaret/pycaret/contents/datasets"
        response = requests.get(url)
        response.raise_for_status()
        
        files = response.json()
        return sorted(
            [f["name"].replace(".csv", "").replace("%20", " ") 
             for f in files 
             if f["name"].endswith(".csv")]
        )[:1]
        
    except Exception as e:
        print(f"Warning: Could not fetch dataset list from GitHub: {str(e)}")
        return [
            "anomaly", "asia_gdp", "CTG", "Traffic Violations",
            "airquality", "amazon", "automobile", "bank", "cancer",
            "credit", "diabetes", "electrical_grid", "employee",
            "heart", "heart_disease", "juice", "nba", "wine",
            "telescope", "us_presidential_election_results"
        ]

class DatasetTester:
    def __init__(self, n_jobs=4, test_size=0.2, random_state=42):
        self.n_jobs = n_jobs
        self.test_size = test_size
        self.random_state = random_state
        self.results = []
        self.error_log = []

    async def process_dataset(self, dataset_row):
        """Process a single dataset asynchronously"""
        try:
            dataset_name = dataset_row['Dataset']
            target_var = dataset_row['Target Variable 1']
            task_type = dataset_row['Default Task']
            
            logging.info(f"Processing dataset: {dataset_name}")
            
            # Load dataset
            df = get_dataset(dataset_name, verbose=False)
            if df is None:
                raise ValueError(f"Dataset {dataset_name} not found")
            
            # Handle special cases
            if dataset_name == 'anomaly':
                X = df.drop('ANOMALY', axis=1) if 'ANOMALY' in df.columns else df
                y = df['ANOMALY'] if 'ANOMALY' in df.columns else None
            else:
                X = df.drop(target_var, axis=1)
                y = df[target_var]
            
            # Split data FIRST to prevent leakage
            X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
                X, y, 
                test_size=self.test_size, 
                random_state=self.random_state
            )

            # Identify feature types from TRAINING DATA
            numeric_features = X_train_raw.select_dtypes(include=np.number).columns.tolist()
            categorical_features = X_train_raw.select_dtypes(exclude=np.number).columns.tolist()

            # Create preprocessor based on training data features
            preprocessor = self._create_preprocessor(
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                has_missing=dataset_row['Missing Values'] == 'Y'
            )

            # Encode target variable before you send it to preprocessor otherwise error!
            le = LabelEncoder()
            y_train = pd.Series(le.fit_transform(y_train_raw), 
                                index=y_train_raw.index, name=target_var)
            y_test = pd.Series(le.transform(y_test_raw),
                                index=y_test_raw.index, name=target_var)


            # Fit on TRAINING DATA only
            X_train = preprocessor.fit_transform(X_train_raw, y_train)
            
            # Transform TEST DATA with fitted preprocessor
            X_test = preprocessor.transform(X_test_raw)

            # Convert to DataFrames with proper column names
            columns = numeric_features + categorical_features
            X_train = pd.DataFrame(X_train, columns=columns)
            X_test = pd.DataFrame(X_test, columns=columns)

            # Get appropriate models and metrics
            model, metric = self._get_models_and_metrics(task_type)
            
            # Create pipelines
            base_pipe, selector_pipe = self._create_pipelines(model, task_type)
            
            # Test with and without feature selection
            base_score = await self._test_pipeline(base_pipe, X_train, y_train, X_test, y_test, metric)
            selector_score = await self._test_pipeline(selector_pipe, X_train, y_train, X_test, y_test, metric)
            
            # Store results
            self.results.append({
                'dataset': dataset_name,
                'task_type': task_type,
                'base_score': base_score,
                'selector_score': selector_score,
                'num_rows': X.shape[0],
                'n_features_original': X.shape[1],
                'n_features_selected': len(selector_pipe.named_steps['selector'].selected_features_),
                'error': None
            })
            
        except Exception as e:
            self.error_log.append({
                'dataset': dataset_row['Dataset'],
                'error': str(e)
            })
            logging.error(f"Error processing {dataset_row['Dataset']}: {str(e)}")

    def _create_preprocessor(self, numeric_features, categorical_features, has_missing):
        """Create a ColumnTransformer with separate processing for numeric/categorical features"""
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(
                strategy='median' if has_missing else 'constant',
                fill_value=0
            )),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(
                strategy='constant', 
                fill_value='missing'
            )),
            ('encoder', CatBoostEncoder())
        ])

        return ColumnTransformer([
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    def _get_models_and_metrics(self, task_type):
        """Get appropriate models and metrics based on task type"""
        if 'Classification' in task_type:
            model = LGBMClassifier(random_state=self.random_state,verbose=-1,num_leaves=8)
            metrics = {
                'ap': average_precision_score,
                'roc_auc': roc_auc_score,
                'accuracy': accuracy_score
            }
        elif 'Regression' in task_type:
            model = LGBMRegressor(random_state=self.random_state,verbose=-1,num_leaves=8)
            metrics = {
                'r2': r2_score,
                'mse': mean_squared_error
            }
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
        
        return model, metrics

    def _create_pipelines(self, model, task_type):
        """Create comparison pipelines"""
        base_pipe = Pipeline([
            ('model', model)
        ])
        
        selector_pipe = Pipeline([
            ('selector', AP_Feature_Selector(
                estimator=model,
                n_iter=100,
                random_state=self.random_state
            )),
            ('model', model)
        ])
        
        return base_pipe, selector_pipe

    async def _test_pipeline(self, pipe, X_train, y_train, X_test, y_test, metric):
        """Test a pipeline and return scores"""
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            
            # Train pipeline
            await loop.run_in_executor(executor, pipe.fit, X_train, y_train)
            
            # Predict
            y_pred = await loop.run_in_executor(executor, pipe.predict, X_test)
            
            # Calculate metrics
            scores = {}
            for name, fn in metric.items():
                if name == 'ap' and 'predict_proba' in dir(pipe):
                    y_score = await loop.run_in_executor(executor, pipe.predict_proba, X_test)
                    scores[name] = fn(y_test, y_score[:, 1])
                else:
                    scores[name] = fn(y_test, y_pred)
            
            return scores

    async def run_tests(self, datasets):
        """Run tests on all datasets"""
        tasks = [self.process_dataset(row) for _, row in datasets.iterrows()]
        await asyncio.gather(*tasks)
        
        # Create results dataframe
        results_df = pd.DataFrame(self.results)
        error_df = pd.DataFrame(self.error_log)
        
        # Save results
        results_df.to_csv('ap_feature_selection_results.csv', index=False)
        error_df.to_csv('ap_feature_selection_errors.csv', index=False)
        
        return results_df, error_df

def get_data():
    """Load your datasets here"""
    sep=','
    df = pd.read_csv('./data/datasets.csv', sep=sep)
    print(df.shape)
    return df[df['Default Task'].str.contains('Classification')]

# Usage example
if __name__ == '__main__':
    datasets = get_data()  # Load your datasets
    tester = DatasetTester(n_jobs=4)
    
    # Run async tests
    results_df, error_df = asyncio.run(tester.run_tests(datasets.head(10)))  # Test first 10 datasets
    results_df.to_csv('ap_feature_selection_results.csv', index=False)
    error_df.to_csv('ap_feature_selection_errors.csv', index=False)  
