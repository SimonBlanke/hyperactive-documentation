# Machine Learning Examples

## Introduction

This page demonstrates comprehensive machine learning optimization scenarios using Hyperactive. From simple classifier tuning to complex ensemble optimization and neural network hyperparameter search, these examples show practical applications across different ML domains.

## Classification Examples

### Binary Classification with Imbalanced Data

```python
from hyperactive.experiment.integrations import SklearnCvExperiment
from hyperactive.opt.gfo import BayesianOptimizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

# Create imbalanced dataset
X, y = make_classification(
    n_samples=1000, 
    n_features=20, 
    n_informative=10,
    n_redundant=5,
    n_classes=2,
    weights=[0.9, 0.1],  # Imbalanced classes
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Parameter space for imbalanced classification
param_grid = {
    "n_estimators": [50, 100, 200, 300],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "class_weight": ["balanced", "balanced_subsample", None],
    "max_features": ["sqrt", "log2", None]
}

# Use stratified CV for imbalanced data
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

experiment = SklearnCvExperiment(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    X=X_train, y=y_train,
    cv=cv,
    scoring="roc_auc",  # Better for imbalanced data than accuracy
    n_jobs=-1
)

optimizer = BayesianOptimizer(experiment=experiment)
best_params = optimizer.solve()

# Train final model and evaluate
final_model = RandomForestClassifier(**best_params, random_state=42)
final_model.fit(X_train, y_train)

# Comprehensive evaluation
y_pred = final_model.predict(X_test)
y_pred_proba = final_model.predict_proba(X_test)[:, 1]

print("Best parameters:", best_params)
print("CV ROC-AUC:", experiment.score(best_params)[0])
print("Test ROC-AUC:", roc_auc_score(y_test, y_pred_proba))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

### Multi-Class Classification

```python
from sklearn.datasets import load_wine
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load multi-class dataset
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Create pipeline with preprocessing
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC(random_state=42, probability=True))
])

# Parameter space for SVM pipeline
param_grid = {
    'scaler__with_mean': [True, False],
    'scaler__with_std': [True, False],
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'classifier__kernel': ['rbf', 'linear', 'poly'],
    'classifier__degree': [2, 3, 4]  # Only used for poly kernel
}

experiment = SklearnCvExperiment(
    estimator=pipeline,
    param_grid=param_grid,
    X=X_train, y=y_train,
    cv=5,
    scoring="f1_weighted"
)

optimizer = BayesianOptimizer(experiment=experiment)
best_params = optimizer.solve()

# Evaluate final model
final_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC(random_state=42, probability=True))
])
final_pipeline.set_params(**best_params)
final_pipeline.fit(X_train, y_train)

# Multi-class metrics
from sklearn.metrics import accuracy_score, f1_score
y_pred = final_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1_weighted = f1_score(y_test, y_pred, average='weighted')
f1_macro = f1_score(y_test, y_pred, average='macro')

print("Best pipeline parameters:", best_params)
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 (weighted): {f1_weighted:.4f}")
print(f"Test F1 (macro): {f1_macro:.4f}")
```

## Regression Examples

### Linear Regression with Feature Selection

```python
from sklearn.datasets import load_diabetes
from sklearn.linear_model import ElasticNet
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler

# Load regression dataset
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline with feature selection
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(f_regression)),
    ('regressor', ElasticNet(random_state=42))
])

# Parameter space for regularized regression
param_grid = {
    'feature_selection__k': [5, 7, 10, 'all'],
    'regressor__alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
    'regressor__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
    'regressor__fit_intercept': [True, False],
    'regressor__max_iter': [1000, 2000, 5000]
}

experiment = SklearnCvExperiment(
    estimator=pipeline,
    param_grid=param_grid,
    X=X_train, y=y_train,
    cv=5,
    scoring="neg_mean_squared_error"
)

optimizer = BayesianOptimizer(experiment=experiment)
best_params = optimizer.solve()

# Evaluate regression model
final_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(f_regression)),
    ('regressor', ElasticNet(random_state=42))
])
final_pipeline.set_params(**best_params)
final_pipeline.fit(X_train, y_train)

predictions = final_pipeline.predict(X_test)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Best regression parameters:", best_params)
print(f"Test MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test R²: {r2:.4f}")
```

### Gradient Boosting Regression

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_regression

# Create regression dataset
X, y = make_regression(n_samples=1000, n_features=15, n_informative=10, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Comprehensive parameter space for gradient boosting
param_grid = {
    "n_estimators": [50, 100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 4, 5, 6, 7],
    "min_samples_split": [2, 5, 10, 15],
    "min_samples_leaf": [1, 2, 4, 6],
    "subsample": [0.8, 0.9, 1.0],
    "max_features": ["sqrt", "log2", None]
}

experiment = SklearnCvExperiment(
    estimator=GradientBoostingRegressor(random_state=42),
    param_grid=param_grid,
    X=X_train, y=y_train,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

# Use different optimizer for comparison
from hyperactive.opt.optuna import TPEOptimizer
optimizer = TPEOptimizer(experiment=experiment)
best_params = optimizer.solve()

# Train and evaluate
final_model = GradientBoostingRegressor(**best_params, random_state=42)
final_model.fit(X_train, y_train)
predictions = final_model.predict(X_test)

# Feature importance analysis
feature_importance = final_model.feature_importances_
top_features = np.argsort(feature_importance)[-5:][::-1]

print("Best GBR parameters:", best_params)
print(f"Test MAE: {mean_absolute_error(y_test, predictions):.4f}")
print(f"Test R²: {r2_score(y_test, predictions):.4f}")
print("Top 5 feature indices:", top_features)
```

## Neural Network Optimization

### Multi-Layer Perceptron Hyperparameter Tuning

```python
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

# Load digit classification dataset
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scale features for neural network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Neural network parameter space
param_grid = {
    "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50), (100, 100), (100, 50, 25)],
    "activation": ["relu", "tanh", "logistic"],
    "solver": ["adam", "lbfgs", "sgd"],
    "alpha": [0.0001, 0.001, 0.01, 0.1],
    "learning_rate": ["constant", "invscaling", "adaptive"],
    "learning_rate_init": [0.001, 0.01, 0.1],
    "max_iter": [200, 500, 1000]
}

experiment = SklearnCvExperiment(
    estimator=MLPClassifier(random_state=42, early_stopping=True, validation_fraction=0.1),
    param_grid=param_grid,
    X=X_train_scaled, y=y_train,
    cv=3,  # Fewer folds due to computational cost
    scoring="accuracy"
)

optimizer = BayesianOptimizer(experiment=experiment)
best_params = optimizer.solve()

# Train final neural network
final_mlp = MLPClassifier(**best_params, random_state=42, early_stopping=True, validation_fraction=0.1)
final_mlp.fit(X_train_scaled, y_train)

# Evaluate
accuracy = final_mlp.score(X_test_scaled, y_test)
y_pred = final_mlp.predict(X_test_scaled)

print("Best MLP parameters:", best_params)
print(f"Test accuracy: {accuracy:.4f}")
print(f"Number of iterations: {final_mlp.n_iter_}")
print(f"Training loss: {final_mlp.loss_:.4f}")
```

## Ensemble Methods

### Random Forest vs Gradient Boosting Comparison

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer

# Load binary classification dataset
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Define algorithms and their parameter spaces
algorithms = {
    "Random Forest": {
        "estimator": RandomForestClassifier(random_state=42),
        "param_grid": {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
            "bootstrap": [True, False]
        }
    },
    "Gradient Boosting": {
        "estimator": GradientBoostingClassifier(random_state=42),
        "param_grid": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.05, 0.1, 0.15, 0.2],
            "max_depth": [3, 4, 5, 6],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "subsample": [0.8, 0.9, 1.0]
        }
    }
}

# Compare algorithms
results = {}
for algorithm_name, config in algorithms.items():
    print(f"\nOptimizing {algorithm_name}...")
    
    experiment = SklearnCvExperiment(
        estimator=config["estimator"],
        param_grid=config["param_grid"],
        X=X_train, y=y_train,
        cv=5,
        scoring="roc_auc"
    )
    
    optimizer = BayesianOptimizer(experiment=experiment)
    best_params = optimizer.solve()
    
    # Train final model
    final_model = config["estimator"].__class__(**best_params, random_state=42)
    final_model.fit(X_train, y_train)
    
    # Evaluate
    train_score = final_model.score(X_train, y_train)
    test_score = final_model.score(X_test, y_test)
    test_auc = roc_auc_score(y_test, final_model.predict_proba(X_test)[:, 1])
    
    results[algorithm_name] = {
        "best_params": best_params,
        "cv_score": experiment.score(best_params)[0],
        "train_accuracy": train_score,
        "test_accuracy": test_score,
        "test_auc": test_auc
    }
    
    print(f"CV AUC: {results[algorithm_name]['cv_score']:.4f}")
    print(f"Test AUC: {test_auc:.4f}")

# Find best algorithm
best_algorithm = max(results.items(), key=lambda x: x[1]["test_auc"])
print(f"\nBest algorithm: {best_algorithm[0]} with AUC {best_algorithm[1]['test_auc']:.4f}")

# Compare overfitting
for name, result in results.items():
    overfitting = result['train_accuracy'] - result['test_accuracy']
    print(f"{name} overfitting: {overfitting:.4f}")
```

### Voting Classifier Ensemble

```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# Create base estimators
rf = RandomForestClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)
lr = LogisticRegression(random_state=42, max_iter=1000)
nb = GaussianNB()

# Create voting classifier
voting_clf = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb), ('lr', lr), ('nb', nb)],
    voting='soft'  # Use probability-based voting
)

# Parameter space for ensemble
param_grid = {
    # Random Forest parameters
    'rf__n_estimators': [50, 100, 150],
    'rf__max_depth': [5, 10, None],
    
    # Gradient Boosting parameters
    'gb__n_estimators': [50, 100, 150],
    'gb__learning_rate': [0.05, 0.1, 0.15],
    'gb__max_depth': [3, 4, 5],
    
    # Logistic Regression parameters
    'lr__C': [0.1, 1, 10],
    'lr__solver': ['liblinear', 'lbfgs'],
    
    # Naive Bayes parameters (fewer options)
    'nb__var_smoothing': [1e-9, 1e-8, 1e-7]
}

experiment = SklearnCvExperiment(
    estimator=voting_clf,
    param_grid=param_grid,
    X=X_train, y=y_train,
    cv=3,  # Fewer folds due to computational complexity
    scoring="roc_auc"
)

optimizer = BayesianOptimizer(experiment=experiment)
best_params = optimizer.solve()

# Train ensemble
final_ensemble = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb), ('lr', lr), ('nb', nb)],
    voting='soft'
)
final_ensemble.set_params(**best_params)
final_ensemble.fit(X_train, y_train)

# Evaluate ensemble vs individual models
ensemble_auc = roc_auc_score(y_test, final_ensemble.predict_proba(X_test)[:, 1])

print("Best ensemble parameters:", best_params)
print(f"Ensemble Test AUC: {ensemble_auc:.4f}")

# Compare with individual model performance
for name, estimator in final_ensemble.named_estimators_.items():
    individual_auc = roc_auc_score(y_test, estimator.predict_proba(X_test)[:, 1])
    print(f"{name.upper()} Individual AUC: {individual_auc:.4f}")
```

## Custom ML Experiments

### Cross-Validation Strategy Optimization

```python
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedKFold
from hyperactive.base import BaseExperiment

class CVStrategyExperiment(BaseExperiment):
    """Optimize both model parameters and CV strategy"""
    
    def __init__(self, X, y, base_estimator):
        super().__init__()
        self.X = X
        self.y = y
        self.base_estimator = base_estimator
    
    def _paramnames(self):
        return [
            # Model parameters
            "n_estimators", "max_depth", "min_samples_split",
            # CV strategy parameters
            "cv_type", "cv_folds", "cv_shuffle"
        ]
    
    def _evaluate(self, params):
        try:
            # Create model with parameters
            model = self.base_estimator(
                n_estimators=int(params["n_estimators"]),
                max_depth=int(params["max_depth"]) if params["max_depth"] > 0 else None,
                min_samples_split=int(params["min_samples_split"]),
                random_state=42
            )
            
            # Create CV strategy based on parameters
            cv_folds = int(params["cv_folds"])
            cv_shuffle = params["cv_shuffle"] > 0.5
            
            if params["cv_type"] < 0.33:  # KFold
                cv = KFold(n_splits=cv_folds, shuffle=cv_shuffle, random_state=42)
            elif params["cv_type"] < 0.66:  # StratifiedKFold
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=cv_shuffle, random_state=42)
            else:  # RepeatedKFold
                cv = RepeatedKFold(n_splits=cv_folds, n_repeats=2, random_state=42)
            
            # Perform cross-validation
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, self.X, self.y, cv=cv, scoring='accuracy')
            
            return scores.mean(), {
                "cv_std": scores.std(),
                "cv_type_used": "KFold" if params["cv_type"] < 0.33 else 
                              "StratifiedKFold" if params["cv_type"] < 0.66 else "RepeatedKFold",
                "cv_folds": cv_folds
            }
            
        except Exception as e:
            return float('-inf'), {"error": str(e)}

# Use the custom experiment
experiment = CVStrategyExperiment(X_train, y_train, RandomForestClassifier)
optimizer = BayesianOptimizer(experiment=experiment)
best_params = optimizer.solve()

print("Best parameters (including CV strategy):", best_params)
print("Best performance:", experiment.score(best_params))
```

### Feature Engineering Optimization

```python
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

class FeatureEngineeringExperiment(BaseExperiment):
    """Optimize feature engineering pipeline"""
    
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y
    
    def _paramnames(self):
        return [
            "use_polynomial", "poly_degree", "use_scaling", 
            "feature_selection", "selection_k", "classifier_C"
        ]
    
    def _evaluate(self, params):
        try:
            from sklearn.pipeline import Pipeline
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score
            
            # Build pipeline based on parameters
            pipeline_steps = []
            
            # Optional polynomial features
            if params["use_polynomial"] > 0.5:
                degree = max(2, min(3, int(params["poly_degree"])))
                pipeline_steps.append(('poly', PolynomialFeatures(degree=degree, include_bias=False)))
            
            # Optional scaling
            if params["use_scaling"] > 0.5:
                pipeline_steps.append(('scaler', StandardScaler()))
            
            # Optional feature selection
            if params["feature_selection"] > 0.5:
                k = max(1, min(self.X.shape[1], int(params["selection_k"])))
                pipeline_steps.append(('selector', SelectKBest(f_classif, k=k)))
            
            # Classifier
            pipeline_steps.append(('classifier', LogisticRegression(
                C=params["classifier_C"], 
                random_state=42, 
                max_iter=1000
            )))
            
            # Create and evaluate pipeline
            pipeline = Pipeline(pipeline_steps)
            scores = cross_val_score(pipeline, self.X, self.y, cv=5, scoring='accuracy')
            
            return scores.mean(), {
                "pipeline_length": len(pipeline_steps),
                "includes_poly": params["use_polynomial"] > 0.5,
                "includes_scaling": params["use_scaling"] > 0.5,
                "includes_selection": params["feature_selection"] > 0.5
            }
            
        except Exception as e:
            return float('-inf'), {"error": str(e)}

# Run feature engineering optimization
experiment = FeatureEngineeringExperiment(X_train, y_train)
optimizer = BayesianOptimizer(experiment=experiment)
best_params = optimizer.solve()

print("Best feature engineering parameters:", best_params)
print("Feature engineering performance:", experiment.score(best_params))
```

## Time Series Forecasting

### ARIMA Parameter Optimization

```python
# Note: This example requires additional packages (statsmodels, sktime)
try:
    from sktime.forecasting.arima import ARIMA
    from sktime.forecasting.model_selection import temporal_train_test_split
    from sktime.datasets import load_airline
    
    # Load time series data
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=24)  # Last 24 months for testing
    
    class ARIMAExperiment(BaseExperiment):
        """Optimize ARIMA parameters"""
        
        def __init__(self, y_train, y_test):
            super().__init__()
            self.y_train = y_train
            self.y_test = y_test
        
        def _paramnames(self):
            return ["p", "d", "q", "seasonal_p", "seasonal_d", "seasonal_q"]
        
        def _evaluate(self, params):
            try:
                # Convert to integers
                p = max(0, min(5, int(params["p"])))
                d = max(0, min(2, int(params["d"])))
                q = max(0, min(5, int(params["q"])))
                P = max(0, min(2, int(params["seasonal_p"])))
                D = max(0, min(1, int(params["seasonal_d"])))
                Q = max(0, min(2, int(params["seasonal_q"])))
                
                # Create and fit ARIMA model
                forecaster = ARIMA(
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, 12),  # Monthly seasonality
                    suppress_warnings=True
                )
                
                forecaster.fit(self.y_train)
                
                # Make predictions
                y_pred = forecaster.predict(fh=range(1, len(self.y_test) + 1))
                
                # Calculate error metric (negative for maximization)
                from sklearn.metrics import mean_absolute_error
                mae = mean_absolute_error(self.y_test, y_pred)
                
                return -mae, {
                    "mae": mae,
                    "arima_order": (p, d, q),
                    "seasonal_order": (P, D, Q, 12)
                }
                
            except Exception as e:
                return float('-inf'), {"error": str(e)}
    
    # Optimize ARIMA parameters
    experiment = ARIMAExperiment(y_train, y_test)
    optimizer = BayesianOptimizer(experiment=experiment)
    best_params = optimizer.solve()
    
    print("Best ARIMA parameters:", best_params)
    print("Best MAE:", -experiment.score(best_params)[0])

except ImportError:
    print("Time series example requires 'sktime' package: pip install sktime")
```

## Model Interpretability

### Feature Importance Optimization

```python
from sklearn.inspection import permutation_importance

class InterpretableModelExperiment(BaseExperiment):
    """Optimize for both performance and interpretability"""
    
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y
    
    def _paramnames(self):
        return ["n_estimators", "max_depth", "min_samples_leaf", "interpretability_weight"]
    
    def _evaluate(self, params):
        try:
            # Create model
            model = RandomForestClassifier(
                n_estimators=int(params["n_estimators"]),
                max_depth=int(params["max_depth"]) if params["max_depth"] > 0 else None,
                min_samples_leaf=int(params["min_samples_leaf"]),
                random_state=42
            )
            
            # Cross-validation for performance
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(model, self.X, self.y, cv=5, scoring='accuracy')
            performance_score = cv_scores.mean()
            
            # Fit model for interpretability analysis
            model.fit(self.X, self.y)
            
            # Calculate interpretability metrics
            feature_importance = model.feature_importances_
            
            # Interpretability score (higher when fewer features are dominant)
            importance_entropy = -np.sum(feature_importance * np.log(feature_importance + 1e-10))
            max_entropy = np.log(len(feature_importance))
            interpretability_score = 1 - (importance_entropy / max_entropy)
            
            # Combined score
            weight = params["interpretability_weight"]
            combined_score = (1 - weight) * performance_score + weight * interpretability_score
            
            return combined_score, {
                "performance": performance_score,
                "interpretability": interpretability_score,
                "n_important_features": np.sum(feature_importance > 0.05),
                "max_feature_importance": feature_importance.max()
            }
            
        except Exception as e:
            return float('-inf'), {"error": str(e)}

# Run interpretable model optimization
experiment = InterpretableModelExperiment(X_train, y_train)
optimizer = BayesianOptimizer(experiment=experiment)
best_params = optimizer.solve()

print("Best interpretable model parameters:", best_params)
print("Combined score breakdown:", experiment.score(best_params)[1])
```

## Best Practices Summary

1. **Data Preparation**:
   - Always split data properly (train/validation/test)
   - Use appropriate preprocessing for each algorithm
   - Handle missing values and outliers

2. **Parameter Space Design**:
   - Start with reasonable ranges based on domain knowledge
   - Include important preprocessing parameters
   - Consider computational constraints

3. **Cross-Validation Strategy**:
   - Use stratified CV for classification
   - Use time series CV for temporal data
   - Consider grouped CV for clustered data

4. **Evaluation Metrics**:
   - Choose metrics aligned with business objectives
   - Use multiple metrics for comprehensive evaluation
   - Consider class imbalance in metric selection

5. **Computational Efficiency**:
   - Use parallel processing when available
   - Start with smaller parameter spaces for exploration
   - Consider early stopping for iterative algorithms

6. **Model Validation**:
   - Always evaluate on held-out test data
   - Use nested CV for unbiased performance estimation
   - Check for overfitting patterns

## References

- Scikit-learn User Guide: [https://scikit-learn.org/stable/user_guide.html](https://scikit-learn.org/stable/user_guide.html)
- Model Selection and Evaluation: [https://scikit-learn.org/stable/model_selection.html](https://scikit-learn.org/stable/model_selection.html)
- Time Series with sktime: [https://www.sktime.org/](https://www.sktime.org/)