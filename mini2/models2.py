from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler
from sklearn.pipeline import make_pipeline
import preprocessing

# Preprocess the data
df = preprocessing.preprocess()

X = df[['Estimate (Land)']]  # Single feature for simple regression
y = df['Sale Price']  # Target variable

# Split the data into train and test sets for simple regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function for simple linear regression
def simple_linear_regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return model, y_pred, mse


# Prepare data for multiple regression
X_full = df.drop(columns=['Sale Price'])  # All features except target
# Re-split data for multiple regression
X_full_train, X_full_test, y_full_train, y_full_test = train_test_split(X_full, y, test_size=0.2, random_state=42)

# Function for multiple linear regression
def multiple_linear_regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return model, y_pred, mse


# Feature selection using RFE
def feature_selection(X_train, y_train, n_features):
    model = LinearRegression()
    rfe = RFE(estimator=model, n_features_to_select=n_features)
    rfe.fit(X_train, y_train)
    selected_features_mask = rfe.support_
    feature_names = X_train.columns
    selected_features = feature_names[selected_features_mask].tolist()
    return selected_features, rfe

# Set the number of features to select
n_features = 3
selected_features, rfe_model = feature_selection(X_full_train, y_full_train, n_features)

print("Selected Features:", selected_features)

# Fit the regression model on the selected features
X_selected_train = X_full_train[selected_features]
X_selected_test = X_full_test[selected_features]

# Fit the model with selected features
model_with_selected_features = LinearRegression()
model_with_selected_features.fit(X_selected_train, y_full_train)

# Make predictions and evaluate the model
y_pred_selected = model_with_selected_features.predict(X_selected_test)
mse_selected = mean_squared_error(y_full_test, y_pred_selected)

print("Mean Squared Error with Selected Features:", mse_selected)


# Lasso Regression function
def lasso(X_train, X_test, y_train, y_test, alpha):
    lasso_model = Lasso(alpha=alpha, max_iter=20000)  # Increased max_iter
    lasso_model.fit(X_train, y_train)
    y_pred_lasso = lasso_model.predict(X_test)
    mse_lasso = mean_squared_error(y_test, y_pred_lasso)
    return lasso_model, y_pred_lasso, mse_lasso


# Ridge Regression function
def ridge(X_train, X_test, y_train, y_test, alpha):
    ridge_model = Ridge(alpha=alpha, max_iter=20000)  # Increased max_iter
    ridge_model.fit(X_train, y_train)
    y_pred_ridge = ridge_model.predict(X_test)
    mse_ridge = mean_squared_error(y_test, y_pred_ridge)
    return ridge_model, y_pred_ridge, mse_ridge


# Polynomial Regression function
def poly(X_train, X_test, y_train, y_test, degrees):
    poly = PolynomialFeatures(degree=degrees)
    model_poly = make_pipeline(poly, LinearRegression())
    model_poly.fit(X_train, y_train)
    y_pred_poly = model_poly.predict(X_test)
    mse_poly = mean_squared_error(y_test, y_pred_poly)
    return model_poly, y_pred_poly, mse_poly


# Spline Regression function
def spline(X_train, X_test, y_train, y_test, degrees):
    spline = SplineTransformer(degree=degrees)
    model_spline = make_pipeline(spline, LinearRegression())
    model_spline.fit(X_train, y_train)
    y_pred_spline = model_spline.predict(X_test)
    mse_spline = mean_squared_error(y_test, y_pred_spline)
    return model_spline, y_pred_spline, mse_spline


# Scale the training and test sets
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Train models
simple_model, y_pred_simple, mse_simple = simple_linear_regression(X_train, X_test, y_train, y_test)
multiple_model, y_pred_multiple, mse_multiple = multiple_linear_regression(X_full_train, X_full_test, y_full_train, y_full_test)
lasso_model, y_pred_lasso, mse_lasso = lasso(X_train_scaled, X_test_scaled, y_train, y_test, 1.0)
ridge_model, y_pred_ridge, mse_ridge = ridge(X_train_scaled, X_test_scaled, y_train, y_test, 1.0)
model_poly, y_pred_poly, mse_poly = poly(X_full_train, X_full_test, y_full_train, y_full_test, 2)
model_spline, y_pred_spline, mse_spline = spline(X_full_train, X_full_test, y_full_train, y_full_test, 3)


# PCA
pca = PCA(n_components=0.95)  # Keep 95% variance
X_pca = pca.fit_transform(X_full)
# Re-split the PCA data
X_train_pca, X_test_pca, y_full_train_pca, y_full_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=42)


# Re-train models on PCA-transformed data
pca_simple_model, pca_y_pred_simple, pca_mse_simple = simple_linear_regression(X_train_pca, X_test_pca, y_full_train_pca, y_full_test_pca)
pca_multiple_model, pca_y_pred_multiple, pca_mse_multiple = multiple_linear_regression(X_train_pca, X_test_pca, y_full_train_pca, y_full_test_pca)
pca_lasso_model, pca_y_pred_lasso, pca_mse_lasso = lasso(X_train_pca, X_test_pca, y_full_train_pca, y_full_test_pca, 1.0)
pca_ridge_model, pca_y_pred_ridge, pca_mse_ridge = ridge(X_train_pca, X_test_pca, y_full_train_pca, y_full_test_pca, 1.0)
pca_model_poly, pca_y_pred_poly, pca_mse_poly = poly(X_train_pca, X_test_pca, y_full_train_pca, y_full_test_pca, 2)
pca_model_spline, pca_y_pred_spline, pca_mse_spline = spline(X_train_pca, X_test_pca, y_full_train_pca, y_full_test_pca, 3)


# Results
results = {
    'Simple Regression': mse_simple,
    'Multiple Regression': mse_multiple,
    'Lasso Regression': mse_lasso,
    'Ridge Regression': mse_ridge,
    'Polynomial Model': mse_poly,
    'Spline Model': mse_spline,
}
print("Regression Results:", results)


# PCA Results
pca_results = {
    'Simple Regression': pca_mse_simple,
    'Multiple Regression': pca_mse_multiple,
    'Lasso Regression': pca_mse_lasso,
    'Ridge Regression': pca_mse_ridge,
    'Polynomial Model': pca_mse_poly,
    'Spline Model': pca_mse_spline,
}
print("PCA Regression Results:", pca_results)


# Cross-validation
cv_scores_simple = cross_val_score(simple_model, X, y, cv=5, scoring='neg_mean_squared_error')
cv_scores_multiple = cross_val_score(multiple_model, X, y, cv=5, scoring='neg_mean_squared_error')
cv_scores_lasso = cross_val_score(lasso_model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
cv_scores_ridge = cross_val_score(ridge_model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')

    
# Cross-validation results
cv_results = {
    'Simple Regression': cv_scores_simple,
    'Multiple Regression': cv_scores_multiple,
    'Lasso Regression': cv_scores_lasso,
    'Ridge Regression': cv_scores_ridge,
}
print("Cross-validation Results:", cv_results)


# PCA cross-validation
pca_cv_scores_simple = cross_val_score(pca_simple_model, X, y, cv=5, scoring='neg_mean_squared_error')
pca_cv_scores_multiple = cross_val_score(pca_multiple_model, X, y, cv=5, scoring='neg_mean_squared_error')
pca_cv_scores_lasso = cross_val_score(pca_lasso_model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
pca_cv_scores_ridge = cross_val_score(pca_ridge_model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')

    
# Cross-validation results
pca_cv_results = {
    'Simple Regression': pca_cv_scores_simple,
    'Multiple Regression': pca_cv_scores_multiple,
    'Lasso Regression': pca_cv_scores_lasso,
    'Ridge Regression': pca_cv_scores_ridge,
}
print("PCA cross-validation Results:", pca_cv_results)
