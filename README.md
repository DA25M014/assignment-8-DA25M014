🧾 [DA5401] DA Lab – Assignment 8 – Ensemble Learning for Complex Regression Modeling on Bike Share Data

Name: Jigarahemad K Shaikh  Roll Number: DA25M014

📂 File Required for Evaluation – DA5401_DA25M014_Assignment_8_Final_Notebook.ipynb

Dataset: 🔗 [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset)

OR

📁 Data Folder: All input datasets required for this assignment are stored in the “Data” folder within the repository.


Title: Ensemble Learning for Complex Regression Modeling on Bike Share Data

Objective: This assignment will challenge the solver to apply and compare three primary ensemble techniques (Bagging, Boosting, and Stacking) to solve a complex, time-series-based regression problem. You will demonstrate your understanding of how these methods address model variance and bias, and how a diverse stack of models can yield superior performance to any single model.

The Notebook (DA5401_DA25M014_Assignment_8_Final_Notebook.ipynb) investigates how ensemble learning techniques—Bagging, Boosting, and Stacking—enhance regression accuracy on the Bike Sharing Demand dataset by addressing variance and bias in hourly bike rental predictions. The workflow spans preprocessing, baseline modeling, ensemble experimentation, and comparative analysis using Root Mean Squared Error (RMSE) as the key evaluation metric.

⚙️ Libraries Used

pandas, numpy — data handling & numerical operations
matplotlib.pyplot, seaborn — custom visualizations and color-blind-friendly plots
scikit-learn — regression models, pipelines, metrics, and preprocessing (DecisionTreeRegressor, BaggingRegressor, GradientBoostingRegressor, StackingRegressor, KNeighborsRegressor, Ridge, train_test_split, StandardScaler, OneHotEncoder, mean_squared_error)

🎨 Color Palettes & Visualization Design

Palettes: colorblind, Set2, Spectral, husl, Paul Tol TOL Colors

Colormaps: viridis, coolwarm, YlGnBu

Accessibility: All plots include hatching/patterns, distinguishable markers, and labeled legends for color-blind safety.

Consistency: Unified axis labeling, fixed RMSE scales, and subtle gridlines for visual readability.

🧠 Assignment Overview

The task involves forecasting hourly bike rentals (cnt) based on weather, temporal, and seasonal factors.
Through three ensemble paradigms, the notebook explores how model aggregation mitigates overfitting and improves generalization.
The exercise provides a hands-on understanding of the bias–variance trade-off in regression contexts.

🔧 Workflow Summary


Part A – Data Preprocessing & Baseline

Loaded hour.csv and removed irrelevant columns: instant, dteday, casual, registered.

Applied One-Hot Encoding to categorical variables (season, mnth, hr, weathersit, etc.).

Scaled numerical features with StandardScaler.

Trained two baselines — Decision Tree Regressor (max_depth = 6) and Linear Regression.

Evaluated both using RMSE and chose the better performer as the baseline.


Part B – Ensemble Techniques

Bagging (Variance Reduction)

Implemented BaggingRegressor with Decision Tree base learners (100 estimators).

Reduced variance and improved stability compared to the single tree.

Boosting (Bias Reduction)

Implemented GradientBoostingRegressor to sequentially correct bias.

Achieved significant RMSE improvement over baseline and Bagging.

Included comparative plots and commentary on error trends.


Part C – Stacking for Optimal Performance

Combined KNN Regressor, Bagging Regressor, and Gradient Boosting Regressor as Level-0 base models.

Used Ridge Regression as Level-1 meta-learner.

Implemented StackingRegressor to aggregate model predictions.

Achieved the lowest RMSE among all tested models.


Part D – Final Analysis & Discussion

Constructed a comparative RMSE table for all models with an added ranking column.

Provided a conceptual explanation connecting performance gains to ensemble diversity and bias–variance balance.

Concluded with interpretation of why Stacking generalizes best for this regression task.



📊 Result Summary

Model	Concept	Test RMSE	Rank
Decision Tree Regressor	Baseline (non-linear, high variance)	115.2	5
Linear Regression	Baseline (linear, high bias)	108.5	4
Bagging Regressor	Variance reduction via bootstrapping	89.7	3
Gradient Boosting Regressor	Sequential bias correction	76.4	2
Stacking Regressor (Ridge meta)	Combines diverse learners for optimal trade-off	71.3	1 🥇


💡 Key Insights

Bagging → Reduces variance by training multiple resampled models.

Boosting → Sequentially reduces bias by focusing on residuals.

Stacking → Learns optimal meta-weights, combining strengths of base models.

Ensemble approaches consistently outperform single regressors on non-linear, high-variance data.


🧩 Outcome & Takeaways

Ensemble models demonstrate robustness and improved generalization on complex regression tasks.

Stacking yielded the lowest test RMSE, highlighting the power of model diversity.

Clear visualization and interpretability principles make the analysis reproducible.

Reinforces the theoretical link between bias–variance decomposition and ensemble learning efficiency.

Date: 10/Nov/2025
