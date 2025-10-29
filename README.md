# Erythemato-Squamous Disease Diagnosis using Stacking Ensemble

## Problem Description

The differential diagnosis of **erythemato-squamous diseases** is a significant challenge in dermatology. These diseases, including **psoriasis**, **seboreic dermatitis**, **lichen planus**, **pityriasis rosea**, **cronic dermatitis**, and **pityriasis rubra pilaris**, share very similar clinical features like erythema and scaling. While a biopsy is often necessary, even histopathological features can overlap, making accurate diagnosis difficult. Additionally, symptoms may evolve, further complicating the process.

This project proposes a machine learning solution using a **stacking ensemble model** to improve the accuracy of diagnosing these six types of erythemato-squamous diseases based on clinical and histopathological data.

## Dataset

The project utilizes the **Dermatology Database** originally provided by Nilsel Ilter, M.D., Ph.D. (Gazi University) and H. Altay Guvenir, PhD. (Bilkent University).

* **Source**: Donated by H. Altay Guvenir, Bilkent University, January 1998.
* **Instances**: 366 patient records.
* **Attributes**: 34 features in total:
    * 12 Clinical Attributes (e.g., erythema, scaling, itching, family history, age).
    * 22 Histopathological Attributes (e.g., melanin incontinence, fibrosis, exocytosis).
    * Most features are ordinal, rated on a 0-3 scale indicating absence to the largest possible amount. Family history is binary (0 or 1), and Age is linear.
* **Missing Values**: The 'Age' attribute has 8 missing values, denoted by '?'.
* **Classes**: 6 types of Eryhemato-Squamous Diseases:
    1.  Psoriasis (112 instances)
    2.  Seboreic Dermatitis (61 instances)
    3.  Lichen Planus (72 instances)
    4.  Pityriasis Rosea (49 instances)
    5.  Cronic Dermatitis (52 instances)
    6.  Pityriasis Rubra Pilaris (20 instances)

_For detailed attribute information, refer to `input/dermatology.names`._

## Methodology

### 1. Preprocessing

* **Data Loading**: Raw data (`dermatology.data`) is loaded using Pandas, with column names assigned based on `dermatology.names`.
* **Imputation**: Missing 'Age' values are imputed using the mean (or median, configurable) of the column. The cleaned data is saved as `dataset.csv`.
* **Correlation Analysis**: Polychoric correlation (suitable for ordinal data) is calculated between each feature and the target 'disease' class using an R script (`feature.R`). Results are stored in `feature info.csv`.
* **Feature Selection**: Features with low correlation (absolute value between -0.1 and 0.1) to the target variable are excluded to reduce noise and dimensionality.
* **Data Splitting**: The preprocessed data is split into training (80%) and testing (20%) sets, stratified by the disease class to maintain class distribution.

### 2. Model Building: Stacking Ensemble

**Stacking (Stacked Generalization)** is employed because different types of models can capture different aspects of the complex relationships within the data. This project uses a two-level stacking approach:

* **Level 0 (Base Learners - Linear):**
    * Linear Discriminant Analysis (LDA)
    * Linear Support Vector Classifier (SVC) with StandardScaler
    * K-Nearest Neighbors (KNN, n=3)
* **Level 1 (Meta-Learner 1 - Combining Non-Linear):**
    * **Base Learners:**
        * Decision Tree Classifier (DT)
        * Random Forest Classifier (RF)
        * Multi-layer Perceptron (MLP) Neural Network
    * **Meta-Learner:** Logistic Regression (combines outputs of DT, RF, NN)
* **Level 2 (Final Meta-Learner - Combining Linear and Level 1):**
    * The predictions from the Level 0 linear models (LDA, SVC, KNN) and the output of the Level 1 Meta-Learner are fed into a final **Logistic Regression** model.

This hierarchical structure allows the model to leverage the strengths of both linear and non-linear classifiers. The final model (`stack_model.pkl`) is saved using pickle.

### 3. Evaluation

The model's performance is evaluated on the test set using **accuracy**.

## Technologies Used

* **Python 3.9**
* **Pandas**: Data manipulation and loading.
* **Scikit-learn**: Machine learning models (LDA, SVC, KNN, DT, RF, MLP, Logistic Regression), StackingClassifier, pipelines, preprocessing (StandardScaler, imputation - although handled manually here), train-test splitting, metrics.
* **NumPy**: Numerical operations.
* **Matplotlib & Seaborn**: Data visualization (used in `analytics.ipynb`).
* **Pickle**: Saving the trained model.
* **R** (with `polycor` package): Used specifically for polychoric correlation analysis (`feature.R`).
* **Jupyter Notebook**: For exploratory data analysis and rationale documentation (`storybooks/`).

## Setup and Usage ⚙️

1.  **Prerequisites**:
    * Python 3.9+ environment.
    * R environment with the `polycor` package installed.
    * Required Python libraries: `pandas`, `scikit-learn`, `numpy`. (Install via pip: `pip install pandas scikit-learn numpy`)

2.  **Correlation (Optional - Precomputed)**:
    * If you need to recalculate correlations, navigate to `src/R/` and run `feature.R`. Ensure the working directory is set correctly within the script or run it from the project root.

3.  **Run the Model**:
    * Navigate to the `src/python/` directory.
    * Execute the main script: `python main.py`
    * The script will:
        * Load raw data.
        * Perform imputation.
        * Perform feature selection based on precomputed correlations.
        * Split data into train/test sets.
        * Build, train, and save the stacking model.
        * Make predictions on the test set.
        * Evaluate and print the model's accuracy.

## Results

The stacking ensemble model achieves high accuracy in diagnosing the six erythemato-squamous diseases on the test set. The exact accuracy score is printed upon running `main.py`. _(Note: Based on the `reason.ipynb` notebook, individual models like Decision Tree achieved ~95% accuracy, while SVM was lower at ~70%. Stacking aims to potentially improve upon the best individual model)_.
