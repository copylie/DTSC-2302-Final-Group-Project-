# DTSC-2302 Final Group Project  
## Modeling Neighborhood Factors Influencing Housing Prices

**Pavle Dimitrijevic, Ikumi Uemura, Michael Forshay, Sonia Sun, and James Harris**

---

## Overview
For our final project, we analyze how neighborhood-level economic, educational, housing, and public safety factors influence housing prices in Charlotte, North Carolina. Using data merged at the Neighborhood Profile Area (NPA) level, our goal is to both predict housing prices and classify neighborhoods into high and low-value areas.  

Housing prices serve as a meaningful outcome variable because they reflect broader neighborhood conditions and overall quality of life.

---

## Data & Merging
We use data from multiple sources, including:
- Charlotte-Mecklenburg Quality of Life dataset  
- City of Charlotte Open Data portal  

### Key variables include:
- Employment rate  
- Graduation rate  
- Job density  
- Home sales prices  
- CMPD homicide records  

All datasets are merged using the **NPA identifier**.  

Homicide data is aggregated to the NPA level by counting incidents per neighborhood before merging.

---

## Regression Model
The objective of the regression model is to **predict 2023 housing prices** across neighborhoods.

### Target Variable:
- `home_price_2023`

### Features:
- `employment_2023`  
- `grad_2023`  
- `job_density_2022`  
- `homicide_count`  
- `home_price_2021`  

### Models Tested:
- Linear Regression (baseline, interpretable)  
- K-Nearest Neighbors (captures nonlinear patterns)  
- Tree-based models (Decision Tree / Random Forest)  

### Evaluation Metrics:
- RMSE (Root Mean Squared Error)  
- R² (Coefficient of Determination)  

Focus will also be placed on identifying the most important predictors.

---

## Classification Model
The objective of the classification model is to **classify neighborhoods as high-value or low-value**.

### Target Variable:
- `high_price_2023`  
  - 1 = Above median 2023 home price  
  - 0 = Below median  

### Features:
- `employment_2023`  
- `grad_2023`  
- `job_density_2022`  
- `homicide_count`  
- `home_price_2021`  

### Models Tested:
- Logistic Regression  
- K-Nearest Neighbors  
- Tree-based models  

### Evaluation Metrics:
- Accuracy  
- ROC-AUC  
- Confusion Matrix  

---

## Summary
This project develops both a regression and classification model centered on understanding and predicting housing prices at the neighborhood level.  

The analysis emphasizes **practical implications**, particularly which neighborhood factors are most strongly associated with housing value differences. Detailed modeling steps and outputs will be included in the appendix.

---