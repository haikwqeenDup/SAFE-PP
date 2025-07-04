# SAFE-PP
A Spatial-Aware FramEwork for Land Suitability  Classification

# Cultivated Land Dataset

## Dataset Overview  
The cultivated land dataset comprises categorical features and numerical features, which collectively reflect the land characteristics of specific cultivated areas and serve as a basis for cultivated land decision-making.  


## Categorical Features (12 Attributes)  
- Soil pollution grade of classification unit  (3 types of values)
- Soil layer thickness of classification unit  (3 types of values)
- Soil texture of classification unit  (3 types of values)
- Soil pollution grade of expanded classification unit  (3 types of values)
- Soil layer thickness of expanded classification unit  (3 types of values)
- Soil texture of expanded classification unit  (3 types of values)
- Whether the slope is above 2°  (2 types of values)
- Whether the slope is above 6°  (2 types of values)
- Whether the slope is above 15°  (2 types of values)
- Main functional area  (4 types of values)
- Drinking water protection area grade  (3 types of values)
- (Additional categorical attributes as applicable)

## Numerical Features (56 Attributes)  
- Cultivated land fragmentation index  (numerical value between 0 and 1)
- Proportion of high-standard farmland area  (numerical value between 0 and 1)
- Proportion of cultivated land area  (numerical value between 0 and 1)
- Area of permanent basic farmland  (numerical value with the unit of m²)
- Area of cultivated land irrigation  (numerical value with the unit of m²)
- Area of cultivated land inflow  (numerical value with the unit of m²)
- Distance between cultivated land and rivers  (numerical value with the unit of m)
- Area of forest land  (numerical value with the unit of m²)
- Area of paddy field  (numerical value with the unit of m²)
- Area of cultivated land reserve resources  (numerical value with the unit of m²)
- (56 total numerical attributes, including additional metrics such as spatial distances, area proportions, and topographical indices)  


## Dataset Structure Summary  
- **Categorical Features**: 12 attributes, capturing qualitative characteristics like soil properties, slope classifications, and functional zoning.

These features collectively characterize the physical, environmental, and spatial attributes of cultivated land, enabling comprehensive analysis for agricultural planning, land management, ecological protection, and policy-making.


# SAFE++ Model Code


## Model Overview  
**SAFE++** is a model designed for cultivated land decision-making, with generalizable applications to broader land decision tasks. It integrates spatial data processing, feature engineering, and machine learning to support informed land management decisions.  


## Code Structure and Functionality  
### 1. Preprocessing Scripts  
- **step1_preprocess_csv_to_pkl.py**  
  First-stage preprocessing: Parses geohash codes of land data into definite latitude-longitude coordinates, then stores the processed data as pickle (`.pkl`) files.  

- **step2_preprocess_pkl_to_tensor.py**  
  Second-stage preprocessing: Converts data into tensors suitable for direct model input. Operations include spatial information aggregation, regional data partitioning, categorical attribute encoding, and numerical attribute standardization.  


### 2. Model and Training Tools  
- **tool_model.py**  
  Contains PyTorch model class definitions required for model execution, defining network architectures and forward propagation logic.  

- **tool_test_and_draw.py**  
  Includes functions for testing model performance, supporting evaluation metrics calculation and result visualization.  

- **tool_train.py**  
  Contains training functions with implemented enhancement strategies: variable learning rates, boundary contrastive learning, and QRAS ( Quad Rotation Augmentation Strategy).  


### 3. Main Execution Script  
- **main.py**  
  The primary entry point of the program, integrating scripts for model training and testing.  


### 4. Data Examples  
Three provided `.pt` files contain preprocessed data samples (after step1 and step2) that can be directly input into the model, demonstrating the data format and structure expected by the model.  


## Model Applications  
- Core application: Cultivated land decision-making (e.g., land use planning, soil protection strategies).  
- Generalizable to: Various land-related decision tasks, including urban land development, ecological conservation, and agricultural resource management.  


## Technical Features  
- Spatial data processing pipeline for geohash and coordinate conversion.  
- End-to-end workflow from raw data to model inference.  
- Enhanced training strategies for improved generalization and accuracy.  
- Modular code structure for easy extension and customization.  


## Runtime Environment  
- **PyTorch**: 1.8.1  
- **Python**: 3.8  
