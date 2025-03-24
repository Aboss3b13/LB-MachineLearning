# LB-MachineLearning


### Dataset Description
This dataset provides detailed information about the prices of houses across various locations, including the state, city, and zip code of the property. Additionally, it includes key features of the properties, such as the number of bedrooms, bathrooms, and the size of the lot, which is measured in acres. The dataset is structured to offer a comprehensive overview of factors that may influence property prices, allowing for analysis and prediction modeling.

P.S The original CVS file conrtains over a million different entries. This one has roughle 300000, because the file was too big to upload.

### Privacy and Measures
The dataset contains no personally identifiable information (PII) beyond the general location and property details.


### My goal
The objective of this project is to develop a model that can predict property prices based on various factors such as location (state, city, zip code) and property characteristics (number of bedrooms, bathrooms, and lot size). By analyzing the relationships between these attributes and property values, I aim to build a machine learning model that can accurately estimate house prices.

### 2.4

In this dataset, scaling fields like price or bed is not required because decision trees (such as Random Forest) and other algorithms are not sensitive to the different scales of input values. For linear models, scaling might be helpful, but since we are mainly focused on price predictions and simple classifications, we have opted not to scale the data.
