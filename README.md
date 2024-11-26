Market Forecasting Using Machine Learning
Team Members
Hemanth Borra - 801428928
Yeswanth Kumar Muttha - 801398789
Adarsh Kodumuru - 801365902
Project Description
This project aims to optimize pricing strategies in the mobile phone industry by applying advanced machine learning techniques. By analyzing market demand, competition, and pricing elasticity, we developed a data-driven model that accurately predicts the optimal price for maximizing revenue while maintaining customer satisfaction.

Features
Dynamic Pricing Optimization: Adjust prices based on demand and market trends.
Machine Learning Models: Utilizes Linear Regression, Ridge Regression, Lasso Regression, and Random Forest for price prediction.
High Accuracy: Random Forest achieved the highest accuracy of 93.9%.
Scalability: Handles large datasets efficiently through automated preprocessing and model training.
Technologies Used
Programming Language: Python
Libraries:
pandas, numpy - Data manipulation and analysis
matplotlib.pyplot, seaborn - Data visualization
sklearn - Machine learning models
imblearn.under_sampling - Handling imbalanced datasets
pickle - Model serialization
warnings - Suppressing warnings
Dataset
The dataset used for this project is named Generated_Mobile_Phone_Data and includes:


Average Price/Unit: Current selling price of the product
Cost/Unit: Manufacturing cost per unit
Average Profit/Unit: Profit earned on each unit
Incremental Price Impact: Effect of price changes on customer return rate
Increase in Sales Volume: Price elasticity and demand response
The dataset file, Flipkart_Mobiles.csv, is included in the zip file provided.

Instructions to Run the Code
Step 1: Install Prerequisites
Ensure the following tools and libraries are installed:

Jupyter Notebook
Python libraries: pandas, numpy, matplotlib.pyplot, seaborn, sklearn, pickle, warnings, imblearn.under_sampling
You can install these libraries using pip:

bash
Copy code
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
Step 2: Prepare the Dataset
Download the Generated_Mobile_Phone_Data.csv file from the zip folder provided.
Copy its file path for use in the code.
Step 3: Run the Code
Open the .ipynb file provided in the zip folder using Jupyter Notebook.
Paste the file path of Flipkart_Mobiles.csv into the code where prompted to upload the dataset.
Run the code cells line by line to preprocess the data, train the machine learning models, and view the results.
Results
Linear and Ridge Regression: Achieved an accuracy of 88%.
Lasso Regression: Lower accuracy of 45%.
Random Forest: Best performance with 93.9% accuracy.
The Random Forest model provided the most reliable predictions for optimal pricing, outperforming other algorithms.

Future Scope
Incorporate real-time data for dynamic pricing updates.
Explore more advanced algorithms, such as neural networks, to further improve accuracy.
Expand the model to include additional factors like customer reviews and competitor pricing.
Acknowledgment
This project was completed as part of our academic coursework. We extend our thanks to everyone who supported us in this endeavor.

Contact Information

Hemanth Borra: hborra@charlotte.edu
Yeswanth Kumar Muttha: ymuttha@charlotte.edu
Adarsh Kodumuru: akodumur@uncc.edu