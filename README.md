# Chillindo_Data_Science
Data Science Engineer position test.

This repository contains 2 main folders: 
1. Test-1: It includes the theoritical and conceptual questions will answers.

File: test-1.docx

2. Test-2: Implementation of the given problem.

In test-2:

Packages that have been used in Python 3.5: 
Numpy 1.15.2,
Pandas 0.23.4,
Matplotlib 2.1.1,
Scikit-learn 0.19.2,
Psycopg2 2.7.5,
Redis 2.10.6

Files in the folder:

1.Database: It creates a postgresql database and load the pokemon.xslv file into it.

2.Classification: It contains the python code to load dataset from postgresql database and then the modeling and computation is done on it to predict the legendary pokemon in the form of 0 (False) & 1 (True) and store the value of true positive rate, false negative rate and precision into redis database. An ROC curve is also plotted. 

3. pokemon.xslv: The given dataset

4. Test-2.docx: The package and enviornment used.

How to Run?

1.Setup and install postgresql database.

2.Run file database.py

3.Execute file classification.py

## Comments have been added in the code itself.

##The task is completed as per the instructions with full functionalities.
