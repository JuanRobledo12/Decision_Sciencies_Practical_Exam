# Entering Practical Exam for the Decision Sciences Research Center at Tecnologico de Monterrey

Welcome to my Practical Exam repository. Below you can find links to the files related to each of the exam questions' deliverables that I was able to complete.

## Question 1: Comprehensive Data Acquisition and Preprocessing
- Q1 Jupyter Notebook: [q1.ipynb](q1.ipynb)
- Markdown file: [q1.md](/deliverables/q1.md)

## Question 2: Predictive Modeling and Scenario Analysis
- Q2 Jupyter Notebook: [q2.ipynb](q2.ipynb)
- Markdown file: [q2.md](/deliverables/q2.md)

## Question 4: Classification and Policy Implications
- Q4 Jupyter Notebook: [q4.ipynb](q4.ipynb)
- Markdown file: [q4.md](/deliverables/q4.md)

## Additional Files
- [helpers.py](helpers.py): A set of classes and methods that I developed to assist with data preprocessing, analysis, and evaluation.

## Observations
- I decided to only use data of countries from the OECD to familiarize myself with the data and problems. However, a further analysis is required with the inclusion of more countries.
- In Q4 I need to further explore why I am getting overly optimistic metrics. In addition, I need to check if the dataset format I selected is correct and test the models with some modifications.
## Requirements

To set up the environment for this project, follow these steps:

### 1. Create and Activate a Conda Environment

1. **Create a New Conda Environment:**

   ```bash
   conda create --name myenv python=3.12
   ```

   Replace `myenv` with your preferred environment name.

2. **Activate the Environment:**

   Once the environment is created, activate it using:
   ```bash
   conda activate myenv
   ```

### 2. Install Dependencies from `requirements.txt`

With the environment activated, install the required packages listed in your `requirements.txt` file:
```bash
pip install -r requirements.txt
```
