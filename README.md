# Customer Support Ticket Classification

An AI-powered application that automatically classifies customer support tickets into predefined categories using Natural Language Processing (NLP) and Machine Learning.

## Project Overview

Customer support teams receive large numbers of tickets every day. Manually reviewing and organizing these tickets can be time-consuming and may delay responses.

This project aims to develop an intelligent system that analyzes customer support messages and predicts the appropriate support category based on their textual content.

The system will process ticket subjects and descriptions, apply text preprocessing techniques, extract text features, and use machine learning models to classify incoming tickets.

## Project Objectives

- Automatically classify customer support tickets into predefined categories.
- Apply data processing and Natural Language Processing techniques.
- Develop and evaluate machine learning classification models.
- Explore ticket distributions and identify common customer issues.
- Build an interactive application using Streamlit.
- Reduce manual ticket organization and support faster customer service operations.

## Dataset

**Dataset Name:** Multilingual Customer Support Tickets

**Source:** [Kaggle – Customer Support Tickets](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets)

**License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

The dataset contains customer support tickets in multiple languages, including English and German.

For this project, the English-language records will be selected for the classification task.

**Main Features:**

| Column | Description |
|---|---|
| `subject` | Ticket subject |
| `body` | Ticket description |
| `queue` | Support department/category |
| `type` | Ticket type |
| `priority` | Ticket priority |
| `language` | Ticket language |

**Classification Setup:**

- Input Features: `subject` and `body`
- Target Variable: `queue`

The dataset will be inspected and validated before preprocessing and model development.

## Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- Natural Language Processing (NLP)
- TF-IDF
- Matplotlib and Seaborn
- Streamlit
- Jupyter Notebook

## Project Workflow

The project follows these main stages:

1. **Data Loading:** Load the dataset and inspect its structure.
2. **Data Validation:** Check data types, missing values, duplicates, and required columns.
3. **Data Cleaning:** Handle missing values and prepare ticket text.
4. **Exploratory Data Analysis:** Analyze category distributions and ticket characteristics.
5. **Feature Extraction:** Convert ticket text into numerical features using TF-IDF.
6. **Model Training:** Train and compare machine learning classification models.
7. **Model Evaluation:** Evaluate model performance using appropriate classification metrics.
8. **Application Development:** Build an interactive Streamlit application.
9. **Export and Documentation:** Save the trained model and document the project results.

## Repository Structure

```text
Customer-Support-Ticket-Classification/
│
├── Data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── customer_support_analysis.ipynb
│
├── src/
│
├── models/
│
├── tests/
│
├── requirements.txt
├── .gitignore
├── README.md
└── app.py
```

*The repository structure will be expanded as the project development progresses.*

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Raneem04H/Arti-303-Project-.git
cd Arti-303-Project-
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Environment

For Windows:

```bash
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Notebook

```bash
jupyter notebook
```

Open `notebooks/customer_support_analysis.ipynb` to explore the dataset and follow the data processing workflow.

## Team Contributions

This is a collaborative academic project developed as part of the Programming for Artificial Intelligence course.

Each team member is responsible for a specific part of the project, including data collection, preprocessing, exploratory data analysis, model development, evaluation, and application integration.

## Data Usage and Ethics

The dataset is obtained from a publicly available Kaggle source and is listed under the CC BY 4.0 license.

The original dataset source and creator must be appropriately credited. Any modifications to the dataset should be documented.

Personal or confidential information must not be uploaded or shared through the project repository.

## Project Status

**Currently in progress.**

The initial dataset collection, loading, and validation stage has been prepared. Further development will include data preprocessing, exploratory analysis, machine learning model development, evaluation, and Streamlit integration.