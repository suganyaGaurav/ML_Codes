# Data Visualization & EDA using Seaborn and Plotly Express

## Overview

This module focuses on **Exploratory Data Analysis (EDA)** and **data visualization**
using **Seaborn** and **Plotly Express**.

The goal of this work is to:
- Understand how visualization helps uncover patterns in data
- Practice univariate, bivariate, and multivariate analysis
- Learn when to use static vs interactive visualizations
- Interpret business-relevant insights from visual patterns

This is a **learning and experimentation module**, not a production pipeline.

---

## Dataset Used

### IBM HR Analytics – Employee Attrition & Performance

This dataset contains employee-related attributes such as:
- Age
- Gender
- Department
- Job Role
- Monthly Income
- Years at Company
- Attrition status

### Business Context

Employee attrition is a critical HR metric that directly impacts:
- Hiring cost
- Productivity
- Organizational stability

The analysis aims to **identify factors contributing to employee attrition**.

---

## Problem Statement

Uncover the key factors that lead to employee attrition and understand:
- Who is more likely to leave
- Why employees leave
- Which organizational factors influence attrition

---

## Analysis Tasks Covered

### Task 1 – Univariate Analysis
- Distribution of **Age**
- Distribution of **Monthly Income**
- Visualized using histograms and KDE plots

---

### Task 2 – Attrition Overview
- Overall attrition percentage
- Attrition rate across departments
- Visualized using count plots

---

### Task 3 – Bivariate Analysis
- Gender vs Attrition
- Distance from Home vs Attrition
- Monthly Income vs Attrition
- Visualized using bar plots

---

### Task 4 – Experience & Attrition
- Years at Company vs Monthly Income
- Senior employees vs recent joiners
- Visualized using scatter plots

---

### Task 5 – Job Role & Job Level Impact
- Attrition by Job Role
- Attrition by Job Level
- Visualized using swarm plots

---

### Task 6 – Department-wise Income Analysis
- Monthly income distribution across departments
- Visualized using box plots

---

### Task 7 – Age vs Monthly Income
- Relationship between Age and Monthly Income
- Visualized using joint plots

---

## Key Findings

- Majority of employees are between **20 and 45 years**
- Most employees earn between **2000 and 7000** monthly
- Attrition is highest in **Sales** and **R&D** departments
- Employees with **low income and fewer years at company** are more likely to leave
- Higher job level and higher income correlate with **lower attrition**
- Distance from home impacts attrition, especially for female employees

---

## Plotly Express – Interactive Visualization Practice

In addition to Seaborn, this module explores **Plotly Express** for:
- Scatter plots & bubble charts
- Line plots
- Bar charts
- Pie charts
- Area plots
- Box plots
- Histograms
- Heatmaps
- Violin plots
- Animated visualizations using Gapminder dataset
- Geographic visualizations (choropleth maps)

These examples demonstrate how **interactive plots** improve storytelling
and exploratory analysis.

---

## Learning Outcomes

- When to use Seaborn vs Plotly Express
- How visualization supports data-driven decisions
- How to interpret patterns beyond summary statistics
- How to communicate insights visually

---

## Notes

- This module is intended for **practice and conceptual understanding**
- No predictive modeling or deployment is included
- Larger ML and GenAI systems are maintained in separate repositories

---

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly Express
