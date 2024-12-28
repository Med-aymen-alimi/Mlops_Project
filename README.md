

# MLOps Project: Movie Rating Classification


Welcome to our MLOps project! This repository automates the end-to-end machine learning pipeline, from raw, uncleaned movie ratings data to the deployment and monitoring of the best models. We’ve integrated MLOps tools, FastAPI backend, and Streamlit frontend for a complete solution. Our goal is to classify movie ratings efficiently and track models from development to deployment.


### Project Overview


This project is dedicated to classifying movie ratings using machine learning techniques and MLOps. We started with raw, uncleaned data and tested multiple preprocessing, modeling, and deployment approaches. Our pipeline is fully automated, leveraging the power of MLOps technologies to ensure smooth data processing, model training, deployment, and monitoring.


### Technologies Used
- **MLflow**: For tracking experiments, managing models, and automating the deployment pipeline.
- **DVC (Data Version Control)**: For managing large datasets and model versions.
- **Deepchecks**: For testing data quality and model performance.
- **Docker**: To containerize the entire environment, ensuring consistency across all stages.
- **Jenkins**: For Continuous Integration and Continuous Deployment (CI/CD) automation of the pipeline.
- **Arise**: For orchestrating the workflow between different MLOps technologies.
- **DGSHub**: For sharing and managing datasets effectively.
- **FastAPI**: As the backend service for serving models and managing API requests.
- **Streamlit**: For creating the frontend web interface for easy interaction with the models.


### Project Structure
- **modeling.ipynb**: Jupyter notebook for training, testing, and selecting the best model.
- **deepcheck.ipynb**: Jupyter notebook for validating the model and data.
- **testing**: Folder for unit tests and model validation.
- **monitoring**: Folder for monitoring model performance post-deployment.
- **jenkins**: Folder containing Jenkins container when building it you find the configurations for automating the CI/CD pipeline.


### Key Features
- **FastAPI Backend**: Serves the models and allows for real-time prediction via API endpoints.
- **Streamlit Frontend**: Interactive web interface for users to upload data and interact with the models.
- **Automated Preprocessing**: Handles data cleaning and transformation automatically.
- **Multiple Model Testing**: Tests several machine learning models to identify the best one for movie rating classification.
- **Seamless Deployment**: Automatically deploys the best model using MLflow and ensures the models are easily accessible through the FastAPI service.
- **Model Monitoring**: Continuously monitors the deployed models to track performance and identify any model drift.
- **CI/CD Pipeline**: Fully automated Continuous Integration and Continuous Deployment using Jenkins, ensuring code changes are integrated and deployed efficiently.


### How to Use


1. **Clone the Repository**:
   Clone the repository to your local machine:
   ```bash
   git clone https://dagshub.com/aymenalimii4070/Ml_OPS_Movies.git

Made With Love By
Hedi Aloulou
Mohamed Aymen Alimi
Yessine Karray
![alt text](notebooks/images/workflow.png)













