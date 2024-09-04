## Integrated-Disease-Prediction-Platform

### 🚀Introduction
This project is a web application that predicts multiple diseases based on the symptoms entered by the user. The user can enter the symptoms and the system will predict the diseases based on the symptoms entered. The system uses a machine learning model to predict the diseases

### 💻Technologies
- Python
- Docker
- Streamlit

### 🖱️How to run the project using Docker :-

1. Clone the repository
2. Run the following command to build the docker image

    ```bash 
    docker build -t disease-prediction-system:v1.0 .
    ```
3. Run the following command to run the docker container

    ```bash
    docker run -d -p 8501:8501 disease-prediction-system:v1.0
    ```
4. Open the browser and go to the following URL

    ```
    http://localhost:8501
    ```
5. Enter the Field values and click on the predict button to get the predicted diseases.(Note: The system will predict the disease based on the symptoms values entered by the user.)
   

### 📝Future Scope
- The system can be improved by adding more diseases and symptoms to the system.
- The system can be improved by adding more features to the system.

 
