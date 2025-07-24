# water-potablity-detector

This project is a Flask-based web application that allows users to predict water potability based on various water quality parameters. Users can upload datasets for training a machine learning model or input individual water quality parameters to test potability.

Features
- Dataset Upload: Upload a CSV dataset containing water quality parameters to train the machine learning model.
- Predict Potability: Enter specific water quality parameters to predict whether the water is potable.
- Validation: Input fields are validated to ensure correct data entry.

Tech Stack
- Backend: Flask (Python)
- Frontend: HTML, CSS, JavaScript
- Machine Learning: XGBoost

How It Works
1. Upload a dataset with water quality parameters and potability labels.
2. The application trains an XGBoost model using the uploaded dataset.
3. Input water quality parameters to test potability.
4. The application predicts and displays whether the water is safe to drink.

Prerequisites
- Python 3.7
- Pip

Installation

1. Clone the repository:
   bash
   git clone https://github.com/Dheeraj5604/water-potability-predictor.git
   cd water-potability-predictor
   

2. Install dependencies:
bash
   pip install -r requirements.txt

3. Create the required folders:
bash
   mkdir templates static uploads
 
   - Place your index.html file in the templates folder.
   

4. Start the vs code:
terminal
   python app.py
  

5. Open your browser and visit:
   
   http://127.0.0.1:5000/

6.File Structure

water-potability-predictor/
├── app.py               # Flask backend 
├── templates/           # HTML templates with css and javascript  
│   └── index.html
├── uploads/             # Folder for uploaded datasets
└── README.md            # Project documentation


7.Dataset Format
The dataset must be in CSV format and include the following columns:
- ph
- Hardness
- Solids
- Chloramines
- Sulfate
- Conductivity
- Organic_carbon
- Trihalomethanes
- Turbidity
- Potability

8.Output 
Give output as :
1.water is safe to drink or water is not safe to drink

9.Usage
1. Train the Model:
   - Upload a dataset via the "Upload Dataset" form on the home page.
2. Predict Potability:
   - Enter values for each parameter in the "Test Water Parameters" form.
   - Submit to see the prediction result.

10.Contributing
Contributions are welcome! If you'd like to contribute, please fork the repository and create a pull request.

 11.Acknowledgments
- Dataset: [Kaggle Water Potability Dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability)
- Framework: Flask
- Model: XGBoost
