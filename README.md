# Phishing Website Detector

A machine learning-based phishing website detector that uses Logistic Regression and Support Vector Machine (SVM) models to identify and classify phishing websites with high accuracy.

## Overview

This project implements and compares multiple machine learning algorithms to detect phishing websites. The detector analyzes various features from websites and classifies them as either legitimate or phishing websites with over 92% accuracy.

## Features

- **Multiple ML Models**: Implements both Logistic Regression and SVM (with linear and RBF kernels)
- **High Accuracy**: Achieves 92%+ accuracy in detecting phishing websites
- **Feature Scaling**: Uses StandardScaler for optimal model performance
- **Data Preprocessing**: Includes comprehensive data extraction and normalization
- **Pre-trained Models**: Includes serialized models for quick predictions
- **Jupyter Notebook**: Complete analysis and training pipeline in `base_line.ipynb`

## Project Structure

```
phishing_detector/
├── base_line.ipynb           # Main notebook with model training and evaluation
├── phishing.csv              # Dataset with phishing website features
├── LR_PHISHING_MODEL         # Trained Logistic Regression model
├── SVM_PHISHING_MODEL        # Trained SVM (Linear) model
└── README.md                 # This file
```

## Models Performance

### Logistic Regression
- **Accuracy**: 92.09%
- **Precision (Phishing)**: 0.93
- **Recall (Phishing)**: 0.93
- **F1-Score (Phishing)**: 0.93

### SVM (Linear Kernel)
- **Accuracy**: 92.40%
- **Precision (Phishing)**: 0.94
- **Recall (Phishing)**: 0.92
- **F1-Score (Phishing)**: 0.93

## Dataset

The project uses a phishing detection dataset with multiple features extracted from websites:
- **Size**: 2211 test samples (80-20 train-test split)
- **Classes**: Binary classification (Legitimate: 0, Phishing: 1)
- **Features**: Various URL and website characteristics

## Installation

### Prerequisites
- Python 3.7+
- Jupyter Notebook
- pandas
- scikit-learn
- joblib

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/phishing_detector.git
cd phishing_detector
```

2. Install required packages:
```bash
pip install pandas scikit-learn joblib jupyter
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook base_line.ipynb
```

## Usage

### Training the Models

Run the `base_line.ipynb` notebook to:
1. Load and explore the phishing dataset
2. Preprocess and scale the features
3. Train Logistic Regression model
4. Train SVM models (Linear and RBF kernels)
5. Evaluate and compare model performance

### Making Predictions

```python
import joblib as jb
from sklearn.preprocessing import StandardScaler

# Load pre-trained model
model = jb.load("LR_PHISHING_MODEL")

# Load and scale your data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_new_data)

# Make predictions
predictions = model.predict(X_scaled)
# 0 = Legitimate, 1 = Phishing
```

## How It Works

1. **Data Preprocessing**: Features are extracted from website URLs and characteristics
2. **Feature Scaling**: StandardScaler normalizes features to improve model performance
3. **Model Training**: Two different algorithms are trained on the preprocessed data:
   - Logistic Regression: Fast, interpretable linear model
   - SVM: Powerful non-linear classifier with excellent generalization
4. **Evaluation**: Models are evaluated using accuracy, precision, recall, and F1-scores

## Results

Both models achieved strong performance on the test set:
- **Best Model**: SVM (Linear) with 92.40% accuracy
- **Most Balanced**: Both models show excellent precision and recall for phishing detection
- **Test Set Size**: 2211 samples

## Technologies Used

- **Python**: Core programming language
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms and metrics
- **joblib**: Model serialization and deserialization
- **Jupyter Notebook**: Interactive development environment

## Future Improvements

- Add more advanced features (deep learning models)
- Implement cross-validation for better model assessment
- Create a web application for real-time phishing detection
- Add explainability features (SHAP values, feature importance)
- Implement ensemble methods combining multiple models
- Add API endpoint for predictions

## Contributing

Contributions are welcome! Feel free to:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**Your Name** - Initial work

## Acknowledgments

- Dataset source and contributors
- scikit-learn documentation and community
- Machine learning best practices and resources

## Contact

For questions or inquiries, please reach out via:
- Email: chitranshushekhawat@gmail.com
- GitHub Issues: [Project Issues](https://github.com/chitranshushekhawat/phishing_detector/issues)

---

**Note**: This is a demonstration project for educational purposes. For production use, consider implementing additional security measures and regular model updates.
