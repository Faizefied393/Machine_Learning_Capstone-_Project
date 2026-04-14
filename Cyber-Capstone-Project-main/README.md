# Whooping Crane Surveillance ML Platform

## Overview
This project, developed by Team 3 – *Birds* at the University of Tulsa, supports the **International Crane Foundation (ICF)** by automating the detection of endangered **Whooping Cranes** from camera trap images using a machine learning-based web application. The platform provides an easy way for biologists to upload images, analyze them using a custom CNN model, and receive results in an Excel format.

---

## Key Features
- Upload multiple images through a simple web interface
- CNN model classifies images as "bird" or "not bird"
- Excel report includes filenames and detection results
- Designed for non-technical users (e.g. field biologists)
- Includes basic login system (in development)
- Supports optional AWS S3 cloud storage
- Processes thousands of images in a single run

---

## Technology Stack

| Component         | Technology         | Purpose                                  |
|------------------|--------------------|------------------------------------------|
| Frontend         | HTML, CSS, Jinja2  | User interface for image uploads         |
| Backend          | Flask (Python)     | Handles file uploads, ML processing      |
| Machine Learning | PyTorch CNN        | Detects presence of Whooping Cranes      |
| Output           | pandas, openpyxl   | Generates Excel report                   |
| Cloud Storage    | AWS S3 (Optional)  | Handles large image batches              |
| Security         | Login system (WIP) | Basic authentication and access control  |

---

## File Structure
```
project/
├── app.py                 # Flask application logic
├── model.py               # ML model and prediction code
├── templates/             # HTML templates (LandingPage.html)
├── static/                # CSS and static images
├── uploads/               # Temporary image upload storage
├── requirements.txt       # Python dependencies
├── .env                   # AWS credentials (not tracked)
└── README.md              # Project documentation
```

---

## Setup Instructions

1. **Clone the Repository**
```bash
git clone https://github.com/your-org/whooping-crane-detector.git
cd whooping-crane-detector
```

2. **Create Virtual Environment**
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **(Optional) Configure AWS S3**
Create a `.env` file in the root directory:
```
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
AWS_REGION=us-east-1
S3_BUCKET_NAME=whooping-crane-uploads
```

5. **Run the App**
```bash
python app.py
```
Open your browser and go to `http://localhost:5000`

---

## Machine Learning Model
- Type: Convolutional Neural Network (CNN)
- Input: 128x128 RGB images
- Output: Binary classification ("Bird"/"No Bird")
- Accuracy: ~91% overall
- Current focus: Improving F1-score for rare crane class using:
  - Oversampling & Undersampling
  - Class weighting
  - New image datasets

---

## Security Features (Planned)
- User login with password validation
- Admin access layer for report control
- File size and type restrictions
- Secure .env for API keys and AWS credentials

---

## Future Development
- Expand detection to include crane counting
- Real-time or scheduled uploads via cloud
- Visual analytics and dashboards
- Hosting on a domain with SSL encryption

---

## Team
**Team 3 – Birds**  
University of Tulsa, Cybersecurity Capstone

| Name     | Role                                  |
|----------|---------------------------------------|
| Sarah    | Presentation & Communications         |
| Jalen    | UI Design & Security System           |
| Helaina  | Application Development & Docs        |
| Faiz     | Machine Learning, GitHub, & Client Liaison |
| Zade     | Machine Learning Development          |

**Client:** Raymond Kirkwood, International Crane Foundation
|
**Support Contact:** Stephen Flowerday, University of Tulsa
