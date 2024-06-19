# Team-59
# VOPA Teaching Assessment Automation

## Project Overview
This project aims to automate the assessment process for teachers at VOPA (Vowels of The People Association), an NGO focused on providing education to underprivileged children. The solution leverages machine learning models, audio-to-text recognition, and various text processing techniques to evaluate teacher assessments efficiently.

## Technologies Used
- Machine Learning Models: Utilized machine learning models for automated assessment, including Longest Common Sequence (LCS), TFIDF (Term Frequency-Inverse Document Frequency), and Spacy for natural language processing.
- Audio-to-Text Recognition: Implemented audio-to-text recognition for converting spoken assessments into text for further analysis.
- Flask: Used Flask, a Python web framework, to connect the backend machine learning models with the frontend developed in React.
- MongoDB: Employed MongoDB, a NoSQL database, to store teacher login/signup information securely.

## Features
- Automated Assessment: Provides automated assessment of teacher evaluations using machine learning models, reducing manual effort and improving efficiency.
- User Authentication: Implements user authentication for teachers, ensuring secure access to the platform.
- Data Storage: Stores assessment data securely in a MongoDB database, allowing for easy retrieval and analysis.
- User Interface: Offers a user-friendly interface developed in React for easy interaction with the assessment system.


## Installation
1. Clone the repository: 
   ```bash
   git clone https://github.com/cfgmumbai24/Team-59.git
   ```
   ```bash
   cd TEAM 59
   ```
2. Install frontend dependencies: 
    ```bash
   cd client
   ```
   ```bash
   npm install
   ```
3. Install Python backend dependencies:
    ```bash
    cd ml_model
    ``` 
    ```bash
    pip install -r requirements.txt
    ```
4. Configure MongoDB: 
   - Set up a MongoDB database.
   - Update the connection string in the .env file in the server folder.
5. Run the Flask application: 
   ```bash
   python app.py
   ```
6. Run the React frontend: 
   ```bash
   npm start
   ```
   For building the Frontend
   ```bash
   npm run build
   ```
7. Run the backend: 
   ```bash
   node app.js
   ```
   If you have nodemon installed
   ```bash
   nodemon app.js
   ```

Make sure to follow these steps in order to properly set up and run the VOPA Teaching Assessment Automation project.
## Usage
1. Sign up or log in as a teacher.
2. Automated assessments through audio.
3. View automated assessment results based on LCS, TFIDF, and Spacy.
4. Track assessment history and progress over time.


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- Special thanks to VOPA for the opportunity to work on this project and improve the teaching assessment process.

---
