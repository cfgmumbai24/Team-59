# Team-59 - Code-a-Cola
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
    Create a Virtual Environment (For MacOS)
     ```bash
    python3 -m venv myenv
    ```
      ```bash
    source myenv/bin/activate
    ```
    ```bash
    pip install -r requirements.txt
    ```
5. Configure MongoDB: 
   - Set up a MongoDB database.
   - Update the connection string in the .env file in the server folder.
6. Run the Flask application: 
   ```bash
   python app.py
   ```
7. Run the React frontend: 
   ```bash
   npm start
   ```
   For building the Frontend
   ```bash
   npm run build
   ```
8. Run the backend: 
   ```bash
   node app.js
   ```
   If you have nodemon installed
   ```bash
   nodemon app.js
   ```
## Running using Docker and Docker Compose

1. Clone the repository: 
   ```bash
   git clone https://github.com/cfgmumbai24/Team-59.git
   ```
   ```bash
   cd TEAM 59
   ```
## Prerequisites

- Docker daemon installed on your local machine or Docker Desktop
- Docker
- Docker Compose


## Running the Application

Follow the steps below to pull the Docker images and run the application on your local machine.

### Step 1: Set Environment Variables

Create a `.env` file in the root directory of your project and add the following environment variables:

```env
MONGO_URL=mongodb://mongo:27017/your_database_name
JWT_SECRET=your_jwt_secret
```

### Step 2: Run Docker Compose

Navigate to the root directory where your `docker-compose.yml` file is present and run the following command:

```sh
docker-compose up
```

This command will pull the necessary Docker images and start the frontend, backend, and MongoDB services.

### Access the Application

- Frontend: Open your browser and go to `http://localhost:3000`
- Backend: The backend API will be accessible at `http://localhost:4000`

### Stopping the Application

To stop the running containers, press `Ctrl+C` in the terminal where `docker-compose up` is running. To remove the containers, networks, and volumes created by Docker Compose, run:

```sh
docker-compose down
```

![image](https://github.com/harshnayangithub/Striver_Sheet_DSA/assets/126700987/4fb75d29-a01f-4315-ac31-85dbc6d0e8be)
![image](https://github.com/harshnayangithub/Striver_Sheet_DSA/assets/126700987/a7e3adb7-9ba1-4234-bdae-6f97f0ad3ad2)
![image](https://github.com/harshnayangithub/Striver_Sheet_DSA/assets/126700987/2ea1a657-49ec-4bb2-8a79-b77d6f41dc31)

---

Make sure to follow these steps in order to properly set up and run the VOPA Teaching Assessment Automation project.

## Usage
1. Sign up or log in as a teacher.
2. Automated assessments through audio.
3. View automated assessment results based on LCS, TFIDF, and Spacy.
4. Track assessment history and progress over time.

---
![image](https://github.com/harshnayangithub/Navjeevan/assets/126700987/c0995c6e-4929-4962-831a-dcb16eeb6032)
![image](https://github.com/harshnayangithub/Navjeevan/assets/126700987/4ea7753d-0ce5-4900-8b2b-9363a018568f)
![image](https://github.com/harshnayangithub/Navjeevan/assets/126700987/fdae9aaa-d174-41dc-adbd-02e6e02d7a8b)
![image](https://github.com/harshnayangithub/Navjeevan/assets/126700987/e3f39034-0978-4322-8eb7-e1698995dd09)
![image](https://github.com/harshnayangithub/Navjeevan/assets/126700987/2a2eeeb4-5592-4a8c-9e18-f2befc041b4a)
![image](https://github.com/harshnayangithub/Navjeevan/assets/126700987/641242f6-6dc8-4853-a3e9-72a04879eec0)
![image](https://github.com/harshnayangithub/Navjeevan/assets/126700987/60ad5e9a-fe03-49bf-8fbd-40408f071e82)

# Meet Our Team

![team](https://github.com/harshnayangithub/Navjeevan/assets/126700987/22ceba2a-a421-4c08-a909-e33131bc15aa)
![image](https://github.com/harshnayangithub/Striver_Sheet_DSA/assets/126700987/ce2e7d6a-1527-4ead-884d-d10da8e8eca9)
![image](https://github.com/harshnayangithub/Striver_Sheet_DSA/assets/126700987/6e3542a5-5150-48ac-bc14-eff3295eb113)

---
## License
This project is licensed under the MIT License. For more details, see the [LICENSE](LICENSE) file.

## Acknowledgements
- Special thanks to VOPA for providing us with this opportunity to contribute to this project and enhance the teaching assessment process.
---
