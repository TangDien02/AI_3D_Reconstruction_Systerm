# 🚀 AI 3D Reconstruction System (ReconApp)

This repository hosts an **end-to-end system designed for 3D model reconstruction of objects from 2D images or 360-degree video scans**. The system comprises a mobile application (Frontend) for visual data collection and an AI server (Backend) for computer vision analysis and 3D rendering.

Initially conceived as a team project to learn and enhance Python development skills through game creation, this system has evolved to tackle the complex challenge of AI-driven 3D reconstruction, providing valuable experience in cutting-edge technologies.

---

## ✨ Key Features & Benefits

*   **End-to-End 3D Reconstruction:** A complete pipeline from data capture to 3D model generation.
*   **Mobile Data Acquisition:** A dedicated mobile application built with React Native and Expo for user-friendly collection of 2D images and 360-degree videos. It includes guidance for users during the data capture process to ensure optimal results.
*   **Powerful AI Backend:** Utilizes Python with deep learning frameworks (PyTorch) for advanced computer vision tasks, processing raw visual data, and generating intricate 3D models.
*   **Scalable Architecture:** Designed with a clear separation between frontend and backend, allowing for independent development and deployment.
*   **Dockerized Deployment:** Simplified setup and consistent environment across different machines using Docker and `docker-compose`.
*   **Interactive Simulation:** Includes an auxiliary HTML-based pixel office simulation, potentially for visualization, testing, or a separate interactive component.
*   **Educational Focus:** Serves as a practical learning platform for mastering Python, AI, mobile development, and system integration.

---

## 🏗️ System Architecture

The project is structured into two primary processing flows:

### 1. 📱 Mobile App (Frontend)

Developed using **React Native** and **Expo**, this acts as the client responsible for:
*   **Data Collection:** Capturing 2D images and 360-degree video scans using the device's camera.
*   **User Guidance:** Providing instructions to the user during the scanning process for optimal data capture.
*   **Data Upload:** Sending collected visual data to the AI Backend for processing.

### 2. 🧠 AI Server (Backend)

The core processing unit, built primarily with **Python**, handling:
*   **Computer Vision Analysis:** Applying advanced algorithms to interpret and process the incoming image/video data.
*   **3D Reconstruction:** Utilizing techniques to reconstruct accurate 3D models from the processed visual input.
*   **Model Storage & Retrieval:** Managing the generated 3D models.

---

## 🛠️ Technologies Used

### Languages
*   **Python** (Backend)
*   **TypeScript** (Frontend)

### Frameworks & Libraries
*   **React** / **React Native** (Frontend for mobile application)
*   **Expo** (Frontend for simplified React Native development)
*   **PyTorch** (Backend for AI/Deep Learning)
*   **trimesh** (Backend for 3D model processing)
*   **numpy**, **pandas**, **pillow**, **matplotlib**, **scipy** (Backend for data manipulation and scientific computing)

### Tools & Platforms
*   **Docker** / **Docker Compose** (Containerization and orchestration)
*   **Node.js** / **npm** (Frontend development environment)
*   **Jest** (Frontend testing framework)

---

## ⚙️ Prerequisites & Dependencies

Before you begin, ensure you have the following installed:

*   **Git**: For cloning the repository.
*   **Node.js** (LTS version recommended) & **npm** (comes with Node.js): For the frontend development.
*   **Expo CLI**: Install globally via npm: `npm install -g expo-cli`.
*   **Python 3.8+**: For the backend development.
*   **pip**: Python package installer (comes with Python 3.4+).
*   **Docker** & **Docker Compose**: For containerized deployment.

---

## 🚀 Installation & Setup

Follow these steps to get the project up and running on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/ChienPM-27/AI_3D_Reconstruction_Systerm.git
cd AI_3D_Reconstruction_Systerm
```

### 2. Frontend Setup (Mobile App)

Navigate to the project root directory (where `package.json` is located):

```bash
npm install
```

### 3. Backend Setup (AI Server)

Navigate to the `project/` directory:

```bash
cd project/
pip install -r requirements.txt
cd .. # Go back to the project root
```

### 4. Dockerized Setup (Optional, but Recommended for Production)

Ensure Docker and Docker Compose are running on your system.
From the project root:

```bash
docker-compose build
docker-compose up
```
This will build and start both the frontend (if configured in `docker-compose.yml`) and backend services in isolated containers.

---

## 💡 Usage Examples

### Mobile App Development & Testing

1.  **Start the Expo Development Server:**
    From the project root:
    ```bash
    npm start
    # or
    expo start
    ```
    This will open the Expo Dev Tools in your browser. You can then scan the QR code with the Expo Go app on your mobile device (iOS/Android) or run on an emulator/simulator:
    *   `expo start --android`
    *   `expo start --ios`
    *   `expo start --web` (for web preview)

2.  **Using the Camera:** The app will leverage `expo-camera` for capturing images and videos, guiding the user through the 360-degree scan process.

### Running Tests

To run the frontend tests (Jest):

```bash
npm test
```

### Pixel Office Simulation

The `pixel_simulation.html` file is a standalone HTML page. You can open it directly in a web browser to view the simulation. It might serve as a visual demo or a separate component.

```bash
open pixel_simulation.html
# or simply navigate to the file in your browser
```

### Interacting with the AI Backend

The mobile app is designed to communicate with the AI backend to send captured data and receive processed 3D models. Specific API endpoints and communication protocols would be defined within the application's source code.

---

## ⚙️ Configuration Options

*   **Environment Variables (`.env`)**:
    Create a `.env` file in the project root to manage environment-specific variables, such as API endpoints for the backend server, API keys, etc.
    ```
    # Example .env content
    API_URL=http://localhost:5000/api
    ```
*   **Mobile App Configuration (`app.json`)**:
    The `app.json` file in the root directory contains configuration settings for the Expo project, including app name, icon, splash screen, and permissions.
*   **Docker Compose Configuration (`docker-compose.yml`)**:
    Modify `docker-compose.yml` to adjust service ports, volumes, environment variables for containers, and scaling options for both frontend and backend services.

---

## 🤝 Contributing Guidelines

We welcome contributions to enhance this AI 3D Reconstruction System! Please follow these guidelines:

1.  **Fork the repository.**
2.  **Create a new branch** for your feature or bug fix: `git checkout -b feature/your-feature-name`.
3.  **Make your changes**, ensuring they adhere to the project's coding style (e.g., ESLint for TypeScript/React, Black/Flake8 for Python).
4.  **Write comprehensive tests** for new features or bug fixes.
5.  **Commit your changes** with a clear and descriptive message: `git commit -m "feat: Add new feature for X"`.
6.  **Push your branch** to your forked repository: `git push origin feature/your-feature-name`.
7.  **Open a Pull Request** to the `main` branch of this repository, describing your changes and their benefits.

---

## 📄 License Information

This project is currently **unlicensed**. Please contact the repository owner, ChienPM-27, for information regarding usage and distribution.

---

## 🙏 Acknowledgments

A special thanks to:
*   The team members involved in this learning and development journey.
*   The creators and communities of Python, React Native, Expo, PyTorch, and Docker for providing robust and open-source tools that make projects like this possible.
