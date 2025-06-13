<img src="https://github.com/user-attachments/assets/1853f0e0-9631-4b6f-8358-6c94bf166e2f" width=350/>

> ## Advanced Behavioral Authentication Engine

### SuRaksha Cyber Hackathon | Canara Bank - Team Bindas Code

## Project Overview
True You will be a cutting-edge security solution that leverages AI and behavioral biometrics to provide continuous and intrusive authentication. This system analyzes over 50 behavioral parameters. Each parameter captures a unique aspect of user behavior, creating a comprehensive 'behavioral DNA' that's nearly impossible to replicate.

## Key Features

### 1. Multi-Modal Authentication
- **Behavioral Biometrics**: Analyzes typing patterns, touch interactions, and device usage
- **Contextual Analysis**: Considers temporal patterns, location stability, and environmental factors
- **Cognitive Metrics**: Monitors stress levels, attention focus, and emotional state patterns

### 2. AI-Powered Security
- **Isolation Forest**: Detects anomalies in user behavior
- **LSTM Networks**: Analyzes sequential patterns and temporal dependencies
- **Fusion Model**: Combines multiple AI models for enhanced accuracy
- **Real-time Risk Assessment**: Provides continuous security monitoring

### 3. User-Centric Design
- **Adaptive Profiles**: Supports different user types (normal young, normal middle, elderly, disabled, fraudulent, frequent traveller, stressed user and tech expert)
- **Personalized Thresholds**: Adjusts security parameters based on user characteristics
- **Comprehensive Reporting**: Detailed session analysis and risk scoring

## Technical Innovation

### 1. Advanced Training Data Collection
- 50+ behavioral parameters
- 7-11 sessions per user for pattern establishment with 8 distinct user types supported.

### 2. Sophisticated Analysis
- Typing dynamics (speed, rhythm, error patterns)
- Touch interactions (pressure, area, duration)
- Device usage patterns (angles, stability, orientation)
- Environmental context (time, location, network)

### 3. Security Features
- Continuous authentication
- Risk score calculation
- Anomaly detection
- Pattern recognition

## Business Value

### 1. Enhanced Security
- Reduces reliance on traditional passwords
- Provides continuous authentication

### 2. User Experience
- Passive authentication (no user intervention)
- Support for diverse user types
- Minimal false positives
- Easy integration with existing systems

## Tech Stack

### Langauge & Deployment
- Python
- Streamlit 

### Frameworks & Librarires
- TensorFlow
- Scikit-learn
- Plotly for data visualization
- Multi-model fusion
- Numpy and Pandas

## Project Structure 

```
TrueYou/
├── app.py                 # Main file
├── live_auth_engine.py    # Real-time authentication 
├── models/               # Directory for trained models
├── behavioral_data.csv   # Training dataset      
└── requirements.txt              

```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Brindha-m/TrueYou-BehavioralAuth.git
cd TrueYou-BehavioralAuth
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

## Future Enhancements Planned Features in Protoype Phase

- Mobile app  API integration capabilities
- Additional behavioral parameters
- Enhanced machine learning models


## Conclusion
The True You, Advanced Behavioral Authentication Engine represents a significant step forward in security technology, combining sophisticated AI with user-centric design to create a robust, adaptable authentication system. Its ability to analyze multiple behavioral parameters while maintaining user convenience makes it a valuable solution for modern security challenges.
