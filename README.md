<img src="https://github.com/user-attachments/assets/a1bd3fef-2e1e-4c76-bb70-5e3489d9e8cb" width=350/>

> ## Advanced Behavioral Authentication Engine

### SuRaksha Cyber Hackathon | Canara Bank

## Project Overview
 True You will be a cutting-edge security solution that leverages AI and behavioral biometrics to provide continuous and intrusive authentication. This system analyzes over 50 behavioral parameters. Each parameter captures a unique aspect of user behavior, creating a comprehensive 'behavioral DNA' that's nearly impossible to replicate.

<img width="874" alt="{11857CD3-3FB2-4FC5-BE5B-34E0064D3891}" src="https://github.com/user-attachments/assets/6b8e18e5-8832-4843-853c-a4cab89a80e8" />
<img width="874" alt="{F562FA08-43ED-4F30-AD6B-3BD6F2769264}" src="https://github.com/user-attachments/assets/2a9476ca-9774-4ea1-8f6a-93292cc7602e" />

## Key Features

### 1. Multi-Modal Authentication
- **Behavioral Biometrics**: Analyzes typing patterns, touch interactions, and device usage
 <img width="392" alt="{B5AA3378-0E62-4D4F-81B4-7A7DC78CC90E}" src="https://github.com/user-attachments/assets/a1d76b9d-5e94-44e6-aadb-51174ee81034" />

- **Contextual Analysis**: Considers temporal patterns, location stability, and environmental factors
- **Cognitive Metrics**: Monitors stress levels, attention focus, and emotional state patterns
- **Isolation Forest**: Detects anomalies in user behavior
- **LSTM Networks**: Analyzes sequential patterns and temporal dependencies
- **Fusion Model**: Combines multiple AI models for enhanced accuracy
- **Real-time Risk Assessment**: Provides continuous security monitoring

### 2. User-Centric Design
- **Adaptive Profiles**: Supports different user types (normal young, normal middle, elderly, disabled, fraudulent, frequent traveler, stressed user and tech expert)
  <img width="848" alt="{436CA847-A3C0-4896-BADF-C00768C136CC}" src="https://github.com/user-attachments/assets/a2574972-8057-4870-b07d-cf31cec90ea1" />

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
 True You, Advanced Behavioral Authentication Engine represents a significant step forward in security technology, combining sophisticated AI with user-centric design to create a robust, adaptable authentication system. Its ability to analyze multiple behavioral parameters while maintaining user convenience makes it a valuable solution for modern security challenges.
