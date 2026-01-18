import React, { useState } from 'react';
import './DietPlannerUI.css';

const DietPlanner = () => {
  const [step, setStep] = useState(1);
  const [uploadType, setUploadType] = useState(null);
  const [file, setFile] = useState(null);
  const [preferences, setPreferences] = useState({
    foodType: 'veg',
    budget: 'medium',
    days: 7
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleFileUpload = (e, type) => {
    const uploadedFile = e.target.files[0];
    if (uploadedFile) {
      setFile(uploadedFile);
      setUploadType(type);
      setStep(2);
    }
  };

  const handleGenerate = async () => {
    setLoading(true);
    
    // Replace this with your actual API call
    const formData = new FormData();
    formData.append('file', file);
    formData.append('food_type', preferences.foodType);
    formData.append('budget', preferences.budget);
    formData.append('days', preferences.days);

    try {
      // Simulated API call - replace with actual endpoint
      setTimeout(() => {
        setResult({
          ml_prediction: {
            predicted_disease: "3",
            confidence: 0.9896
          },
          assessment: {
            conditions: ["Overweight"],
            severity: "High",
            metrics: [
              { metric: "BMI", value: 26.4, interpretation: "Overweight" },
              { metric: "Cholesterol", value: 177, interpretation: "High" },
              { metric: "PPBS", value: 70, interpretation: "Normal" },
              { metric: "Hemoglobin", value: 12.6, interpretation: "Normal" }
            ]
          },
          diet_plan: {
            day_1: ["Oatmeal", "Banana", "Carrot"],
            day_2: ["Brown Rice", "Lentils", "Spinach"],
            day_3: ["Quinoa", "Broccoli", "Apple"],
            day_4: ["Whole Wheat Bread", "Avocado", "Orange"],
            day_5: ["Sweet Potato", "Green Beans", "Berries"],
            day_6: ["Barley", "Tomato", "Grapes"],
            day_7: ["Millet", "Cucumber", "Watermelon"]
          },
          justification: [
            { food: "Oatmeal", reason: "High in fiber, helps lower cholesterol" },
            { food: "Banana", reason: "Rich in potassium, helps lower blood pressure" },
            { food: "Brown Rice", reason: "High fiber, low fat content" }
          ]
        });
        setLoading(false);
        setStep(3);
      }, 2000);

      /* 
      // Actual API call example:
      const response = await fetch('http://127.0.0.1:8000/ML/Predict', {
        method: 'POST',
        body: formData
      });
      const data = await response.json();
      setResult(data);
      setLoading(false);
      setStep(3);
      */
    } catch (error) {
      console.error('Error:', error);
      setLoading(false);
      alert('Failed to generate diet plan. Please try again.');
    }
  };

  const resetFlow = () => {
    setStep(1);
    setUploadType(null);
    setFile(null);
    setPreferences({ foodType: 'veg', budget: 'medium', days: 7 });
    setResult(null);
  };

  return (
    <div className="diet-planner">
      {/* Header */}
      <header className="header">
        <div className="container">
          <div className="header-content">
            <div className="logo-section">
              <div className="logo-icon">🥗</div>
              <div className="logo-text">
                <h1>DietPlanner AI</h1>
                <p>Personalized nutrition guidance</p>
              </div>
            </div>
            {step > 1 && (
              <button onClick={resetFlow} className="btn-secondary">
                Start Over
              </button>
            )}
          </div>
        </div>
      </header>

      <main className="main-content">
        <div className="container">
          
          {/* Step 1: Upload Options */}
          {step === 1 && (
            <div className="step-container">
              <div className="step-header">
                <h2>Upload Your Medical Report</h2>
                <p>Choose how you'd like to share your medical information with our AI system</p>
              </div>

              <div className="upload-grid">
                <label className="upload-card">
                  <input
                    type="file"
                    accept=".txt,.doc,.docx"
                    onChange={(e) => handleFileUpload(e, 'text')}
                    style={{ display: 'none' }}
                  />
                  <div className="upload-icon blue">📄</div>
                  <h3>Text Document</h3>
                  <p>Upload .txt, .doc, or .docx files</p>
                </label>

                <label className="upload-card">
                  <input
                    type="file"
                    accept=".pdf"
                    onChange={(e) => handleFileUpload(e, 'pdf')}
                    style={{ display: 'none' }}
                  />
                  <div className="upload-icon indigo">📋</div>
                  <h3>PDF Document</h3>
                  <p>Upload PDF medical reports</p>
                </label>

                <label className="upload-card">
                  <input
                    type="file"
                    accept="image/*"
                    onChange={(e) => handleFileUpload(e, 'image')}
                    style={{ display: 'none' }}
                  />
                  <div className="upload-icon violet">🖼️</div>
                  <h3>Scanned Image</h3>
                  <p>Upload scanned report images</p>
                </label>
              </div>
            </div>
          )}

          {/* Step 2: Preferences */}
          {step === 2 && (
            <div className="step-container">
              <div className="preferences-card">
                <div className="file-upload-success">
                  <div className="success-icon">✓</div>
                  <div>
                    <h3>File Uploaded Successfully</h3>
                    <p>{file?.name}</p>
                  </div>
                </div>

                <div className="preferences-form">
                  <h2>Set Your Preferences</h2>

                  {/* Food Type */}
                  <div className="form-group">
                    <label className="form-label">
                      <span className="icon">🥗</span>
                      Food Preference
                    </label>
                    <div className="button-group two-col">
                      <button
                        onClick={() => setPreferences({...preferences, foodType: 'veg'})}
                        className={`option-btn ${preferences.foodType === 'veg' ? 'active green' : ''}`}
                      >
                        Vegetarian
                      </button>
                      <button
                        onClick={() => setPreferences({...preferences, foodType: 'nonveg'})}
                        className={`option-btn ${preferences.foodType === 'nonveg' ? 'active orange' : ''}`}
                      >
                        Non-Vegetarian
                      </button>
                    </div>
                  </div>

                  {/* Budget */}
                  <div className="form-group">
                    <label className="form-label">
                      <span className="icon">💰</span>
                      Budget Range
                    </label>
                    <div className="button-group three-col">
                      {['low', 'medium', 'high'].map((budget) => (
                        <button
                          key={budget}
                          onClick={() => setPreferences({...preferences, budget})}
                          className={`option-btn ${preferences.budget === budget ? 'active blue' : ''}`}
                        >
                          {budget.charAt(0).toUpperCase() + budget.slice(1)}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Days */}
                  <div className="form-group">
                    <label className="form-label">
                      <span className="icon">📅</span>
                      Diet Plan Duration (Days)
                    </label>
                    <input
                      type="range"
                      min="1"
                      max="30"
                      value={preferences.days}
                      onChange={(e) => setPreferences({...preferences, days: parseInt(e.target.value)})}
                      className="range-slider"
                    />
                    <div className="range-labels">
                      <span>1 day</span>
                      <span className="range-value">{preferences.days} days</span>
                      <span>30 days</span>
                    </div>
                  </div>

                  {/* Generate Button */}
                  <button
                    onClick={handleGenerate}
                    disabled={loading}
                    className="btn-primary full-width"
                  >
                    {loading ? (
                      <>
                        <span className="spinner"></span>
                        Generating Your Plan...
                      </>
                    ) : (
                      'Generate Diet Plan'
                    )}
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Step 3: Results */}
          {step === 3 && result && (
            <div className="step-container">
              <div className="results-header">
                <h2>Your Personalized Diet Plan</h2>
                <p>Based on your medical analysis and preferences</p>
              </div>

              {/* ML Prediction */}
              <div className="result-card">
                <h3 className="card-title">🔬 Medical Analysis</h3>
                <div className="prediction-grid">
                  <div className="prediction-item">
                    <p className="label">Predicted Condition</p>
                    <p className="value large">Condition #{result.ml_prediction.predicted_disease}</p>
                  </div>
                  <div className="prediction-item">
                    <p className="label">Confidence Level</p>
                    <div className="confidence-bar">
                      <div className="progress-bar">
                        <div 
                          className="progress-fill"
                          style={{ width: `${result.ml_prediction.confidence * 100}%` }}
                        ></div>
                      </div>
                      <span className="confidence-value">
                        {(result.ml_prediction.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Health Metrics */}
              <div className="result-card">
                <h3 className="card-title">📊 Health Metrics</h3>
                <div className="metrics-grid">
                  {result.assessment.metrics.map((metric, idx) => (
                    <div key={idx} className="metric-item">
                      <div className="metric-header">
                        <p className="metric-name">{metric.metric}</p>
                        <span className={`badge ${metric.interpretation.toLowerCase()}`}>
                          {metric.interpretation}
                        </span>
                      </div>
                      <p className="metric-value">{metric.value}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Diet Plan */}
              <div className="result-card">
                <h3 className="card-title">🍽️ Your {preferences.days}-Day Diet Plan</h3>
                <div className="diet-grid">
                  {Object.entries(result.diet_plan).slice(0, preferences.days).map(([day, foods], idx) => (
                    <div key={day} className="diet-day">
                      <h4>Day {idx + 1}</h4>
                      <ul>
                        {foods.map((food, i) => (
                          <li key={i}>{food}</li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>
              </div>

              {/* Justification */}
              <div className="result-card">
                <h3 className="card-title">📋 Why These Foods?</h3>
                <div className="justification-list">
                  {result.justification.map((item, idx) => (
                    <div key={idx} className="justification-item">
                      <h4>{item.food}</h4>
                      <p>{item.reason}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Action Buttons */}
              <div className="action-buttons">
                <button onClick={() => window.print()} className="btn-secondary">
                  Download PDF
                </button>
                <button onClick={resetFlow} className="btn-primary">
                  Create New Plan
                </button>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
};

export default DietPlanner;
