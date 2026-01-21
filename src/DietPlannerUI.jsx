import React, { useState } from 'react';
import './DietPlannerUI.css';

const DietPlanner = () => {
  const [step, setStep] = useState(1);
  const [uploadType, setUploadType] = useState(null);
  const [file, setFile] = useState(null);
  const [uploadedFilePath, setUploadedFilePath] = useState(null);
  const [preferences, setPreferences] = useState({
    foodType: 'veg',
    budget: 'medium',
    days: 7
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [selectedDay, setSelectedDay] = useState('day_1');

 const API_BASE_URL = import.meta.env.PROD ? '/api' : 'http://127.0.0.1:8000';

  // Parse the nested JSON structure from API response
  const parseDietPlan = (apiResponse) => {
    try {
      let dietPlanData = apiResponse.diet_plan;
      
      // If diet_plan is a string containing JSON, parse it
      if (typeof dietPlanData === 'string') {
        // Remove markdown code blocks if present
        dietPlanData = dietPlanData.replace(/```json\n?/g, '').replace(/```\n?/g, '');
        dietPlanData = JSON.parse(dietPlanData);
      }
      
      return dietPlanData;
    } catch (e) {
      console.error('Failed to parse diet plan:', e);
      return null;
    }
  };

  const handleFileUpload = async (e, type) => {
    const uploadedFile = e.target.files[0];
    if (!uploadedFile) return;

    setFile(uploadedFile);
    setUploadType(type);
    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', uploadedFile);

      const uploadResponse = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!uploadResponse.ok) {
        throw new Error('File upload failed');
      }

      const uploadData = await uploadResponse.json();
      setUploadedFilePath(uploadData.file_path);
      setStep(2);
    } catch (err) {
      setError(`Upload failed: ${err.message}`);
      setFile(null);
    } finally {
      setLoading(false);
    }
  };

  const handleGenerate = async () => {
    setLoading(true);
    setError(null);

    try {
      // Build query parameters with user preferences
      const params = new URLSearchParams({
        file_path: uploadedFilePath,
        food_type: preferences.foodType,
        budget: preferences.budget,
        days: preferences.days.toString()
      });

      const response = await fetch(`${API_BASE_URL}/ML/Predict?${params}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        }
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to generate diet plan');
      }

      const data = await response.json();
      
      // Parse the nested JSON structure
      const parsedPlan = parseDietPlan(data);
      
      setResult({
        ml_prediction: data.ml_prediction,
        parsedData: parsedPlan
      });
      setStep(3);
    } catch (err) {
      setError(`Generation failed: ${err.message}`);
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const resetFlow = () => {
    setStep(1);
    setUploadType(null);
    setFile(null);
    setUploadedFilePath(null);
    setPreferences({ foodType: 'veg', budget: 'medium', days: 7 });
    setResult(null);
    setError(null);
    setSelectedDay('day_1');
  };

  const renderMealTime = (mealData, timeLabel) => {
    if (!mealData || mealData.length === 0) return null;
    
    return (
      <div className="meal-time-section">
        <h5 className="meal-time-label">{timeLabel}</h5>
        <div className="meal-items">
          {mealData.map((item, idx) => (
            <div key={idx} className="meal-item">
              <div className="meal-item-header">
                <span className="food-name">{item.food}</span>
                <span className="calories">{item.approx_calories}</span>
              </div>
              {item.portion && (
                <p className="portion">Portion: {item.portion}</p>
              )}
              <p className="benefit">{item.nutritional_benefit}</p>
              {item.source && (
                <p className="source">Source: {item.source}</p>
              )}
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="app-container">
      <header className="header">
        <div className="container">
          <div className="header-content">
            <div className="logo-section">
              <div className="logo-icon">🥗</div>
              <div>
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
          {error && (
            <div className="error-box">
              <div className="error-icon">⚠️</div>
              <div>
                <p className="error-title">Error</p>
                <p className="error-message">{error}</p>
              </div>
            </div>
          )}

          {step === 1 && (
            <div className="step-container">
              <div className="text-center mb-large">
                <h2 className="page-title">Upload Your Medical Report</h2>
                <p className="page-subtitle">
                  Choose how you'd like to share your medical information
                </p>
              </div>

              <div className="upload-grid">
                {[
                  { icon: '📄', type: 'text', label: 'Text Document', accept: '.txt,.doc,.docx' },
                  { icon: '📋', type: 'pdf', label: 'PDF Document', accept: '.pdf' },
                  { icon: '🖼️', type: 'image', label: 'Scanned Image', accept: 'image/*' }
                ].map(({ icon, type, label, accept }) => (
                  <label key={type} className="upload-card">
                    <input
                      type="file"
                      accept={accept}
                      onChange={(e) => handleFileUpload(e, type)}
                      style={{ display: 'none' }}
                      disabled={loading}
                    />
                    <div className="upload-icon">{icon}</div>
                    <h3>{label}</h3>
                    <p>Upload {accept.split(',')[0]} files</p>
                  </label>
                ))}
              </div>

              {loading && (
                <div className="loading-box">
                  <div className="spinner"></div>
                  <p>Uploading file...</p>
                </div>
              )}
            </div>
          )}

          {step === 2 && (
            <div className="step-container">
              <div className="preferences-card">
                <div className="success-header">
                  <div className="success-icon">✓</div>
                  <div>
                    <h3>File Uploaded Successfully</h3>
                    <p>{file?.name}</p>
                  </div>
                </div>

                <div className="preferences-form">
                  <h2 className="form-title">Set Your Preferences</h2>

                  <div className="form-group">
                    <label className="form-label">
                      <span className="label-icon">🥗</span>
                      Food Preference
                    </label>
                    <div className="button-group two-col">
                      {['veg', 'nonveg'].map((type) => (
                        <button
                          key={type}
                          onClick={() => setPreferences({...preferences, foodType: type})}
                          className={`option-btn ${preferences.foodType === type ? `active ${type}` : ''}`}
                        >
                          {type === 'veg' ? 'Vegetarian' : 'Non-Vegetarian'}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="form-group">
                    <label className="form-label">
                      <span className="label-icon">💰</span>
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

                  <div className="form-group">
                    <label className="form-label">
                      <span className="label-icon">📅</span>
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

          {step === 3 && result && result.parsedData && (
            <div className="step-container">
              <div className="text-center mb-large">
                <h2 className="page-title">Your Personalized Diet Plan</h2>
                <p className="page-subtitle">Based on your medical analysis</p>
              </div>

              {/* ML Prediction */}
              <div className="result-card">
                <div className="prediction-box">
                  <p className="prediction-label">Predicted Condition</p>
                  <p className="prediction-value">{result.ml_prediction.predicted_disease}</p>
                  <div className="confidence-bar-container">
                    <span className="confidence-label">Confidence:</span>
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

                {result.parsedData.medical_assessment?.ml_prediction?.explanation && (
                  <p className="explanation-text">
                    {result.parsedData.medical_assessment.ml_prediction.explanation}
                  </p>
                )}
              </div>

              {/* Health Metrics */}
              {result.parsedData.medical_assessment?.metric_analysis && 
               result.parsedData.medical_assessment.metric_analysis.length > 0 && (
                <div className="result-card">
                  <h3 className="card-title">📊 Health Metrics Analysis</h3>
                  <div className="metrics-grid">
                    {result.parsedData.medical_assessment.metric_analysis.map((metric, idx) => (
                      <div key={idx} className="metric-card">
                        <p className="metric-label">{metric.metric}</p>
                        <p className="metric-value">{metric.value}</p>
                        <span className={`metric-status ${(metric.status || metric.interpretation || '').toLowerCase()}`}>
                          {metric.status || metric.interpretation}
                        </span>
                        {metric.interpretation && metric.interpretation !== metric.status && (
                          <p className="metric-interpretation">
                            {metric.interpretation}
                          </p>
                        )}
                        {metric.action_needed && (
                          <p className="metric-action">
                            💡 {metric.action_needed}
                          </p>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Diet Plan */}
              {result.parsedData.diet_plan && (
                <div className="result-card">
                  <h3 className="card-title">🍽️ Your Weekly Diet Plan</h3>
                  
                  <div className="diet-tabs">
                    {Object.keys(result.parsedData.diet_plan).map((day, idx) => (
                      <button
                        key={day}
                        onClick={() => setSelectedDay(day)}
                        className={`diet-tab ${selectedDay === day ? 'active' : ''}`}
                      >
                        Day {idx + 1}
                      </button>
                    ))}
                  </div>

                  <div className="day-content">
                    {result.parsedData.diet_plan[selectedDay] && (
                      <>
                        {renderMealTime(result.parsedData.diet_plan[selectedDay].morning, '🌅 Morning')}
                        {renderMealTime(result.parsedData.diet_plan[selectedDay].afternoon, '☀️ Afternoon')}
                        {renderMealTime(result.parsedData.diet_plan[selectedDay].evening, '🌆 Evening')}
                        {renderMealTime(result.parsedData.diet_plan[selectedDay].night, '🌙 Night')}
                      </>
                    )}
                  </div>

                  {result.parsedData.daily_estimated_calories && (
                    <div className="calorie-summary">
                      <p><strong>Daily Calorie Target:</strong> {result.parsedData.daily_estimated_calories[selectedDay] || result.parsedData.daily_estimated_calories.target_range}</p>
                    </div>
                  )}
                </div>
              )}

              {/* Dietary Recommendations */}
              {result.parsedData.dietary_recommendations && (
                <div className="result-card">
                  <h3 className="card-title">📋 Dietary Recommendations</h3>
                  <div className="recommendations-section">
                    {result.parsedData.dietary_recommendations.foods_to_favor && (
                      <div className="recommendation-card favor">
                        <h4>✅ Foods to Favor</h4>
                        <ul>
                          {result.parsedData.dietary_recommendations.foods_to_favor.map((food, idx) => (
                            <li key={idx}>{food}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {result.parsedData.dietary_recommendations.foods_to_limit && (
                      <div className="recommendation-card limit">
                        <h4>⚠️ Foods to Limit</h4>
                        <ul>
                          {result.parsedData.dietary_recommendations.foods_to_limit.map((food, idx) => (
                            <li key={idx}>{food}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {result.parsedData.dietary_recommendations.key_nutrients && (
                      <div className="recommendation-card nutrients">
                        <h4>🔬 Key Nutrients</h4>
                        <ul>
                          {result.parsedData.dietary_recommendations.key_nutrients.map((nutrient, idx) => (
                            <li key={idx}>{nutrient}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {result.parsedData.dietary_recommendations.lifestyle_tips && (
                      <div className="recommendation-card lifestyle">
                        <h4>💪 Lifestyle Tips</h4>
                        <ul>
                          {result.parsedData.dietary_recommendations.lifestyle_tips.map((tip, idx) => (
                            <li key={idx}>{tip}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Diet Justification */}
              {result.parsedData.diet_justification && result.parsedData.diet_justification.length > 0 && (
                <div className="result-card">
                  <h3 className="card-title">🔍 Why These Foods?</h3>
                  <div className="justification-list">
                    {result.parsedData.diet_justification.map((item, idx) => (
                      <div key={idx} className="justification-item">
                        <h4>{item.food}</h4>
                        <p className="justification-reason">
                          <strong>Benefit:</strong> {item.reason || item.mechanism}
                        </p>
                        {item.condition_addressed && (
                          <p className="justification-condition">
                            <strong>Addresses:</strong> {item.condition_addressed}
                          </p>
                        )}
                        {item.source && (
                          <p className="justification-source">
                            <em>Source: {item.source}</em>
                          </p>
                        )}
                        {item.frequency && (
                          <p className="justification-frequency">
                            <strong>Frequency:</strong> {item.frequency}
                          </p>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Data Sources */}
              {result.parsedData.data_sources && (
                <div className="result-card">
                  <h3 className="card-title">📚 Data Sources</h3>
                  <div className="data-sources-info">
                    <p><strong>From RAG Context:</strong> {result.parsedData.data_sources.from_rag_context}</p>
                    <p><strong>From Clinical Guidelines:</strong> {result.parsedData.data_sources.from_clinical_guidelines}</p>
                    {result.parsedData.data_sources.rationale && (
                      <p className="rationale"><em>{result.parsedData.data_sources.rationale}</em></p>
                    )}
                  </div>
                </div>
              )}

              {/* Disclaimer */}
              {result.parsedData.medical_note && (
                <div className="disclaimer-box">
                  <p><strong>⚠️ Medical Disclaimer:</strong> {result.parsedData.medical_note}</p>
                </div>
              )}

              <div className="action-buttons">
                <button onClick={() => window.print()} className="btn-secondary">
                  📄 Download PDF
                </button>
                <button onClick={resetFlow} className="btn-primary">
                  🔄 Create New Plan
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