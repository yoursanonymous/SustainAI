import React, { useState } from 'react';
import { Brain, TrendingUp, Activity, Zap } from 'lucide-react';

const MLPrediction = () => {
  // --- THIS STATE IS NOW UPDATED TO MATCH YOUR MODEL'S FEATURE NAMES ---
  const [inputData, setInputData] = useState({
    air_quality_Nitrogen_dioxide: 45,
    air_quality_Carbon_Monoxide: 2.5,
    air_quality_Ozone: 120,
    air_quality_Sulphur_dioxide: 15,
    temperature_celsius: 25,
    humidity: 65,
    wind_kph: 12,
    pressure_mb: 1013,
    visibility_km: 10,
    hour: 14,
    month: 6,
    latitude: 28.6139,
    longitude: 77.2090
  });

  const [nowcastResult, setNowcastResult] = useState(null);
  const [forecastResult, setForecastResult] = useState(null);
  const [shapValues, setShapValues] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleInputChange = (field, value) => {
    setInputData(prev => ({
      ...prev,
      [field]: parseFloat(value) || 0
    }));
  };

  const makePrediction = async (type) => {
    setIsLoading(true);
    setShapValues(null); // Clear SHAP on new prediction
    try {
      const response = await fetch(`http://localhost:5000/api/predict/${type}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(inputData),
      });

      if (response.ok) {
        const result = await response.json();
        if (type === 'nowcast') {
          setNowcastResult(result);
        } else {
          setForecastResult(result);
        }
      } else {
        console.error('Prediction failed');
        // Optionally show error to user
      }
    } catch (error) {
      console.error('Error making prediction:', error);
      // Optionally show error to user
    }
    setIsLoading(false);
  };

  const getExplanation = async () => {
    setIsLoading(true); // Add loading state
    setShapValues(null); // Clear old values
    try {
      const response = await fetch('http://localhost:5000/api/explain', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(inputData),
      });

      if (response.ok) {
        const result = await response.json();
        setShapValues(result.shap_values);
      } else {
        console.error('SHAP Explanation failed');
        // Optionally show error to user
      }
    } catch (error) {
      console.error('Error getting explanation:', error);
      // Optionally show error to user
    }
    setIsLoading(false); // Stop loading
  };

  const getAqiCategory = (pm25) => {
    if (!pm25 && pm25 !== 0) return { label: '-', color: 'bg-gray-400', text: 'text-gray-400' };
    if (pm25 <= 12) return { label: 'Good', color: 'bg-green-500', text: 'text-green-500' };
    if (pm25 <= 35.4) return { label: 'Moderate', color: 'bg-yellow-500', text: 'text-yellow-500' };
    if (pm25 <= 55.4) return { label: 'Unhealthy for Sensitive Groups', color: 'bg-orange-500', text: 'text-orange-500' };
    if (pm25 <= 150.4) return { label: 'Unhealthy', color: 'bg-red-500', text: 'text-red-500' };
    if (pm25 <= 250.4) return { label: 'Very Unhealthy', color: 'bg-purple-500', text: 'text-purple-500' };
    return { label: 'Hazardous', color: 'bg-red-900', text: 'text-red-900' };
  };

  // --- THIS LIST MATCHES YOUR MODEL'S FEATURE NAMES ---
  const inputFields = [
    { key: 'air_quality_Nitrogen_dioxide', label: 'Nitrogen Dioxide (NO₂)', unit: 'ppb', min: 0, max: 200, step: 1 },
    { key: 'air_quality_Carbon_Monoxide', label: 'Carbon Monoxide (CO)', unit: 'ppm', min: 0, max: 50, step: 0.1 },
    { key: 'air_quality_Ozone', label: 'Ozone (O₃)', unit: 'ppb', min: 0, max: 300, step: 1 },
    { key: 'air_quality_Sulphur_dioxide', label: 'Sulfur Dioxide (SO₂)', unit: 'ppb', min: 0, max: 100, step: 1 },
    { key: 'temperature_celsius', label: 'Temperature', unit: '°C', min: -50, max: 50, step: 0.1 },
    { key: 'humidity', label: 'Humidity', unit: '%', min: 0, max: 100, step: 1 },
    { key: 'wind_kph', label: 'Wind Speed', unit: 'km/h', min: 0, max: 100, step: 0.1 },
    { key: 'pressure_mb', label: 'Pressure', unit: 'mb', min: 900, max: 1100, step: 1 },
    { key: 'visibility_km', label: 'Visibility', unit: 'km', min: 0, max: 20, step: 0.1 },
    { key: 'hour', label: 'Hour of Day', unit: '24h', min: 0, max: 23, step: 1 },
    { key: 'month', label: 'Month', unit: '', min: 1, max: 12, step: 1 },
    { key: 'latitude', label: 'Latitude', unit: '°', min: -90, max: 90, step: 0.0001 },
    { key: 'longitude', label: 'Longitude', unit: '°', min: -180, max: 180, step: 0.0001 }
  ];

  // Find max absolute SHAP value for scaling, default to 1 if none
  const maxAbsShap = shapValues
      ? Math.max(1, ...Object.values(shapValues).map(v => Math.abs(v)))
      : 1;

  return (
      <div className="space-y-8">
        {/* Header */}
        <div className="bg-white rounded-2xl shadow-lg p-6 md:p-8">
          <h2 className="text-3xl font-bold text-gray-800 mb-4">AI-Powered Air Quality Prediction</h2>
          <p className="text-gray-600 mb-6">
            Use our advanced machine learning models to predict current and future PM2.5 levels.
            Our models use explainable AI to provide insights into the factors driving air quality predictions.
          </p>

          {/* Model Status */}
          <div className="flex items-center gap-4 mb-6">
            <div className="flex items-center gap-2">
              <Brain className="w-5 h-5 text-green-500" />
              <span className="text-sm font-medium">Nowcast Model: Active</span>
            </div>
            <div className="flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-blue-500" />
              <span className="text-sm font-medium">Forecast Model: Active</span>
            </div>
          </div>
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Input Panel */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-2xl shadow-lg p-6">
              <h3 className="text-xl font-bold text-gray-800 mb-4">Input Parameters</h3>

              <div className="space-y-4">
                {inputFields.map((field) => (
                    <div key={field.key}>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        {field.label} {field.unit && `(${field.unit})`}
                      </label>
                      <input
                          type="number"
                          min={field.min}
                          max={field.max}
                          step={field.step || "0.1"} // Use specified step
                          value={inputData[field.key]}
                          onChange={(e) => handleInputChange(field.key, e.target.value)}
                          className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-green-500"
                      />
                    </div>
                ))}
              </div>

              {/* Prediction Buttons */}
              <div className="mt-6 space-y-3">
                <button
                    onClick={() => makePrediction('nowcast')}
                    disabled={isLoading}
                    className="w-full p-3 bg-green-500 hover:bg-green-600 disabled:bg-green-300 text-white rounded-lg font-semibold transition-colors"
                >
                  {isLoading ? 'Predicting...' : 'Predict Current PM2.5'}
                </button>

                <button
                    onClick={() => makePrediction('forecast')}
                    disabled={isLoading}
                    className="w-full p-3 bg-blue-500 hover:bg-blue-600 disabled:bg-blue-300 text-white rounded-lg font-semibold transition-colors"
                >
                  {isLoading ? 'Predicting...' : 'Predict Future PM2.5'}
                </button>

                <button
                    onClick={getExplanation}
                    disabled={isLoading} // Disable while loading
                    className="w-full p-3 bg-purple-500 hover:bg-purple-600 disabled:bg-purple-300 text-white rounded-lg font-semibold transition-colors"
                >
                  {isLoading ? 'Working...' : 'Get SHAP Explanation'}
                </button>
              </div>
            </div>
          </div>

          {/* Results Panel */}
          <div className="lg:col-span-2 space-y-6">
            {/* Nowcast Results */}
            {nowcastResult && (
                <div className="bg-white rounded-2xl shadow-lg p-6">
                  <div className="flex items-center gap-2 mb-4">
                    <Activity className="w-5 h-5 text-green-500" />
                    <h3 className="text-xl font-bold text-gray-800">Current PM2.5 Prediction (Nowcast)</h3>
                  </div>

                  <div className="grid md:grid-cols-2 gap-6">
                    <div className="text-center">
                      <div className="text-6xl font-bold text-gray-800 mb-2">
                        {nowcastResult.prediction.toFixed(1)}
                      </div>
                      <div className="text-lg text-gray-600">μg/m³</div>
                      <div className={`inline-block px-4 py-2 rounded-full text-white font-semibold mt-2 ${
                          getAqiCategory(nowcastResult.prediction).color
                      }`}>
                        {getAqiCategory(nowcastResult.prediction).label}
                      </div>
                    </div>

                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <span className="text-gray-600">Confidence:</span>
                        <span className="font-semibold">{(nowcastResult.confidence * 100).toFixed(1)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Model Type:</span>
                        <span className="font-semibold">{nowcastResult.model_type}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Prediction Time:</span>
                        <span className="font-semibold">
                      {new Date(nowcastResult.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit'})}
                    </span>
                      </div>
                    </div>
                  </div>
                </div>
            )}

            {/* Forecast Results */}
            {forecastResult && (
                <div className="bg-white rounded-2xl shadow-lg p-6">
                  <div className="flex items-center gap-2 mb-4">
                    <TrendingUp className="w-5 h-5 text-blue-500" />
                    <h3 className="text-xl font-bold text-gray-800">Future PM2.5 Prediction (Forecast)</h3>
                  </div>

                  <div className="grid md:grid-cols-2 gap-6">
                    <div className="text-center">
                      <div className="text-6xl font-bold text-gray-800 mb-2">
                        {forecastResult.prediction.toFixed(1)}
                      </div>
                      <div className="text-lg text-gray-600">μg/m³</div>
                      <div className={`inline-block px-4 py-2 rounded-full text-white font-semibold mt-2 ${
                          getAqiCategory(forecastResult.prediction).color
                      }`}>
                        {getAqiCategory(forecastResult.prediction).label}
                      </div>
                    </div>

                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <span className="text-gray-600">Confidence:</span>
                        <span className="font-semibold">{(forecastResult.confidence * 100).toFixed(1)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Model Type:</span>
                        <span className="font-semibold">{forecastResult.model_type}</span>
                      </div>
                      {/* --- START of Time Adjustment --- */}
                      <div className="flex justify-between">
                        <span className="text-gray-600">Prediction Time:</span>
                        <span className="font-semibold">
                          {(() => {
                            // Create a date object from the backend timestamp
                            const forecastDate = new Date(forecastResult.timestamp);
                            // Add 1 hour (in milliseconds)
                            forecastDate.setTime(forecastDate.getTime() + (60 * 60 * 1000));
                            // Format the NEW time for display
                            return forecastDate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit'});
                          })()}
                        </span>
                      </div>
                      {/* --- END of Time Adjustment --- */}
                    </div>
                  </div>
                </div>
            )}

            {/* SHAP Explanation */}
            {shapValues && (
                <div className="bg-white rounded-2xl shadow-lg p-6">
                  <div className="flex items-center gap-2 mb-4">
                    <Zap className="w-5 h-5 text-purple-500" />
                    <h3 className="text-xl font-bold text-gray-800">SHAP Feature Importance</h3>
                  </div>

                  <p className="text-gray-600 mb-4">
                    The following features contribute most to the PM2.5 prediction:
                  </p>

                  <div className="space-y-3">
                    {Object.entries(shapValues)
                        .sort(([,a], [,b]) => Math.abs(b) - Math.abs(a)) // Sort by absolute value
                        .slice(0, 8) // Take top 8
                        .map(([feature, importance]) => (
                            <div key={feature} className="flex items-center">
                              {/* Feature Name (Truncated) */}
                              <span
                                  className="w-48 truncate text-sm font-medium text-gray-600 pr-2"
                                  title={feature} // Show full name on hover
                              >
                                {/* Clean up name for display */}
                                {feature.replace('air_quality_', '').replace('_', ' ').replace('dioxide', 'Dioxide').replace('monoxide', 'Monoxide').replace('celsius', '°C')}
                              </span>

                              {/* --- CORRECTED BAR RENDERING --- */}
                              <div className="flex-1 mx-2 bg-gray-200 rounded-full h-4 relative overflow-hidden">
                                <div
                                    className={`absolute top-0 h-4 rounded-full ${
                                        importance >= 0 ? 'bg-red-500' : 'bg-blue-500' // Red for positive, Blue for negative
                                    }`}
                                    style={{
                                      // Width based on importance relative to max absolute importance
                                      width: `${Math.min((Math.abs(importance) / maxAbsShap) * 100, 100)}%`,
                                      // Position blue bars starting from the right, red from the left
                                      left: importance >= 0 ? '0' : 'auto',
                                      right: importance < 0 ? '0' : 'auto',
                                    }}
                                />
                              </div>
                              {/* --- END OF CORRECTION --- */}

                              {/* Importance Value */}
                              <span className="w-16 text-sm font-semibold text-gray-800 text-right">
                                {importance.toFixed(3)}
                              </span>
                            </div>
                        ))}
                  </div>
                </div>
            )}
          </div>
        </div>
      </div>
  );
};

export default MLPrediction;