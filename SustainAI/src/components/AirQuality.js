import React, { useState } from 'react';

const AirQuality = () => {
  const [cityInput, setCityInput] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [error, setError] = useState(null); // <-- NEW: State for errors

  const getAqiCategory = (pm25) => {
    // PM2.5 to AQI conversion is complex, this is a simplified mapping
    if (pm25 <= 12) return { label: 'Good', color: 'bg-green-500', text: 'text-green-500' };
    if (pm25 <= 35.4) return { label: 'Moderate', color: 'bg-yellow-500', text: 'text-yellow-500' };
    if (pm25 <= 55.4) return { label: 'Unhealthy for Sensitive Groups', color: 'bg-orange-500', text: 'text-orange-500' };
    if (pm25 <= 150.4) return { label: 'Unhealthy', color: 'bg-red-500', text: 'text-red-500' };
    if (pm25 <= 250.4) return { label: 'Very Unhealthy', color: 'bg-purple-500', text: 'text-purple-500' };
    return { label: 'Hazardous', color: 'bg-red-900', text: 'text-red-900' };
  };

  // --- THIS FUNCTION IS NOW REWRITTEN ---
  const handlePrediction = async () => {
    if (!cityInput.trim()) return;

    setIsPredicting(true);
    setPrediction(null);
    setError(null); // Clear previous errors

    try {
      const response = await fetch('http://localhost:5000/api/predict/city', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ city: cityInput.trim() }),
      });

      const result = await response.json();

      if (response.ok) {
        const pm25 = result.prediction;
        const category = getAqiCategory(pm25);
        setPrediction({
          // Note: Your model predicts PM2.5, not AQI. We'll display the PM2.5 value.
          pm25: pm25,
          category: category,
          city: result.city
        });
      } else {
        // Handle errors from the backend (e.g., city not found)
        setError(result.error || 'Prediction failed. Please try again.');
      }
    } catch (e) {
      console.error('Error making prediction:', e);
      setError('Could not connect to the prediction server.');
    } finally {
      setIsPredicting(false);
    }
  };
  // ------------------------------------------

  const citiesData = [
    {
      name: 'Delhi, India', aqi: 356, category: 'Hazardous', color: 'text-red-500',
      pm25: '250 µg/m³', ozone: '180 ppb', bars: [60, 80, 70, 90, 85, 95, 92]
    },
    {
      name: 'Los Angeles, USA', aqi: 95, category: 'Moderate', color: 'text-yellow-500',
      pm25: '35 µg/m³', ozone: '105 ppb', bars: [30, 45, 50, 40, 25, 55, 48]
    },
    {
      name: 'Beijing, China', aqi: 152, category: 'Unhealthy', color: 'text-orange-500',
      pm25: '88 µg/m³', ozone: '90 ppb', bars: [75, 80, 65, 70, 85, 60, 78]
    }
  ];

  return (
      <div className="space-y-8">
        {/* Main Air Quality Section */}
        <div className="bg-white rounded-2xl shadow-lg p-6 md:p-8">
          <h2 className="text-3xl font-bold text-gray-800 mb-6">Air Quality Pattern Analysis</h2>
          <p className="text-gray-600 mb-6">
            Uncover air quality insights with our interactive tools. Explore AQI levels and pollutant
            concentrations across global cities, and use our prediction tool to see real-time data trends.
          </p>

          {/* AQI Prediction Tool */}
          <div className="max-w-xl mx-auto p-6 bg-gray-50 rounded-2xl shadow-inner mb-8">
            <h3 className="text-xl font-semibold mb-4 text-center">Live City PM2.5 Prediction</h3>
            <p className="text-gray-600 text-center mb-4">
              Enter a city name to get a PM2.5 prediction based on its latest available weather data.
            </p>
            <div className="flex flex-col gap-4">
              <input
                  type="text"
                  value={cityInput}
                  onChange={(e) => setCityInput(e.target.value)}
                  placeholder="Enter city name (e.g., Delhi)"
                  className="p-3 rounded-lg border border-gray-300 bg-white text-gray-900 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-green-500"
              />
              <button
                  onClick={handlePrediction}
                  disabled={isPredicting}
                  className="w-full p-3 rounded-lg font-semibold transition-colors duration-200 bg-green-500 hover:bg-green-600 disabled:bg-green-300 text-white"
              >
                {isPredicting ? 'Predicting...' : 'Get Prediction'}
              </button>
            </div>

            {/* --- UPDATED DISPLAY LOGIC --- */}
            {error && (
                <div className="mt-4 p-4 rounded-lg bg-red-100 border border-red-200 text-center text-red-700">
                  {error}
                </div>
            )}
            {prediction && (
                <div className="mt-4 p-4 rounded-lg bg-gray-100 border border-gray-200 text-center">
                  <p className="text-sm font-light text-gray-500 mb-1">
                    Predicted PM2.5 for {prediction.city}
                  </p>
                  <div className="flex flex-col items-center justify-center">
                <span className={`text-5xl font-extrabold mb-1 ${prediction.category.text}`}>
                  {prediction.pm25.toFixed(1)}
                </span>
                    <span className="text-lg text-gray-600 -mt-2 mb-2">μg/m³</span>
                    <span className={`text-xl font-bold px-3 py-1 rounded-full text-white ${prediction.category.color}`}>
                  {prediction.category.label}
                </span>
                  </div>
                </div>
            )}
            {/* ----------------------------- */}
          </div>
        </div>

        {/* City Cards Grid (This section remains static) */}
        <div className="grid md:grid-cols-3 gap-6">
          {citiesData.map((city, index) => (
              <div key={index} className="bg-gray-50 rounded-xl p-4 shadow-md">
                <h3 className="text-xl font-semibold mb-2">{city.name}</h3>
                <div className="flex items-center mb-2">
                  <span className={`text-4xl font-bold mr-2 ${city.color}`}>{city.aqi}</span>
                  <span className={`text-lg font-semibold ${city.color}`}>{city.category}</span>
                </div>
                <p className="text-sm text-gray-500">PM2.5: {city.pm25}</p>
                <p className="text-sm text-gray-500">Ozone: {city.ozone}</p>
                <div className="flex items-end gap-1 mt-4 h-32">
                  {city.bars.map((height, barIndex) => (
                      <div
                          key={barIndex}
                          className="w-full rounded"
                          style={{
                            height: `${height}%`,
                            backgroundColor: city.color === 'text-red-500' ? '#ef4444' :
                                city.color === 'text-yellow-500' ? '#f59e0b' : '#f97316'
                          }}
                      />
                  ))}
                </div>
              </div>
          ))}
        </div>
      </div>
  );
};

export default AirQuality;