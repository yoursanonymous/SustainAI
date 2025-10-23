import React, { useState, useEffect } from 'react'; // Import hooks
import { Leaf, Brain, CheckCircle, Users, TrendingUp } from 'lucide-react';

const Dashboard = () => {
  // --- NEW: State for accuracy data ---
  const [accuracyData, setAccuracyData] = useState({ nowcast_r2: null, forecast_r2: null });
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  // ------------------------------------

  // --- NEW: Fetch accuracy on mount ---
  useEffect(() => {
    const fetchAccuracy = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const response = await fetch('http://localhost:5000/api/model-accuracy');
        if (response.ok) {
          const data = await response.json();
          // Basic validation
          if (data && data.nowcast_r2 !== undefined) {
            setAccuracyData({
              nowcast_r2: data.nowcast_r2,
              forecast_r2: data.forecast_r2 // Handle potential null
            });
          } else {
            setError("Incomplete accuracy data received.");
          }
        } else {
          setError("Failed to fetch model accuracy.");
        }
      } catch (e) {
        console.error("Error fetching accuracy:", e);
        setError("Could not connect to accuracy server.");
      } finally {
        setIsLoading(false);
      }
    };

    fetchAccuracy();
  }, []); // Empty dependency array means run once on mount
  // ------------------------------------

  // Helper to format R2 score
  const formatR2 = (r2Value) => {
    if (isLoading) return "Loading...";
    if (error || r2Value === null || r2Value === undefined) return "N/A";
    // Convert R2 score (0 to 1) to percentage string
    return `${(r2Value * 100).toFixed(1)}% R² Score`;
  };


  return (
      <div className="space-y-12">
        {/* (Keep Header and Project Components sections as they are) */}
        <div className="bg-white rounded-2xl shadow-lg p-6 md:p-8">
          <h2 className="text-3xl font-bold text-gray-800 mb-4">
            SustainAI: Explainable Intelligence for a Cleaner Tomorrow
          </h2>
          <p className="text-lg text-gray-600 mb-6">
            Harnessing explainable AI to empower environmental action. Our platform provides clear,
            data-driven insights for air quality and promotes a more sustainable future by tracking
            the carbon footprint of our models.
          </p>
        </div>

        <div className="text-center mb-12">
        <span className="inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold bg-gray-200 text-green-600 border-green-600/20">
          Project Components
        </span>
          <h3 className="text-3xl font-bold mt-2 mb-8 text-gray-900">Core Project Features</h3> {/* Added mt-2 */}
        </div>


        {/* --- Core Features Grid --- */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {/* Air Quality Analysis Card (Keep as is) */}
          <div className="group hover:shadow-xl transition-all duration-300 bg-white border border-gray-200 hover:border-green-400 relative overflow-hidden rounded-xl">
            {/* (Keep inner content) */}
            <div className="absolute top-0 right-0 w-32 h-32 opacity-10 bg-gradient-to-br from-green-400 to-teal-500 rounded-full -translate-y-16 translate-x-16 transition-transform group-hover:scale-110"></div>
            <div className="p-6 relative">
              <div className="flex items-center gap-4 mb-4">
                <div className="relative p-2 bg-green-100 rounded-lg"> {/* Placeholder icon background */}
                  <Leaf className="w-8 h-8 text-green-600" />
                </div>
                <div>
                  <h3 className="text-xl font-bold">Air Quality Analysis</h3>
                </div>
              </div>
              <div className="flex flex-wrap gap-2 mb-4">
              <span className="inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold border-green-400 text-green-500">
                Weather Data
              </span>
                <span className="inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold border-green-400 text-green-500">
                PM2.5 Prediction
              </span>
              </div>
              <p className="text-base mb-4 leading-relaxed text-gray-700">
                Predict air quality using real-world weather data. Our models provide insights into current (nowcast) and future (forecast) PM2.5 levels.
              </p>
            </div>
          </div>

          {/* Explainable AI Card (Keep as is) */}
          <div className="group hover:shadow-xl transition-all duration-300 bg-white border border-gray-200 hover:border-green-400 relative overflow-hidden rounded-xl">
            {/* (Keep inner content) */}
            <div className="absolute top-0 right-0 w-32 h-32 opacity-10 bg-gradient-to-br from-purple-400 to-indigo-500 rounded-full -translate-y-16 translate-x-16 transition-transform group-hover:scale-110"></div>
            <div className="p-6 relative">
              <div className="flex items-center gap-4 mb-4">
                <div className="relative p-2 bg-purple-100 rounded-lg">
                  <Brain className="w-8 h-8 text-purple-600" />
                </div>
                <div>
                  <h3 className="text-xl font-bold">Explainable AI</h3>
                </div>
              </div>
              <div className="flex flex-wrap gap-2 mb-4">
               <span className="inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold border-purple-400 text-purple-500">
                SHAP Integration
              </span>
                <span className="inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold border-purple-400 text-purple-500">
                Model Transparency
              </span>
              </div>
              <p className="text-base mb-4 leading-relaxed text-gray-700">
                Understand *why* a prediction is made. Our SHAP-powered engine reveals the key factors influencing each PM2.5 result.
              </p>
            </div>
          </div>

          {/* --- MODIFIED Model Accuracy Card --- */}
          <div className="flex flex-col justify-between group hover:shadow-xl transition-all duration-300 bg-white border border-gray-200 hover:border-blue-400 relative overflow-hidden rounded-xl">
            <div className="absolute top-0 right-0 w-32 h-32 opacity-10 bg-gradient-to-br from-blue-400 to-cyan-500 rounded-full -translate-y-16 translate-x-16 transition-transform group-hover:scale-110"></div>
            <div className="p-6 relative">
              <div className="flex items-center gap-4 mb-4">
                <div className="relative p-2 bg-blue-100 rounded-lg">
                  <CheckCircle className="w-8 h-8 text-blue-600" />
                </div>
                <div>
                  <h3 className="text-xl font-bold">Model Performance</h3>
                </div>
              </div>
              <p className="text-base text-gray-700 mb-4">Real-world performance metrics for our trained models based on test data.</p>
            </div>
            {/* Display fetched accuracy */}
            <div className="p-6 pt-0 relative flex-grow">
              <ul className="space-y-3"> {/* Reduced spacing */}
                {/* Nowcast Accuracy */}
                <li className="flex items-center space-x-3">
                  <div className="flex-shrink-0">
                  <span className="h-5 w-5 text-green-500"> {/* Smaller icon */}
                    <Leaf className="w-5 h-5" />
                  </span>
                  </div>
                  <div className="flex-1">
                    <h4 className="text-md font-semibold">Nowcast Model</h4> {/* Slightly smaller heading */}
                    {/* Display fetched R2 or loading/error state */}
                    <p className={`text-gray-600 text-sm font-medium ${error ? 'text-red-500' : ''}`}>
                      {formatR2(accuracyData.nowcast_r2)}
                    </p>
                  </div>
                </li>
                {/* Forecast Accuracy */}
                <li className="flex items-center space-x-3">
                  <div className="flex-shrink-0">
                  <span className="h-5 w-5 text-blue-500">
                    <TrendingUp className="w-5 h-5" /> {/* Use TrendingUp icon */}
                  </span>
                  </div>
                  <div className="flex-1">
                    <h4 className="text-md font-semibold">Forecast Model</h4>
                    <p className={`text-gray-600 text-sm font-medium ${error ? 'text-red-500' : ''}`}>
                      {formatR2(accuracyData.forecast_r2)}
                    </p>
                  </div>
                </li>
              </ul>
              {error && <p className="text-xs text-red-500 mt-2 text-center">{error}</p>}
            </div>
          </div>
          {/* ---------------------------------- */}

        </div>

        {/* Team Section (Keep as is) */}
        <div className="text-center mt-20">
        <span className="inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold bg-gray-200 text-green-600 border-green-600/20">
          Our Team
        </span>
          <h3 className="text-3xl font-bold mt-2 mb-8 text-gray-900">Meet the Creators</h3>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-6">
            {['Harthik MV', 'Tejas', 'Sridhar M', 'Vinayak Sharma', 'Bhanu Sharma'].map((name) => (
                <div key={name} className="p-4 bg-white border border-gray-200 hover:border-green-400 transition-all duration-300 hover:shadow-md rounded-xl text-center">
                  <Users className="w-12 h-12 text-green-500 mx-auto mb-2" />
                  <p className="font-semibold text-gray-800">{name}</p>
                </div>
            ))}
          </div>
        </div>
      </div>
  );
};

export default Dashboard;