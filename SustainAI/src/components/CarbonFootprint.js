import React, { useState, useEffect } from 'react'; // Import useEffect

const CarbonFootprint = () => {
  // Initialize carbonData to null
  const [carbonData, setCarbonData] = useState(null);
  const [isLoading, setIsLoading] = useState(true); // Start loading immediately
  const [error, setError] = useState(null); // State for errors

  // Fetch data when component mounts
  useEffect(() => {
    fetchCarbonData();
  }, []);

  const fetchCarbonData = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch('http://localhost:5000/api/carbon-footprint');
      if (response.ok) {
        const data = await response.json();
        // --- Add a check for expected data ---
        if (data && data.models_trained !== undefined && data.predictions_made !== undefined && data.carbon_per_prediction !== undefined) {
          setCarbonData(data);
        } else {
          console.error("Received incomplete carbon data:", data);
          setError("Failed to load complete carbon footprint data.");
        }
        // ------------------------------------
      } else {
        setError("Failed to fetch carbon footprint data from server.");
      }
    } catch (error) {
      console.error('Error fetching carbon data:', error);
      setError("Could not connect to the carbon footprint server.");
    } finally {
      setIsLoading(false); // Ensure loading stops even on error
    }
  };

  // Define default metrics separately
  const defaultMetrics = [
    { title: 'Total Carbon Emissions', value: '1.52 kg', description: 'equivalent to 6.3 km driven by an average car', color: 'text-green-700' },
    { title: 'Energy Consumed', value: '5.4 kWh', description: 'equivalent to charging a smartphone 440 times', color: 'text-green-700' },
    { title: 'Project Duration', value: '2 Months', description: 'for all model training and development', color: 'text-green-700' }
  ];

  // --- Calculate mlMetrics ONLY if carbonData is valid ---
  let mlMetrics = []; // Initialize as empty
  if (carbonData) {
    mlMetrics = [
      { title: 'Models Trained', value: carbonData.models_trained.toString(), description: 'machine learning models in production', color: 'text-blue-700' },
      { title: 'Predictions Made', value: carbonData.predictions_made.toLocaleString(), description: 'total predictions served', color: 'text-purple-700' },
      { title: 'Carbon per Prediction', value: `${(carbonData.carbon_per_prediction * 1000).toFixed(3)} mg`, description: 'average carbon footprint per prediction', color: 'text-orange-700' }
    ];
  }
  // ----------------------------------------------------

  return (
      <div className="space-y-8">
        <div className="bg-white rounded-2xl shadow-lg p-6 md:p-8">
          <h2 className="text-3xl font-bold text-gray-800 mb-6">Carbon-Conscious Computing</h2>
          <p className="text-gray-600 mb-6">
            In line with our mission, we track the carbon footprint of our AI models using CodeCarbon.
            This promotes sustainable machine learning practices and reduces our environmental impact.
          </p>
        </div>

        {/* Project Metrics Grid (Always shown) */}
        <div>
          <h3 className="text-2xl font-bold text-gray-800 mb-4">Project Carbon Footprint</h3>
          <div className="grid md:grid-cols-3 gap-6 mb-8">
            {defaultMetrics.map((metric, index) => (
                <div key={index} className="bg-gray-50 rounded-xl p-4 shadow-md text-center">
                  <p className="text-xl font-semibold mb-2">{metric.title}</p>
                  <p className={`text-4xl font-bold mt-2 ${metric.color}`}>{metric.value}</p>
                  <p className="text-gray-500 text-sm mt-1">{metric.description}</p>
                </div>
            ))}
          </div>
        </div>

        {/* ML Model Metrics Grid (Conditional Rendering) */}
        <div>
          <h3 className="text-2xl font-bold text-gray-800 mb-4">ML Model Carbon Tracking</h3>
          {/* --- Updated Loading/Error/Success Handling --- */}
          {isLoading ? (
              <div className="text-center py-8">
                <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-green-500"></div>
                <p className="mt-2 text-gray-600">Loading carbon data...</p>
              </div>
          ) : error ? (
              <div className="text-center py-8 p-4 rounded-lg bg-red-100 border border-red-200 text-red-700">
                {error}
              </div>
          ) : carbonData ? ( // Only render if carbonData is loaded successfully
              <div className="grid md:grid-cols-3 gap-6">
                {mlMetrics.map((metric, index) => (
                    <div key={index} className="bg-gray-50 rounded-xl p-4 shadow-md text-center">
                      <p className="text-xl font-semibold mb-2">{metric.title}</p>
                      <p className={`text-4xl font-bold mt-2 ${metric.color}`}>{metric.value}</p>
                      <p className="text-gray-500 text-sm mt-1">{metric.description}</p>
                    </div>
                ))}
              </div>
          ) : (
              // Fallback message if data loaded but was invalid
              <div className="text-center py-8 text-gray-500">
                Could not display ML Model Carbon Tracking data.
              </div>
          )}
          {/* ------------------------------------------- */}
        </div>
      </div>
  );
};

export default CarbonFootprint;