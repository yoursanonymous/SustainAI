import React, { useState, useEffect } from 'react';

// Helper component for the SHAP bar chart
const ShapChart = ({ shapValues }) => {
    if (!shapValues || Object.keys(shapValues).length === 0) {
        return <p className="text-gray-500 text-center">No SHAP data available.</p>;
    }

    // Find max absolute SHAP value for scaling
    const maxAbsShap = Math.max(1, ...Object.values(shapValues).map(v => Math.abs(v)));

    // Sort features by absolute importance and take top 8
    const sortedFeatures = Object.entries(shapValues)
        .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
        .slice(0, 8);

    return (
        <div className="space-y-3">
            {sortedFeatures.map(([feature, importance]) => (
                <div key={feature} className="flex items-center">
                    {/* Feature Name */}
                    <span
                        className="w-48 truncate text-sm font-medium text-gray-600 pr-2"
                        title={feature}
                    >
            {feature.replace('air_quality_', '').replace('_', ' ').replace('dioxide', 'Dioxide').replace('monoxide', 'Monoxide').replace('celsius', '°C')}
          </span>

                    {/* Bar Section */}
                    <div className="flex-1 mx-2 bg-gray-200 rounded-full h-4 relative overflow-hidden">
                        <div
                            className={`absolute top-0 h-4 rounded-full ${
                                importance >= 0 ? 'bg-red-500' : 'bg-blue-500'
                            }`}
                            style={{
                                width: `${Math.min((Math.abs(importance) / maxAbsShap) * 100, 100)}%`,
                                left: importance >= 0 ? '0' : 'auto',
                                right: importance < 0 ? '0' : 'auto',
                            }}
                        />
                    </div>

                    {/* Importance Value */}
                    <span className="w-16 text-sm font-semibold text-gray-800 text-right">
            {importance.toFixed(3)}
          </span>
                </div>
            ))}
        </div>
    );
};


const Explainability = () => {
    // Use city names that likely match the CSV (lowercase will be handled by backend)
    const [selectedCity, setSelectedCity] = useState('New Delhi'); // Default city
    const [shapValues, setShapValues] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);

    // Static text explanations (kept for simplicity)
    const cityExplanations = {
        'New Delhi': "High PM2.5 and NO₂ levels in Delhi are often linked to traffic congestion, industrial activities, and seasonal crop residue burning in nearby regions.",
        'Los Angeles': "High O₃ (Ozone) and NO₂ levels in Los Angeles are primarily due to vehicle emissions reacting with sunlight (photochemical smog), common in sunny, high-traffic urban areas.",
        'Beijing': "Significant PM2.5 and SO₂ levels in Beijing are frequently associated with industrial emissions from surrounding regions and coal combustion, especially during colder months.",
    };

    // Fetch explanation when selectedCity changes
    useEffect(() => {
        const fetchCityExplanation = async () => {
            if (!selectedCity) return;

            setIsLoading(true);
            setError(null);
            setShapValues(null); // Clear previous graph

            try {
                const response = await fetch('http://localhost:5000/api/explain/city', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ city: selectedCity }),
                });

                const result = await response.json();

                if (response.ok) {
                    setShapValues(result.shap_values);
                } else {
                    setError(result.error || 'Failed to fetch explanation for this city.');
                }
            } catch (e) {
                console.error("Error fetching city explanation:", e);
                setError("Could not connect to the explanation server.");
            } finally {
                setIsLoading(false);
            }
        };

        fetchCityExplanation();
    }, [selectedCity]); // Re-run effect when selectedCity changes

    return (
        <div className="bg-white rounded-2xl shadow-lg p-6 md:p-8">
            <h2 className="text-3xl font-bold text-gray-800 mb-4">Explainability: SHAP Analysis</h2>
            <p className="text-gray-600 mb-6">
                Select a city to view the SHAP feature importance for its latest PM2.5 prediction,
                along with common reasons for pollution in that location.
            </p>

            <div className="mb-6">
                <select
                    value={selectedCity}
                    onChange={(e) => setSelectedCity(e.target.value)}
                    className="p-2 rounded-lg border border-gray-300 bg-white text-gray-900 focus:outline-none focus:ring-2 focus:ring-green-500"
                >
                    {/* Make sure these values match city names in your CSV or backend logic */}
                    <option value="New Delhi">New Delhi</option>
                    <option value="Beijing">Beijing</option>
                    {/* Add more cities here if needed */}
                </select>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-start">
                {/* SHAP Chart Area */}
                <div>
                    {isLoading ? (
                        <div className="text-center py-8">
                            <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-green-500"></div>
                            <p className="mt-2 text-gray-600">Loading explanation...</p>
                        </div>
                    ) : error ? (
                        <div className="p-4 rounded-lg bg-red-100 border border-red-200 text-red-700">
                            {error}
                        </div>
                    ) : (
                        <>
                            <ShapChart shapValues={shapValues} />
                            <p className="text-sm text-gray-500 mt-4 text-center">
                                Feature impact on the latest PM2.5 prediction for {selectedCity}. Red increases prediction, Blue decreases it.
                            </p>
                        </>
                    )}
                </div>

                {/* Static Explanation Text Area */}
                <div className="bg-gray-50 rounded-xl p-6 shadow-inner">
                    <h4 className="text-lg font-bold text-gray-800 mb-2">Why is there pollution in {selectedCity}?</h4>
                    <p className="text-gray-700">
                        {cityExplanations[selectedCity] || "General pollution factors include traffic, industry, and weather patterns."}
                    </p>
                </div>
            </div>
        </div>
    );
};

export default Explainability;