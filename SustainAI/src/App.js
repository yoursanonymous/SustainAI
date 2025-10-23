import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import Dashboard from './components/Dashboard';
import AirQuality from './components/AirQuality';
import Explainability from './components/Explainability';
import CarbonFootprint from './components/CarbonFootprint';
import MLPrediction from './components/MLPrediction';

function App() {
  const [activeSection, setActiveSection] = useState('dashboard');
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const renderActiveSection = () => {
    switch (activeSection) {
      case 'dashboard':
        return <Dashboard />;
      case 'air-quality':
        return <AirQuality />;
      case 'explainability':
        return <Explainability />;
      case 'carbon-footprint':
        return <CarbonFootprint />;
      case 'ml-prediction':
        return <MLPrediction />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <div className="flex flex-col md:flex-row min-h-screen">
      <button 
        onClick={() => setSidebarOpen(!sidebarOpen)}
        className="md:hidden fixed top-4 left-4 z-50 p-2 bg-white rounded-lg shadow-md focus:outline-none"
      >
      </button>
      <Sidebar 
        activeSection={activeSection}
        onSectionChange={setActiveSection}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      {/* Main Content Area */}
      <main className="main-content flex-grow p-4 md:p-8 flex items-start justify-center">
        <div className="max-w-4xl w-full">
          {renderActiveSection()}
        </div>
      </main>
    </div>
  );
}

export default App;
