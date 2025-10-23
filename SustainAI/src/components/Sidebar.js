import React from 'react';
import { LayoutDashboard, Leaf, Lightbulb, CloudLightning, Brain } from 'lucide-react';

const Sidebar = ({ activeSection, onSectionChange, isOpen, onClose }) => {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'air-quality', label: 'Air Quality', icon: Leaf },
    { id: 'explainability', label: 'Explainability', icon: Lightbulb },
    { id: 'ml-prediction', label: 'ML Prediction', icon: Brain },
    { id: 'carbon-footprint', label: 'Carbon Footprint', icon: CloudLightning },
  ];

  const handleSectionChange = (sectionId) => {
    onSectionChange(sectionId);
    onClose(); 
  };

  return (
    <>
      {isOpen && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 z-30 md:hidden"
          onClick={onClose}
        />
      )}
      
      <aside className={`sidebar bg-white w-64 fixed inset-y-0 left-0 z-40 p-4 shadow-lg md:relative md:shadow-none ${isOpen ? 'open' : ''}`}>
        <div className="flex items-center justify-center p-4">
          <h1 className="text-2xl font-bold text-gray-800">
            Sustain<span className="text-green-500">AI</span>
          </h1>
        </div>
        
        <nav className="mt-8">
          {navItems.map((item) => {
            const IconComponent = item.icon;
            return (
              <button
                key={item.id}
                onClick={() => handleSectionChange(item.id)}
                className={`nav-item flex items-center p-4 rounded-xl text-gray-600 hover:bg-gray-100 font-medium transition-colors mb-2 w-full text-left ${
                  activeSection === item.id ? 'bg-gray-100' : ''
                }`}
              >
                <IconComponent className="h-5 w-5 mr-3" />
                {item.label}
              </button>
            );
          })}
        </nav>
      </aside>
    </>
  );
};

export default Sidebar;

