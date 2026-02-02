'use client';

import { useState } from 'react';
import { Locality } from '@/lib/queries';

interface ComparisonPanelProps {
  selectedLocations: Locality[];
  onRemoveLocation: (id: number) => void;
  onClearAll: () => void;
}

export function ComparisonPanel({ 
  selectedLocations, 
  onRemoveLocation, 
  onClearAll, 
}: ComparisonPanelProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  if (selectedLocations.length === 0) return null;

  const exportToCSV = () => {
    const headers = ['Name', 'Type', 'Population', 'Latitude', 'Longitude', 'Municipality ID'];
    const csvData = selectedLocations.map(loc => [
      loc.name,
      loc.featureType,
      loc.population || '',
      loc.latitude,
      loc.longitude,
      loc.municipalityId
    ]);
    
    const csvContent = [headers, ...csvData]
      .map(row => row.map(field => `"${field}"`).join(','))
      .join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'portugal-locations.csv';
    link.click();
    URL.revokeObjectURL(url);
  };

  const exportToJSON = () => {
    const jsonContent = JSON.stringify(selectedLocations, null, 2);
    const blob = new Blob([jsonContent], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'portugal-locations.json';
    link.click();
    URL.revokeObjectURL(url);
  };

  const shareLocations = async () => {
    const locationIds = selectedLocations.map(loc => loc.id).join(',');
    const shareUrl = `${window.location.origin}${window.location.pathname}?locations=${locationIds}`;
    
    try {
      await navigator.clipboard.writeText(shareUrl);
      // You could show a toast notification here
      alert('Share URL copied to clipboard!');
    } catch {
      // Fallback for browsers without clipboard API
      const textArea = document.createElement('textarea');
      textArea.value = shareUrl;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand('copy');
      document.body.removeChild(textArea);
      alert('Share URL copied to clipboard!');
    }
  };

  return (
    <div className="fixed bottom-4 right-4 bg-white shadow-2xl border rounded-lg max-w-sm z-40">
      {/* Header */}
      <div 
        className="p-3 bg-blue-600 text-white rounded-t-lg cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            <span className="font-medium">Comparison ({selectedLocations.length})</span>
          </div>
          <svg 
            className={`w-4 h-4 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </div>

      {/* Content */}
      {isExpanded && (
        <div className="p-3">
          {/* Location List */}
          <div className="max-h-48 overflow-y-auto mb-3 space-y-2">
            {selectedLocations.map((location) => (
              <div key={location.id} className="flex items-center justify-between bg-gray-50 p-2 rounded">
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium text-gray-900 truncate">
                    {location.name}
                  </div>
                  <div className="text-xs text-gray-500 capitalize">
                    {location.featureType.replace('_', ' ')}
                    {location.population && ` • ${location.population.toLocaleString()}`}
                  </div>
                </div>
                <button
                  onClick={() => onRemoveLocation(location.id)}
                  className="ml-2 text-red-600 hover:text-red-800 transition-colors"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            ))}
          </div>

          {/* Comparison Stats */}
          {selectedLocations.length >= 2 && (
            <div className="mb-3 p-2 bg-blue-50 rounded border-l-4 border-blue-400">
              <div className="text-xs text-blue-700 space-y-1">
                <div>Population range: {
                  (() => {
                    const populations = selectedLocations.filter(l => l.population).map(l => l.population!);
                    if (populations.length === 0) return 'N/A';
                    const min = Math.min(...populations);
                    const max = Math.max(...populations);
                    return min === max ? min.toLocaleString() : `${min.toLocaleString()} - ${max.toLocaleString()}`;
                  })()
                }</div>
                <div>Feature types: {[...new Set(selectedLocations.map(l => l.featureType))].length} different</div>
              </div>
            </div>
          )}

          {/* Actions */}
          <div className="flex space-x-2">
            <button
              onClick={exportToCSV}
              className="flex-1 bg-green-600 text-white text-xs px-2 py-1 rounded hover:bg-green-700 transition-colors"
            >
              Export CSV
            </button>
            <button
              onClick={exportToJSON}
              className="flex-1 bg-blue-600 text-white text-xs px-2 py-1 rounded hover:bg-blue-700 transition-colors"
            >
              Export JSON
            </button>
          </div>
          
          <div className="flex space-x-2 mt-2">
            <button
              onClick={shareLocations}
              className="flex-1 bg-purple-600 text-white text-xs px-2 py-1 rounded hover:bg-purple-700 transition-colors"
            >
              Share URL
            </button>
            <button
              onClick={onClearAll}
              className="flex-1 bg-red-600 text-white text-xs px-2 py-1 rounded hover:bg-red-700 transition-colors"
            >
              Clear All
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
