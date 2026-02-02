'use client';

import { useState } from 'react';
import { useQuery } from '@apollo/client';
import { GET_NEARBY_LOCALITIES, Locality } from '@/lib/queries';
import { Breadcrumb } from './Breadcrumb';

interface LocationDetailsPanelProps {
  selectedLocation: Locality | null;
  onClose: () => void;
  onNearbyLocationClick: (location: Locality) => void;
  onAddToComparison?: (location: Locality) => void;
}

export function LocationDetailsPanel({ 
  selectedLocation, 
  onClose, 
  onNearbyLocationClick,
  onAddToComparison
}: LocationDetailsPanelProps) {
  const [nearbyRadius, setNearbyRadius] = useState(10);
  
  const { data: nearbyData, loading: nearbyLoading } = useQuery(GET_NEARBY_LOCALITIES, {
    variables: {
      lat: selectedLocation?.latitude,
      lng: selectedLocation?.longitude,
      radiusKm: nearbyRadius,
    },
    skip: !selectedLocation,
  });

  if (!selectedLocation) return null;

  const nearbyLocalities = nearbyData?.localitiesNear || [];
  // Filter out the current location from nearby results
  const filteredNearbyLocalities = nearbyLocalities.filter(
    (loc: Locality) => loc.id !== selectedLocation.id
  );

  return (
    <div className="fixed inset-y-0 right-0 w-96 bg-white shadow-2xl border-l z-50 overflow-y-auto">
      {/* Header */}
      <div className="bg-blue-600 text-white p-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold">Location Details</h2>
          <button
            onClick={onClose}
            className="text-white hover:text-gray-200 transition-colors"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      </div>

      {/* Location Information */}
      <div className="p-4 border-b">
        <h3 className="text-xl font-bold text-gray-900 mb-3">
          {selectedLocation.name}
        </h3>
        
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-black">Type:</span>
            <span className="font-medium text-black capitalize">
              {selectedLocation.featureType.replace('_', ' ')}
            </span>
          </div>
          
          <div className="flex justify-between">
            <span className="text-black">Population:</span>
            <span className="font-medium text-black">
              {selectedLocation.population
                ? selectedLocation.population.toLocaleString()
                : 'Not available'}
            </span>
          </div>
          
          <div className="flex justify-between">
            <span className="text-black">Coordinates:</span>
            <span className="font-medium text-black font-mono text-xs">
              {selectedLocation.latitude.toFixed(4)}, {selectedLocation.longitude.toFixed(4)}
            </span>
          </div>
          
          {selectedLocation.municipalityId ? (
            <div className="flex justify-between">
              <span className="text-black">Municipality ID:</span>
              <span className="font-medium text-black">
                {selectedLocation.municipalityId}
              </span>
            </div>
          ) : (
            <div className="flex justify-between">
              <span className="text-black">Municipality:</span>
              <span className="font-medium text-black">Not available</span>
            </div>
          )}
        </div>
      </div>

      {/* Administrative Hierarchy */}
      <div className="p-4 border-b">
        <Breadcrumb locality={selectedLocation} />
      </div>

      {/* Nearby Locations */}
      <div className="p-4">
        <div className="flex items-center justify-between mb-3">
          <h4 className="font-semibold text-gray-900">Nearby Locations</h4>
          <select
            value={nearbyRadius}
            onChange={(e) => setNearbyRadius(Number(e.target.value))}
            className="text-xs border border-gray-300 rounded px-2 py-1"
          >
            <option value={5}>5 km</option>
            <option value={10}>10 km</option>
            <option value={20}>20 km</option>
            <option value={50}>50 km</option>
          </select>
        </div>

        {nearbyLoading ? (
          <div className="flex items-center justify-center py-4">
            <div className="animate-spin h-6 w-6 border-2 border-blue-600 border-t-transparent rounded-full"></div>
          </div>
        ) : filteredNearbyLocalities.length > 0 ? (
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {filteredNearbyLocalities.slice(0, 10).map((location: Locality) => (
              <div
                key={location.id}
                className="p-2 border border-gray-200 rounded hover:bg-gray-50 cursor-pointer transition-colors"
                onClick={() => onNearbyLocationClick(location)}
              >
                <div className="font-medium text-sm text-gray-900">
                  {location.name}
                </div>
                <div className="text-xs text-gray-700">
                  {location.featureType.replace('_', ' ')} 
                  {location.population && ` • ${location.population.toLocaleString()}`}
                </div>
                <div className="text-xs text-gray-600">
                  {location.latitude.toFixed(3)}, {location.longitude.toFixed(3)}
                </div>
              </div>
            ))}
            {filteredNearbyLocalities.length > 10 && (
              <div className="text-xs text-gray-500 text-center py-2">
                Showing first 10 of {filteredNearbyLocalities.length} nearby locations
              </div>
            )}
          </div>
        ) : (
          <p className="text-sm text-gray-500 py-4">
            No nearby locations found within {nearbyRadius} km.
          </p>
        )}
      </div>

      {/* Actions */}
      <div className="p-4 border-t bg-gray-50 space-y-2">
        {onAddToComparison && (
          <button
            onClick={() => onAddToComparison(selectedLocation)}
            className="w-full bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700 transition-colors text-sm flex items-center justify-center space-x-2"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
            </svg>
            <span>Add to Comparison</span>
          </button>
        )}
        <button
          onClick={() => {
            const url = `https://www.openstreetmap.org/?mlat=${selectedLocation.latitude}&mlon=${selectedLocation.longitude}&zoom=15`;
            window.open(url, '_blank');
          }}
          className="w-full bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition-colors text-sm flex items-center justify-center space-x-2"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
          </svg>
          <span>View on OpenStreetMap</span>
        </button>
      </div>
    </div>
  );
}
