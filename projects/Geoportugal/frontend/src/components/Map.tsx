'use client';

import { useEffect, useState, useRef, useCallback } from 'react';
import dynamic from 'next/dynamic';
import type L from 'leaflet';

// Dynamically import Leaflet components to avoid SSR issues
const MapContainer = dynamic(
  () => import('react-leaflet').then((mod) => mod.MapContainer),
  { ssr: false }
);

const TileLayer = dynamic(
  () => import('react-leaflet').then((mod) => mod.TileLayer),
  { ssr: false }
);

const Marker = dynamic(
  () => import('react-leaflet').then((mod) => mod.Marker),
  { ssr: false }
);

const Popup = dynamic(
  () => import('react-leaflet').then((mod) => mod.Popup),
  { ssr: false }
);



interface LocalityData {
  id: number;
  name: string;
  latitude: number;
  longitude: number;
  featureType: string;
  population?: number;
  municipalityId: number;
}

interface MapProps {
  localities?: LocalityData[];
  onLocationClick?: (locality: LocalityData) => void;
  selectedLocationId?: number;
  nearbyLocation?: LocalityData | null;
  showControls?: boolean;
}

// Create custom marker icons (client-side only)
const createCustomIcon = (color: string, size: 'small' | 'medium' | 'large') => {
  if (typeof window === 'undefined') return null;
  
  // Dynamically import Icon to avoid SSR issues
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const L = require('leaflet');
  
  const sizes = {
    small: [8, 8],
    medium: [12, 12], 
    large: [16, 16]
  };
  
  const [width, height] = sizes[size];
  
  return new L.Icon({
    iconUrl: `data:image/svg+xml;base64,${btoa(`
      <svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">
        <circle cx="${width/2}" cy="${height/2}" r="${width/2 - 1}" fill="${color}" stroke="white" stroke-width="1"/>
      </svg>
    `)}`,
    iconSize: [width, height],
    iconAnchor: [width/2, height/2],
    popupAnchor: [0, -height/2],
  });
};

// Create special nearby location indicator icon (pulsing effect)
const createNearbyLocationIcon = () => {
  if (typeof window === 'undefined') return null;
  
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const L = require('leaflet');
  
  return L.divIcon({
    html: `
      <div style="position: relative; width: 24px; height: 24px;">
        <!-- Pulsing outer ring -->
        <div style="
          position: absolute;
          top: 0; left: 0;
          width: 24px; height: 24px;
          background-color: rgba(255, 59, 48, 0.3);
          border-radius: 50%;
          animation: pulse 2s infinite;
        "></div>
        <!-- Inner marker -->
        <div style="
          position: absolute;
          top: 4px; left: 4px;
          width: 16px; height: 16px;
          background-color: #FF3B30;
          border: 3px solid white;
          border-radius: 50%;
          box-shadow: 0 2px 6px rgba(0,0,0,0.4);
        "></div>
        <!-- Center dot -->
        <div style="
          position: absolute;
          top: 9px; left: 9px;
          width: 6px; height: 6px;
          background-color: white;
          border-radius: 50%;
        "></div>
      </div>
      <style>
        @keyframes pulse {
          0% { transform: scale(1); opacity: 0.7; }
          50% { transform: scale(1.3); opacity: 0.3; }
          100% { transform: scale(1); opacity: 0.7; }
        }
      </style>
    `,
    className: 'nearby-location-icon',
    iconSize: [24, 24],
    iconAnchor: [12, 12]
  });
};

// Get marker color based on feature type
const getMarkerColor = (featureType: string): string => {
  switch (featureType.toLowerCase()) {
    case 'district': return '#dc2626'; // Red for districts
    case 'municipality': return '#059669'; // Green for municipalities  
    case 'locality':
    case 'populated_place': return '#2563eb'; // Blue for localities/populated places
    case 'parish': return '#7c3aed'; // Purple for parishes
    default: return '#6b7280'; // Gray for unknown
  }
};

// Get marker size based on population
const getMarkerSize = (population?: number): 'small' | 'medium' | 'large' => {
  if (!population) return 'small';
  if (population > 100000) return 'large';
  if (population > 10000) return 'medium';
  return 'small';
};

// Get population-based color (heat map style)
const getPopulationColor = (population?: number): string => {
  if (!population || population === 0) return '#e5e7eb'; // Gray for no population
  if (population > 500000) return '#991b1b'; // Dark red for very large cities
  if (population > 100000) return '#dc2626'; // Red for large cities
  if (population > 50000) return '#f97316'; // Orange for medium cities
  if (population > 10000) return '#eab308'; // Yellow for small cities
  if (population > 1000) return '#22c55e'; // Green for towns
  return '#3b82f6'; // Blue for small localities
};

// Get population-based size (more dramatic than regular size)
const getPopulationSize = (population?: number): 'small' | 'medium' | 'large' => {
  if (!population || population === 0) return 'small';
  if (population > 250000) return 'large'; // Lower threshold for large
  if (population > 25000) return 'medium'; // Lower threshold for medium
  return 'small';
};

export function Map({ 
  localities = [], 
  onLocationClick, 
  selectedLocationId, 
  nearbyLocation,
  showControls = true
}: MapProps) {
  const [isClient, setIsClient] = useState(false);
  const [currentTileLayer, setCurrentTileLayer] = useState('streets');
  const [showPopulationDensity, setShowPopulationDensity] = useState(false);
  const mapRef = useRef<L.Map | null>(null);

  // Callback ref to properly set the map instance
  const setMapRef = useCallback((map: L.Map | null) => {
    mapRef.current = map;
  }, []);

  useEffect(() => {
    setIsClient(true);
  }, []);

  // Force map to resize when component mounts and cleanup on unmount
  useEffect(() => {
    if (isClient && mapRef.current) {
      const timer = setTimeout(() => {
        mapRef.current?.invalidateSize();
      }, 100);
      return () => {
        clearTimeout(timer);
        mapRef.current = null;
      };
    }
    return () => {
      mapRef.current = null;
    };
  }, [isClient]);

  // Auto-fit map bounds when localities change
  useEffect(() => {
    if (isClient && mapRef.current && localities.length > 0) {
      const bounds = localities.map(loc => [loc.latitude, loc.longitude] as [number, number]);
      
      // Add a small delay to ensure map is fully loaded
      setTimeout(() => {
        if (mapRef.current) {
          try {
            if (bounds.length === 1) {
              // Single location - center and zoom
              mapRef.current.setView(bounds[0], 13);
            } else if (bounds.length > 1) {
              // Multiple locations - fit bounds
              mapRef.current.fitBounds(bounds, { padding: [30, 30] });
            }
          } catch (error) {
            console.warn('Map centering failed:', error);
          }
        }
      }, 100);
    }
  }, [localities, isClient]);

  if (!isClient) {
    return (
      <div className="w-full h-96 bg-gray-200 flex items-center justify-center rounded-lg">
        <p className="text-gray-600">Loading interactive map...</p>
      </div>
    );
  }

  // Portugal center coordinates
  const portugalCenter = [39.5, -8.0] as [number, number];

  const tileLayerConfigs = {
    streets: {
      url: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
      attribution: '&copy; OpenStreetMap contributors'
    },
    satellite: {
      url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      attribution: 'Tiles &copy; Esri'
    },
    terrain: {
      url: "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
      attribution: '&copy; OpenTopoMap contributors'
    }
  };

  const currentConfig = tileLayerConfigs[currentTileLayer as keyof typeof tileLayerConfigs];

  return (
    <div className="relative w-full h-96 rounded-lg overflow-hidden shadow-lg border" style={{ height: '400px' }}>
      {/* Map Controls */}
      {showControls && (
        <div className="absolute top-4 right-4 z-[1000] space-y-2">
          {/* Layer Selector */}
          <div className="bg-white rounded-lg shadow-md border p-2">
            <div className="text-xs font-medium text-gray-800 mb-2">Map Style</div>
            <div className="flex flex-col space-y-1">
              {Object.entries(tileLayerConfigs).map(([key]) => (
                <button
                  key={key}
                  onClick={() => setCurrentTileLayer(key)}
                  className={`text-xs px-2 py-1 rounded transition-colors ${
                    currentTileLayer === key
                      ? 'bg-blue-500 text-white'
                      : 'bg-gray-100 text-gray-800 hover:bg-gray-200'
                  }`}
                >
                  {key.charAt(0).toUpperCase() + key.slice(1)}
                </button>
              ))}
            </div>
          </div>

          {/* View Options */}
          <div className="bg-white rounded-lg shadow-md border p-2">
            <div className="text-xs font-medium text-gray-800 mb-2">Options</div>
            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="checkbox"
                checked={showPopulationDensity}
                onChange={(e) => setShowPopulationDensity(e.target.checked)}
                className="rounded text-blue-500 focus:ring-blue-500 text-xs"
              />
              <span className="text-xs text-gray-800">Population Overlay</span>
            </label>
          </div>

          {/* Stats */}
          {localities.length > 0 && (
            <div className="bg-white rounded-lg shadow-md border p-2">
              <div className="text-xs font-medium text-gray-800 mb-1">Quick Stats</div>
              <div className="text-xs text-gray-700 space-y-1">
                <div>{localities.length} locations</div>
                <div>{[...new Set(localities.map(l => l.featureType))].length} types</div>
                {localities.some(l => l.population) && (
                  <div>
                    {localities.filter(l => l.population).reduce((sum, l) => sum + (l.population || 0), 0).toLocaleString()} people
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      <MapContainer
        center={portugalCenter}
        zoom={7}
        style={{ height: '100%', width: '100%' }}
        className="z-0"
        ref={setMapRef}
        key={`map-${currentTileLayer}`}
      >
        <TileLayer
          key={currentTileLayer}
          url={currentConfig.url}
          attribution={currentConfig.attribution + ' | Data from GeoNames'}
        />
        
        {localities.map((locality) => {
          const color = showPopulationDensity 
            ? getPopulationColor(locality.population) 
            : getMarkerColor(locality.featureType);
          const size = showPopulationDensity 
            ? getPopulationSize(locality.population)
            : getMarkerSize(locality.population);
          const isSelected = locality.id === selectedLocationId;
          const icon = createCustomIcon(isSelected ? '#fbbf24' : color, size);

          // Skip rendering if icon creation failed (SSR)
          if (!icon) return null;

          return (
            <Marker
              key={locality.id}
              position={[locality.latitude, locality.longitude]}
              icon={icon}
              eventHandlers={{
                click: () => onLocationClick?.(locality),
              }}
            />
          );
        })}
        
        {/* Nearby Location Indicator */}
        {nearbyLocation && isClient && (
          <Marker
            position={[nearbyLocation.latitude, nearbyLocation.longitude]}
            icon={createNearbyLocationIcon()}
          >
            <Popup>
              <div className="p-3 min-w-48">
                <h3 className="font-semibold text-lg text-red-600 mb-2 flex items-center">
                  <svg className="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M5.05 4.05a7 7 0 119.9 9.9L10 18.9l-4.95-4.95a7 7 0 010-9.9zM10 11a2 2 0 100-4 2 2 0 000 4z" clipRule="evenodd" />
                  </svg>
                  📍 {nearbyLocation.name}
                </h3>
                
                <div className="space-y-1 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-700">Type:</span>
                    <span className="font-medium capitalize">
                      {nearbyLocation.featureType.replace('_', ' ')}
                    </span>
                  </div>
                  
                  {nearbyLocation.population && (
                    <div className="flex justify-between">
                      <span className="text-gray-700">Population:</span>
                      <span className="font-medium">
                        {nearbyLocation.population.toLocaleString()}
                      </span>
                    </div>
                  )}
                  
                  <div className="flex justify-between">
                    <span className="text-gray-700">Coordinates:</span>
                    <span className="font-medium text-xs">
                      {nearbyLocation.latitude.toFixed(4)}, {nearbyLocation.longitude.toFixed(4)}
                    </span>
                  </div>
                </div>
                
                <div className="mt-3 p-2 bg-red-50 rounded border border-red-200">
                  <p className="text-xs text-red-700 text-center">
                    🎯 Selected from nearby locations
                  </p>
                </div>
              </div>
            </Popup>
          </Marker>
        )}
        
        {localities.length === 0 && (
          <div className="absolute top-4 left-4 bg-white p-3 rounded shadow-md z-1000 border">
            <p className="text-sm text-gray-700">
              🔍 Search for Portuguese locations to see them on the map
            </p>
          </div>
        )}
      </MapContainer>
      
      {/* Map Legend */}
      <div className="absolute bottom-4 left-4 bg-white p-3 rounded shadow-md text-xs border z-[1000] text-gray-900">
        <h4 className="font-semibold mb-2 text-gray-900">Legend</h4>
        <div className="space-y-1">
          {showPopulationDensity ? (
            // Population-based legend
            <>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#991b1b' }}></div>
                <span className="text-gray-900">500K+ people</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#dc2626' }}></div>
                <span className="text-gray-900">100K+ people</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#f97316' }}></div>
                <span className="text-gray-900">50K+ people</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#eab308' }}></div>
                <span className="text-gray-900">10K+ people</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#22c55e' }}></div>
                <span className="text-gray-900">1K+ people</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#3b82f6' }}></div>
                <span className="text-gray-900">&lt;1K people</span>
              </div>
              <div className="text-gray-900 mt-1">
                Population Density View
              </div>
              {nearbyLocation && (
                <>
                  <hr className="my-2 border-gray-300" />
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-red-500 animate-pulse"></div>
                    <span className="text-gray-900 text-xs">Nearby Location</span>
                  </div>
                </>
              )}
            </>
          ) : (
            // Feature type-based legend
            <>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-red-600"></div>
                <span className="text-gray-900">Districts</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-green-600"></div>
                <span className="text-gray-900">Municipalities</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-blue-600"></div>
                <span className="text-gray-900">Localities</span>
              </div>
              <div className="text-gray-900 mt-1">
                Size = Population
              </div>
              {nearbyLocation && (
                <>
                  <hr className="my-2 border-gray-300" />
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-red-500 animate-pulse"></div>
                    <span className="text-gray-900 text-xs">Nearby Location</span>
                  </div>
                </>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
