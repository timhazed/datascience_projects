'use client';

import { useState, useMemo, useEffect, useCallback } from 'react';
import { useQuery } from '@apollo/client';
import { Layout } from '@/components/Layout';
import { Map } from '@/components/Map';
import { SearchBar } from '@/components/SearchBar';
import { LocationDetailsPanel } from '@/components/LocationDetailsPanel';
import { FilterPanel, SearchFilters } from '@/components/FilterPanel';
import { ComparisonPanel } from '@/components/ComparisonPanel';
import { StatsDashboard } from '@/components/StatsDashboard';
import { ErrorBoundary } from '@/components/ErrorBoundary';
import {
  CITY_SEARCH,
  GENERAL_SEARCH,
  Locality,
} from '@/lib/queries';

export default function Home() {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchMode, setSearchMode] = useState<'city' | 'general'>('general');
  const [selectedLocalities, setSelectedLocalities] = useState<Locality[]>([]);
  const [selectedLocation, setSelectedLocation] = useState<Locality | null>(null);
  const [nearbyLocation, setNearbyLocation] = useState<Locality | null>(null); // Track nearby location indicator
  const [comparisonLocations, setComparisonLocations] = useState<Locality[]>([]);
  const [searchResults, setSearchResults] = useState<{
    localities: Locality[];
    message?: string;
    resultType?: string;
  }>({ localities: [] });
  const [filters, setFilters] = useState<SearchFilters>({
    featureTypes: [],
    populationMin: undefined,
    populationMax: undefined
  });
  const [isFilterOpen, setIsFilterOpen] = useState(false);

  // City Search Query
  const { data: cityData, loading: cityLoading, error: cityError } = useQuery(CITY_SEARCH, {
    variables: { query: searchQuery },
    skip: !searchQuery || searchQuery.length < 2 || searchMode !== 'city',
  });

  // General Search Query  
  const { data: generalData, loading: generalLoading, error: generalError } = useQuery(GENERAL_SEARCH, {
    variables: { query: searchQuery },
    skip: !searchQuery || searchQuery.length < 2 || searchMode !== 'general',
  });

  const loading = cityLoading || generalLoading;
  // Only show errors for user-initiated searches
  const error = searchQuery.length >= 2 ? (cityError || generalError) : null;

  // Process search results based on mode
  useEffect(() => {
    if (searchMode === 'city' && cityData?.citySearch) {
      setSearchResults({
        localities: cityData.citySearch,
        message: `Found ${cityData.citySearch.length} cities matching "${searchQuery}"`,
        resultType: 'city_search'
      });
    } else if (searchMode === 'general' && generalData?.generalSearch) {
      setSearchResults({
        localities: generalData.generalSearch.localities,
        message: generalData.generalSearch.message,
        resultType: generalData.generalSearch.resultType
      });
    } else if (!searchQuery) {
      // Clear results when no search query (clean slate)
      setSearchResults({ localities: [] });
    }
  }, [cityData, generalData, searchMode, searchQuery]);

  // Filter search results based on active filters
  const filteredLocalities = useMemo(() => {
    if (!searchResults.localities) return [];
    
    let filtered = searchResults.localities;
    
    // Filter by feature types
    if (filters.featureTypes.length > 0) {
      filtered = filtered.filter((locality: Locality) => 
        filters.featureTypes.includes(locality.featureType)
      );
    }
    
    // Filter by population range
    if (filters.populationMin !== undefined || filters.populationMax !== undefined) {
      filtered = filtered.filter((locality: Locality) => {
        if (!locality.population) return filters.populationMin === undefined;
        
        const meetsMin = filters.populationMin === undefined || locality.population >= filters.populationMin;
        const meetsMax = filters.populationMax === undefined || locality.population <= filters.populationMax;
        
        return meetsMin && meetsMax;
      });
    }
    
    return filtered;
  }, [searchResults.localities, filters]);

  const handleCloseDetailsPanel = () => {
    setSelectedLocation(null);
    setNearbyLocation(null);
  };

  // Handle new search - clear previous selection state
  // Memoized to prevent SearchBar's useEffect from firing on unrelated re-renders
  const handleSearch = useCallback((query: string, mode: 'city' | 'general') => {
    setSearchQuery(query);
    setSearchMode(mode);
    setSelectedLocation(null);
    setNearbyLocation(null);
  }, []);

  // Update selected localities when filtered results change
  useEffect(() => {
    setSelectedLocalities(filteredLocalities);
    // Clear nearby location indicator when showing new search results
    if (filteredLocalities.length > 0) {
      setNearbyLocation(null);
    }
  }, [filteredLocalities]);

  const handleLocationClick = (locality: Locality) => {
    setSelectedLocation(locality);
    setNearbyLocation(null); // Clear nearby location indicator when selecting a regular location
  };

  const handleNearbyLocationClick = (location: Locality) => {
    setSelectedLocation(location);
    setNearbyLocation(location); // Set the nearby location indicator
    // Keep original search results visible AND add the nearby location if it's not already in the list
    if (!selectedLocalities.find(loc => loc.id === location.id)) {
      setSelectedLocalities([...selectedLocalities, location]);
    }
  };

  const handleAddToComparison = (locality: Locality) => {
    if (!comparisonLocations.find(loc => loc.id === locality.id)) {
      setComparisonLocations([...comparisonLocations, locality]);
    }
  };

  const handleRemoveFromComparison = (id: number) => {
    setComparisonLocations(comparisonLocations.filter(loc => loc.id !== id));
  };

  const handleClearComparison = () => {
    setComparisonLocations([]);
  };

  // Helper functions for styling search result messages
  const getMessageStyle = (resultType?: string) => {
    switch (resultType) {
      case 'city_found': return 'bg-green-50 border border-green-200 text-green-800';
      case 'city_search': return 'bg-blue-50 border border-blue-200 text-blue-800';
      case 'district_found': return 'bg-blue-50 border border-blue-200 text-blue-800';
      case 'municipality_found': return 'bg-purple-50 border border-purple-200 text-purple-800';
      case 'partial_matches': return 'bg-orange-50 border border-orange-200 text-orange-800';
      default: return 'bg-gray-50 border border-gray-200 text-gray-800';
    }
  };

  const getMessageIcon = (resultType?: string) => {
    switch (resultType) {
      case 'city_found': return '🎯';
      case 'city_search': return '🏙️';
      case 'district_found': return '🗺️';
      case 'municipality_found': return '🏛️';
      case 'partial_matches': return '🔍';
      default: return 'ℹ️';
    }
  };

  return (
    <Layout>
      <ErrorBoundary>
        <div className="space-y-6">
          <div className="text-center animate-fade-in">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4 leading-tight">
              Explore Portugal&apos;s Geography
            </h2>
            <p className="text-lg md:text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
              Search and discover Portuguese districts, municipalities, and localities with interactive maps
            </p>
            
            <div className="flex justify-center mb-8 px-4">
              <div className="w-full max-w-2xl">
                <SearchBar 
                  onSearch={handleSearch}
                  placeholder={searchMode === 'city' 
                    ? "Search for Portuguese cities and towns..." 
                    : "Search districts, municipalities, or cities..."
                  }
                  isLoading={loading}
                />
              </div>
            </div>
          </div>

          {/* Search Results Message */}
          {searchResults.message && filteredLocalities.length > 0 && (
            <div className={`rounded-lg p-4 ${getMessageStyle(searchResults.resultType)}`}>
              <div className="flex items-center space-x-2">
                <span className="text-lg">{getMessageIcon(searchResults.resultType)}</span>
                <p className="font-medium">{searchResults.message}</p>
              </div>
            </div>
          )}

          {/* Filters Panel */}
          <FilterPanel 
            filters={filters}
            onFiltersChange={setFilters}
            isOpen={isFilterOpen}
            onToggle={() => setIsFilterOpen(!isFilterOpen)}
          />

        <div className="bg-white rounded-lg shadow-lg p-4 md:p-6 animate-scale-in border border-gray-100">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-4 space-y-2 sm:space-y-0">
            <h3 className="text-lg md:text-xl font-semibold text-gray-800 flex items-center">
              <svg className="w-5 h-5 md:w-6 md:h-6 text-blue-600 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-1.447-.894L15 4m0 13V4m-6 3l6-3" />
              </svg>
              Interactive Map
            </h3>
            {selectedLocalities.length > 0 && (
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-sm text-gray-600 font-medium">
                  Showing {selectedLocalities.length} location{selectedLocalities.length !== 1 ? 's' : ''}
                </span>
              </div>
            )}
          </div>
          
          <div className="relative overflow-hidden rounded-lg">
            <Map 
              localities={selectedLocalities} 
              onLocationClick={handleLocationClick}
              selectedLocationId={selectedLocation?.id}
              nearbyLocation={nearbyLocation}
            />
            {selectedLocalities.length === 0 && !loading && (
              <div className="absolute inset-0 bg-gray-50 bg-opacity-90 flex items-center justify-center">
                <div className="text-center p-8 animate-pulse-subtle">
                  <svg className="w-16 h-16 text-gray-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                  {searchQuery && filteredLocalities.length === 0 ? (
                    <div>
                      <p className="text-gray-600 font-medium mb-2">No locations found for &quot;{searchQuery}&quot;</p>
                      <p className="text-gray-500 text-sm">Try searching for Portuguese cities like Lisboa or Porto</p>
                    </div>
                  ) : (
                    <p className="text-gray-600 font-medium">Search for Portuguese locations to see them on the map</p>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

        {loading && searchQuery && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="flex items-center space-x-3">
              <div className="animate-spin h-5 w-5 border-2 border-blue-600 border-t-transparent rounded-full"></div>
              <p className="text-blue-800">
                Searching for &quot;{searchQuery}&quot;...
              </p>
            </div>
          </div>
        )}

        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4">
            <p className="text-red-800">
              Error loading data. Please make sure the backend server is running.
            </p>
          </div>
        )}

        {searchQuery && filteredLocalities.length === 0 && !loading && (
          <div className="bg-orange-50 border border-orange-200 rounded-lg p-6 text-center">
            <div className="flex flex-col items-center space-y-3">
              <svg className="w-12 h-12 text-orange-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v2m0 4h.01" />
              </svg>
              <div>
                <h3 className="text-lg font-semibold text-orange-800 mb-2">
                  No locations found for &quot;{searchQuery}&quot;
                </h3>
                <p className="text-orange-700 mb-4">
                  We couldn&apos;t find any Portuguese locations matching your search.
                </p>
                <div className="bg-white p-4 rounded-lg border border-orange-200">
                  <h4 className="font-medium text-orange-800 mb-2">💡 Search Tips:</h4>
                  <ul className="text-sm text-orange-700 space-y-1 text-left">
                    <li>• Try popular cities: <strong>Lisboa</strong>, <strong>Porto</strong>, <strong>Coimbra</strong></li>
                    <li>• Use Portuguese spellings: <strong>Óbidos</strong> (with accent marks)</li>
                    <li>• Try partial names: <strong>Tav</strong> instead of <strong>Tavira</strong></li>
                    <li>• Check spelling and try variations</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        )}

          {/* Statistics Dashboard */}
          <StatsDashboard 
            localities={selectedLocalities}
            isVisible={selectedLocalities.length > 0 && searchQuery.length >= 2}
          />

          {/* Location Details Panel */}
          <LocationDetailsPanel
            selectedLocation={selectedLocation}
            onClose={handleCloseDetailsPanel}
            onNearbyLocationClick={handleNearbyLocationClick}
            onAddToComparison={handleAddToComparison}
          />

          {/* Comparison Panel */}
          <ComparisonPanel
            selectedLocations={comparisonLocations}
            onRemoveLocation={handleRemoveFromComparison}
            onClearAll={handleClearComparison}
          />
        </div>
      </ErrorBoundary>
    </Layout>
  );
}
