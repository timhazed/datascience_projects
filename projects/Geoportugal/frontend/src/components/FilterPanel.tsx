'use client';

export interface SearchFilters {
  featureTypes: string[];
  populationMin?: number;
  populationMax?: number;
}

interface FilterPanelProps {
  filters: SearchFilters;
  onFiltersChange: (filters: SearchFilters) => void;
  isOpen: boolean;
  onToggle: () => void;
}

const FEATURE_TYPES = [
  { value: 'district', label: 'Districts', color: 'bg-red-500' },
  { value: 'municipality', label: 'Municipalities', color: 'bg-green-500' },
  { value: 'locality', label: 'Localities', color: 'bg-blue-500' },
  { value: 'populated_place', label: 'Populated Places', color: 'bg-blue-500' },
  { value: 'parish', label: 'Parishes', color: 'bg-purple-500' },
];

const POPULATION_RANGES = [
  { label: 'Any', min: undefined, max: undefined },
  { label: 'Small (< 1K)', min: 0, max: 1000 },
  { label: 'Medium (1K - 10K)', min: 1000, max: 10000 },
  { label: 'Large (10K - 100K)', min: 10000, max: 100000 },
  { label: 'Very Large (> 100K)', min: 100000, max: undefined },
];

export function FilterPanel({ filters, onFiltersChange, isOpen, onToggle }: FilterPanelProps) {
  const handleFeatureTypeChange = (featureType: string, checked: boolean) => {
    const newFeatureTypes = checked 
      ? [...filters.featureTypes, featureType]
      : filters.featureTypes.filter(t => t !== featureType);
    
    onFiltersChange({
      ...filters,
      featureTypes: newFeatureTypes
    });
  };

  const handlePopulationRangeChange = (range: typeof POPULATION_RANGES[0]) => {
    onFiltersChange({
      ...filters,
      populationMin: range.min,
      populationMax: range.max
    });
  };

  const clearAllFilters = () => {
    onFiltersChange({
      featureTypes: [],
      populationMin: undefined,
      populationMax: undefined
    });
  };

  const hasActiveFilters = filters.featureTypes.length > 0 || 
    filters.populationMin !== undefined || 
    filters.populationMax !== undefined;

  return (
    <div className="bg-white rounded-lg shadow-md border">
      {/* Filter Toggle Button */}
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between p-4 hover:bg-gray-50 transition-colors"
      >
        <div className="flex items-center space-x-2">
          <svg className="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.707A1 1 0 013 7V4z" />
          </svg>
          <span className="font-medium text-gray-700">Filters</span>
          {hasActiveFilters && (
            <span className="bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded-full">
              Active
            </span>
          )}
        </div>
        <svg 
          className={`w-5 h-5 text-gray-400 transition-transform ${isOpen ? 'rotate-180' : ''}`}
          fill="none" 
          stroke="currentColor" 
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Filter Content */}
      {isOpen && (
        <div className="px-4 pb-4 border-t border-gray-100">
          {/* Feature Types */}
          <div className="mb-6">
            <h4 className="font-medium text-gray-700 mb-3">Location Types</h4>
            <div className="space-y-2">
              {FEATURE_TYPES.map((type) => (
                <label key={type.value} className="flex items-center space-x-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={filters.featureTypes.includes(type.value)}
                    onChange={(e) => handleFeatureTypeChange(type.value, e.target.checked)}
                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                  <div className="flex items-center space-x-2">
                    <div className={`w-3 h-3 rounded-full ${type.color}`}></div>
                    <span className="text-sm text-gray-700">{type.label}</span>
                  </div>
                </label>
              ))}
            </div>
          </div>

          {/* Population Range */}
          <div className="mb-6">
            <h4 className="font-medium text-gray-700 mb-3">Population Size</h4>
            <div className="space-y-2">
              {POPULATION_RANGES.map((range, index) => (
                <label key={index} className="flex items-center space-x-3 cursor-pointer">
                  <input
                    type="radio"
                    name="populationRange"
                    checked={filters.populationMin === range.min && filters.populationMax === range.max}
                    onChange={() => handlePopulationRangeChange(range)}
                    className="border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="text-sm text-gray-700">{range.label}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Clear Filters */}
          {hasActiveFilters && (
            <button
              onClick={clearAllFilters}
              className="w-full bg-gray-100 text-gray-700 px-4 py-2 rounded hover:bg-gray-200 transition-colors text-sm"
            >
              Clear All Filters
            </button>
          )}
        </div>
      )}
    </div>
  );
}
