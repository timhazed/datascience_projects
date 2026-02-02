'use client';

import { useState, useEffect } from 'react';

interface SearchBarProps {
  onSearch: (query: string, searchMode: 'city' | 'general') => void;
  placeholder?: string;
  isLoading?: boolean;
}

// Custom hook for debouncing
function useDebounce(value: string, delay: number) {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);

  return debouncedValue;
}

export function SearchBar({ 
  onSearch, 
  placeholder = "Search Portuguese locations...",
  isLoading = false 
}: SearchBarProps) {
  const [query, setQuery] = useState('');
  const [searchMode, setSearchMode] = useState<'city' | 'general'>('general');
  const debouncedQuery = useDebounce(query, 300);

  // Effect to trigger search when debounced query or search mode changes
  useEffect(() => {
    if (debouncedQuery.trim().length >= 2) {
      onSearch(debouncedQuery.trim(), searchMode);
    }
  }, [debouncedQuery, searchMode, onSearch]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim().length >= 2) {
      onSearch(query.trim(), searchMode);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setQuery(value);
  };

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-2xl">
      {/* Search Mode Toggle */}
      <div className="flex mb-3 bg-gray-100 rounded-lg p-1">
        <button
          type="button"
          onClick={() => setSearchMode('general')}
          className={`flex-1 px-4 py-2 text-sm font-medium rounded-md transition-all ${
            searchMode === 'general'
              ? 'bg-white text-blue-600 shadow-sm'
              : 'text-gray-600 hover:text-gray-900'
          }`}
        >
          🔍 General Search
        </button>
        <button
          type="button"
          onClick={() => setSearchMode('city')}
          className={`flex-1 px-4 py-2 text-sm font-medium rounded-md transition-all ${
            searchMode === 'city'
              ? 'bg-white text-blue-600 shadow-sm'
              : 'text-gray-600 hover:text-gray-900'
          }`}
        >
          🏙️ Cities Only
        </button>
      </div>

      {/* Search Input */}
      <div className="relative">
        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
          <svg 
            className="h-5 w-5 text-gray-400" 
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth={2} 
              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" 
            />
          </svg>
        </div>
        <input
          type="text"
          value={query}
          onChange={handleInputChange}
          className="block w-full pl-10 pr-3 py-3 border border-gray-300 rounded-lg leading-5 bg-white placeholder-gray-500 focus:outline-none focus:placeholder-gray-400 focus:ring-1 focus:ring-blue-500 focus:border-blue-500 text-sm text-gray-900"
          placeholder={placeholder}
          disabled={isLoading}
        />
        {isLoading && (
          <div className="absolute inset-y-0 right-0 pr-3 flex items-center">
            <div className="animate-spin h-4 w-4 border-2 border-blue-600 border-t-transparent rounded-full"></div>
          </div>
        )}
      </div>

      {/* Help Text */}
      <div className="mt-2 text-sm text-gray-600">
        {searchMode === 'city' ? (
          <p>🏙️ <strong>Cities Only:</strong> Search for specific cities and towns with exact coordinates</p>
        ) : (
          <p>🔍 <strong>General Search:</strong> Search districts, municipalities, and cities. Shows all related locations</p>
        )}
      </div>

      {query.length > 0 && query.length < 2 && (
        <p className="text-sm text-gray-700 mt-1">
          Type at least 2 characters to search
        </p>
      )}
    </form>
  );
}