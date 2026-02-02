'use client';

import { useMemo } from 'react';
import { Locality } from '@/lib/queries';

interface StatsDashboardProps {
  localities: Locality[];
  isVisible: boolean;
}

type ByTypeStats = Record<
  string,
  {
    count: number;
    totalPopulation: number;
    avgPopulation: number;
    locations: Locality[];
  }
>;

interface PopulationStats {
  min: number;
  max: number;
  avg: number;
  median: number;
  total: number;
}

interface GeographicSpread {
  latRange: number;
  lngRange: number;
  centerLat: number;
  centerLng: number;
}

interface ChartBarProps {
  label: string;
  value: number;
  maxValue: number;
  color: string;
  count?: number;
}

function ChartBar({ label, value, maxValue, color, count }: ChartBarProps) {
  const percentage = maxValue > 0 ? (value / maxValue) * 100 : 0;
  
  return (
    <div className="mb-3">
      <div className="flex justify-between items-center mb-1">
        <span className="text-sm font-medium text-gray-800">{label}</span>
        <div className="text-xs text-gray-700">
          {count !== undefined && `${count} locations • `}
          {value > 0 ? value.toLocaleString() : 'N/A'}
        </div>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div
          className={`h-2 rounded-full transition-all duration-300 ${color}`}
          style={{ width: `${percentage}%` }}
        ></div>
      </div>
    </div>
  );
}

export function StatsDashboard({ localities, isVisible }: StatsDashboardProps) {
  const stats = useMemo(() => {
    if (localities.length === 0) {
      return {
        total: 0,
        byType: {} as ByTypeStats,
        populationStats: null as PopulationStats | null,
        geographicSpread: null as GeographicSpread | null,
      };
    }

    // Group by feature type
    const byType = localities.reduce<ByTypeStats>((acc, locality) => {
      const type = locality.featureType;
      if (!acc[type]) {
        acc[type] = {
          count: 0,
          totalPopulation: 0,
          avgPopulation: 0,
          locations: []
        };
      }
      acc[type].count++;
      acc[type].locations.push(locality);
      if (locality.population) {
        acc[type].totalPopulation += locality.population;
      }
      return acc;
    }, {} as ByTypeStats);

    // Calculate averages
    Object.keys(byType).forEach(type => {
      const locationsWithPopulation = byType[type].locations.filter((l: Locality) => l.population);
      if (locationsWithPopulation.length > 0) {
        byType[type].avgPopulation = byType[type].totalPopulation / locationsWithPopulation.length;
      }
    });

    // Population statistics
    const locationsWithPopulation = localities.filter(l => l.population);
    const populations = locationsWithPopulation.map(l => l.population!);
    
    const populationStats: PopulationStats | null = populations.length > 0 ? {
      min: Math.min(...populations),
      max: Math.max(...populations),
      avg: populations.reduce((sum, pop) => sum + pop, 0) / populations.length,
      median: populations.sort((a, b) => a - b)[Math.floor(populations.length / 2)],
      total: populations.reduce((sum, pop) => sum + pop, 0)
    } : null;

    // Geographic spread
    const latitudes = localities.map(l => l.latitude);
    const longitudes = localities.map(l => l.longitude);
    
    const geographicSpread: GeographicSpread = {
      latRange: Math.max(...latitudes) - Math.min(...latitudes),
      lngRange: Math.max(...longitudes) - Math.min(...longitudes),
      centerLat: (Math.max(...latitudes) + Math.min(...latitudes)) / 2,
      centerLng: (Math.max(...longitudes) + Math.min(...longitudes)) / 2
    };

    return {
      total: localities.length,
      byType,
      populationStats,
      geographicSpread
    };
  }, [localities]);

  if (!isVisible || localities.length === 0) return null;

  const typeColors = {
    district: 'bg-red-500',
    municipality: 'bg-green-500', 
    locality: 'bg-blue-500',
    populated_place: 'bg-blue-400',
    parish: 'bg-purple-500'
  };

  const maxTypeCount = Math.max(...Object.values(stats.byType).map((t) => t.count));
  const maxPopulation = stats.populationStats ? Math.max(
    ...Object.values(stats.byType)
      .filter((t) => t.totalPopulation > 0)
      .map((t) => t.totalPopulation)
  ) : 0;

  return (
    <div className="bg-white rounded-lg shadow-md p-6 mt-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-semibold text-gray-800">Search Results Analytics</h3>
        <div className="text-sm text-gray-800">
          {stats.total} location{stats.total !== 1 ? 's' : ''}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Location Types Distribution */}
        <div>
          <h4 className="font-medium text-gray-800 mb-4">Distribution by Type</h4>
          {Object.entries(stats.byType).map(([type, data]) => (
            <ChartBar
              key={type}
              label={type.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
              value={data.count}
              maxValue={maxTypeCount}
              color={typeColors[type as keyof typeof typeColors] || 'bg-gray-500'}
              count={data.count}
            />
          ))}
        </div>

        {/* Population Analysis */}
        {stats.populationStats && (
          <div>
            <h4 className="font-medium text-gray-800 mb-4">Population by Type</h4>
            {Object.entries(stats.byType)
              .filter(([, data]) => data.totalPopulation > 0)
              .map(([type, data]) => (
                <ChartBar
                  key={`pop-${type}`}
                  label={type.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  value={data.totalPopulation}
                  maxValue={maxPopulation}
                  color={typeColors[type as keyof typeof typeColors] || 'bg-gray-500'}
                />
              ))}
          </div>
        )}

        {/* Summary Statistics */}
        <div className="lg:col-span-2">
          <h4 className="font-medium text-gray-800 mb-4">Summary Statistics</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-blue-50 p-4 rounded-lg text-center">
              <div className="text-2xl font-bold text-blue-600">{stats.total}</div>
              <div className="text-sm text-gray-700">Total Locations</div>
            </div>
            
            {stats.populationStats && (
              <>
                <div className="bg-green-50 p-4 rounded-lg text-center">
                  <div className="text-2xl font-bold text-green-600">
                    {stats.populationStats.total.toLocaleString()}
                  </div>
                  <div className="text-sm text-gray-700">Total Population</div>
                </div>
                
                <div className="bg-purple-50 p-4 rounded-lg text-center">
                  <div className="text-2xl font-bold text-purple-600">
                    {Math.round(stats.populationStats.avg).toLocaleString()}
                  </div>
                  <div className="text-sm text-gray-700">Avg Population</div>
                </div>
                
                <div className="bg-red-50 p-4 rounded-lg text-center">
                  <div className="text-2xl font-bold text-red-600">
                    {stats.populationStats.max.toLocaleString()}
                  </div>
                  <div className="text-sm text-gray-700">Largest City</div>
                </div>
              </>
            )}
            
            {!stats.populationStats && (
              <div className="bg-gray-50 p-4 rounded-lg text-center col-span-3">
                <div className="text-lg text-gray-500">No population data available</div>
              </div>
            )}
          </div>
        </div>

        {/* Geographic Spread */}
        {stats.geographicSpread && (
          <div className="lg:col-span-2">
            <h4 className="font-medium text-gray-800 mb-4">Geographic Distribution</h4>
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <div className="font-medium text-gray-800">Latitude Range</div>
                  <div className="text-gray-700">{stats.geographicSpread.latRange.toFixed(3)}°</div>
                </div>
                <div>
                  <div className="font-medium text-gray-800">Longitude Range</div>
                  <div className="text-gray-700">{stats.geographicSpread.lngRange.toFixed(3)}°</div>
                </div>
                <div>
                  <div className="font-medium text-gray-800">Center Point</div>
                  <div className="text-gray-700">
                    {stats.geographicSpread.centerLat.toFixed(3)}°, {stats.geographicSpread.centerLng.toFixed(3)}°
                  </div>
                </div>
                <div>
                  <div className="font-medium text-gray-800">Unique Types</div>
                  <div className="text-gray-700">{Object.keys(stats.byType).length}</div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
