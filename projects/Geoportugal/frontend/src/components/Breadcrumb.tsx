'use client';

import { useState, useEffect } from 'react';
import { useQuery } from '@apollo/client';
import { GET_DISTRICT_BY_ID, GET_MUNICIPALITY_BY_ID, District, Municipality, Locality } from '@/lib/queries';

interface BreadcrumbProps {
  locality: Locality;
}

export function Breadcrumb({ locality }: BreadcrumbProps) {
  const [municipality, setMunicipality] = useState<Municipality | null>(null);
  const [district, setDistrict] = useState<District | null>(null);

  // Fetch municipality data
  const { data: municipalityData } = useQuery(GET_MUNICIPALITY_BY_ID, {
    variables: { id: locality.municipalityId },
    skip: !locality.municipalityId,
  });

// Fetch district data based on municipality's districtId
const { data: districtData } = useQuery(GET_DISTRICT_BY_ID, {
  variables: { id: municipality?.districtId },
  skip: !municipality?.districtId,
});

  useEffect(() => {
    if (municipalityData?.municipality) {
      setMunicipality(municipalityData.municipality);
    }
  }, [municipalityData]);

  useEffect(() => {
    if (districtData?.district) {
      setDistrict(districtData.district);
    }
  }, [districtData]);

  const breadcrumbItems = [
    district && { 
      name: district.name, 
      type: 'District',
      color: 'text-red-600',
      bgColor: 'bg-red-50',
      population: district.population
    },
    municipality && { 
      name: municipality.name, 
      type: 'Municipality',
      color: 'text-green-600',
      bgColor: 'bg-green-50',
      population: municipality.population
    },
    { 
      name: locality.name, 
      type: locality.featureType.replace('_', ' '),
      color: locality.featureType === 'locality' || locality.featureType === 'populated_place' 
        ? 'text-blue-600' 
        : 'text-purple-600',
      bgColor: locality.featureType === 'locality' || locality.featureType === 'populated_place'
        ? 'bg-blue-50'
        : 'bg-purple-50',
      population: locality.population,
      current: true
    }
  ].filter(Boolean);

  return (
    <div className="mb-4">
      <h4 className="font-medium text-gray-700 mb-3">Administrative Hierarchy</h4>
      <nav className="flex flex-wrap items-center space-x-2 text-sm">
        {breadcrumbItems.map((item, index) => (
          <div key={index} className="flex items-center">
            {index > 0 && (
              <svg 
                className="w-4 h-4 text-gray-400 mx-2" 
                fill="none" 
                stroke="currentColor" 
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            )}
            <div className={`px-3 py-1 rounded-full text-xs font-medium ${item!.bgColor} ${item!.color} ${item!.current ? 'ring-2 ring-offset-1 ring-current' : ''}`}>
              <div className="flex items-center space-x-1">
                <span className="capitalize">{item!.type}</span>
                <span>•</span>
                <span className="font-semibold">{item!.name}</span>
                {item!.population && (
                  <>
                    <span>•</span>
                    <span>{item!.population.toLocaleString()}</span>
                  </>
                )}
              </div>
            </div>
          </div>
        ))}
      </nav>
      
      {/* Alternative vertical layout for mobile */}
      <div className="mt-3 md:hidden">
        <div className="space-y-2">
          {breadcrumbItems.map((item, index) => (
            <div key={`mobile-${index}`} className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${
                item!.type === 'District' ? 'bg-red-500' :
                item!.type === 'Municipality' ? 'bg-green-500' :
                'bg-blue-500'
              }`}></div>
              <div className="text-sm">
                <span className="font-medium capitalize">{item!.type}:</span>{' '}
                <span className={`${item!.current ? 'font-semibold' : ''}`}>
                  {item!.name}
                </span>
                {item!.population && (
                  <span className="text-gray-500 ml-2">
                    ({item!.population.toLocaleString()})
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
