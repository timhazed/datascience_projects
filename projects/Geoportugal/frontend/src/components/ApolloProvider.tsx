'use client';

import { ApolloProvider } from '@apollo/client';
import { useState, useEffect } from 'react';
import { apolloClient } from '@/lib/apollo';

interface ApolloProviderWrapperProps {
  children: React.ReactNode;
}

export function ApolloProviderWrapper({ children }: ApolloProviderWrapperProps) {
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
  }, []);

  if (!isClient) {
    return <div className="min-h-screen bg-gray-50 flex items-center justify-center">
      <div className="text-center">
        <div className="animate-spin h-8 w-8 border-2 border-blue-600 border-t-transparent rounded-full mx-auto mb-4"></div>
        <p className="text-gray-600">Loading GeoPortugal...</p>
      </div>
    </div>;
  }

  return (
    <ApolloProvider client={apolloClient}>
      {children}
    </ApolloProvider>
  );
}