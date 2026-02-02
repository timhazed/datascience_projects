#!/usr/bin/env python3
"""
GeoNames Portugal Data Loader

Downloads and loads Portugal geographical data from GeoNames into the database.
Handles the hierarchical relationship: Country → Districts → Municipalities → Localities

Usage:
    python scripts/load_portugal_data.py
"""

import asyncio
import csv
import urllib.request
from pathlib import Path

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.core.logging import configure_logging
from app.db.models import AlternateName, Base, District, Locality, Municipality

logger = structlog.get_logger(__name__)

# GeoNames download URLs
GEONAMES_URLS = {
    "portugal": "http://download.geonames.org/export/dump/PT.zip",
    "admin1": "http://download.geonames.org/export/dump/admin1CodesASCII.txt",
    "admin2": "http://download.geonames.org/export/dump/admin2Codes.txt",
    "hierarchy": "http://download.geonames.org/export/dump/hierarchy.zip",
    "alternate_names": "http://download.geonames.org/export/dump/alternateNamesV2.zip",
}

# Feature codes for Portuguese administrative divisions
PORTUGAL_ADMIN_CODES = {
    "ADM1": "district",  # Districts
    "ADM2": "municipality",  # Municipalities
    "ADM3": "parish",  # Parishes
    "ADM4": "locality",  # Localities
}

# Portuguese district codes (official INE codes)
PORTUGAL_DISTRICTS = {
    "01": "Aveiro",
    "02": "Beja",
    "03": "Braga",
    "04": "Bragança",
    "05": "Castelo Branco",
    "06": "Coimbra",
    "07": "Évora",
    "08": "Faro",
    "09": "Guarda",
    "10": "Leiria",
    "11": "Lisboa",
    "12": "Portalegre",
    "13": "Porto",
    "14": "Santarém",
    "15": "Setúbal",
    "16": "Viana do Castelo",
    "17": "Vila Real",
    "18": "Viseu",
    "20": "Região Autónoma dos Açores",
    "30": "Região Autónoma da Madeira",
}


class GeoNamesLoader:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.data_dir = Path("./data/raw")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Cache for lookups
        self.districts_cache: dict[str, District] = {}
        self.municipalities_cache: dict[str, Municipality] = {}
        self.geonames_hierarchy: dict[int, list[int]] = {}

    async def load_all_data(self) -> None:
        """Load all Portugal data from GeoNames"""
        logger.info("Starting Portugal data load from GeoNames")

        try:
            # Download data files
            await self.download_data_files()

            # Create districts first (administrative level 1)
            await self.create_districts()

            # Load main Portugal data
            await self.load_portugal_data()

            logger.info("Portugal data load completed successfully")

        except Exception as e:
            logger.error("Error loading Portugal data", error=str(e))
            raise

    async def download_data_files(self) -> None:
        """Download required data files from GeoNames"""
        logger.info("Downloading GeoNames data files")

        for name, url in GEONAMES_URLS.items():
            file_path = self.data_dir / f"{name}.txt"
            if name in ["portugal", "hierarchy", "alternate_names"]:
                file_path = self.data_dir / f"{name}.zip"

            if not file_path.exists():
                logger.info(f"Downloading {name}", url=url)
                urllib.request.urlretrieve(url, file_path)

                # Extract zip files
                if file_path.suffix == ".zip":
                    import zipfile
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(self.data_dir)
            else:
                logger.info(f"Using cached {name}", file=str(file_path))

    async def create_districts(self) -> None:
        """Create Portuguese districts"""
        logger.info("Creating Portuguese districts")

        for code, name in PORTUGAL_DISTRICTS.items():
            district = District(
                name=name,
                code=code,
                population=None  # Will be updated when processing localities
            )
            self.session.add(district)
            self.districts_cache[name] = district

        await self.session.commit()
        logger.info(f"Created {len(PORTUGAL_DISTRICTS)} districts")

    async def load_portugal_data(self) -> None:
        """Load main Portugal geographical data"""
        logger.info("Loading Portugal geographical data")

        pt_file = self.data_dir / "PT.txt"
        if not pt_file.exists():
            raise FileNotFoundError("PT.txt not found. Please check download.")

        municipalities_count = 0
        localities_count = 0

        with open(pt_file, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')

            for row in reader:
                if len(row) < 19:
                    continue

                geonameid = int(row[0])
                name = row[1]
                asciiname = row[2]
                alternatenames = row[3]
                latitude = float(row[4]) if row[4] else 0.0
                longitude = float(row[5]) if row[5] else 0.0
                feature_class = row[6]
                feature_code = row[7]
                country_code = row[8]
                admin1_code = row[10]  # District code
                admin2_code = row[11]  # Municipality code
                population = int(row[14]) if row[14] else None

                # Skip if not administrative division or populated place
                if feature_class not in ['A', 'P']:
                    continue

                # Handle municipalities (ADM2)
                if feature_code == 'ADM2':
                    municipality = await self.create_municipality(
                        name, admin1_code, population, latitude, longitude
                    )
                    if municipality:
                        municipalities_count += 1

                # Handle localities and populated places
                elif feature_code in ['ADM3', 'ADM4', 'PPL', 'PPLA', 'PPLA2', 'PPLA3', 'PPLC']:
                    locality = await self.create_locality(
                        name, admin1_code, admin2_code, feature_code,
                        population, latitude, longitude
                    )
                    if locality:
                        localities_count += 1

                        # Add alternate names if available
                        if alternatenames:
                            await self.add_alternate_names(
                                locality, alternatenames
                            )

        await self.session.commit()
        logger.info(
            "Loaded Portugal data",
            municipalities=municipalities_count,
            localities=localities_count
        )

    async def create_municipality(
        self, name: str, admin1_code: str, population: int | None,
        latitude: float, longitude: float
    ) -> Municipality | None:
        """Create a municipality"""
        municipality = None

        # Find parent district
        district_name = PORTUGAL_DISTRICTS.get(admin1_code)
        if not district_name:
            logger.warning(f"Unknown district code: {admin1_code}")
        else:
            district = self.districts_cache.get(district_name)
            if not district:
                logger.warning(f"District not found: {district_name}")
            else:
                municipality = Municipality(
                    district_id=district.id,
                    name=name,
                    population=population,
                    area=None  # Could be calculated from coordinates
                )

                self.session.add(municipality)
                self.municipalities_cache[f"{admin1_code}_{name}"] = municipality

        return municipality

    async def create_locality(
        self, name: str, admin1_code: str, admin2_code: str,
        feature_code: str, population: int | None,
        latitude: float, longitude: float
    ) -> Locality | None:
        """Create a locality"""
        locality = None

        # Find parent municipality
        municipality = None

        # Try to find municipality by admin codes
        for key, muni in self.municipalities_cache.items():
            if key.startswith(admin1_code):
                municipality = muni
                break

        # If no municipality found, try to find district and create locality under it
        if not municipality:
            district_name = PORTUGAL_DISTRICTS.get(admin1_code)
            if not district_name:
                logger.warning(f"Unknown district for locality: {name}")
            else:
                district = self.districts_cache.get(district_name)
                if not district:
                    logger.warning(f"District not found for locality: {name}")
                else:
                    # Create a default municipality if none exists
                    municipality = Municipality(
                        district_id=district.id,
                        name=f"{district_name} (Default)",
                        population=None,
                        area=None
                    )
                    self.session.add(municipality)
                    await self.session.flush()  # Get the ID
                    self.municipalities_cache[f"{admin1_code}_default"] = municipality

        # Create locality if we have a valid municipality
        if municipality:
            # Map feature code to type
            feature_type = PORTUGAL_ADMIN_CODES.get(feature_code, "locality")
            if feature_code.startswith('PPL'):
                feature_type = "populated_place"

            locality = Locality(
                municipality_id=municipality.id,
                name=name,
                feature_type=feature_type,
                population=population,
                latitude=latitude,
                longitude=longitude
            )

            self.session.add(locality)
            await self.session.flush()

        return locality

    async def add_alternate_names(
        self, locality: Locality, alternatenames: str
    ) -> None:
        """Add alternate names for a locality"""

        if not alternatenames:
            return

        names = alternatenames.split(',')
        for alt_name in names[:5]:  # Limit to 5 alternate names
            alt_name = alt_name.strip()
            if alt_name and alt_name != locality.name:
                alternate = AlternateName(
                    locality_id=locality.id,
                    location_type="locality",
                    name=alt_name,
                    language=None,  # Could be detected
                    is_preferred=False
                )
                self.session.add(alternate)


async def main():
    """Main function to load Portugal data"""
    configure_logging()
    logger.info("Starting GeoNames Portugal data loader")

    # Create database engine
    engine = create_async_engine(settings.database_url, echo=False)

    # Create tables if they don't exist
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Create session
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    async with async_session() as session:
        loader = GeoNamesLoader(session)
        await loader.load_all_data()

    await engine.dispose()
    logger.info("Data loading completed")


if __name__ == "__main__":
    asyncio.run(main())
