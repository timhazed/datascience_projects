from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class District(Base):
    __tablename__ = "districts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    code: Mapped[str] = mapped_column(String(10), nullable=False, unique=True)
    population: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )

    municipalities: Mapped[list["Municipality"]] = relationship(
        "Municipality", back_populates="district", cascade="all, delete-orphan"
    )


class Municipality(Base):
    __tablename__ = "municipalities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    district_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("districts.id"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    population: Mapped[int | None] = mapped_column(Integer, nullable=True)
    area: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )

    district: Mapped["District"] = relationship(
        "District", back_populates="municipalities"
    )
    localities: Mapped[list["Locality"]] = relationship(
        "Locality", back_populates="municipality", cascade="all, delete-orphan"
    )


class Locality(Base):
    __tablename__ = "localities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    municipality_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("municipalities.id"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    feature_type: Mapped[str] = mapped_column(String(50), nullable=False)
    population: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )

    municipality: Mapped["Municipality"] = relationship(
        "Municipality", back_populates="localities"
    )
    alternate_names: Mapped[list["AlternateName"]] = relationship(
        "AlternateName", back_populates="locality", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_localities_coordinates", "latitude", "longitude"),
        Index("idx_search_name", "name"),
    )


class AlternateName(Base):
    __tablename__ = "alternate_names"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    locality_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("localities.id"), nullable=False, index=True
    )
    location_type: Mapped[str] = mapped_column(String(20), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    language: Mapped[str | None] = mapped_column(String(10), nullable=True)
    is_preferred: Mapped[bool] = mapped_column(Boolean, default=False)

    locality: Mapped["Locality"] = relationship(
        "Locality", back_populates="alternate_names"
    )
