#!/usr/bin/env python3
"""
Production Configuration Validation Script

This script validates that production deployments have secure configuration.
Run this before deploying to production to catch security issues.

Usage:
    python scripts/validate_production_config.py

Environment Variables:
    ENVIRONMENT=production  # Enables production validation mode
"""

import os
import sys
from pathlib import Path

# Add the app directory to the path so we can import the config
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from app.core.config import Settings
except ImportError as e:
    print(f"❌ Failed to import settings: {e}")
    print("Make sure you're running this from the backend directory")
    sys.exit(1)


class ProductionValidator:
    """Validates production configuration for security compliance"""

    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.settings = None

    def validate(self) -> bool:
        """Run all validation checks"""
        print("🔍 Validating production configuration...\n")

        try:
            self.settings = Settings()
        except Exception as e:
            self.errors.append(f"Configuration loading failed: {e}")
            self._print_results()
            return False

        # Run all validation checks
        self._validate_environment()
        self._validate_secrets()
        self._validate_database()
        self._validate_cors()
        self._validate_security_headers()
        self._validate_monitoring()

        # Print results
        self._print_results()

        return len(self.errors) == 0

    def _validate_environment(self):
        """Validate environment settings"""
        if self.settings.debug:
            self.errors.append("DEBUG=true in production - must be false")

        if os.getenv("ENVIRONMENT", "development").lower() != "production":
            self.warnings.append("ENVIRONMENT != 'production' - set for proper validation")

        if self.settings.log_level == "DEBUG":
            self.warnings.append("LOG_LEVEL=DEBUG in production - consider INFO or WARNING")

    def _validate_secrets(self):
        """Validate secret keys and passwords"""
        # Check SECRET_KEY
        if len(self.settings.secret_key) < 32:
            self.errors.append(f"SECRET_KEY too short ({len(self.settings.secret_key)} chars) - minimum 32 required")

        if "dev-secret" in self.settings.secret_key.lower():
            self.errors.append("SECRET_KEY contains 'dev-secret' - must be changed for production")

        # Check metrics password
        if self.settings.metrics_password:
            if len(self.settings.metrics_password) < 12:
                self.warnings.append("METRICS_PASSWORD is short - consider using 16+ characters")

            if "change-this" in self.settings.metrics_password.lower():
                self.errors.append("METRICS_PASSWORD contains 'change-this' - must be changed")

            if "admin" == self.settings.metrics_password.lower():
                self.errors.append("METRICS_PASSWORD is 'admin' - must be changed")

    def _validate_database(self):
        """Validate database configuration"""
        db_url = self.settings.database_url.lower()

        # Check for localhost in production
        if "localhost" in db_url or "127.0.0.1" in db_url:
            self.warnings.append("DATABASE_URL uses localhost - ensure this is correct for production")

        # Check for default credentials
        if "postgres:postgres@" in db_url:
            self.errors.append("DATABASE_URL uses default postgres:postgres credentials - must be changed")

        # Check connection pool settings
        if self.settings.max_connections < 10:
            self.warnings.append(f"MAX_CONNECTIONS={self.settings.max_connections} is low for production")

    def _validate_cors(self):
        """Validate CORS configuration"""
        localhost_origins = [origin for origin in self.settings.cors_origins if "localhost" in origin]
        if localhost_origins:
            self.errors.append(f"CORS_ORIGINS contains localhost URLs in production: {localhost_origins}")

        if "http://" in str(self.settings.cors_origins):
            self.warnings.append("CORS_ORIGINS contains HTTP (not HTTPS) URLs - consider security implications")

        if not self.settings.cors_origins:
            self.warnings.append("CORS_ORIGINS is empty - this might block legitimate requests")

    def _validate_security_headers(self):
        """Validate security-related settings"""
        if self.settings.rate_limit_requests_per_minute > 1000:
            self.warnings.append(f"Rate limit ({self.settings.rate_limit_requests_per_minute}/min) is very high")

        if self.settings.rate_limit_requests_per_minute < 10:
            self.warnings.append(f"Rate limit ({self.settings.rate_limit_requests_per_minute}/min) might be too restrictive")

    def _validate_monitoring(self):
        """Validate monitoring configuration"""
        if not self.settings.enable_metrics:
            self.warnings.append("ENABLE_METRICS=false - monitoring might be limited")

        if self.settings.metrics_username == "admin":
            self.warnings.append("METRICS_USERNAME='admin' is default - consider changing")

    def _print_results(self):
        """Print validation results"""
        print("=" * 60)

        if self.errors:
            print(f"❌ CRITICAL ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"   • {error}")
            print()

        if self.warnings:
            print(f"⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   • {warning}")
            print()

        if not self.errors and not self.warnings:
            print("✅ All validation checks passed!")
        elif not self.errors:
            print(f"✅ No critical errors found (but {len(self.warnings)} warnings)")
        else:
            print(f"❌ Production deployment BLOCKED - {len(self.errors)} critical error(s)")

        print("=" * 60)


def main():
    """Main entry point"""
    validator = ProductionValidator()

    # Set environment for validation
    if not os.getenv("ENVIRONMENT"):
        print("💡 Tip: Set ENVIRONMENT=production for full validation")
        print()

    success = validator.validate()

    if not success:
        print("\n🚫 PRODUCTION DEPLOYMENT BLOCKED")
        print("Fix the critical errors above before deploying to production.")
        sys.exit(1)
    else:
        print("\n✅ PRODUCTION VALIDATION PASSED")
        print("Configuration is secure for production deployment.")
        sys.exit(0)


if __name__ == "__main__":
    main()
