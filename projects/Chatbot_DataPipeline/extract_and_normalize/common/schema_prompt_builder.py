import json
import logging
import jsonschema

SCHEMA_DEFAULTS = {
    "document_type": "manual",
    "sustainability_dimensions": ["environmental"],
    "key_topics": ["sustainable agriculture"],
    "contains_harmful_practices": False,
    "intended_audience": ["other"],
    "region_or_country": "global",
    "language": "en",
}

class SchemaPromptBuilder:
    """Extracts and formats field descriptions from JSON schemas for metadata extraction."""

    def __init__(self, schema_path, logger=None):
        """
        Initialize with schema path and optional logger.
        
        Args:
            schema_path (str): Path to the JSON schema file
            logger (Logger, optional): Custom logger instance
        """
        self.schema_path = schema_path
        self.logger = logger or logging.getLogger(__name__)
        self.schema = self._load_schema()
        self.json_fields = self._extract_schema_fields()
    
    def _load_schema(self):
        """Load JSON schema for metadata validation."""
        with open(self.schema_path) as f:
            return json.load(f)
    
    def _extract_schema_fields(self):
        """Extract field definitions from JSON schema for use in prompt construction."""
        fields = {}
        properties = self.schema.get("properties", {})
        
        for field_name, field_def in properties.items():
            # Skip fields that are added later in the process
            if field_name in ["source_url", "source_name"]:
                continue
                
            field_type = field_def.get("type")
            description = field_def.get("description", "")
            
            if field_type == "string" and "enum" in field_def:
                fields[field_name] = {
                    "type": "string",
                    "description": description,
                    "enum": field_def["enum"]
                }
            elif field_type == "array" and "enum" in field_def.get("items", {}):
                fields[field_name] = {
                    "type": "array",
                    "description": description,
                    "enum": field_def["items"]["enum"]
                }
            elif field_type == "boolean":
                fields[field_name] = {
                    "type": "boolean",
                    "description": description
                }
            else:
                fields[field_name] = {
                    "type": field_type,
                    "description": description
                }
                
        return fields
    
    def get_field_descriptions(self):
        """
        Get formatted field descriptions extracted from the schema.
        
        Returns:
            tuple: (field_descriptions_string, array_fields_list)
                - field_descriptions_string: Formatted string with all field descriptions
                - array_fields_list: List of field names that are arrays with enums
        """
        # Get array fields that need special handling
        array_fields = [field for field, info in self.json_fields.items() 
                      if info.get("type") == "array" and "enum" in info]
        
        # Build the target JSON fields section
        field_descriptions = []
        for field_name, field_info in self.json_fields.items():
            if field_info.get("type") == "string" and "enum" in field_info:
                # String enum field
                enum_list = '", "'.join(field_info["enum"])
                field_descriptions.append(f'- {field_name}: one of ["{enum_list}"]')
            elif field_info.get("type") == "array" and "enum" in field_info:
                # Array enum field
                enum_items = [f'"{item}"' for item in field_info["enum"]]
                enum_text = ",\n            ".join(enum_items)
                field_descriptions.append(f'- {field_name}: list of 1 or more from [\n            {enum_text}\n        ]')
            elif field_name == "language":
                field_descriptions.append(f'- {field_name}: 2-letter ISO code (e.g., "en")')
            elif field_name == "date_published":
                field_descriptions.append(f'- {field_name}: Use format "YYYY-MM-DD" ONLY if clearly stated in the text. If not present, omit the field.')
            elif field_name == "region_or_country":
                field_descriptions.append(f'- {field_name}: country or region mentioned; if not mentioned or multiple regions are mentioned, use "global"')
            elif field_info.get("type") == "boolean":
                field_descriptions.append(f'- {field_name}: boolean')
            else:
                field_descriptions.append(f'- {field_name}: {field_info.get("type", "string")}')
        
        fields_section = "\n        ".join(field_descriptions)
        
        return fields_section, array_fields

    def enforce_schema_and_apply_defaults(self, metadata):
        """
        Ensures metadata conforms to schema:
        - Removes invalid enum values
        - Applies defaults where necessary
        - Logs all corrections
        
        Args:
            metadata (dict): Metadata to validate and fix
            
        Returns:
            dict: Cleaned metadata with defaults applied
        """
        cleaned = metadata.copy()

        for field, rules in self.schema.get("properties", {}).items():
            # Handle scalar enum fields
            if rules.get("type") == "string" and "enum" in rules:
                val = cleaned.get(field)
                if val is not None and val not in rules["enum"]:
                    self.logger.info(f"Invalid value for '{field}': '{val}' — replacing with default: {SCHEMA_DEFAULTS.get(field)}")
                    cleaned[field] = SCHEMA_DEFAULTS.get(field)
                elif val is None and field in SCHEMA_DEFAULTS:
                    self.logger.info(f"Missing '{field}' — using default: {SCHEMA_DEFAULTS[field]}")
                    cleaned[field] = SCHEMA_DEFAULTS[field]

            # Handle list-of-enum fields
            elif rules.get("type") == "array" and "enum" in rules.get("items", {}):
                val = cleaned.get(field, [])
                allowed = set(rules["items"]["enum"])
                filtered = [v for v in val if v in allowed]

                if not filtered and field in SCHEMA_DEFAULTS:
                    self.logger.info(f"No valid values for list field '{field}' — using default: {SCHEMA_DEFAULTS[field]}")
                    cleaned[field] = SCHEMA_DEFAULTS[field]
                else:
                    if set(val) != set(filtered):
                        self.logger.info(f"Removed invalid values from '{field}': {set(val) - allowed}")
                    cleaned[field] = filtered

            # Handle booleans
            elif rules.get("type") == "boolean":
                if field not in cleaned and field in SCHEMA_DEFAULTS:
                    self.logger.info(f"Missing boolean '{field}' — using default: {SCHEMA_DEFAULTS[field]}")
                    cleaned[field] = SCHEMA_DEFAULTS[field]

            # Fallback: fill any known default if missing
            if field not in cleaned and field in SCHEMA_DEFAULTS:
                self.logger.info(f"Missing field '{field}' — using default: {SCHEMA_DEFAULTS[field]}")
                cleaned[field] = SCHEMA_DEFAULTS[field]

        return cleaned
        
    def validate_metadata(self, metadata):
        """
        Validate metadata against the schema.
        
        Args:
            metadata (dict): Metadata to validate
            
        Raises:
            jsonschema.ValidationError: If validation fails
        """
        jsonschema.validate(instance=metadata, schema=self.schema) 