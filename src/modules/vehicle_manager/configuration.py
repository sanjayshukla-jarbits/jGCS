"""
Vehicle Configuration - Vehicle configuration management and validation

This module provides:
- Vehicle configuration templates
- Configuration validation
- Parameter management
- Configuration inheritance
"""

import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path

from src.utils.logger import get_logger

class ConfigurationLevel(Enum):
    """Configuration inheritance levels"""
    GLOBAL = "global"          # System-wide defaults
    VEHICLE_TYPE = "vehicle_type"  # Vehicle type specific
    VEHICLE_MODEL = "vehicle_model"  # Vehicle model specific
    VEHICLE_INSTANCE = "vehicle_instance"  # Individual vehicle

class ParameterType(Enum):
    """Parameter data types"""
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"

@dataclass
class ConfigurationParameter:
    """Individual configuration parameter"""
    name: str
    value: Any
    parameter_type: ParameterType
    description: str = ""
    unit: str = ""
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    required: bool = True
    validation_rules: List[str] = field(default_factory=list)
    
    def validate(self) -> Dict[str, Any]:
        """Validate parameter value"""
        
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Type validation
        if self.parameter_type == ParameterType.INTEGER and not isinstance(self.value, int):
            try:
                self.value = int(self.value)
            except (ValueError, TypeError):
                validation_result["valid"] = False
                validation_result["errors"].append(f"{self.name}: Must be an integer")
        
        elif self.parameter_type == ParameterType.FLOAT and not isinstance(self.value, (int, float)):
            try:
                self.value = float(self.value)
            except (ValueError, TypeError):
                validation_result["valid"] = False
                validation_result["errors"].append(f"{self.name}: Must be a number")
        
        elif self.parameter_type == ParameterType.BOOLEAN and not isinstance(self.value, bool):
            if isinstance(self.value, str):
                self.value = self.value.lower() in ["true", "1", "yes", "on"]
            else:
                validation_result["valid"] = False
                validation_result["errors"].append(f"{self.name}: Must be a boolean")
        
        # Range validation
        if self.parameter_type in [ParameterType.INTEGER, ParameterType.FLOAT]:
            if self.min_value is not None and self.value < self.min_value:
                validation_result["valid"] = False
                validation_result["errors"].append(f"{self.name}: Value {self.value} below minimum {self.min_value}")
            
            if self.max_value is not None and self.value > self.max_value:
                validation_result["valid"] = False
                validation_result["errors"].append(f"{self.name}: Value {self.value} above maximum {self.max_value}")
        
        # Allowed values validation
        if self.allowed_values and self.value not in self.allowed_values:
            validation_result["valid"] = False
            validation_result["errors"].append(f"{self.name}: Value {self.value} not in allowed values {self.allowed_values}")
        
        # Custom validation rules
        for rule in self.validation_rules:
            try:
                if not eval(rule.replace("value", str(self.value))):
                    validation_result["valid"] = False
                    validation_result["errors"].append(f"{self.name}: Failed validation rule: {rule}")
            except Exception:
                validation_result["warnings"].append(f"{self.name}: Could not evaluate validation rule: {rule}")
        
        return validation_result

@dataclass
class VehicleConfigurationTemplate:
    """Vehicle configuration template"""
    template_name: str
    vehicle_type: str
    description: str
    parameters: Dict[str, ConfigurationParameter]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def validate_configuration(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration against template"""
        
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "missing_required": [],
            "unknown_parameters": []
        }
        
        # Check required parameters
        for param_name, param in self.parameters.items():
            if param.required and param_name not in config_data:
                validation_result["missing_required"].append(param_name)
                validation_result["valid"] = False
        
        # Validate provided parameters
        for param_name, value in config_data.items():
            if param_name in self.parameters:
                param = self.parameters[param_name]
                param.value = value
                param_validation = param.validate()
                
                if not param_validation["valid"]:
                    validation_result["valid"] = False
                    validation_result["errors"].extend(param_validation["errors"])
                
                validation_result["warnings"].extend(param_validation["warnings"])
                
            else:
                validation_result["unknown_parameters"].append(param_name)
                validation_result["warnings"].append(f"Unknown parameter: {param_name}")
        
        return validation_result

class VehicleConfiguration:
    """Vehicle configuration management system"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # Configuration storage
        self.templates: Dict[str, VehicleConfigurationTemplate] = {}
        self.configurations: Dict[str, Dict[str, Any]] = {}
        self.configuration_inheritance: Dict[str, List[ConfigurationLevel]] = {}
        
        # Initialize default templates
        self._initialize_default_templates()
        
        # Configuration validation
        self.validation_enabled = True
        self.strict_mode = False  # If True, unknown parameters cause errors
    
    def _initialize_default_templates(self) -> None:
        """Initialize default configuration templates"""
        
        # Multi-rotor template
        multirotor_params = {
            "max_speed": ConfigurationParameter(
                "max_speed", 20.0, ParameterType.FLOAT,
                "Maximum horizontal speed", "m/s", 5.0, 30.0
            ),
            "max_climb_rate": ConfigurationParameter(
                "max_climb_rate", 5.0, ParameterType.FLOAT,
                "Maximum climb rate", "m/s", 1.0, 10.0
            ),
            "max_descent_rate": ConfigurationParameter(
                "max_descent_rate", 3.0, ParameterType.FLOAT,
                "Maximum descent rate", "m/s", 1.0, 8.0
            ),
            "hover_accuracy": ConfigurationParameter(
                "hover_accuracy", 2.0, ParameterType.FLOAT,
                "Position hold accuracy", "meters", 0.5, 10.0
            ),
            "battery_capacity": ConfigurationParameter(
                "battery_capacity", 16000, ParameterType.INTEGER,
                "Battery capacity", "mAh", 1000, 50000
            ),
            "max_payload": ConfigurationParameter(
                "max_payload", 2.5, ParameterType.FLOAT,
                "Maximum payload weight", "kg", 0.1, 20.0
            ),
            "rotor_count": ConfigurationParameter(
                "rotor_count", 4, ParameterType.INTEGER,
                "Number of rotors", "", 3, 8
            ),
            "frame_type": ConfigurationParameter(
                "frame_type", "X", ParameterType.STRING,
                "Frame configuration", "", allowed_values=["X", "+", "H", "Y6", "V"]
            )
        }
        
        self.templates["multirotor"] = VehicleConfigurationTemplate(
            "multirotor", "multi_rotor", "Multi-rotor vehicle configuration", multirotor_params
        )
        
        # Fixed-wing template
        fixedwing_params = {
            "max_speed": ConfigurationParameter(
                "max_speed", 30.0, ParameterType.FLOAT,
                "Maximum speed", "m/s", 12.0, 50.0
            ),
            "stall_speed": ConfigurationParameter(
                "stall_speed", 12.0, ParameterType.FLOAT,
                "Stall speed", "m/s", 8.0, 20.0
            ),
            "cruise_speed": ConfigurationParameter(
                "cruise_speed", 20.0, ParameterType.FLOAT,
                "Cruise speed", "m/s", 10.0, 35.0
            ),
            "min_turn_radius": ConfigurationParameter(
                "min_turn_radius", 50.0, ParameterType.FLOAT,
                "Minimum turn radius", "meters", 20.0, 200.0
            ),
            "service_ceiling": ConfigurationParameter(
                "service_ceiling", 1000.0, ParameterType.FLOAT,
                "Service ceiling", "meters", 100.0, 3000.0
            ),
            "wingspan": ConfigurationParameter(
                "wingspan", 1.5, ParameterType.FLOAT,
                "Wingspan", "meters", 0.5, 5.0
            ),
            "wing_area": ConfigurationParameter(
                "wing_area", 0.3, ParameterType.FLOAT,
                "Wing area", "m²", 0.1, 2.0
            )
        }
        
        self.templates["fixedwing"] = VehicleConfigurationTemplate(
            "fixedwing", "fixed_wing", "Fixed-wing vehicle configuration", fixedwing_params
        )
        
        # VTOL template
        vtol_params = {
            "hover_speed_max": ConfigurationParameter(
                "hover_speed_max", 15.0, ParameterType.FLOAT,
                "Maximum hover mode speed", "m/s", 5.0, 25.0
            ),
            "forward_speed_max": ConfigurationParameter(
                "forward_speed_max", 25.0, ParameterType.FLOAT,
                "Maximum forward flight speed", "m/s", 15.0, 40.0
            ),
            "transition_speed": ConfigurationParameter(
                "transition_speed", 18.0, ParameterType.FLOAT,
                "Transition speed between modes", "m/s", 12.0, 25.0
            ),
            "vtol_type": ConfigurationParameter(
                "vtol_type", "quadplane", ParameterType.STRING,
                "VTOL configuration type", "", 
                allowed_values=["quadplane", "tiltrotor", "tailsitter"]
            ),
            "hover_efficiency": ConfigurationParameter(
                "hover_efficiency", 0.7, ParameterType.FLOAT,
                "Hover mode efficiency", "ratio", 0.3, 1.0
            ),
            "forward_efficiency": ConfigurationParameter(
                "forward_efficiency", 0.85, ParameterType.FLOAT,
                "Forward flight efficiency", "ratio", 0.5, 1.0
            ),
            "transition_time": ConfigurationParameter(
                "transition_time", 5.0, ParameterType.FLOAT,
                "Transition time between modes", "seconds", 2.0, 15.0
            )
        }
        
        self.templates["vtol"] = VehicleConfigurationTemplate(
            "vtol", "vtol", "VTOL vehicle configuration", vtol_params
        )
    
    def create_configuration(self, vehicle_id: str, template_name: str, 
                           config_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create vehicle configuration from template"""
        
        if template_name not in self.templates:
            raise ValueError(f"Unknown template: {template_name}")
        
        template = self.templates[template_name]
        
        # Start with default values from template
        configuration = {}
        for param_name, param in template.parameters.items():
            configuration[param_name] = param.value
        
        # Override with provided configuration data
        if config_data:
            if self.validation_enabled:
                validation_result = template.validate_configuration(config_data)
                
                if not validation_result["valid"]:
                    error_msg = f"Configuration validation failed: {validation_result['errors']}"
                    if validation_result["missing_required"]:
                        error_msg += f", Missing required: {validation_result['missing_required']}"
                    raise ValueError(error_msg)
                
                if self.strict_mode and validation_result["unknown_parameters"]:
                    raise ValueError(f"Unknown parameters in strict mode: {validation_result['unknown_parameters']}")
            
            configuration.update(config_data)
        
        # Store configuration
        self.configurations[vehicle_id] = {
            "vehicle_id": vehicle_id,
            "template": template_name,
            "parameters": configuration,
            "created_at": datetime.now(),
            "modified_at": datetime.now()
        }
        
        self.logger.info(f"Created configuration for vehicle {vehicle_id} using template {template_name}")
        
        return self.configurations[vehicle_id]
    
    def update_configuration(self, vehicle_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update existing vehicle configuration"""
        
        if vehicle_id not in self.configurations:
            raise ValueError(f"No configuration found for vehicle {vehicle_id}")
        
        config = self.configurations[vehicle_id]
        template_name = config["template"]
        
        if self.validation_enabled and template_name in self.templates:
            # Validate updates against template
            template = self.templates[template_name]
            validation_result = template.validate_configuration(updates)
            
            if not validation_result["valid"]:
                raise ValueError(f"Configuration update validation failed: {validation_result['errors']}")
        
        # Apply updates
        config["parameters"].update(updates)
        config["modified_at"] = datetime.now()
        
        self.logger.info(f"Updated configuration for vehicle {vehicle_id}")
        
        return config
    
    def get_configuration(self, vehicle_id: str) -> Optional[Dict[str, Any]]:
        """Get vehicle configuration"""
        return self.configurations.get(vehicle_id)
    
    def get_parameter(self, vehicle_id: str, parameter_name: str, default: Any = None) -> Any:
        """Get specific parameter value"""
        
        config = self.configurations.get(vehicle_id)
        if not config:
            return default
        
        return config["parameters"].get(parameter_name, default)
    
    def set_parameter(self, vehicle_id: str, parameter_name: str, value: Any) -> bool:
        """Set specific parameter value"""
        
        if vehicle_id not in self.configurations:
            return False
        
        # Validate single parameter if template exists
        config = self.configurations[vehicle_id]
        template_name = config["template"]
        
        if self.validation_enabled and template_name in self.templates:
            template = self.templates[template_name]
            
            if parameter_name in template.parameters:
                param = template.parameters[parameter_name]
                param.value = value
                validation_result = param.validate()
                
                if not validation_result["valid"]:
                    self.logger.error(f"Parameter validation failed: {validation_result['errors']}")
                    return False
        
        # Set parameter
        config["parameters"][parameter_name] = value
        config["modified_at"] = datetime.now()
        
        return True
    
    def delete_configuration(self, vehicle_id: str) -> bool:
        """Delete vehicle configuration"""
        
        if vehicle_id in self.configurations:
            del self.configurations[vehicle_id]
            self.logger.info(f"Deleted configuration for vehicle {vehicle_id}")
            return True
        
        return False
    
    def list_configurations(self) -> List[Dict[str, Any]]:
        """List all vehicle configurations"""
        
        return [
            {
                "vehicle_id": vehicle_id,
                "template": config["template"],
                "created_at": config["created_at"].isoformat(),
                "modified_at": config["modified_at"].isoformat(),
                "parameter_count": len(config["parameters"])
            }
            for vehicle_id, config in self.configurations.items()
        ]
    
    def create_template(self, template_name: str, vehicle_type: str, 
                       description: str, parameters: Dict[str, ConfigurationParameter]) -> None:
        """Create new configuration template"""
        
        template = VehicleConfigurationTemplate(
            template_name, vehicle_type, description, parameters
        )
        
        self.templates[template_name] = template
        self.logger.info(f"Created template {template_name}")
    
    def get_template(self, template_name: str) -> Optional[VehicleConfigurationTemplate]:
        """Get configuration template"""
        return self.templates.get(template_name)
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """List available templates"""
        
        return [
            {
                "template_name": name,
                "vehicle_type": template.vehicle_type,
                "description": template.description,
                "parameter_count": len(template.parameters),
                "created_at": template.created_at.isoformat()
            }
            for name, template in self.templates.items()
        ]
    
    def export_configuration(self, vehicle_id: str, file_path: str = None) -> Dict[str, Any]:
        """Export vehicle configuration to file or return as dict"""
        
        if vehicle_id not in self.configurations:
            raise ValueError(f"No configuration found for vehicle {vehicle_id}")
        
        config = self.configurations[vehicle_id].copy()
        
        # Convert datetime objects to strings for JSON serialization
        config["created_at"] = config["created_at"].isoformat()
        config["modified_at"] = config["modified_at"].isoformat()
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(config, f, indent=2)
                self.logger.info(f"Exported configuration to {file_path}")
            except Exception as e:
                self.logger.error(f"Failed to export configuration: {e}")
                raise
        
        return config
    
    def import_configuration(self, vehicle_id: str, file_path: str = None, 
                           config_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Import vehicle configuration from file or dict"""
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    config_data = json.load(f)
                self.logger.info(f"Imported configuration from {file_path}")
            except Exception as e:
                self.logger.error(f"Failed to import configuration: {e}")
                raise
        
        if not config_data:
            raise ValueError("No configuration data provided")
        
        # Validate required fields
        if "template" not in config_data or "parameters" not in config_data:
            raise ValueError("Invalid configuration format")
        
        # Create configuration
        return self.create_configuration(
            vehicle_id, 
            config_data["template"], 
            config_data["parameters"]
        )
    
    def clone_configuration(self, source_vehicle_id: str, target_vehicle_id: str) -> Dict[str, Any]:
        """Clone configuration from one vehicle to another"""
        
        source_config = self.configurations.get(source_vehicle_id)
        if not source_config:
            raise ValueError(f"No configuration found for source vehicle {source_vehicle_id}")
        
        # Create new configuration with same template and parameters
        return self.create_configuration(
            target_vehicle_id,
            source_config["template"],
            source_config["parameters"].copy()
        )
    
    def compare_configurations(self, vehicle_id_1: str, vehicle_id_2: str) -> Dict[str, Any]:
        """Compare two vehicle configurations"""
        
        config1 = self.configurations.get(vehicle_id_1)
        config2 = self.configurations.get(vehicle_id_2)
        
        if not config1 or not config2:
            raise ValueError("One or both vehicles not found")
        
        comparison = {
            "vehicle_1": vehicle_id_1,
            "vehicle_2": vehicle_id_2,
            "template_1": config1["template"],
            "template_2": config2["template"],
            "same_template": config1["template"] == config2["template"],
            "differences": {},
            "common_parameters": {},
            "unique_to_1": {},
            "unique_to_2": {}
        }
        
        params1 = config1["parameters"]
        params2 = config2["parameters"]
        
        all_params = set(params1.keys()) | set(params2.keys())
        
        for param in all_params:
            if param in params1 and param in params2:
                if params1[param] == params2[param]:
                    comparison["common_parameters"][param] = params1[param]
                else:
                    comparison["differences"][param] = {
                        "vehicle_1": params1[param],
                        "vehicle_2": params2[param]
                    }
            elif param in params1:
                comparison["unique_to_1"][param] = params1[param]
            else:
                comparison["unique_to_2"][param] = params2[param]
        
        return comparison
    
    def validate_all_configurations(self) -> Dict[str, Dict[str, Any]]:
        """Validate all stored configurations"""
        
        validation_results = {}
        
        for vehicle_id, config in self.configurations.items():
            template_name = config["template"]
            
            if template_name in self.templates:
                template = self.templates[template_name]
                validation_result = template.validate_configuration(config["parameters"])
                validation_results[vehicle_id] = validation_result
            else:
                validation_results[vehicle_id] = {
                    "valid": False,
                    "errors": [f"Template {template_name} not found"],
                    "warnings": [],
                    "missing_required": [],
                    "unknown_parameters": []
                }
        
        return validation_results
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get summary of all configurations"""
        
        total_configs = len(self.configurations)
        templates_used = set(config["template"] for config in self.configurations.values())
        
        # Validate all configurations
        validation_results = self.validate_all_configurations()
        valid_configs = sum(1 for result in validation_results.values() if result["valid"])
        
        return {
            "total_configurations": total_configs,
            "valid_configurations": valid_configs,
            "invalid_configurations": total_configs - valid_configs,
            "templates_used": list(templates_used),
            "available_templates": list(self.templates.keys()),
            "validation_enabled": self.validation_enabled,
            "strict_mode": self.strict_mode
        }
    
    def backup_configurations(self, backup_file: str) -> None:
        """Backup all configurations to file"""
        
        backup_data = {
            "timestamp": datetime.now().isoformat(),
            "configurations": {},
            "templates": {}
        }
        
        # Export configurations
        for vehicle_id, config in self.configurations.items():
            config_copy = config.copy()
            config_copy["created_at"] = config_copy["created_at"].isoformat()
            config_copy["modified_at"] = config_copy["modified_at"].isoformat()
            backup_data["configurations"][vehicle_id] = config_copy
        
        # Export custom templates (not built-in ones)
        builtin_templates = {"multirotor", "fixedwing", "vtol"}
        for template_name, template in self.templates.items():
            if template_name not in builtin_templates:
                template_data = asdict(template)
                template_data["created_at"] = template.created_at.isoformat()
                # Convert ConfigurationParameter objects to dicts
                template_data["parameters"] = {
                    name: asdict(param) for name, param in template.parameters.items()
                }
                backup_data["templates"][template_name] = template_data
        
        try:
            with open(backup_file, 'w') as f:
                json.dump(backup_data, f, indent=2)
            self.logger.info(f"Backed up configurations to {backup_file}")
        except Exception as e:
            self.logger.error(f"Failed to backup configurations: {e}")
            raise
    
    def restore_configurations(self, backup_file: str, overwrite: bool = False) -> Dict[str, Any]:
        """Restore configurations from backup file"""
        
        try:
            with open(backup_file, 'r') as f:
                backup_data = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load backup file: {e}")
            raise
        
        restore_results = {
            "configurations_restored": 0,
            "configurations_skipped": 0,
            "templates_restored": 0,
            "errors": []
        }
        
        # Restore templates first
        for template_name, template_data in backup_data.get("templates", {}).items():
            try:
                # Reconstruct ConfigurationParameter objects
                parameters = {}
                for param_name, param_data in template_data["parameters"].items():
                    param_data["parameter_type"] = ParameterType(param_data["parameter_type"])
                    parameters[param_name] = ConfigurationParameter(**param_data)
                
                # Create template
                template = VehicleConfigurationTemplate(
                    template_data["template_name"],
                    template_data["vehicle_type"],
                    template_data["description"],
                    parameters,
                    template_data.get("metadata", {}),
                    datetime.fromisoformat(template_data["created_at"])
                )
                
                self.templates[template_name] = template
                restore_results["templates_restored"] += 1
                
            except Exception as e:
                restore_results["errors"].append(f"Failed to restore template {template_name}: {e}")
        
        # Restore configurations
        for vehicle_id, config_data in backup_data.get("configurations", {}).items():
            try:
                if vehicle_id in self.configurations and not overwrite:
                    restore_results["configurations_skipped"] += 1
                    continue
                
                # Restore configuration
                config = config_data.copy()
                config["created_at"] = datetime.fromisoformat(config["created_at"])
                config["modified_at"] = datetime.fromisoformat(config["modified_at"])
                
                self.configurations[vehicle_id] = config
                restore_results["configurations_restored"] += 1
                
            except Exception as e:
                restore_results["errors"].append(f"Failed to restore configuration {vehicle_id}: {e}")
        
        self.logger.info(f"Restored {restore_results['configurations_restored']} configurations and {restore_results['templates_restored']} templates")
        
        return restore_results