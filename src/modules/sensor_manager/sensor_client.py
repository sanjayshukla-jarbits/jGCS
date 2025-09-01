"""
Sensor Manager Module - Interface with external sensor library

This module handles:
- External sensor library API integration
- Vehicle BOM (Bill of Materials) fetching
- SOP (Standard Operating Procedures) retrieval
- Sensor bounds validation
- Caching and performance optimization
"""

import aiohttp
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json

from src.modules.base_module import BaseModule
from src.models.sensor import SensorDefinition, SensorBounds
from src.models.process import ProcessStep
from src.orchestrator.execution_context import ExecutionContext
from src.utils.logger import get_logger

class SensorManagerModule(BaseModule):
    """Module for managing sensor data from external library"""
    
    def __init__(self, module_name: str, implementation: str):
        super().__init__(module_name, implementation)
        self.external_api_base = None
        self.api_key = None
        self._sensor_cache: Dict[str, List[SensorDefinition]] = {}
        self._cache_expiry: Dict[str, datetime] = {}
        self.cache_duration = timedelta(hours=1)  # Cache for 1 hour
    
    async def _execute_implementation(self, step_config: ProcessStep, context: ExecutionContext) -> Dict[str, Any]:
        """Execute sensor manager implementation"""
        
        implementation_map = {
            "external_client": self._fetch_vehicle_sensors,
            "bounds_validator": self._validate_sensor_bounds
        }
        
        if self.implementation not in implementation_map:
            raise ValueError(f"Unknown sensor manager implementation: {self.implementation}")
        
        return await implementation_map[self.implementation](step_config, context)
    
    async def _validate_parameters(self, parameters: Dict[str, Any]) -> None:
        """Validate sensor manager parameters"""
        
        if self.implementation == "external_client":
            required_params = ["external_api_base", "api_key", "vehicle_bom", "sop_id"]
        elif self.implementation == "bounds_validator":
            required_params = ["vehicle_id", "sensor_data"]
        else:
            raise ValueError(f"Unknown implementation: {self.implementation}")
        
        self._validate_required_parameters(parameters, required_params)
    
    async def _fetch_vehicle_sensors(self, step_config: ProcessStep, context: ExecutionContext) -> Dict[str, Any]:
        """Fetch vehicle sensor configuration from external API"""
        
        params = step_config.parameters
        self.external_api_base = params["external_api_base"]
        self.api_key = params["api_key"]
        vehicle_bom = params["vehicle_bom"]
        sop_id = params["sop_id"]
        
        # Check cache first
        cache_key = f"{vehicle_bom}_{sop_id}"
        if self._is_cache_valid(cache_key):
            self.logger.info(f"Using cached sensor data for {vehicle_bom}")
            return {
                "sensor_data": self._sensor_cache[cache_key],
                "source": "cache",
                "vehicle_bom": vehicle_bom,
                "sop_id": sop_id
            }
        
        try:
            async with aiohttp.ClientSession() as session:
                # Fetch BOM data
                bom_url = f"{self.external_api_base}/bom/{vehicle_bom}"
                headers = {"Authorization": f"Bearer {self.api_key}"}
                
                async with session.get(bom_url, headers=headers) as response:
                    if response.status != 200:
                        raise Exception(f"BOM API request failed: {response.status}")
                    
                    bom_data = await response.json()
                
                # Fetch SOP data
                sop_url = f"{self.external_api_base}/sop/{sop_id}"
                async with session.get(sop_url, headers=headers) as response:
                    if response.status != 200:
                        raise Exception(f"SOP API request failed: {response.status}")
                    
                    sop_data = await response.json()
                
                # Process and combine sensor data
                sensors = self._process_sensor_data(bom_data, sop_data)
                
                # Cache the results
                self._sensor_cache[cache_key] = sensors
                self._cache_expiry[cache_key] = datetime.now() + self.cache_duration
                
                # Store in execution context for other steps
                context.set_shared_data("sensor_data", sensors)
                context.set_shared_data("vehicle_bom", vehicle_bom)
                
                return {
                    "sensor_data": sensors,
                    "source": "external_api",
                    "vehicle_bom": vehicle_bom,
                    "sop_id": sop_id,
                    "sensors_count": len(sensors),
                    "api_response_time": datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Failed to fetch sensor data: {str(e)}")
            raise Exception(f"Sensor API integration failed: {str(e)}")
    
    async def _validate_sensor_bounds(self, step_config: ProcessStep, context: ExecutionContext) -> Dict[str, Any]:
        """Validate sensor data against operational bounds"""
        
        params = step_config.parameters
        vehicle_id = params["vehicle_id"]
        sensor_data = params.get("sensor_data") or context.get_shared_data("sensor_data", [])
        
        if not sensor_data:
            raise ValueError("No sensor data available for validation")
        
        validation_results = []
        violations = []
        warnings = []
        
        for sensor in sensor_data:
            result = self._validate_individual_sensor(sensor)
            validation_results.append(result)
            
            if not result["valid"]:
                violations.append(result)
                
            if result.get("warnings"):
                warnings.extend(result["warnings"])
        
        # Calculate summary statistics
        total_sensors = len(sensor_data)
        valid_sensors = len([r for r in validation_results if r["valid"]])
        validation_rate = (valid_sensors / total_sensors) * 100 if total_sensors > 0 else 0
        
        return {
            "vehicle_id": vehicle_id,
            "validation_summary": {
                "total_sensors": total_sensors,
                "valid_sensors": valid_sensors,
                "invalid_sensors": total_sensors - valid_sensors,
                "validation_rate": validation_rate
            },
            "sensor_results": validation_results,
            "violations": violations,
            "warnings": warnings,
            "validation_timestamp": datetime.now().isoformat()
        }
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        
        if cache_key not in self._cache_expiry:
            return False
        
        return datetime.now() < self._cache_expiry[cache_key]
    
    def _process_sensor_data(self, bom_data: Dict[str, Any], sop_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process and combine BOM and SOP sensor data"""
        
        sensors = []
        
        # Extract sensors from BOM
        for component in bom_data.get("components", []):
            if component.get("type") == "sensor":
                sensor = {
                    "sensor_id": component.get("id"),
                    "name": component.get("name"),
                    "type": component.get("sensor_type"),
                    "manufacturer": component.get("manufacturer"),
                    "model": component.get("model"),
                    "specifications": component.get("specifications", {}),
                    "operational_bounds": component.get("operational_bounds", {}),
                    "calibration_data": component.get("calibration", {})
                }
                
                # Merge SOP data
                sop_sensor = next(
                    (s for s in sop_data.get("sensors", []) if s.get("id") == sensor["sensor_id"]), 
                    {}
                )
                
                if sop_sensor:
                    sensor.update({
                        "operating_procedures": sop_sensor.get("procedures", []),
                        "safety_limits": sop_sensor.get("safety_limits", {}),
                        "maintenance_schedule": sop_sensor.get("maintenance", {}),
                        "emergency_procedures": sop_sensor.get("emergency", [])
                    })
                
                sensors.append(sensor)
        
        return sensors
    
    def _validate_individual_sensor(self, sensor: Dict[str, Any]) -> Dict[str, Any]:
        """Validate individual sensor against bounds"""
        
        sensor_id = sensor.get("sensor_id", "unknown")
        bounds = sensor.get("operational_bounds", {})
        current_values = sensor.get("current_readings", {})
        
        violations = []
        warnings = []
        
        # Validate temperature bounds
        if "temperature" in bounds and "temperature" in current_values:
            temp_min = bounds["temperature"].get("min")
            temp_max = bounds["temperature"].get("max")
            current_temp = current_values["temperature"]
            
            if temp_min and current_temp < temp_min:
                violations.append(f"Temperature {current_temp}°C below minimum {temp_min}°C")
            elif temp_max and current_temp > temp_max:
                violations.append(f"Temperature {current_temp}°C above maximum {temp_max}°C")
            elif temp_min and current_temp < temp_min + 5:  # Warning threshold
                warnings.append(f"Temperature {current_temp}°C approaching minimum")
        
        # Validate voltage bounds  
        if "voltage" in bounds and "voltage" in current_values:
            volt_min = bounds["voltage"].get("min")
            volt_max = bounds["voltage"].get("max")
            current_volt = current_values["voltage"]
            
            if volt_min and current_volt < volt_min:
                violations.append(f"Voltage {current_volt}V below minimum {volt_min}V")
            elif volt_max and current_volt > volt_max:
                violations.append(f"Voltage {current_volt}V above maximum {volt_max}V")
        
        # Validate operational time
        if "max_operation_time" in bounds:
            max_time = bounds["max_operation_time"]
            current_time = current_values.get("operation_time", 0)
            
            if current_time > max_time:
                violations.append(f"Operation time {current_time}s exceeds maximum {max_time}s")
            elif current_time > max_time * 0.8:  # 80% warning threshold
                warnings.append(f"Operation time approaching maximum ({current_time}/{max_time}s)")
        
        return {
            "sensor_id": sensor_id,
            "sensor_name": sensor.get("name", "unknown"),
            "valid": len(violations) == 0,
            "violations": violations,
            "warnings": warnings,
            "validated_at": datetime.now().isoformat()
        }