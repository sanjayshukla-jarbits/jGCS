"""
Base Module - Abstract base class for all SkyrouteX modules

This module provides the foundation for all SkyrouteX modules with:
- Standardized execution interface
- Error handling and timing
- Parameter validation framework
- Result standardization
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

from src.models.process import ProcessStep
from src.orchestrator.execution_context import ExecutionContext
from src.utils.logger import get_logger

@dataclass
class ModuleResult:
    """Result object returned by module execution"""
    step_name: str
    module_name: str
    implementation: str
    success: bool
    execution_time: float
    data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    warnings: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format"""
        return {
            "step_name": self.step_name,
            "module_name": self.module_name,
            "implementation": self.implementation,
            "success": self.success,
            "execution_time": self.execution_time,
            "data": self.data,
            "error_message": self.error_message,
            "warnings": self.warnings
        }

class BaseModule(ABC):
    """Abstract base class for all SkyrouteX modules"""
    
    def __init__(self, module_name: str, implementation: str):
        self.module_name = module_name
        self.implementation = implementation
        self.logger = get_logger(f"{module_name}.{implementation}")
    
    async def execute(self, step_config: ProcessStep, context: ExecutionContext) -> ModuleResult:
        """Execute module with error handling and timing"""
        
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Starting {self.module_name}.{self.implementation} execution")
            
            # Validate parameters
            await self._validate_parameters(step_config.parameters)
            
            # Execute module-specific logic
            result_data = await self._execute_implementation(step_config, context)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(f"Completed {self.module_name}.{self.implementation} in {execution_time:.2f}s")
            
            return ModuleResult(
                step_name=step_config.name,
                module_name=self.module_name,
                implementation=self.implementation,
                success=True,
                execution_time=execution_time,
                data=result_data
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Module {self.module_name}.{self.implementation} failed: {str(e)}")
            
            return ModuleResult(
                step_name=step_config.name,
                module_name=self.module_name,
                implementation=self.implementation,
                success=False,
                execution_time=execution_time,
                error_message=str(e)
            )
    
    @abstractmethod
    async def _execute_implementation(self, step_config: ProcessStep, context: ExecutionContext) -> Dict[str, Any]:
        """Module-specific implementation logic - Must be implemented by subclasses"""
        pass
    
    @abstractmethod
    async def _validate_parameters(self, parameters: Dict[str, Any]) -> None:
        """Validate module-specific parameters - Must be implemented by subclasses"""
        pass
    
    def _get_parameter(self, parameters: Dict[str, Any], key: str, required: bool = True, default: Any = None) -> Any:
        """Helper method to get parameter with validation"""
        
        if key not in parameters:
            if required:
                raise ValueError(f"Required parameter '{key}' not found")
            return default
        
        return parameters[key]
    
    def _validate_required_parameters(self, parameters: Dict[str, Any], required_params: List[str]) -> None:
        """Helper method to validate required parameters exist"""
        
        missing_params = []
        for param in required_params:
            if param not in parameters:
                missing_params.append(param)
        
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")
    
    def _log_warning(self, message: str, warnings_list: List[str] = None) -> None:
        """Helper method to log warnings"""
        
        self.logger.warning(message)
        if warnings_list is not None:
            warnings_list.append(message)