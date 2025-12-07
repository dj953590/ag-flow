"""
Pydantic models for CDE extraction system
"""
from pydantic import BaseModel, Field, validator, ConfigDict
from typing import Dict, List, Any, Optional, Union, Literal
from enum import Enum
from datetime import datetime
from uuid import UUID, uuid4
import json

class ToolCategory(str, Enum):
    """Categories of available tools"""
    DOCUMENT_RETRIEVAL = "document_retrieval"
    DATA_EXTRACTION = "data_extraction"
    CALCULATION = "calculation"
    VALIDATION = "validation"
    SYNTHESIS = "synthesis"
    FORMATTING = "formatting"
    UTILITY = "utility"

class ToolCapability(str, Enum):
    """Capabilities of tools"""
    PAGE_SCANNING = "page_scanning"
    KEYWORD_SEARCH = "keyword_search"
    ENTITY_EXTRACTION = "entity_extraction"
    SEMANTIC_SEARCH = "semantic_search"
    DATE_CALCULATION = "date_calculation"
    MATH_CALCULATION = "math_calculation"
    VALIDATION = "validation"
    FORMATTING = "formatting"
    NORMALIZATION = "normalization"
    AGGREGATION = "aggregation"
    TRANSFORMATION = "transformation"

class ToolParameter(BaseModel):
    """Parameter definition for a tool"""
    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type (string, integer, boolean, list, dict)")
    description: str = Field(..., description="Parameter description")
    required: bool = Field(True, description="Whether parameter is required")
    default: Optional[Any] = Field(None, description="Default value")
    constraints: Optional[Dict[str, Any]] = Field(None, description="Parameter constraints")

    model_config = ConfigDict(frozen=True)

class MCPTool(BaseModel):
    """Definition of a tool available through MCP"""
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    category: ToolCategory = Field(..., description="Tool category")
    capabilities: List[ToolCapability] = Field(..., description="Tool capabilities")
    parameters: List[ToolParameter] = Field(default_factory=list, description="Tool parameters")
    mcp_server: str = Field(..., description="MCP server name")
    version: str = Field("1.0.0", description="Tool version")
    example_usage: Optional[str] = Field(None, description="Example usage")
    tags: List[str] = Field(default_factory=list, description="Tool tags")

    model_config = ConfigDict(frozen=True)

class WorkflowStep(BaseModel):
    """A step in the workflow execution plan"""
    step_id: str = Field(default_factory=lambda: f"step_{uuid4().hex[:8]}")
    step_number: int = Field(..., ge=1, description="Step execution order")
    objective: str = Field(..., description="Step objective")

    # Tool selection
    selected_tool: str = Field(..., description="Tool to execute")
    tool_justification: str = Field(..., description="Why this tool was selected")

    # Parameters
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")

    # Dependencies and flow
    dependencies: List[str] = Field(default_factory=list, description="Dependent steps")
    parallel_group: Optional[str] = Field(None, description="Parallel execution group")

    # Expected outputs
    expected_output_type: str = Field(..., description="Expected output type")
    output_mapping: Optional[Dict[str, str]] = Field(None, description="Output variable mapping")

    # Validation and error handling
    success_criteria: List[str] = Field(default_factory=list, description="Success criteria")
    fallback_tool: Optional[str] = Field(None, description="Fallback tool")
    retry_policy: Optional[Dict[str, Any]] = Field(None, description="Retry policy")

    # Metadata
    timeout: int = Field(60, ge=1, le=300, description="Step timeout in seconds")
    priority: int = Field(1, ge=1, le=10, description="Step priority")
    tags: List[str] = Field(default_factory=list, description="Step tags")

    @validator('dependencies')
    def validate_dependencies(cls, v, values):
        """Ensure step doesn't depend on itself"""
        if 'step_id' in values and values['step_id'] in v:
            raise ValueError("Step cannot depend on itself")
        return v

class SynthesisStrategy(BaseModel):
    """Strategy for synthesizing workflow results"""
    name: str = Field(..., description="Strategy name")
    description: str = Field(..., description="Strategy description")
    input_steps: List[str] = Field(..., description="Input step IDs")
    transformation_rules: List[Dict[str, Any]] = Field(default_factory=list, description="Transformation rules")
    validation_rules: List[Dict[str, Any]] = Field(default_factory=list, description="Validation rules")
    output_schema: Dict[str, Any] = Field(..., description="Output schema")
    confidence_calculation: Dict[str, Any] = Field(default_factory=dict, description="Confidence calculation method")

class WorkflowPlan(BaseModel):
    """Complete workflow plan for CDE extraction"""
    plan_id: str = Field(default_factory=lambda: f"plan_{uuid4().hex[:8]}")
    cde_name: str = Field(..., description="Critical Data Element name")
    created_at: datetime = Field(default_factory=datetime.now)
    version: str = Field("1.0.0", description="Plan version")

    # Context
    user_query: str = Field(..., description="User query")
    instructions: str = Field(..., description="Processing instructions")
    output_spec: Dict[str, Any] = Field(..., description="Output specification")

    # Available tools at generation time
    available_tools: List[MCPTool] = Field(..., description="Available tools")

    # Workflow structure
    steps: List[WorkflowStep] = Field(..., min_items=1, description="Workflow steps")
    synthesis: SynthesisStrategy = Field(..., description="Synthesis strategy")

    # Execution context
    document_context: Optional[Dict[str, Any]] = Field(None, description="Document context")
    assumptions: List[str] = Field(default_factory=list, description="Assumptions made")
    constraints: List[str] = Field(default_factory=list, description="Constraints")

    # Quality metrics
    estimated_confidence: float = Field(0.8, ge=0.0, le=1.0, description="Estimated confidence")
    estimated_duration: int = Field(..., ge=1, description="Estimated duration in seconds")
    complexity_score: float = Field(0.5, ge=0.0, le=1.0, description="Complexity score")
    risk_level: Literal["low", "medium", "high"] = Field("medium", description="Risk level")

    # Metadata
    tags: List[str] = Field(default_factory=list, description="Plan tags")
    author: str = Field("system", description="Plan author")
    notes: Optional[str] = Field(None, description="Additional notes")

    @validator('steps')
    def validate_steps(cls, v):
        """Validate step ordering and dependencies"""
        step_ids = [step.step_id for step in v]

        # Check for duplicate step numbers
        step_numbers = [step.step_number for step in v]
        if len(set(step_numbers)) != len(step_numbers):
            raise ValueError("Duplicate step numbers found")

        # Check dependencies exist
        for step in v:
            for dep in step.dependencies:
                if dep not in step_ids:
                    raise ValueError(f"Step {step.step_id} depends on non-existent step {dep}")

        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return json.loads(self.model_dump_json())

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return self.model_dump_json(indent=indent)

class ExecutionResult(BaseModel):
    """Result of a workflow step execution"""
    step_id: str
    workflow_id: str
    tool_used: str
    status: Literal["success", "partial", "failed", "skipped", "timeout"]
    output: Optional[Any] = None
    error: Optional[str] = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    execution_time: float = Field(..., ge=0.0, description="Execution time in seconds")
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('end_time', always=True)
    def set_end_time(cls, v, values):
        """Set end time if not provided"""
        if v is None and 'start_time' in values:
            return datetime.now()
        return v

class CDEExtractionResult(BaseModel):
    """Final CDE extraction result"""
    result_id: str = Field(default_factory=lambda: f"result_{uuid4().hex[:8]}")
    cde_name: str
    workflow_plan: WorkflowPlan
    extracted_data: Dict[str, Any]
    execution_summary: Dict[str, ExecutionResult]
    overall_confidence: float = Field(0.0, ge=0.0, le=1.0)
    validation_status: Literal["valid", "partial", "invalid", "pending"] = "pending"
    issues_found: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    execution_start: datetime = Field(default_factory=datetime.now)
    execution_end: Optional[datetime] = None
    total_execution_time: Optional[float] = Field(None, ge=0.0)

    @validator('total_execution_time', always=True)
    def calculate_total_time(cls, v, values):
        """Calculate total execution time"""
        if 'execution_end' in values and values['execution_end'] and 'execution_start' in values:
            return (values['execution_end'] - values['execution_start']).total_seconds()
        return v

    def to_report(self) -> Dict[str, Any]:
        """Generate a comprehensive report"""
        successful_steps = [
            r for r in self.execution_summary.values()
            if r.status == "success"
        ]

        return {
            "result_id": self.result_id,
            "cde_name": self.cde_name,
            "overall_confidence": self.overall_confidence,
            "validation_status": self.validation_status,
            "successful_steps": len(successful_steps),
            "total_steps": len(self.execution_summary),
            "total_time": self.total_execution_time,
            "extracted_fields": list(self.extracted_data.keys()),
            "issues": self.issues_found,
            "warnings": self.warnings,
            "suggestions": self.suggestions
        }

# Event Models
class WorkflowEvent(BaseModel):
    """Workflow execution event"""
    event_id: str = Field(default_factory=lambda: f"event_{uuid4().hex[:8]}")
    event_type: str = Field(..., description="Event type")
    workflow_id: str = Field(..., description="Workflow ID")
    step_id: Optional[str] = Field(None, description="Step ID")
    timestamp: datetime = Field(default_factory=datetime.now)
    data: Dict[str, Any] = Field(default_factory=dict, description="Event data")
    severity: Literal["info", "warning", "error", "critical"] = "info"

    model_config = ConfigDict(extra="allow")

class ToolRegistry(BaseModel):
    """Registry of all available tools"""
    tools: Dict[str, MCPTool] = Field(default_factory=dict, description="Tools by name")
    servers: Dict[str, List[str]] = Field(default_factory=dict, description="Tools by server")
    categories: Dict[str, List[str]] = Field(default_factory=dict, description="Tools by category")
    capabilities: Dict[str, List[str]] = Field(default_factory=dict, description="Tools by capability")

    def register_tool(self, tool: MCPTool):
        """Register a new tool"""
        self.tools[tool.name] = tool

        # Update indices
        if tool.mcp_server not in self.servers:
            self.servers[tool.mcp_server] = []
        self.servers[tool.mcp_server].append(tool.name)

        if tool.category.value not in self.categories:
            self.categories[tool.category.value] = []
        self.categories[tool.category.value].append(tool.name)

        for capability in tool.capabilities:
            if capability.value not in self.capabilities:
                self.capabilities[capability.value] = []
            self.capabilities[capability.value].append(tool.name)

    def find_tools_by_capability(self, capability: ToolCapability) -> List[MCPTool]:
        """Find tools with specific capability"""
        return [self.tools[name] for name in self.capabilities.get(capability.value, [])]

    def find_tools_by_category(self, category: ToolCategory) -> List[MCPTool]:
        """Find tools by category"""
        return [self.tools[name] for name in self.categories.get(category.value, [])]