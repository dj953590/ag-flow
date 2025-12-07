# agent/mcp_agent.py
"""
MCP-aware agent with dynamic tool discovery and workflow generation
"""
import asyncio
import json
import os

import yaml
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import subprocess
import sys

from mcp import ClientSession, StdioServerParameters
import instructor
from openai import AsyncOpenAI

from .models import *
from .plan_generator import StructuredPlanGenerator
from .workflow_executor import WorkflowExecutor

class MCPToolManager:
    """Manages MCP tool discovery and lifecycle"""

    def __init__(self, config_path: str = "config/mcp_config.yaml"):
        self.config_path = config_path
        self.servers: Dict[str, subprocess.Popen] = {}
        self.sessions: Dict[str, ClientSession] = {}
        self.tools: List[MCPTool] = []
        self.tool_capabilities: Dict[str, Set[ToolCapability]] = {}

    async def initialize(self):
        """Start all MCP servers and discover tools"""
        print("🚀 Initializing MCP Tool Manager...")

        # Load configuration
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Start MCP servers
        server_configs = config.get('mcp_servers', {})

        for server_name, server_config in server_configs.items():
            try:
                print(f"  Starting {server_name}...")
                await self._start_mcp_server(server_name, server_config)
            except Exception as e:
                print(f"  ❌ Failed to start {server_name}: {e}")

        # Discover tools from all servers
        await self._discover_tools()

        print(f"✅ MCP Tool Manager initialized with {len(self.tools)} tools")

    async def _start_mcp_server(self, name: str, config: Dict[str, Any]):
        """Start an MCP server process"""
        command = config['command']
        args = config.get('args', [])
        env = {**config.get('env', {}), **dict(os.environ)}

        # Start server process
        process = subprocess.Popen(
            [command] + args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1
        )

        self.servers[name] = process

        # Create MCP client session
        session = ClientSession(
            StdioServerParameters(
                command=command,
                args=args,
                env=env
            )
        )

        await session.initialize()
        self.sessions[name] = session

    async def _discover_tools(self):
        """Discover all tools from all MCP servers"""
        self.tools = []

        for server_name, session in self.sessions.items():
            try:
                # List tools from this server
                mcp_tools = await session.list_tools()

                for mcp_tool in mcp_tools:
                    # Convert MCP tool to our model
                    tool = MCPTool(
                        name=mcp_tool.name,
                        description=mcp_tool.description or "No description available",
                        category=self._infer_category(mcp_tool.name, mcp_tool.description),
                        capabilities=self._infer_capabilities(mcp_tool.name, mcp_tool.description),
                        parameters=[
                            ToolParameter(
                                name=param.name,
                                type=param.json_schema.get("type", "string"),
                                description=param.description or "",
                                required=param.required,
                                default_value=param.json_schema.get("default")
                            )
                            for param in (mcp_tool.inputSchema or [])
                        ],
                        mcp_server=server_name,
                        example_usage=self._generate_example_usage(mcp_tool)
                    )

                    self.tools.append(tool)

                    # Track capabilities
                    self.tool_capabilities[tool.name] = set(tool.capabilities)

            except Exception as e:
                print(f"  ❌ Failed to discover tools from {server_name}: {e}")

    def _infer_category(self, name: str, description: str) -> ToolCategory:
        """Infer tool category from name and description"""
        desc_lower = (description or "").lower()
        name_lower = name.lower()

        if any(word in name_lower or word in desc_lower
               for word in ["retrieve", "search", "extract", "scan"]):
            return ToolCategory.DOCUMENT_RETRIEVAL
        elif any(word in name_lower or word in desc_lower
                 for word in ["calculate", "compute", "date", "offset"]):
            return ToolCategory.CALCULATION
        elif any(word in name_lower or word in desc_lower
                 for word in ["validate", "check", "verify"]):
            return ToolCategory.VALIDATION
        else:
            return ToolCategory.DATA_EXTRACTION

    def _infer_capabilities(self, name: str, description: str) -> List[ToolCapability]:
        """Infer tool capabilities from name and description"""
        capabilities = []
        desc_lower = (description or "").lower()
        name_lower = name.lower()

        capability_mapping = {
            "page": ToolCapability.PAGE_SCANNING,
            "first": ToolCapability.PAGE_SCANNING,
            "keyword": ToolCapability.KEYWORD_SEARCH,
            "search": ToolCapability.KEYWORD_SEARCH,
            "entity": ToolCapability.ENTITY_EXTRACTION,
            "extract": ToolCapability.ENTITY_EXTRACTION,
            "semantic": ToolCapability.SEMANTIC_SEARCH,
            "similar": ToolCapability.SEMANTIC_SEARCH,
            "date": ToolCapability.DATE_CALCULATION,
            "calculate": ToolCapability.DATE_CALCULATION,
            "validate": ToolCapability.VALIDATION,
            "check": ToolCapability.VALIDATION,
            "format": ToolCapability.FORMATTING
        }

        for keyword, capability in capability_mapping.items():
            if keyword in name_lower or keyword in desc_lower:
                capabilities.append(capability)

        return list(set(capabilities))

    def _generate_example_usage(self, mcp_tool) -> str:
        """Generate example usage for a tool"""
        params = []
        for param in (mcp_tool.inputSchema or []):
            example = {
                "string": "'example_value'",
                "integer": "123",
                "boolean": "true",
                "array": "['item1', 'item2']"
            }.get(param.json_schema.get("type", "string"), "'value'")

            params.append(f"{param.name}={example}")

        return f"{mcp_tool.name}({', '.join(params)})"

    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Call a tool by name with parameters"""
        # Find which server has this tool
        for server_name, session in self.sessions.items():
            try:
                tools = await session.list_tools()
                if any(t.name == tool_name for t in tools):
                    # Call the tool
                    result = await session.call_tool(tool_name, arguments=parameters)
                    return result
            except Exception:
                continue

        raise ValueError(f"Tool {tool_name} not found in any MCP server")

    def get_tools_by_capability(self, capability: ToolCapability) -> List[MCPTool]:
        """Get all tools that have a specific capability"""
        return [tool for tool in self.tools if capability in tool.capabilities]

    def get_tool(self, name: str) -> Optional[MCPTool]:
        """Get a specific tool by name"""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    async def shutdown(self):
        """Shutdown all MCP servers"""
        print("🛑 Shutting down MCP Tool Manager...")

        for session in self.sessions.values():
            await session.close()

        for name, process in self.servers.items():
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

class MCPEnhancedCDEExtractor:
    """
    Main agent that uses MCP tools to extract Critical Data Elements
    """

    def __init__(self, config_path: str = "config/mcp_config.yaml"):
        self.config_path = config_path
        self.tool_manager = MCPToolManager(config_path)

        # Load LLM configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        llm_config = config.get('llm', {})
        self.llm_client = AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            **{k: v for k, v in llm_config.items() if k != 'model'}
        )

        # Patch client with instructor
        self.client = instructor.patch(self.llm_client)

        # Initialize components
        self.plan_generator = StructuredPlanGenerator(self.client)
        self.workflow_executor = WorkflowExecutor(self.tool_manager)

        # Cache for generated plans
        self.plan_cache: Dict[str, WorkflowPlan] = {}

    async def initialize(self):
        """Initialize the agent and all components"""
        print("🔧 Initializing CDE Extractor Agent...")
        await self.tool_manager.initialize()
        print("✅ Agent initialized and ready")

    async def extract_cde(
            self,
            cde_prompt: Dict[str, Any],
            document_id: str = "credit_agreement_001"
    ) -> CDEExtractionResult:
        """
        Extract a Critical Data Element using dynamically generated workflow

        Args:
            cde_prompt: CDE extraction prompt with query, instructions, reasoning
            document_id: Document to analyze

        Returns:
            Extraction result with confidence and validation
        """
        print(f"\n🎯 Extracting CDE: {cde_prompt.get('user_query', 'Unknown')}")
        print("=" * 60)

        # Step 1: Generate workflow plan with current tool context
        workflow_plan = await self._generate_workflow_plan(cde_prompt, document_id)

        # Step 2: Execute the workflow
        execution_results = await self.workflow_executor.execute_workflow(workflow_plan)

        # Step 3: Synthesize final result
        final_result = await self._synthesize_results(
            cde_prompt,
            workflow_plan,
            execution_results
        )

        return final_result

    async def _generate_workflow_plan(
            self,
            cde_prompt: Dict[str, Any],
            document_id: str
    ) -> WorkflowPlan:
        """Generate a workflow plan using current tool context"""

        # Create cache key
        cache_key = f"{cde_prompt.get('user_query')}_{document_id}"

        if cache_key in self.plan_cache:
            print("  Using cached workflow plan")
            return self.plan_cache[cache_key]

        print("  Generating new workflow plan...")

        # Prepare context with available tools
        context = {
            "available_tools": [tool.dict() for tool in self.tool_manager.tools],
            "document_id": document_id,
            "tool_capabilities": {
                tool.name: [c.value for c in tool.capabilities]
                for tool in self.tool_manager.tools
            }
        }

        # Generate plan
        plan = await self.plan_generator.generate_plan(cde_prompt, context)

        # Cache the plan
        self.plan_cache[cache_key] = plan

        # Display plan summary
        self._display_plan_summary(plan)

        return plan

    def _display_plan_summary(self, plan: WorkflowPlan):
        """Display a summary of the generated plan"""
        print(f"\n📋 Generated Workflow Plan: {plan.cde_name}")
        print(f"   Steps: {len(plan.steps)} | Estimated confidence: {plan.estimated_confidence:.2f}")

        for step in plan.steps:
            deps = f" (depends on: {', '.join(step.dependencies)})" if step.dependencies else ""
            print(f"   Step {step.step_number}: {step.objective}")
            print(f"     → Tool: {step.selected_tool}{deps}")

        print(f"   Synthesis: {plan.synthesis.strategy}")

    async def _synthesize_results(
            self,
            cde_prompt: Dict[str, Any],
            workflow_plan: WorkflowPlan,
            execution_results: Dict[str, ExecutionResult]
    ) -> CDEExtractionResult:
        """Synthesize final result from execution results"""

        # Extract successful results
        successful_results = {
            step_id: result
            for step_id, result in execution_results.items()
            if result.status == "success" and result.output
        }

        # Calculate overall confidence
        confidences = [
            result.confidence
            for result in successful_results.values()
            if result.confidence > 0
        ]

        overall_confidence = (
            sum(confidences) / len(confidences)
            if confidences else 0.0
        )

        # Extract data from results
        extracted_data = await self._extract_and_format_data(
            successful_results,
            cde_prompt.get("output", {})
        )

        # Create final result
        result = CDEExtractionResult(
            cde_name=workflow_plan.cde_name,
            extracted_data=extracted_data,
            workflow_plan=workflow_plan,
            execution_summary=execution_results,
            overall_confidence=overall_confidence,
            validation_status=(
                "valid" if overall_confidence >= 0.8
                else "partial" if overall_confidence >= 0.5
                else "invalid"
            )
        )

        return result

    async def _extract_and_format_data(
            self,
            results: Dict[str, ExecutionResult],
            output_spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract and format data according to output specification"""

        # This would use the synthesis strategy from the workflow plan
        # For simplicity, we'll extract facilities and dates from results

        facilities = []

        for step_id, result in results.items():
            if result.output:
                try:
                    data = json.loads(result.output)

                    # Look for facilities in entity extraction results
                    if "extracted_entities" in data:
                        entities = data["extracted_entities"]
                        if "facility" in entities:
                            for facility in entities["facility"]:
                                facilities.append({
                                    "name": facility["entity"],
                                    "source_step": step_id,
                                    "confidence": result.confidence
                                })

                    # Look for dates
                    if "extracted_dates" in data:
                        for date_info in data["extracted_dates"]:
                            if date_info["date_type"] == "maturity":
                                # Try to match with facilities
                                pass

                except json.JSONDecodeError:
                    # Try to extract from raw text
                    if "facility" in result.output.lower() or "loan" in result.output.lower():
                        # Simple pattern matching
                        import re
                        facility_matches = re.findall(
                            r'(Term [A-B] Loan|Revolving Facility)',
                            result.output
                        )
                        for match in facility_matches:
                            facilities.append({
                                "name": match,
                                "source_step": step_id,
                                "confidence": result.confidence * 0.9  # Reduce confidence for regex extraction
                            })

        # Format according to output spec
        formatted_result = {
            "facilities": [
                {
                    "facility": facility["name"],
                    "maturity_date": "To be extracted",  # Would come from date extraction
                    "extraction_confidence": facility["confidence"],
                    "source": facility["source_step"]
                }
                for facility in facilities
            ],
            "extraction_timestamp": datetime.now().isoformat(),
            "sources_used": list(results.keys()),
            "validation_notes": "Extracted from multiple tool outputs"
        }

        return formatted_result

    async def shutdown(self):
        """Shutdown the agent"""
        await self.tool_manager.shutdown()