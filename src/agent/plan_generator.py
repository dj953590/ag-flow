# agent/plan_generator.py
"""
Generates structured workflow plans using LLM with MCP tool context
"""
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

import instructor
from openai import AsyncOpenAI

from .models import *

class StructuredPlanGenerator:
    """Generates structured workflow plans using LLM with MCP context"""

    def __init__(self, llm_client: AsyncOpenAI):
        self.client = llm_client

    async def generate_plan(
            self,
            cde_prompt: Dict[str, Any],
            context: Dict[str, Any]
    ) -> WorkflowPlan:
        """Generate a workflow plan for CDE extraction"""

        # Prepare prompt with tool context
        prompt = self._create_plan_prompt(cde_prompt, context)

        # Generate plan using LLM with structured output
        plan = await self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            response_model=WorkflowPlan,
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert workflow designer for financial document analysis.
                    Your task is to create optimal workflows for extracting Critical Data Elements (CDEs)
                    from credit agreements using the available tools.
                    
                    Key principles:
                    1. Always start with document context retrieval
                    2. Use specialized tools for specific tasks
                    3. Build progressively from basic to complex extraction
                    4. Include validation steps
                    5. Consider parallel execution where possible
                    6. Handle edge cases and fallbacks"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_retries=3
        )

        # Post-process the plan
        plan = self._enhance_plan(plan, context)

        return plan

    def _create_plan_prompt(
            self,
            cde_prompt: Dict[str, Any],
            context: Dict[str, Any]
    ) -> str:
        """Create prompt for plan generation"""

        user_query = cde_prompt.get("user_query", "")
        instructions = cde_prompt.get("instructions", "")
        reasoning = cde_prompt.get("step_by_step_reasoning", "")
        output_spec = cde_prompt.get("output", {})

        # Format available tools
        tools_info = []
        for tool in context.get("available_tools", []):
            capabilities = context["tool_capabilities"].get(tool["name"], [])

            tools_info.append(f"""
            Tool: {tool['name']}
            Description: {tool['description']}
            Category: {tool['category']}
            Capabilities: {', '.join(capabilities)}
            Parameters: {', '.join([p['name'] for p in tool.get('parameters', [])])}
            Example: {tool.get('example_usage', 'N/A')}
            """)

        tools_section = "\n".join(tools_info)

        prompt = f"""
        CDE EXTRACTION WORKFLOW DESIGN REQUEST
        
        ========================
        EXTRACTION REQUIREMENTS
        ========================
        
        User Query: {user_query}
        
        Specific Instructions:
        {instructions}
        
        Step-by-step Reasoning Provided:
        {reasoning}
        
        Expected Output Format:
        {json.dumps(output_spec, indent=2)}
        
        ========================
        AVAILABLE TOOLS
        ========================
        
        The following tools are available through MCP servers:
        
        {tools_section}
        
        ========================
        DOCUMENT CONTEXT
        ========================
        
        Document ID: {context.get('document_id', 'Unknown')}
        
        ========================
        WORKFLOW DESIGN TASK
        ========================
        
        Design an optimal workflow plan that:
        1. Uses the available tools in the best sequence
        2. Specifies exact parameters for each tool call
        3. Identifies dependencies between steps
        4. Suggests parallel execution where possible
        5. Includes validation and fallback strategies
        6. Estimates confidence and complexity
        
        Consider:
        - Start with document context retrieval
        - Use entity extraction to identify key elements
        - Use calculation tools for date operations
        - Include validation steps
        - Plan for synthesis of results
        
        Provide a complete, executable workflow plan.
        """

        return prompt

    def _enhance_plan(self, plan: WorkflowPlan, context: Dict[str, Any]) -> WorkflowPlan:
        """Enhance the generated plan with additional metadata"""

        # Add missing step IDs if needed
        for i, step in enumerate(plan.steps):
            if not step.step_id:
                step.step_id = f"step_{i+1:03d}"

        # Estimate duration based on steps
        plan.estimated_duration = len(plan.steps)

        # Add document context
        plan.document_context = {
            "document_id": context.get("document_id"),
            "tool_count": len(context.get("available_tools", [])),
            "generation_timestamp": datetime.now().isoformat()
        }

        return plan