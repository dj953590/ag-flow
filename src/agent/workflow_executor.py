# agent/workflow_executor.py
"""
Executes workflow plans using MCP tools
"""
import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import traceback

from .models import *
from .mcp_agent import MCPToolManager

class WorkflowExecutor:
    """Executes workflow plans using MCP tools"""

    def __init__(self, tool_manager: MCPToolManager):
        self.tool_manager = tool_manager
        self.execution_cache: Dict[str, Any] = {}

    async def execute_workflow(self, plan: WorkflowPlan) -> Dict[str, ExecutionResult]:
        """Execute a workflow plan"""

        print(f"\n⚡ Executing workflow: {plan.cde_name}")

        results: Dict[str, ExecutionResult] = {}
        step_queue = asyncio.Queue()

        # Initialize queue with steps that have no dependencies
        initial_steps = [
            step for step in plan.steps
            if not step.dependencies
        ]

        for step in initial_steps:
            await step_queue.put(step)

        # Track completed steps
        completed_steps: Set[str] = set()

        # Process steps
        while not step_queue.empty():
            step = await step_queue.get()

            # Check if dependencies are satisfied
            if not all(dep in completed_steps for dep in step.dependencies):
                # Requeue for later
                await step_queue.put(step)
                await asyncio.sleep(0.1)  # Small delay to prevent busy loop
                continue

            # Execute step
            result = await self._execute_step(step, plan)
            results[step.step_id] = result

            if result.status == "success":
                completed_steps.add(step.step_id)

                # Add dependent steps to queue
                dependent_steps = [
                    s for s in plan.steps
                    if step.step_id in s.dependencies and s not in list(step_queue._queue)
                ]

                for dep_step in dependent_steps:
                    await step_queue.put(dep_step)

            elif result.status == "failed" and step.fallback_tool:
                # Try fallback
                print(f"  Trying fallback for {step.step_id}...")
                fallback_result = await self._execute_fallback(step, plan)
                results[f"{step.step_id}_fallback"] = fallback_result

                if fallback_result.status == "success":
                    completed_steps.add(step.step_id)

        return results

    async def _execute_step(
            self,
            step: WorkflowStep,
            plan: WorkflowPlan
    ) -> ExecutionResult:
        """Execute a single workflow step"""

        start_time = datetime.now()

        try:
            print(f"  Executing {step.step_id}: {step.objective}")

            # Get tool information
            tool = self.tool_manager.get_tool(step.selected_tool)
            if not tool:
                return ExecutionResult(
                    step_id=step.step_id,
                    tool_used=step.selected_tool,
                    status="failed",
                    error=f"Tool {step.selected_tool} not found",
                    confidence=0.0,
                    execution_time=(datetime.now() - start_time).total_seconds()
                )

            # Prepare parameters
            params = step.parameters.copy()
            params["document_id"] = plan.document_context.get("document_id")

            # Call tool through MCP
            raw_result = await self.tool_manager.call_tool(step.selected_tool, params)

            # Parse and validate result
            parsed_result = self._parse_tool_result(raw_result, step)

            # Calculate confidence
            confidence = self._calculate_confidence(parsed_result, step)

            return ExecutionResult(
                step_id=step.step_id,
                tool_used=step.selected_tool,
                status="success",
                output=raw_result,
                confidence=confidence,
                execution_time=(datetime.now() - start_time).total_seconds()
            )

        except Exception as e:
            return ExecutionResult(
                step_id=step.step_id,
                tool_used=step.selected_tool or "unknown",
                status="failed",
                error=f"{type(e).__name__}: {str(e)}",
                confidence=0.0,
                execution_time=(datetime.now() - start_time).total_seconds()
            )

    async def _execute_fallback(
            self,
            step: WorkflowStep,
            plan: WorkflowPlan
    ) -> ExecutionResult:
        """Execute fallback for a failed step"""

        if not step.fallback_tool:
            return ExecutionResult(
                step_id=step.step_id,
                tool_used="none",
                status="failed",
                error="No fallback tool specified",
                confidence=0.0,
                execution_time=0.0
            )

        # Create modified step with fallback tool
        fallback_step = WorkflowStep(
            step_id=f"{step.step_id}_fallback",
            step_number=step.step_number,
            objective=f"Fallback for: {step.objective}",
            selected_tool=step.fallback_tool,
            tool_justification=f"Fallback after {step.selected_tool} failed",
            parameters=step.parameters,
            dependencies=step.dependencies,
            expected_output_type=step.expected_output_type
        )

        return await self._execute_step(fallback_step, plan)

    def _parse_tool_result(self, raw_result: str, step: WorkflowStep) -> Dict[str, Any]:
        """Parse and validate tool result"""
        try:
            result = json.loads(raw_result)

            # Check for errors in result
            if isinstance(result, dict) and "error" in result:
                raise ValueError(f"Tool returned error: {result['error']}")

            return result

        except json.JSONDecodeError:
            # Try to extract structured information from text
            return {"raw_output": raw_result, "parse_warning": "Could not parse as JSON"}

    def _calculate_confidence(
            self,
            result: Dict[str, Any],
            step: WorkflowStep
    ) -> float:
        """Calculate confidence score for execution result"""

        confidence = 0.7  # Base confidence

        # Adjust based on result characteristics
        if "error" not in result:
            confidence += 0.2

        # Check if result has expected structure
        if step.expected_output_type == "entities" and "entities_found" in result:
            entity_count = sum(
                len(v) for v in result.get("entities_found", {}).values()
            )
            if entity_count > 0:
                confidence += 0.1

        elif step.expected_output_type == "dates" and "dates_found" in result:
            if result["dates_found"] > 0:
                confidence += 0.1

        # Cap at 1.0
        return min(confidence, 1.0)