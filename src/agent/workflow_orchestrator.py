# agent/workflow_orchestrator.py
"""
LlamaIndex AgentWorkflow-based orchestrator with step-by-step execution and event handling
"""
from typing import Dict, List, Any, Optional, Callable, AsyncGenerator
from enum import Enum
import asyncio
import json
from datetime import datetime

from llama_index.core.agent import (
    AgentWorkflow,
    WorkflowStep,
    WorkflowResponse,
    StepResult
)
from llama_index.core.tools import BaseTool
from llama_index.core.callbacks import CallbackManager
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    EventStream
)

from .models import *
from .mcp_agent import MCPToolManager

class WorkflowEventType(str, Enum):
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"
    TOOL_CALLED = "tool_called"
    TOOL_RESULT = "tool_result"
    PARALLEL_STARTED = "parallel_started"
    PARALLEL_COMPLETED = "parallel_completed"
    SYNTHESIS_STARTED = "synthesis_started"
    SYNTHESIS_COMPLETED = "synthesis_completed"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"

class WorkflowEvent(Event):
    """Custom workflow events"""
    event_type: WorkflowEventType
    step_id: Optional[str] = None
    data: Dict[str, Any] = {}
    timestamp: datetime = Field(default_factory=datetime.now)

class MCPToolStep(WorkflowStep):
    """Workflow step that calls MCP tools"""

    def __init__(
            self,
            step_id: str,
            tool_name: str,
            parameters: Dict[str, Any],
            tool_manager: MCPToolManager,
            **kwargs
    ):
        super().__init__(step_id, **kwargs)
        self.tool_name = tool_name
        self.parameters = parameters
        self.tool_manager = tool_manager

    async def run(self, context: Context) -> StepResult:
        """Execute the step by calling the MCP tool"""

        # Emit step started event
        await context.emit(WorkflowEvent(
            event_type=WorkflowEventType.STEP_STARTED,
            step_id=self.step_id,
            data={
                "tool": self.tool_name,
                "parameters": self.parameters,
                "timestamp": datetime.now().isoformat()
            }
        ))

        try:
            # Emit tool called event
            await context.emit(WorkflowEvent(
                event_type=WorkflowEventType.TOOL_CALLED,
                step_id=self.step_id,
                data={
                    "tool": self.tool_name,
                    "parameters": self.parameters
                }
            ))

            # Call MCP tool
            start_time = datetime.now()
            result = await self.tool_manager.call_tool(
                self.tool_name,
                self.parameters
            )
            execution_time = (datetime.now() - start_time).total_seconds()

            # Emit tool result event
            await context.emit(WorkflowEvent(
                event_type=WorkflowEventType.TOOL_RESULT,
                step_id=self.step_id,
                data={
                    "tool": self.tool_name,
                    "execution_time": execution_time,
                    "result_preview": str(result)[:500] if result else None,
                    "success": True
                }
            ))

            # Parse and return result
            parsed_result = self._parse_result(result)

            return StepResult(
                step_id=self.step_id,
                output=parsed_result,
                metadata={
                    "execution_time": execution_time,
                    "tool": self.tool_name,
                    "success": True
                }
            )

        except Exception as e:
            # Emit failure event
            await context.emit(WorkflowEvent(
                event_type=WorkflowEventType.STEP_FAILED,
                step_id=self.step_id,
                data={
                    "tool": self.tool_name,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            ))

            return StepResult(
                step_id=self.step_id,
                output={"error": str(e)},
                metadata={
                    "success": False,
                    "error": str(e)
                }
            )

    def _parse_result(self, raw_result: Any) -> Dict[str, Any]:
        """Parse MCP tool result"""
        try:
            if isinstance(raw_result, str):
                return json.loads(raw_result)
            elif isinstance(raw_result, dict):
                return raw_result
            else:
                return {"raw_output": str(raw_result)}
        except json.JSONDecodeError:
            return {"raw_output": raw_result}

class ParallelStepGroup(WorkflowStep):
    """Group of steps that can execute in parallel"""

    def __init__(
            self,
            group_id: str,
            steps: List[WorkflowStep],
            **kwargs
    ):
        super().__init__(group_id, **kwargs)
        self.steps = steps

    async def run(self, context: Context) -> StepResult:
        """Execute steps in parallel"""

        # Emit parallel started event
        await context.emit(WorkflowEvent(
            event_type=WorkflowEventType.PARALLEL_STARTED,
            step_id=self.step_id,
            data={
                "steps_count": len(self.steps),
                "step_ids": [s.step_id for s in self.steps]
            }
        ))

        try:
            # Execute all steps in parallel
            tasks = [step.run(context) for step in self.steps]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            outputs = {}
            successes = []
            failures = []

            for step, result in zip(self.steps, results):
                if isinstance(result, Exception):
                    failures.append({
                        "step_id": step.step_id,
                        "error": str(result)
                    })
                else:
                    successes.append(step.step_id)
                    outputs[step.step_id] = result.output

            # Emit parallel completed event
            await context.emit(WorkflowEvent(
                event_type=WorkflowEventType.PARALLEL_COMPLETED,
                step_id=self.step_id,
                data={
                    "successful_steps": successes,
                    "failed_steps": failures,
                    "total_steps": len(self.steps)
                }
            ))

            return StepResult(
                step_id=self.step_id,
                output=outputs,
                metadata={
                    "successes": len(successes),
                    "failures": len(failures),
                    "is_parallel": True
                }
            )

        except Exception as e:
            await context.emit(WorkflowEvent(
                event_type=WorkflowEventType.STEP_FAILED,
                step_id=self.step_id,
                data={
                    "error": str(e),
                    "is_parallel": True
                }
            ))
            raise

class SynthesisStep(WorkflowStep):
    """Step that synthesizes results from multiple steps"""

    def __init__(
            self,
            step_id: str,
            input_step_ids: List[str],
            synthesis_strategy: str,
            **kwargs
    ):
        super().__init__(step_id, **kwargs)
        self.input_step_ids = input_step_ids
        self.synthesis_strategy = synthesis_strategy

    async def run(self, context: Context) -> StepResult:
        """Synthesize results from previous steps"""

        await context.emit(WorkflowEvent(
            event_type=WorkflowEventType.SYNTHESIS_STARTED,
            step_id=self.step_id,
            data={
                "input_steps": self.input_step_ids,
                "strategy": self.synthesis_strategy
            }
        ))

        try:
            # Collect inputs from context
            inputs = {}
            for step_id in self.input_step_ids:
                step_result = context.state.get(f"step_{step_id}")
                if step_result:
                    inputs[step_id] = step_result.output

            # Apply synthesis strategy
            synthesized = await self._apply_synthesis_strategy(inputs)

            await context.emit(WorkflowEvent(
                event_type=WorkflowEventType.SYNTHESIS_COMPLETED,
                step_id=self.step_id,
                data={
                    "input_count": len(inputs),
                    "output_keys": list(synthesized.keys()),
                    "strategy": self.synthesis_strategy
                }
            ))

            return StepResult(
                step_id=self.step_id,
                output=synthesized,
                metadata={
                    "synthesis_strategy": self.synthesis_strategy,
                    "input_steps": self.input_step_ids
                }
            )

        except Exception as e:
            await context.emit(WorkflowEvent(
                event_type=WorkflowEventType.STEP_FAILED,
                step_id=self.step_id,
                data={
                    "error": str(e),
                    "step_type": "synthesis"
                }
            ))
            raise

    async def _apply_synthesis_strategy(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Apply synthesis strategy to inputs"""
        # This would be more sophisticated in production
        # For now, combine all results
        combined = {
            "synthesized_at": datetime.now().isoformat(),
            "strategy_used": self.synthesis_strategy,
            "inputs_combined": len(inputs)
        }

        # Merge all outputs
        for step_id, output in inputs.items():
            if isinstance(output, dict):
                for key, value in output.items():
                    combined[f"{step_id}_{key}"] = value
            else:
                combined[step_id] = output

        return combined

class AgentWorkflowOrchestrator:
    """
    Orchestrates workflow execution using LlamaIndex AgentWorkflow
    """

    def __init__(
            self,
            tool_manager: MCPToolManager,
            callback_manager: Optional[CallbackManager] = None
    ):
        self.tool_manager = tool_manager
        self.callback_manager = callback_manager
        self.workflow_registry: Dict[str, AgentWorkflow] = {}
        self.event_streams: Dict[str, EventStream] = {}

    async def create_workflow_from_plan(
            self,
            plan: WorkflowPlan,
            workflow_id: Optional[str] = None
    ) -> AgentWorkflow:
        """Create AgentWorkflow from a structured plan"""

        workflow_id = workflow_id or plan.plan_id

        if workflow_id in self.workflow_registry:
            return self.workflow_registry[workflow_id]

        print(f"Creating AgentWorkflow from plan: {plan.cde_name}")

        # Create steps from plan
        steps: List[WorkflowStep] = []

        # Group steps for parallel execution
        parallel_groups = self._identify_parallel_groups(plan)

        # Create workflow steps
        for step in plan.steps:
            if step.step_id in parallel_groups:
                # This step is part of a parallel group
                continue

            # Check if this step is the start of a parallel group
            if any(step.step_id == group[0] for group in parallel_groups.values()):
                group_steps = []
                group_id = f"parallel_group_{step.step_id}"

                # Find all steps in this parallel group
                for step_in_group in plan.steps:
                    if step_in_group.step_id in parallel_groups.get(step.step_id, []):
                        group_steps.append(self._create_mcp_step(step_in_group))

                if len(group_steps) > 1:
                    parallel_step = ParallelStepGroup(
                        step_id=group_id,
                        steps=group_steps
                    )
                    steps.append(parallel_step)
                else:
                    # Single step, just add it directly
                    steps.append(self._create_mcp_step(step))
            else:
                # Regular sequential step
                steps.append(self._create_mcp_step(step))

        # Add synthesis step
        if plan.synthesis:
            synthesis_step = SynthesisStep(
                step_id="synthesis",
                input_step_ids=[s.step_id for s in plan.steps],
                synthesis_strategy=plan.synthesis.strategy
            )
            steps.append(synthesis_step)

        # Create workflow definition
        workflow_definition = self._create_workflow_definition(steps)

        # Create AgentWorkflow
        workflow = AgentWorkflow.from_steps(
            steps=steps,
            workflow=workflow_definition,
            callback_manager=self.callback_manager,
            verbose=True
        )

        self.workflow_registry[workflow_id] = workflow
        return workflow

    def _create_mcp_step(self, step: WorkflowStep) -> MCPToolStep:
        """Create MCPToolStep from workflow step definition"""
        return MCPToolStep(
            step_id=step.step_id,
            tool_name=step.selected_tool,
            parameters=step.parameters,
            tool_manager=self.tool_manager,
            description=step.objective
        )

    def _identify_parallel_groups(self, plan: WorkflowPlan) -> Dict[str, List[str]]:
        """Identify steps that can run in parallel"""
        parallel_groups = {}

        # Group steps with same dependencies
        dependency_groups = {}

        for step in plan.steps:
            dep_key = tuple(sorted(step.dependencies))
            if dep_key not in dependency_groups:
                dependency_groups[dep_key] = []
            dependency_groups[dep_key].append(step.step_id)

        # Create parallel groups for groups with >1 step
        for dep_key, step_ids in dependency_groups.items():
            if len(step_ids) > 1:
                first_step = step_ids[0]
                parallel_groups[first_step] = step_ids

        return parallel_groups

    def _create_workflow_definition(self, steps: List[WorkflowStep]) -> List[Dict[str, Any]]:
        """Create workflow definition for AgentWorkflow"""
        definition = []

        for step in steps:
            step_def = {
                "name": step.step_id,
                "description": getattr(step, 'description', step.step_id)
            }

            # Handle parallel steps
            if isinstance(step, ParallelStepGroup):
                step_def["parallel"] = True
                step_def["steps"] = [s.step_id for s in step.steps]
            else:
                # Determine dependencies
                if hasattr(step, 'dependencies') and step.dependencies:
                    step_def["depends_on"] = step.dependencies

            definition.append(step_def)

        return definition

    async def execute_workflow(
            self,
            workflow: AgentWorkflow,
            input_data: Optional[Dict[str, Any]] = None,
            event_handler: Optional[Callable] = None
    ) -> WorkflowResponse:
        """Execute workflow with event handling"""

        # Create event stream
        event_stream = EventStream()
        self.event_streams[workflow.workflow_id] = event_stream

        # Subscribe event handler if provided
        if event_handler:
            event_stream.subscribe(event_handler)

        # Subscribe to all events
        event_stream.subscribe(self._log_event)

        try:
            # Run workflow
            response = await workflow.run(
                input=input_data or {},
                context=Context(event_stream=event_stream)
            )

            # Emit workflow completed event
            await event_stream.emit(WorkflowEvent(
                event_type=WorkflowEventType.WORKFLOW_COMPLETED,
                data={
                    "workflow_id": workflow.workflow_id,
                    "status": "success",
                    "output_keys": list(response.output.keys()) if response.output else []
                }
            ))

            return response

        except Exception as e:
            # Emit workflow failed event
            await event_stream.emit(WorkflowEvent(
                event_type=WorkflowEventType.WORKFLOW_FAILED,
                data={
                    "workflow_id": workflow.workflow_id,
                    "error": str(e),
                    "status": "failed"
                }
            ))
            raise

    async def _log_event(self, event: Event):
        """Log workflow events"""
        if isinstance(event, WorkflowEvent):
            print(f"\n[EVENT] {event.event_type.value} - Step: {event.step_id or 'N/A'}")
            if event.data:
                for key, value in event.data.items():
                    if key not in ['timestamp']:
                        print(f"  {key}: {value}")

    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get current workflow status"""
        if workflow_id in self.event_streams:
            # Get recent events
            events = list(self.event_streams[workflow_id].get_recent_events(10))

            return {
                "workflow_id": workflow_id,
                "event_count": len(events),
                "recent_events": [
                    {
                        "type": e.event_type.value if isinstance(e, WorkflowEvent) else type(e).__name__,
                        "timestamp": e.timestamp.isoformat() if hasattr(e, 'timestamp') else None,
                        "step_id": e.step_id if isinstance(e, WorkflowEvent) else None
                    }
                    for e in events
                ]
            }
        return {"workflow_id": workflow_id, "status": "not_found"}

class EventDrivenCDEExtractor:
    """
    Event-driven CDE extractor using AgentWorkflow
    """

    def __init__(
            self,
            tool_manager: MCPToolManager,
            orchestrator: AgentWorkflowOrchestrator
    ):
        self.tool_manager = tool_manager
        self.orchestrator = orchestrator
        self.workflow_cache: Dict[str, AgentWorkflow] = {}

    async def extract_with_events(
            self,
            plan: WorkflowPlan,
            event_handler: Optional[Callable] = None,
            context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract CDE using event-driven workflow
        """

        print(f"\n🚀 Starting event-driven extraction for: {plan.cde_name}")

        # Create or get workflow
        workflow = await self.orchestrator.create_workflow_from_plan(plan)

        # Prepare input context
        input_data = {
            "plan": plan.dict(),
            "document_context": plan.document_context or {},
            "execution_start": datetime.now().isoformat()
        }

        if context:
            input_data.update(context)

        # Create custom event handler
        async def handle_event(event: Event):
            # Call user event handler if provided
            if event_handler:
                await event_handler(event)

            # Track progress
            if isinstance(event, WorkflowEvent):
                await self._track_progress(event, plan)

        # Execute workflow with events
        response = await self.orchestrator.execute_workflow(
            workflow=workflow,
            input_data=input_data,
            event_handler=handle_event
        )

        # Process results
        results = await self._process_workflow_response(response, plan)

        return results

    async def _track_progress(self, event: WorkflowEvent, plan: WorkflowPlan):
        """Track workflow progress"""
        if event.event_type == WorkflowEventType.STEP_COMPLETED:
            completed_steps = len([
                s for s in plan.steps
                if any(e.step_id == s.step_id and e.event_type == WorkflowEventType.STEP_COMPLETED
                       for e in getattr(self, '_events', []))
            ])

            progress = (completed_steps / len(plan.steps)) * 100
            print(f"  Progress: {progress:.1f}% ({completed_steps}/{len(plan.steps)} steps)")

        elif event.event_type == WorkflowEventType.TOOL_RESULT:
            if event.data.get("success"):
                print(f"  ✓ Tool executed: {event.data.get('tool')} "
                      f"({event.data.get('execution_time', 0):.2f}s)")

    async def _process_workflow_response(
            self,
            response: WorkflowResponse,
            plan: WorkflowPlan
    ) -> Dict[str, Any]:
        """Process workflow response into structured result"""

        # Extract step results
        step_results = {}
        synthesis_result = None

        if response.output:
            for key, value in response.output.items():
                if key.startswith("step_"):
                    step_results[key] = value
                elif key == "synthesis":
                    synthesis_result = value

        # Calculate overall confidence
        confidences = []
        for step_id, result in step_results.items():
            if isinstance(result, dict) and "confidence" in result:
                confidences.append(result["confidence"])

        overall_confidence = (
            sum(confidences) / len(confidences)
            if confidences else 0.0
        )

        # Extract final data
        extracted_data = await self._extract_final_data(
            synthesis_result or step_results,
            plan
        )

        return {
            "workflow_id": response.workflow_id,
            "plan": plan.dict(),
            "step_results": step_results,
            "synthesis_result": synthesis_result,
            "extracted_data": extracted_data,
            "overall_confidence": overall_confidence,
            "execution_time": response.execution_time,
            "status": "completed"
        }

    async def _extract_final_data(
            self,
            results: Dict[str, Any],
            plan: WorkflowPlan
    ) -> Dict[str, Any]:
        """Extract final CDE data from results"""
        # This would use the synthesis strategy from the plan
        # For now, extract facilities and dates

        facilities = []

        # Look for facilities in results
        for key, value in results.items():
            if isinstance(value, dict):
                # Check for facilities in entity extraction
                if "extracted_entities" in value:
                    entities = value["extracted_entities"]
                    if isinstance(entities, dict) and "facility" in entities:
                        for facility in entities["facility"]:
                            if isinstance(facility, dict):
                                facilities.append({
                                    "name": facility.get("entity", "Unknown"),
                                    "source": key,
                                    "context": facility.get("context", "")
                                })

                # Check for dates
                if "extracted_dates" in value:
                    dates = value["extracted_dates"]
                    if isinstance(dates, list):
                        maturity_dates = [
                            d for d in dates
                            if isinstance(d, dict) and d.get("date_type") == "maturity"
                        ]

        return {
            "facilities": facilities,
            "extraction_time": datetime.now().isoformat(),
            "plan_name": plan.cde_name
        }