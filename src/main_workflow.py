# main_workflow.py
"""
Main application using LlamaIndex AgentWorkflow
"""
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any
import signal

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from agent.mcp_agent import MCPToolManager, MCPEnhancedCDEExtractor
from agent.plan_generator import StructuredPlanGenerator
from agent.workflow_orchestrator import (
    AgentWorkflowOrchestrator,
    WorkflowEvent,
    EventDrivenCDEExtractor
)
from agent.models import *
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler

class WorkflowEventHandler:
    """Handles workflow events for monitoring and logging"""

    def __init__(self, workflow_id: str):
        self.workflow_id = workflow_id
        self.events: List[WorkflowEvent] = []

    async def handle_event(self, event: Any):
        """Handle workflow events"""
        if isinstance(event, WorkflowEvent):
            self.events.append(event)
            await self._process_event(event)

    async def _process_event(self, event: WorkflowEvent):
        """Process individual events"""
        emoji = {
            "step_started": "▶️",
            "step_completed": "✅",
            "step_failed": "❌",
            "tool_called": "🔧",
            "tool_result": "📊",
            "workflow_completed": "🎉",
            "workflow_failed": "💥"
        }.get(event.event_type.value, "📝")

        step_info = f" [Step: {event.step_id}]" if event.step_id else ""
        print(f"\n{emoji} {event.event_type.value}{step_info}")

        # Show additional data for specific events
        if event.event_type.value == "tool_result":
            if event.data.get("success"):
                tool = event.data.get("tool", "unknown")
                time = event.data.get("execution_time", 0)
                print(f"   Tool: {tool} | Time: {time:.2f}s")

        elif event.event_type.value == "step_completed":
            if event.data:
                output_keys = list(event.data.get("output", {}).keys())
                if output_keys:
                    print(f"   Output keys: {', '.join(output_keys[:3])}")

    def get_summary(self) -> Dict[str, Any]:
        """Get event summary"""
        event_counts = {}
        for event in self.events:
            event_type = event.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        return {
            "workflow_id": self.workflow_id,
            "total_events": len(self.events),
            "event_counts": event_counts,
            "last_event": self.events[-1].event_type.value if self.events else None
        }

async def main():
    """Main application with AgentWorkflow"""

    print("\n" + "="*60)
    print("🤖 CDE Extraction System with AgentWorkflow")
    print("="*60 + "\n")

    # Initialize callback manager for debugging
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])

    try:
        # Initialize MCP tool manager
        print("🔧 Initializing MCP Tool Manager...")
        tool_manager = MCPToolManager("config/mcp_config.yaml")
        await tool_manager.initialize()

        # Initialize workflow orchestrator
        print("🔄 Initializing AgentWorkflow Orchestrator...")
        orchestrator = AgentWorkflowOrchestrator(
            tool_manager=tool_manager,
            callback_manager=callback_manager
        )

        # Initialize event-driven extractor
        extractor = EventDrivenCDEExtractor(tool_manager, orchestrator)

        # Load LLM client for plan generation
        import yaml
        from openai import AsyncOpenAI
        import instructor

        with open("config/mcp_config.yaml", 'r') as f:
            config = yaml.safe_load(f)

        llm_config = config.get('llm', {})
        llm_client = AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            **{k: v for k, v in llm_config.items() if k != 'model'}
        )
        client = instructor.patch(llm_client)

        plan_generator = StructuredPlanGenerator(client)

        # Example CDE prompts
        cde_prompts = [
            {
                "user_query": "extract all credit facilities with maturity dates from the credit agreement",
                "instructions": """
                1. First, retrieve the first page to get the closing date
                2. Extract all facility entities from the entire document
                3. For each facility, find its maturity date
                4. If maturity is relative to closing, calculate the actual date
                5. Validate that maturity dates are after the closing date
                """,
                "step_by_step_reasoning": """
                Step 1: Get document context and closing date
                Step 2: Identify all credit facilities mentioned
                Step 3: Locate maturity information for each facility
                Step 4: Calculate dates if needed
                Step 5: Validate and format results
                """,
                "output": {
                    "format": "json",
                    "fields": ["facility_name", "maturity_date", "days_from_closing", "confidence"],
                    "validation_rules": ["maturity_date > closing_date"]
                }
            }
        ]

        # Process each CDE
        for i, cde_prompt in enumerate(cde_prompts):
            print(f"\n📄 Processing CDE {i+1}: {cde_prompt['user_query'][:50]}...")

            # Generate workflow plan
            print("  Generating workflow plan...")
            context = {
                "available_tools": [tool.dict() for tool in tool_manager.tools],
                "document_id": "credit_agreement_001"
            }

            plan = await plan_generator.generate_plan(cde_prompt, context)

            # Create event handler
            event_handler = WorkflowEventHandler(plan.plan_id)

            # Execute with event-driven workflow
            print(f"  Executing workflow with {len(plan.steps)} steps...")
            result = await extractor.extract_with_events(
                plan=plan,
                event_handler=event_handler.handle_event,
                context={"document_id": "credit_agreement_001"}
            )

            # Display results
            print(f"\n✅ Extraction Complete!")
            print(f"   Workflow ID: {result['workflow_id']}")
            print(f"   Overall Confidence: {result['overall_confidence']:.2f}")
            print(f"   Execution Time: {result.get('execution_time', 0):.2f}s")

            if result.get('extracted_data', {}).get('facilities'):
                print(f"\n📊 Extracted Facilities:")
                for facility in result['extracted_data']['facilities']:
                    print(f"   • {facility['name']}")

            # Get event summary
            summary = event_handler.get_summary()
            print(f"\n📈 Event Summary:")
            for event_type, count in summary['event_counts'].items():
                print(f"   {event_type}: {count}")

            # Save results
            output_file = f"results/workflow_result_{plan.plan_id}.json"
            os.makedirs("results", exist_ok=True)

            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)

            print(f"\n💾 Results saved to: {output_file}")

            # Get workflow status
            status = await orchestrator.get_workflow_status(plan.plan_id)
            print(f"\n📋 Workflow Status:")
            print(f"   Event count: {status.get('event_count', 0)}")

        print("\n" + "="*60)
        print("🎉 All workflows completed successfully!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean shutdown
        print("\n🛑 Shutting down...")
        await tool_manager.shutdown()

async def interactive_monitor():
    """Interactive workflow monitor"""

    tool_manager = MCPToolManager("config/mcp_config.yaml")
    await tool_manager.initialize()

    orchestrator = AgentWorkflowOrchestrator(tool_manager)

    # Create a simple workflow for demonstration
    from agent.models import WorkflowPlan, WorkflowStep

    demo_plan = WorkflowPlan(
        cde_name="Demo Facility Extraction",
        user_query="Extract facilities",
        instructions="Demo",
        output_spec={},
        available_tools=[],
        steps=[
            WorkflowStep(
                step_id="step_1",
                step_number=1,
                objective="Get first page",
                selected_tool="retrieve_first_page",
                parameters={"document_id": "credit_agreement_001"}
            ),
            WorkflowStep(
                step_id="step_2",
                step_number=2,
                objective="Extract facilities",
                selected_tool="extract_entities",
                parameters={
                    "document_id": "credit_agreement_001",
                    "entity_types": ["facility"]
                },
                dependencies=["step_1"]
            )
        ],
        synthesis=None,
        estimated_duration=2
    )

    # Create workflow
    workflow = await orchestrator.create_workflow_from_plan(demo_plan)

    # Monitor events in real-time
    print("\n👁️  Starting interactive monitor...")
    print("Press Ctrl+C to stop\n")

    async def monitor_events(event: Any):
        if isinstance(event, WorkflowEvent):
            print(f"[{event.timestamp.strftime('%H:%M:%S')}] "
                  f"{event.event_type.value} - {event.step_id or 'workflow'}")

    # Run workflow with monitoring
    await orchestrator.execute_workflow(
        workflow=workflow,
        event_handler=monitor_events
    )

if __name__ == "__main__":
    # Set OpenAI API key
    if "OPENAI_API_KEY" not in os.environ:
        print("Please set OPENAI_API_KEY environment variable")
        sys.exit(1)

    # Handle graceful shutdown
    async def shutdown(signal, loop):
        print(f"\nReceived exit signal {signal.name}...")
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        [task.cancel() for task in tasks]
        await asyncio.gather(*tasks, return_exceptions=True)
        loop.stop()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            sig,
            lambda s=sig: asyncio.create_task(shutdown(s, loop))
        )

    try:
        loop.run_until_complete(main())
    finally:
        loop.close()