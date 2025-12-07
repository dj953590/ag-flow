# dashboard/app.py
"""
Web dashboard for monitoring workflow execution
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
from typing import Dict, List
from datetime import datetime
import uvicorn

from agent.workflow_orchestrator import AgentWorkflowOrchestrator, WorkflowEvent

app = FastAPI(title="Workflow Monitor")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:
    """Manages WebSocket connections"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

# Store active workflows
active_workflows: Dict[str, Dict] = {}
workflow_history: Dict[str, List] = {}

@app.get("/")
async def get():
    """Serve dashboard HTML"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Workflow Monitor</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .workflow { border: 1px solid #ccc; padding: 15px; margin: 10px 0; }
            .step { background: #f5f5f5; padding: 10px; margin: 5px 0; }
            .event { padding: 5px; margin: 2px; border-left: 3px solid #007bff; }
            .success { border-color: #28a745; }
            .error { border-color: #dc3545; }
            .running { border-color: #007bff; }
        </style>
    </head>
    <body>
        <h1>Workflow Execution Monitor</h1>
        <div id="workflows"></div>
        <script>
            const ws = new WebSocket('ws://' + window.location.host + '/ws');
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                updateDashboard(data);
            };
            
            function updateDashboard(data) {
                let workflowsDiv = document.getElementById('workflows');
                
                // Create or update workflow display
                let workflowDiv = document.getElementById('workflow-' + data.workflow_id);
                if (!workflowDiv) {
                    workflowDiv = document.createElement('div');
                    workflowDiv.id = 'workflow-' + data.workflow_id;
                    workflowDiv.className = 'workflow';
                    workflowsDiv.appendChild(workflowDiv);
                }
                
                workflowDiv.innerHTML = `
                    <h3>${data.workflow_name}</h3>
                    <p>Status: <strong>${data.status}</strong></p>
                    <p>Progress: ${data.progress}%</p>
                    <div>
                        <h4>Recent Events:</h4>
                        ${data.recent_events.map(event => `
                            <div class="event ${event.status}">
                                [${event.timestamp}] ${event.type}
                            </div>
                        `).join('')}
                    </div>
                `;
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)

    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()

    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/workflow/{workflow_id}/event")
async def receive_workflow_event(workflow_id: str, event: dict):
    """Receive workflow events from orchestrator"""

    if workflow_id not in workflow_history:
        workflow_history[workflow_id] = []

    # Store event
    event_with_timestamp = {
        **event,
        "received_at": datetime.now().isoformat()
    }
    workflow_history[workflow_id].append(event_with_timestamp)

    # Update active workflow status
    if workflow_id not in active_workflows:
        active_workflows[workflow_id] = {
            "workflow_id": workflow_id,
            "status": "running",
            "start_time": datetime.now().isoformat(),
            "event_count": 0
        }

    active_workflows[workflow_id]["event_count"] += 1
    active_workflows[workflow_id]["last_event"] = event_with_timestamp

    # Broadcast to WebSocket clients
    broadcast_data = {
        "workflow_id": workflow_id,
        "event": event_with_timestamp,
        "total_events": len(workflow_history[workflow_id])
    }
    await manager.broadcast(broadcast_data)

    return {"status": "received"}

@app.get("/workflows")
async def list_workflows():
    """List all active workflows"""
    return {
        "active_workflows": list(active_workflows.keys()),
        "total_active": len(active_workflows),
        "workflows": active_workflows
    }

@app.get("/workflow/{workflow_id}")
async def get_workflow(workflow_id: str):
    """Get workflow details and history"""
    if workflow_id not in workflow_history:
        return {"error": "Workflow not found"}

    return {
        "workflow_id": workflow_id,
        "history": workflow_history[workflow_id],
        "total_events": len(workflow_history[workflow_id]),
        "active": workflow_id in active_workflows
    }

@app.post("/workflow/{workflow_id}/complete")
async def complete_workflow(workflow_id: str):
    """Mark workflow as completed"""
    if workflow_id in active_workflows:
        active_workflows[workflow_id]["status"] = "completed"
        active_workflows[workflow_id]["end_time"] = datetime.now().isoformat()

    return {"status": "completed"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)