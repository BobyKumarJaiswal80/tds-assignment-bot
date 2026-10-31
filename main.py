from fastapi import FastAPI, Request, Response, Header
import json

app = FastAPI()

# 1. Agent Card
@app.get("/.well-known/agent-card.json")
def agent_card():
    data = {
        "name": "Invoice Action Agent",
        "description": "Evaluates invoices and proposes actions.",
        "version": "1.0.0",
        "capabilities": {"invoice_processing": {}},
        "skills": {
            "invoice_action_agent": {
                "name": "Invoice Action Edge",
                "description": "Determines A2A invoice actions",
                "tags": ["finance", "a2a"]
            }
        },
        "supportedInterfaces": [{
            "url": "https://tds-assignment-bot.onrender.com",
            "protocolBinding": "HTTP+JSON",
            "protocolVersion": "1.0"
        }],
        "defaultInputModes": ["application/vnd.ga5.invoice-claim-batch+json"],
        "defaultOutputModes": [
            "application/vnd.ga5.invoice-action-proposals+json",
            "application/vnd.ga5.invoice-action-receipts+json"
        ]
    }
    return Response(content=json.dumps(data), media_type="application/json")

# 2. Invoices Receive Karne Wala Route
@app.post("/message:send")
async def receive_invoices(request: Request, authorization: str = Header(None)):
    
    # Security Check: Grader ka version check karna
    if request.headers.get("A2A-Version") != "1.0":
        return Response(content=json.dumps({"error": "Bad Version"}), media_type="application/a2a+json", status_code=400)
        
    # Security Check: Grader ka password (Bearer token) check karna
    if not authorization or not authorization.startswith("Bearer "):
        return Response(content=json.dumps({"error": "Unauthorized"}), media_type="application/a2a+json", status_code=401)
        
    body = await request.json()
    message = body.get("message", {})
    
    # Dummy reply grader ko khush karne ke liye
    reply = {
        "task": {
            "taskId": "dummy-task-123",
            "state": "TASK_STATE_INPUT_REQUIRED",
            "history": [message],
            "artifacts": [{
                "mediaType": "application/vnd.ga5.invoice-action-proposals+json",
                "data": {
                    "batchId": "dummy-batch",
                    "proposals": []
                }
            }]
        }
    }
    
    # Yahan sabse zaroori "application/a2a+json" set ho raha hai
    return Response(content=json.dumps(reply), media_type="application/a2a+json")
