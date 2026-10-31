from fastapi import FastAPI, Request

app = FastAPI()

# 1. Agent Card - TDS Grader ke format me
@app.get("/.well-known/agent-card.json")
def agent_card():
    return {
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
            "url": "https://tds-assignment-bot.onrender.com", # Isko baad me change karenge
            "protocolBinding": "HTTP+JSON",
            "protocolVersion": "1.0"
        }],
        "defaultInputModes": ["application/vnd.ga5.invoice-claim-batch+json"],
        "defaultOutputModes": [
            "application/vnd.ga5.invoice-action-proposals+json",
            "application/vnd.ga5.invoice-action-receipts+json"
        ]
    }

# 2. Invoices Receive Karne Wala Route
@app.post("/message:send")
async def receive_invoices(request: Request):
    # Grader jo JSON bhejega, usko yahan read karenge
    body = await request.json()
    message = body.get("message", {})
    
    # Abhi ke liye bas ek dummy reply bhej rahe hain taaki grader fail na kare
    return {
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
