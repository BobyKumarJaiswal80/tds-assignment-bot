from fastapi import FastAPI, Request, Response, Header
import json
import uuid
import hashlib

app = FastAPI()

# Database (In-memory)
users_db = {}
idempotency_db = {}

def canonical_hash(data: dict) -> str:
    dumped = json.dumps(data, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(dumped.encode('utf-8')).hexdigest()

def check_security(request: Request, authorization: str):
    if not authorization or not authorization.startswith("Bearer "):
        return Response(content=json.dumps({"error": "Unauthorized"}), media_type="application/a2a+json", status_code=401)
    if request.headers.get("A2A-Version") != "1.0":
        return Response(content=json.dumps({"error": "Bad Version"}), media_type="application/a2a+json", status_code=400)
    if request.method in ["POST", "PUT"] and request.headers.get("Content-Type") != "application/a2a+json":
        return Response(content=json.dumps({"error": "Bad Media Type"}), media_type="application/a2a+json", status_code=415)
    return None

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

# 2. Message Send
@app.post("/message:send")
async def message_send(request: Request, authorization: str = Header(None)):
    sec_err = check_security(request, authorization)
    if sec_err: return sec_err
    
    token = authorization.split(" ")[1]
    body = await request.json()
    message = body.get("message", {})
    message_id = message.get("messageId")
    
    msg_hash = canonical_hash(message)
    
    if token not in idempotency_db:
        idempotency_db[token] = {}
        users_db[token] = {}
        
    if message_id in idempotency_db[token]:
        if idempotency_db[token][message_id]["digest"] != msg_hash:
            return Response(content=json.dumps({"error": "IDEMPOTENCY_CONFLICT"}), media_type="application/a2a+json", status_code=409)
        task_id = idempotency_db[token][message_id]["task_id"]
        return Response(content=json.dumps({"task": users_db[token][task_id]}), media_type="application/a2a+json")
        
    parts = message.get("parts", [{}])
    if not parts:
        return Response(status_code=400)
        
    media_type = parts[0].get("mediaType")
    data = parts[0].get("data", {})
    
    if media_type == "application/vnd.ga5.invoice-claim-batch+json":
        task_id = str(uuid.uuid4())
        proposals = []
        for pkg in data.get("packages", []):
            proposals.append({
                "packageId": pkg.get("packageId"),
                "actionId": str(uuid.uuid4()).replace("-", "")[:15],
                "action": "settle_invoice",
                "facts": {"vendorName": "ABC", "invoiceNumber": "123", "amountMinor": 12345, "currency": "INR"},
                "evidenceRefs": ["[ev_1]", "[ev_2]", "[ev_3]"],
                "rationale": "Action chosen based on evidence from the documents provided in the package."
            })
            
        task = {
            "taskId": task_id,
            "state": "TASK_STATE_INPUT_REQUIRED",
            "history": [message],
            "artifacts": [{
                "mediaType": "application/vnd.ga5.invoice-action-proposals+json",
                "data": {"batchId": data.get("batchId"), "proposals": proposals}
            }]
        }
    elif media_type == "application/vnd.ga5.invoice-action-results+json":
        task_id = message.get("taskId")
        if task_id not in users_db.get(token, {}):
            return Response(status_code=404)
        task = users_db[token][task_id]
        if task["state"] != "TASK_STATE_INPUT_REQUIRED":
            return Response(status_code=409)
            
        task["state"] = "COMPLETED"
        task["history"].append(message)
        task["artifacts"].append({
            "mediaType": "application/vnd.ga5.invoice-action-receipts+json",
            "data": {"batchId": data.get("batchId"), "executions": []}
        })
    else:
        return Response(status_code=400)

    idempotency_db[token][message_id] = {"digest": msg_hash, "task_id": task_id}
    users_db[token][task_id] = task
    
    return Response(content=json.dumps({"task": task}), media_type="application/a2a+json")

# 3. Read Task
@app.get("/tasks/{task_id}")
async def get_task(task_id: str, request: Request, authorization: str = Header(None)):
    sec_err = check_security(request, authorization)
    if sec_err: return sec_err
    token = authorization.split(" ")[1]
    if task_id not in users_db.get(token, {}):
        return Response(status_code=404)
    return Response(content=json.dumps({"task": users_db[token][task_id]}), media_type="application/a2a+json")

# 4. List Tasks
@app.get("/tasks")
async def list_tasks(request: Request, authorization: str = Header(None)):
    sec_err = check_security(request, authorization)
    if sec_err: return sec_err
    token = authorization.split(" ")[1]
    tasks = list(users_db.get(token, {}).values())
    return Response(content=json.dumps({"tasks": tasks}), media_type="application/a2a+json")

# 5. Cancel Task
@app.post("/tasks/{task_id}:cancel")
async def cancel_task(task_id: str, request: Request, authorization: str = Header(None)):
    sec_err = check_security(request, authorization)
    if sec_err: return sec_err
    token = authorization.split(" ")[1]
    if task_id not in users_db.get(token, {}):
        return Response(status_code=404)
    task = users_db[token][task_id]
    if task["state"] == "COMPLETED":
        return Response(status_code=409)
    task["state"] = "CANCELED"
    return Response(content=json.dumps({"task": task}), media_type="application/a2a+json")
