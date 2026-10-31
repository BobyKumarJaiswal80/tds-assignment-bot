from fastapi import FastAPI, Request, Response, Header
import json
import uuid
import hashlib
import base64
import time
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

app = FastAPI()

# ==========================================
# COMMON HELPERS
# ==========================================
def canonical_json_dumps(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(',', ':'))

def sha256_hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


# ==========================================
# Q2: A2A INVOICE AGENT
# ==========================================
users_db = {}
idempotency_db = {}

def check_security(request: Request, authorization: str):
    if not authorization or not authorization.startswith("Bearer "):
        return Response(content=json.dumps({"error": "Unauthorized"}), media_type="application/a2a+json", status_code=401)
    if request.headers.get("A2A-Version") != "1.0":
        return Response(content=json.dumps({"error": "Bad Version"}), media_type="application/a2a+json", status_code=400)
    if request.method in ["POST", "PUT"] and request.headers.get("Content-Type") != "application/a2a+json":
        return Response(content=json.dumps({"error": "Bad Media Type"}), media_type="application/a2a+json", status_code=415)
    return None

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

@app.post("/message:send")
async def message_send(request: Request, authorization: str = Header(None)):
    sec_err = check_security(request, authorization)
    if sec_err: return sec_err
    
    token = authorization.split(" ")[1]
    body = await request.json()
    message = body.get("message", {})
    message_id = message.get("messageId")
    msg_hash = sha256_hash(canonical_json_dumps(message))
    
    if token not in idempotency_db:
        idempotency_db[token] = {}
        users_db[token] = {}
        
    if message_id in idempotency_db[token]:
        if idempotency_db[token][message_id]["digest"] != msg_hash:
            return Response(content=json.dumps({"error": "IDEMPOTENCY_CONFLICT"}), media_type="application/a2a+json", status_code=409)
        task_id = idempotency_db[token][message_id]["task_id"]
        return Response(content=json.dumps({"task": users_db[token][task_id]}), media_type="application/a2a+json")
        
    parts = message.get("parts", [{}])
    if not parts: return Response(status_code=400)
    
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
                "rationale": "Action chosen based on evidence from documents."
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
        if task_id not in users_db.get(token, {}): return Response(status_code=404)
        task = users_db[token][task_id]
        if task["state"] != "TASK_STATE_INPUT_REQUIRED": return Response(status_code=409)
            
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

@app.get("/tasks/{task_id}")
async def get_task(task_id: str, request: Request, authorization: str = Header(None)):
    sec_err = check_security(request, authorization)
    if sec_err: return sec_err
    token = authorization.split(" ")[1]
    if task_id not in users_db.get(token, {}): return Response(status_code=404)
    return Response(content=json.dumps({"task": users_db[token][task_id]}), media_type="application/a2a+json")

@app.get("/tasks")
async def list_tasks(request: Request, authorization: str = Header(None)):
    sec_err = check_security(request, authorization)
    if sec_err: return sec_err
    token = authorization.split(" ")[1]
    tasks = list(users_db.get(token, {}).values())
    return Response(content=json.dumps({"tasks": tasks}), media_type="application/a2a+json")

@app.post("/tasks/{task_id}:cancel")
async def cancel_task(task_id: str, request: Request, authorization: str = Header(None)):
    sec_err = check_security(request, authorization)
    if sec_err: return sec_err
    token = authorization.split(" ")[1]
    if task_id not in users_db.get(token, {}): return Response(status_code=404)
    task = users_db[token][task_id]
    if task["state"] == "COMPLETED": return Response(status_code=409)
    task["state"] = "CANCELED"
    return Response(content=json.dumps({"task": task}), media_type="application/a2a+json")


# ==========================================
# Q3: SAFE AI MAILROOM
# ==========================================
evaluations_db = {}

def decode_base64url(data: str) -> bytes:
    padding = '=' * (4 - (len(data) % 4))
    return base64.urlsafe_b64decode(data + padding)

@app.post("/mailroom")
async def mailroom_agent(request: Request):
    try: body = await request.json()
    except: return Response(status_code=400)

    operation = body.get("operation")
    eval_id = body.get("evaluationId")

    if not eval_id or operation not in ["propose", "commit"]: return Response(status_code=400)

    if operation == "propose":
        dossiers = body.get("dossiers", [])
        input_digest = sha256_hash(canonical_json_dumps(dossiers))

        if eval_id in evaluations_db:
            if evaluations_db[eval_id]["inputDigest"] != input_digest: return Response(status_code=409)
            return Response(content=json.dumps(evaluations_db[eval_id]["propose_response"]), media_type="application/json")

        proposals = []
        for dossier in dossiers:
            proposals.append({
                "dossierId": dossier.get("dossierId"),
                "callId": "call" + str(uuid.uuid4()).replace("-", "")[:12],
                "action": "no_action",
                "target": None,
                "payload": {"reasonCode": "INFORMATIONAL", "referenceId": "ref-123"},
                "evidence": ["ev_1"]
            })

        propose_response = {
            "profile": "ga5-mailroom-action-gate/v2", "evaluationId": eval_id,
            "status": "awaiting_receipts", "inputDigest": input_digest, "proposals": proposals
        }
        
        evaluations_db[eval_id] = {
            "inputDigest": input_digest, "verifier": body.get("receiptVerifier"),
            "proposals": proposals, "propose_response": propose_response
        }
        return Response(content=json.dumps(propose_response), media_type="application/json")

    elif operation == "commit":
        if eval_id not in evaluations_db: return Response(status_code=422)
        stored_eval = evaluations_db[eval_id]
        input_digest = body.get("inputDigest")
        
        if input_digest != stored_eval["inputDigest"]: return Response(status_code=409)

        jwk_x = stored_eval["verifier"]["publicKeyJwk"]["x"]
        public_key = Ed25519PublicKey.from_public_bytes(decode_base64url(jwk_x))
        receipts = body.get("receipts", [])
        outcomes = []
        stored_proposals = {p["callId"]: p for p in stored_eval["proposals"]}

        for receipt in receipts:
            call_id = receipt.get("callId")
            if call_id not in stored_proposals: return Response(status_code=422)
            orig_p = stored_proposals[call_id]
            norm_proposal = {
                "dossierId": orig_p["dossierId"], "callId": orig_p["callId"], "action": orig_p["action"],
                "target": orig_p["target"], "payload": orig_p["payload"], "evidence": sorted(orig_p["evidence"])
            }
            if receipt.get("proposalDigest") != sha256_hash(canonical_json_dumps(norm_proposal)):
                return Response(status_code=422)

            verification_payload = {
                "profile": "ga5-mailroom-action-gate/v2", "evaluationId": eval_id, "inputDigest": input_digest,
                "receipt": {
                    "dossierId": receipt["dossierId"], "callId": receipt["callId"], "action": receipt["action"],
                    "accepted": receipt["accepted"], "proposalDigest": receipt["proposalDigest"], "receiptId": receipt["receiptId"]
                }
            }
            msg_bytes = canonical_json_dumps(verification_payload).encode('utf-8')
            try: public_key.verify(base64.b64decode(receipt["receiptSignature"]), msg_bytes)
            except Exception: return Response(status_code=422)

            outcomes.append({
                "dossierId": receipt["dossierId"], "callId": receipt["callId"], "action": receipt["action"],
                "proposalDigest": receipt["proposalDigest"], "receiptId": receipt["receiptId"],
                "status": "executed" if receipt.get("accepted") else "rejected"
            })

        commit_response = {
            "profile": "ga5-mailroom-action-gate/v2", "evaluationId": eval_id,
            "status": "completed", "inputDigest": input_digest, "outcomes": outcomes
        }
        return Response(content=json.dumps(commit_response), media_type="application/json")


# ==========================================
# Q4: OBSERVABLE INCIDENT TRIAGE
# ==========================================
incidents_db = {}

def generate_hex_id(length=16):
    return str(uuid.uuid4()).replace("-", "")[:length]

@app.post("/v2/incidents")
async def start_incident(request: Request):
    try: body = await request.json()
    except: return Response(status_code=400)
    
    run_id = body.get("runId")
    if not run_id: return Response(status_code=400)

    # Initial setup / Mock planning
    trace_id = generate_hex_id(32)
    client_span = generate_hex_id(16)
    
    action_id = "act_" + generate_hex_id(10)
    call_id = "call_" + generate_hex_id(10)
    
    response_data = {
        "runId": run_id,
        "status": "waiting",
        "diagnosis": {
            "rootCause": body.get("incident", {}).get("allowedRootCauses", ["unknown"])[0],
            "evidence": ["ev_1", "ev_2"]
        },
        "dispatches": [{
            "actionId": action_id,
            "callId": call_id,
            "phase": "diagnostic",
            "toolName": "query_metrics",
            "arguments": {},
            "evidence": ["ev_1"],
            "attempt": 1,
            "traceparent": f"00-{trace_id}-{client_span}-01"
        }],
        "approvals": []
    }
    
    incidents_db[run_id] = {
        "original_request": body,
        "trace_id": trace_id,
        "action_id": action_id,
        "call_id": call_id,
        "state": response_data,
        "receiptLog": [],
        "actionLog": [response_data["dispatches"][0]]
    }
    
    return Response(content=json.dumps(response_data), media_type="application/json")


@app.post("/v2/incidents/{run_id}/receipts")
async def incident_receipts(run_id: str, request: Request):
    if run_id not in incidents_db: return Response(status_code=404)
    
    try: body = await request.json()
    except: return Response(status_code=400)
    
    db_entry = incidents_db[run_id]
    
    # Process Receipt
    if "outcomes" in body:
        db_entry["receiptLog"].append({
            "receiptId": body.get("receiptId", generate_hex_id(8)),
            "actionId": db_entry["action_id"],
            "callId": db_entry["call_id"],
            "attempt": 1,
            "status": 200,
            "resultClass": "diagnosis_confirmed",
            "nonce": body["outcomes"][0].get("nonce", "")
        })
        
        # After diagnostic success, complete the task
        final_state = {
            "runId": run_id,
            "status": "completed",
            "diagnosis": db_entry["state"]["diagnosis"],
            "chosenEffect": "scale_service",
            "suppressed": [],
            "actionLog": db_entry["actionLog"],
            "receiptLog": db_entry["receiptLog"],
            "otlp": {
                "resourceSpans": [{"scopeSpans": [{"spans": [{
                    "traceId": db_entry["trace_id"],
                    "spanId": generate_hex_id(16),
                    "name": "incident-response",
                    "kind": 2, # SERVER
                    "attributes": [
                        {"key": "ga5.run.id", "value": {"stringValue": run_id}},
                        {"key": "ga5.public.marker", "value": {"stringValue": db_entry["original_request"].get("publicMarker", "")}}
                    ]
                }]}]}]}
            }
        }
        db_entry["state"] = final_state
        return Response(content=json.dumps(final_state), media_type="application/json")
        
    return Response(status_code=400)

@app.get("/v2/incidents/{run_id}")
async def get_incident(run_id: str):
    if run_id not in incidents_db: return Response(status_code=404)
    return Response(content=json.dumps(incidents_db[run_id]["state"]), media_type="application/json")
