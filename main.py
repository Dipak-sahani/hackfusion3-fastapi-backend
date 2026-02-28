import os
import json
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any, Union
from bson import ObjectId
import hashlib
import requests
import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from groq import Groq
import time
import shutil
import langsmith
from langsmith import traceable

# Load environment variables
load_dotenv()

# Initialize LangSmith
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "MediFlow-AI")
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
    print(f"LangSmith Tracing Enabled: {LANGCHAIN_PROJECT}")

# Configure Groq API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OCR_API_KEY = os.getenv("OCR_API_KEY", "helloworld") # Default or helloworld

if not GROQ_API_KEY:
    print("Warning: GROQ_API_KEY not found in environment variables.")

client = None
if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)

# Initialize FastAPI app
app = FastAPI(title="Medical Intelligence Service")

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://hackfusion3-nodejs-backend.onrender.com",
        "https://hackfusion3-fastapi-backend.onrender.com",
        "http://localhost:5173",
        "http://localhost:5174"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Request Logging Middleware
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    path = request.url.path
    method = request.method
    
    print(f"\n[REQUEST] {method} {path} - Processing...")
    
    try:
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        formatted_process_time = "{0:.2f}".format(process_time)
        print(f"[RESPONSE] {method} {path} - Status: {response.status_code} - Time: {formatted_process_time}ms")
        return response
    except Exception as e:
        process_time = (time.time() - start_time) * 1000
        formatted_process_time = "{0:.2f}".format(process_time)
        print(f"[ERROR] {method} {path} - Failed: {str(e)} - Time: {formatted_process_time}ms")
        raise e

# Startup Log
print(f"\n[STARTUP] Medical Intelligence Service initializing...")
print(f"[STARTUP] Detected PORT: {os.getenv('PORT', '8000 (Default)')}")

# Create storage directory for prescriptions
UPLOAD_DIR = "uploads/prescriptions"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Data Models
class MedicineOrder(BaseModel):
    medicine_name: str
    quantity: int
    unit: str
    quantity_converted: Optional[int] = Field(None, description="Quantity converted to base unit")
    daily_consumption: Optional[float] = Field(None, description="Daily consumption rate if mentioned")
    operation: str = Field("add", pattern="^(add|remove|set|replace|update)$")
    confidence: float = 1.0

class MedicineResponse(BaseModel):
    id: str
    name: str
    quantity: int
    status: str
    nextRefillDate: Optional[str] = None
    expiredOn: Optional[str] = None
    lowStockThreshold: int = 50
    isPaused: bool = False

class NormalizationResponse(BaseModel):
    type: str = Field(description="Type of response: 'chat', 'order', 'query_history', or 'cancel'")
    message: Optional[str] = None
    orders: Optional[List[MedicineOrder]] = None
    medicine_filter: Optional[str] = Field(None, description="Medicine name to filter history by")
    reason: Optional[str] = Field(None, description="Reason for cancellation if type is cancel")
    safety_alerts: List[str] = []

class ChatMessage(BaseModel):
    role: str
    parts: List[str]

class InputPayload(BaseModel):
    text: str
    chat_history: Optional[List[ChatMessage]] = []
    medicine_context: Optional[str] = ""

class ResponseGenerationPayload(BaseModel):
    query: str
    context: str

class MedicineDetails(BaseModel):
    name: str = "Unknown"
    dosage: Optional[str] = "N/A"
    frequency: Optional[str] = "N/A"
    quantity: Optional[int] = 1

class PrescriptionExtraction(BaseModel):
    doctorName: Optional[str] = "Not found"
    doctorRegistrationNumber: Optional[str] = "N/A"
    patientName: Optional[str] = "N/A"
    medicines: List[MedicineDetails] = []
    prescriptionDate: Optional[str] = "N/A"
    address: Optional[str] = "N/A"
    suspiciousScore: int = 0
    verdict: str = "VALID"
    reason: Optional[str] = None

class NormalizationRawPayload(BaseModel):
    raw_text: str

class PrescriptionUploadResponse(BaseModel):
    id: str
    extractedData: PrescriptionExtraction
    status: str
    suspiciousScore: int
    rawOcrText: str

# System Instructions
SYSTEM_PROMPT_CLASSIFIER = """
You are a friendly and caring Nurse Assistant at a pharmacy. 
Your tone is professional, supportive, warm, and diligent. You treat users with patience and care, like a nurse would.

Your goal is to classify the user's input as General Chat, Medicine Order, History/Stock Query, or Cancellation.

STRATEGIC RULES:
1. NO PRESCRIPTIONS: For now, do NOT ask the user for a prescription. Ignore any requirement for prescriptions in the order flow.
2. 2-STEP ORDER CONFIRMATION:
   - If the user first mentions wanting to buy or order a medicine:
     * ALWAYS classify as "type": "chat".
     * Provide a warm response asking for confirmation (e.g., "I'd be happy to help you with that, dear. Shall I go ahead and order 1 strip of [Medicine] for you?").
   - If the user says "yes", "confirm", "proceed", "buy it", "ok do it", or similar AFTER you have asked for confirmation in the chat history:
     * ONLY then classify as "type": "order" and provide the structured order details.

INPUT CONTEXT:
1. User's latest message.
2. Chat History (last 10 messages).
3. Medicine Context (User's current stock and history).

ACKNOWLEDGMENT & PIVOTING HANDLING:
- If the user says "ok", "got it", "fine", "alright", "understand", or similar brief acknowledgments (WITHOUT an active order confirmation pending):
  * ALWAYS classify as "type": "chat".
  * Do NOT classify as "order" even if a previous order failed.
- PIVOTING RULE: If the user changes the subject or asks something unrelated to a previous order question, ALWAYS classify as "chat" or "query_history". Do NOT get stuck on the order flow if the user has moved on.

OUTPUT FORMAT:
You MUST return a JSON object with a "type" field. (Requirement: The word 'json' must be used in this instruction).

SCENARIO 1: GENERAL CHAT & HISTORY ANSWERS
If the user greets, says 'how are you', asks for advice, acknowledges something (e.g., "ok"), OR if the input is very short/empty (like "."):
{
    "type": "chat",
    "message": "Hello! 👋 I'm your AI pharmacy assistant. I can help you order medicines or answer your health questions. How can I assist you today?"
}

SCENARIO 2: MEDICINE ORDER (Confirmation Pending)
If the user mentions a medicine to buy/order for the first time:
{
    "type": "chat",
    "message": "Certainly, dear. I see you'd like to order [Medicine Name]. Shall I go ahead and prepare an order for [Quantity] for you?"
}

SCENARIO 3: MEDICINE ORDER (Confirmed)
If the user confirms (e.g., "Yes", "Confirm") after the AI asked for confirmation:
{
    "type": "order",
    "orders": [
        {
            "medicine_name": "Exact Name",
            "quantity": 1,
            "unit": "strip",
            "quantity_converted": 10,
            "daily_consumption": 1.0, 
            "operation": "add",
            "confidence": 0.99
        }
    ]
}

UNIT CONVERSION RULES:
- 1 strip = 10 tablets (UNLESS specified otherwise in context).
- 1 box = 100 tablets.
- ALWAYS calculate "quantity_converted" as (quantity * tablets_per_unit). 
- If unit is "tablet", quantity_converted = quantity.
- If user says "2 strips", quantity=2, unit="strip", quantity_converted=20.

SCENARIO 4: NO RESULT FOUND
If the user mentions a medicine that isn't in their records, treat it as a new request and ask for confirmation to order it.

SCENARIO 5: CANCELLATION
If the user says "cancel", "stop", "abort", or "I don't want this":
{
    "type": "cancel",
    "message": "Order cancelled."
}
"""

SYSTEM_PROMPT_DIET = """
You are a Nutrition and Dietetics AI assistant specializing in medication-aware dietary guidance.
Your goal is to provide general diet recommendations based on the medicines a user is taking.

USER DATA:
- List of Medicines
- Medicine Categories (e.g., diabetes, BP, thyroid)
- Dosages

REQUIRED OUTPUT STRUCTURE (JSON):
{
    "diet_guidance": "Short overview of how their medications might interact with diet.",
    "foods_to_include": ["List", "of", "beneficial", "foods"],
    "foods_to_avoid": ["List", "of", "potentially", "harmful", "foods"],
    "hydration_advice": "Advice on water intake.",
    "lifestyle_tips": ["Activity", "Sleep", "etc."],
    "disclaimer": "This is general guidance. Please consult your doctor."
}

SAFETY RULES:
1. NEVER prescribe medication changes.
2. NEVER give strict medical claims or 'cures'.
3. DO NOT recommend specific dosages or timing for medications.
4. If a drug-food interaction is well-known (e.g., Grapefruit with Statins, Leafy greens with Warfarin), prioritize mentioning it.
5. ALWAYS include the disclaimer.
6. If the medications are for diabetes, focus on glycemic control. For BP, focus on low sodium.
7. Tone: Professional, helpful, and cautious.
"""

SYSTEM_PROMPT_EXERCISE = """
You are a warm and diligent AI Physiotherapist and Nurse. 
Your goal is to provide safe, encouraging exercise suggestions based on the user's health context (medicines).

CORE PRINCIPLES:
1. Encouragement: "It's wonderful that you're thinking about staying active, dear!"
2. Modification: Suggest low-impact alternatives (walking, swimming) if they are on heavy medication.
3. Warnings: If they take BP medicine, remind them to stand up slowly and avoid overexertion.
4. Professionalism: Always include "Please check with your doctor before starting any new strenuous routine."

Tone: Supportive, elder-care focused, and medical-safe.
"""

SYSTEM_PROMPT_PRESCRIPTION = """
You are a medical prescription verification system.

Extract the following in structured JSON:
{
  "doctorName": "string",
  "doctorRegistrationNumber": "string",
  "patientName": "string",
  "medicines": [
    { "name": "string", "dosage": "string", "frequency": "string", "quantity": number }
  ],
  "prescriptionDate": "ISO date string",
  "address": "string",
  "suspiciousScore": number (0-100),
  "verdict": "VALID" or "SUSPICIOUS",
  "reason": "string"
}

Rules for scoring 'suspiciousScore':
- High score (>70, verdict: SUSPICIOUS) if:
    * No doctor name found
    * Missing or invalid registration number
    * Medicine quantity looks unrealistic for the duration
    * Visual layout suggests fake formatting
    * Date is missing or in the far future/past
    * Significant mismatched handwriting or digital fonts

Return ONLY the JSON. No preamble.
"""


# Data Models
class DietPayload(BaseModel):
    medicines: List[Dict[str, Any]]

class DietResponse(BaseModel):
    diet_guidance: str
    foods_to_include: List[str]
    foods_to_avoid: List[str]
    hydration_advice: str
    lifestyle_tips: List[str]
    disclaimer: str

class ExercisePayload(BaseModel):
    medicines: List[Dict[str, Any]]

class ExerciseResponse(BaseModel):
    exercise_plan: str
    safe_exercises: List[str]
    yoga_poses: List[str]
    breathing_exercises: List[str]
    duration_recommendations: str
    safety_notes: List[str]
    disclaimer: str

@app.post("/diet-recommendation", response_model=DietResponse)
async def get_diet_recommendation(payload: DietPayload):
    if not client:
        raise HTTPException(status_code=500, detail="Groq API Key missing")

    try:
        med_context = "\n".join([
            f"- {m.get('name')}: Category: {m.get('category')}, Dosage: {m.get('dosage')} tablets/day"
            for m in payload.medicines
        ])

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_DIET},
                {"role": "user", "content": f"User Medications:\n{med_context}"}
            ],
            temperature=0.3,
            max_tokens=2048,
            response_format={"type": "json_object"}
        )

        response_text = completion.choices[0].message.content
        return DietResponse(**json.loads(response_text))

    except Exception as e:
        print(f"DIET API ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/exercise-recommendation", response_model=ExerciseResponse)
async def get_exercise_recommendation(payload: ExercisePayload):
    if not client:
        raise HTTPException(status_code=500, detail="Groq API Key missing")

    try:
        med_context = "\n".join([
            f"- {m.get('name')}: Category: {m.get('category')}"
            for m in payload.medicines
        ])

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_EXERCISE},
                {"role": "user", "content": f"User Medications/Profile:\n{med_context}"}
            ],
            temperature=0.4,
            max_tokens=2048,
            response_format={"type": "json_object"}
        )

        response_text = completion.choices[0].message.content
        return ExerciseResponse(**json.loads(response_text))

    except Exception as e:
        print(f"EXERCISE API ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

SYSTEM_PROMPT_GENERATOR = """
You are 'Nurse Maya', a caring and professional AI Assistant at a digital pharmacy.
Your tone is like a supportive nurse - warm, polite, and deeply concerned for the user's health.

Your tasks:
1. Answer health questions (diet, exercise, side effects) using the provided medicine context. 
   - Be specific: If they take Metformin, mention glycemic index. If they take BP meds, mention salt.
2. Safety: ALWAYS prioritize safety and follow medical guidelines.
3. Disclaimer: Include a subtle medical disclaimer in your natural response.
4. Empathy: Use phrases like "I understand", "Don't you worry", "It's my pleasure to help".

NO PRESCRIPTION POLICY:
- For now, do NOT ask for or mention prescriptions. Assume we have what we need.

ORDER CONFIRMATION POLICY:
- If the user wants to order something, ALWAYS ask them: "Shall I go ahead and confirm that order for you, dear?". 
- Do NOT place the order until they explicitly say yes in the conversation.

Example: "It's so important that you're taking care of your nutrition while on your BP medication, dear! I'd recommend focusing on low-sodium foods like..."
"""

@traceable(run_type="llm", name="Groq Llama 3.1")
def call_ai(messages, temperature=0.4, max_tokens=8192, response_format=None):
    if not client:
        raise Exception("Groq client not initialized")
    
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        stream=False,
        response_format=response_format
    )
    
    # Explicitly attach usage to metadata for LangSmith token tracking
    run = langsmith.run_helpers.get_current_run_tree()
    if run and hasattr(completion, 'usage'):
        # LangSmith dashboard looks for 'usage' in metadata to show tokens in front
        usage_data = completion.usage.dict() if hasattr(completion.usage, 'dict') else completion.usage
        run.metadata["usage"] = usage_data
        
    return completion

@app.post("/normalize", response_model=NormalizationResponse)
@traceable(run_type="chain", name="Order Normalization")
async def normalize(payload: InputPayload):
    if not client:
        raise HTTPException(status_code=500, detail="Groq API Key missing")

    try:
        # Construct the prompt with context
        context_str = f"Medicine Context:\n{payload.medicine_context}\n\n" if payload.medicine_context else ""
        
        history_str = ""
        if payload.chat_history:
             history_str = "Chat History:\n" + "\n".join([f"{msg.role}: {msg.parts[0]}" for msg in payload.chat_history]) + "\n\n"

        user_message = context_str + history_str + f"User: {payload.text}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_CLASSIFIER},
            {"role": "user", "content": user_message}
        ]

        # Use the traceable helper
        completion = call_ai(
            messages=messages,
            temperature=0.4,
            max_tokens=8192,
            response_format={"type": "json_object"}
        )
        
        # Propagate tokens to the parent trace too
        parent_run = langsmith.run_helpers.get_current_run_tree()
        if parent_run and hasattr(completion, 'usage'):
            parent_run.metadata["usage"] = completion.usage.dict() if hasattr(completion.usage, 'dict') else completion.usage
        
        response_text = completion.choices[0].message.content
        print(f"DEBUG: Raw LLM Response: {response_text}") # LOGGING

        # Parse result
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as e:
             print(f"JSON Error: {e}")
             return NormalizationResponse(
                 type="chat",
                 message="I'm sorry, I encountered an error processing your request."
             )
            
        if data.get("type") == "chat":
            return NormalizationResponse(
                type="chat",
                message=data.get("message", "I am here to help.")
            )

        elif data.get("type") == "cancel":
            return NormalizationResponse(
                type="cancel",
                message=data.get("message", "Order cancelled.")
            )
            
        elif data.get("type") == "query_history":
            return NormalizationResponse(
                type="query_history",
                medicine_filter=data.get("medicine_filter")
            )
            
        elif data.get("type") == "order":
            validated_orders = []
            safety_alerts = []
            
            orders_list = data.get("orders", [])
            for item in orders_list:
                # Fill missing default values if not present in LLM response
                if 'confidence' not in item: item['confidence'] = 1.0
                if 'operation' not in item: item['operation'] = 'add'
                if item.get('quantity_converted') is None: 
                     # Fallback conversion logic
                     q = item.get('quantity')
                     if q is None: q = 1
                     u = (item.get('unit') or 'tablet').lower()
                     
                     if u in ['strip', 'strips']: item['quantity_converted'] = q * 10
                     elif u in ['box', 'boxes']: item['quantity_converted'] = q * 100
                     else: item['quantity_converted'] = q
                
                # Double check after assignment or if it was already there
                qty_conv = item.get('quantity_converted')
                if qty_conv is None: qty_conv = 0
                
                # Validation Logic
                if qty_conv > 100:
                    safety_alerts.append(f"High quantity detected for {item.get('medicine_name')}. Please verify.")
                validated_orders.append(item)

            try:
                return NormalizationResponse(
                    type="order",
                    orders=validated_orders,
                    safety_alerts=safety_alerts
                )
            except Exception as ve:
                print(f"Validation Error: {ve}")
                return NormalizationResponse(
                    type="chat",
                    message="I identified your request but couldn't structure the order correctly. Please try again with simpler terms."
                )
        
        else:
             return NormalizationResponse(
                 type="chat", 
                 message="I am not sure how to process that. I can help you order medicines."
             )

    except Exception as e:
        print(f"SERVER ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-response")
@traceable(run_type="chain", name="Response Generation")
async def generate_response(payload: ResponseGenerationPayload):
    if not client:
        raise HTTPException(status_code=500, detail="Groq API Key missing")

    try:
        user_message = f"Context (Order History):\n{payload.context}\n\nUser Question: {payload.query}"
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_GENERATOR},
            {"role": "user", "content": user_message}
        ]

        completion = call_ai(
            messages=messages,
            temperature=0.7,
            max_tokens=1024
        )
        
        # Propagate tokens to parent
        parent_run = langsmith.run_helpers.get_current_run_tree()
        if parent_run and hasattr(completion, 'usage'):
            parent_run.metadata["usage"] = completion.usage.dict() if hasattr(completion.usage, 'dict') else completion.usage
        
        return {"message": completion.choices[0].message.content}
    except Exception as e:
        print(f"Groq Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "Medical Intelligence (Groq Powered)"}

@app.post("/api/prescription/process", response_model=PrescriptionUploadResponse)
@traceable(run_type="chain", name="Prescription Processing")
async def process_prescription(file: UploadFile = File(...), userId: str = Form("default_user")):
    if not client:
        raise HTTPException(status_code=500, detail="Groq API Key missing")

    print(f"\n[PROCESS] Starting processing for user: {userId}")
    print(f"[DEBUG] Raw UserId type: {type(userId)}")
    
    # Rename UPLOAD_DIR for Node.js temp if needed, but here we just process
    # We will save to a temp file for OCR
    temp_file_path = f"temp_{int(time.time())}_{file.filename}"
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"[PROCESS] Temp file saved for OCR: {temp_file_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save temp file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File saving failed: {str(e)}")

    # 2. Call OCR.Space API
    print(f"[PROCESS] Starting OCR extraction via OCR.Space...")
    try:
        with open(temp_file_path, 'rb') as f:
            r = requests.post(
                'https://api.ocr.space/parse/image',
                files={'file': f},
                data={'apikey': OCR_API_KEY, 'language': 'eng', 'isOverlayRequired': False},
                timeout=90
            )
        
        if not r.ok:
            print(f"[ERROR] OCR API failed with status {r.status_code}")
            raise Exception(f"OCR Server returned status {r.status_code}: {r.text}")

        try:
            ocr_result = r.json()
        except Exception:
            print(f"[ERROR] OCR Response is not valid JSON")
            raise Exception(f"Failed to parse OCR response as JSON: {r.text[:200]}")
            
        print(f"[DEBUG] OCR Full Response: {ocr_result}")
        
        if ocr_result.get("IsErroredOnProcessing"):
            err_msg = ocr_result.get("ErrorMessage", "OCR Error")
            print(f"[ERROR] OCR processing error: {err_msg}")
            raise Exception(err_msg)
        
        parsed_results = ocr_result.get("ParsedResults", [])
        if not parsed_results:
            print(f"[ERROR] No text found in OCR result")
            raise Exception("No text found in prescription")
        
        raw_ocr_text = parsed_results[0].get("ParsedText", "")
        print(f"[PROCESS] OCR Successful. Extracted {len(raw_ocr_text)} characters.")

        if len(raw_ocr_text.strip()) < 10:
            print(f"[ERROR] OCR text too short ({len(raw_ocr_text)} chars)")
            raise Exception("The text extracted from the image is too sparse. Please provide a clearer image.")
        
    except Exception as e:
        if os.path.exists(temp_file_path): os.remove(temp_file_path)
        print(f"[ERROR] OCR Pipeline Failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OCR Processing failed: {str(e)}")

    # 3. AI Data Extraction
    print(f"[PROCESS] Starting AI analysis (Llama 3.1)...")
    try:
        user_prompt = f"Extract medicine details from this OCR text:\n\n{raw_ocr_text}"
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_PRESCRIPTION},
            {"role": "user", "content": user_prompt}
        ]

        completion = call_ai(
            messages=messages,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        # Propagate tokens to parent
        parent_run = langsmith.run_helpers.get_current_run_tree()
        if parent_run and hasattr(completion, 'usage'):
            parent_run.metadata["usage"] = completion.usage.dict() if hasattr(completion.usage, 'dict') else completion.usage
        
        ai_raw_response = completion.choices[0].message.content
        extracted_data = json.loads(ai_raw_response)
        print(f"[PROCESS] AI analysis complete. Verdict: {extracted_data.get('verdict', 'UNKNOWN')}")
        
        # 6. Final Status Logic
        suspicious_score = extracted_data.get("suspiciousScore", 0)
        final_status = "AI_APPROVED"
        
        if suspicious_score > 70:
            final_status = "MANUAL_REVIEW"
            print(f"[PROCESS] High risk detected (Score: {suspicious_score}). Flagging for Manual Review.")

        if os.path.exists(temp_file_path): os.remove(temp_file_path)
        print(f"[PROCESS] Prescription record processed successfully.")

        return PrescriptionUploadResponse(
            id="AI_PROCESSED",
            extractedData=extracted_data,
            status=final_status,
            suspiciousScore=suspicious_score,
            rawOcrText=raw_ocr_text
        )

    except Exception as e:
        if os.path.exists(temp_file_path): os.remove(temp_file_path)
        print(f"[ERROR] AI Extraction Failed: {str(e)}")
        # Return OCR result even if AI fails
        return PrescriptionUploadResponse(
            id="ocr_only",
            extractedData=PrescriptionExtraction(),
            status="AI_REJECTED",
            suspiciousScore=0,
            rawOcrText=raw_ocr_text
        )

@app.post("/api/voice/transcribe")
@traceable(run_type="chain", name="Voice Transcription")
async def transcribe_audio(file: UploadFile = File(...)):
    if not client:
        raise HTTPException(status_code=500, detail="Groq API Key missing")

    temp_file_path = f"temp_voice_{int(time.time())}_{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        file_size = os.path.getsize(temp_file_path)
        print(f"[VOICE] Received file: {file.filename}, Size: {file_size} bytes")

        if file_size < 100:
             print("[VOICE] Warning: Audio file is extremely small. Might be empty.")

        with open(temp_file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=audio_file, # Pass the file-like object directly
                model="whisper-large-v3",
                response_format="json",
                language="en",
                prompt="Short greetings: Hi, Hello, Hey. Pharmacy help: I need medicine, order dolo, how are you."
            )
        
        text = transcription.text.strip()
        print(f"[VOICE] Transcribed text: {text}")

        # Catch common Whisper hallucinations for silence/noise
        hallucinations = ["Subtitles by", "Amara.org", "Thank you for watching"]
        if any(h.lower() in text.lower() for h in hallucinations) and len(text) < 50:
             print(f"[VOICE] Hallucination detected: '{text}'. Returning empty dot.")
             text = "."

        return {"text": text}
    except Exception as e:
        print(f"TRANSCRIPTION ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/api/prescription/normalize-raw")
async def normalize_raw_prescription(payload: NormalizationRawPayload):
    if not client:
        raise HTTPException(status_code=500, detail="Groq API Key missing")

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_PRESCRIPTION},
                {"role": "user", "content": f"Raw OCR Text:\n{payload.raw_text}"}
            ],
            temperature=0.2,
            max_tokens=2048,
            response_format={"type": "json_object"}
        )

        response_text = completion.choices[0].message.content
        extracted_data = json.loads(response_text)
        
        return {"extractedData": extracted_data}
    except Exception as e:
        print(f"RE-NORMALIZATION ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
