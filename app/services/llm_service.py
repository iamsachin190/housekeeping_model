import base64
import logging
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from app.config import settings
from app.models import InspectionResult

logger = logging.getLogger("BIMS_LLM")

SYSTEM_PROMPT = """
You are the 'Master Inspector' for a Building Management System. 
Your job is to evaluate the cleanliness of a facility based on an image.

Evaluate based on these 4 criteria:
1. Spills (Liquids on floor/surfaces)
2. Dust (Accumulation on vents, surfaces)
3. Trash (Debris, paper, cups)
4. Stains (Discoloration on carpets/walls)

You will be provided with:
1. An image (or a grid of images).
2. Context from similar past images (RAG).

Return STRICT JSON matching this schema:
{
    "status": "Clean" or "Dirty",
    "confidence": 0.0 to 1.0,
    "reasoning": "string explanation",
    "issues_detected": ["list", "of", "issues"]
}
"""

parser = JsonOutputParser(pydantic_object=InspectionResult)

def get_groq_client():
    if not settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set. Please check your .env file.")
    return ChatGroq(
        model_name="llama-3.2-11b-vision-preview", 
        temperature=0,
        api_key=settings.GROQ_API_KEY,
        model_kwargs={"response_format": {"type": "json_object"}}
    )

def get_gemini_client():
    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set. Please check your .env file.")
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=settings.GOOGLE_API_KEY,
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }
    )

async def analyze_image_with_failover(image_path: str, rag_context: str) -> dict:
    """
    Tries Groq first. If 429 or error, switches to Gemini.
    """
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    
    image_url = f"data:image/jpeg;base64,{encoded_string}"
    
    human_message = HumanMessage(
        content=[
            {"type": "text", "text": f"Context from database:\n{rag_context}\n\nAnalyze this image:"},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    )
    
    sys_msg = SystemMessage(content=SYSTEM_PROMPT)

    # 1. Try Groq
    try:
        logger.info("Attempting analysis via Groq...")
        llm = get_groq_client()
        chain = llm | parser
        result = await chain.ainvoke([sys_msg, human_message])
        return result
    except Exception as e:
        logger.error(f"Groq failed: {str(e)}. Switching to Gemini...")
        
        # 2. Failover to Gemini
        try:
            llm = get_gemini_client()
            chain = llm | parser
            result = await chain.ainvoke([sys_msg, human_message])
            return result
        except Exception as e2:
            logger.critical(f"Gemini also failed: {str(e2)}")
            raise e2
