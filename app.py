from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv
load_dotenv()
import json
import re

from groq import Groq
import os

app = FastAPI()

# CORS configuration
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
    "*",  # REMOVE IN PRODUCTION
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)


def extract_json(results):
    json_content = re.sub(r'```json\s+|\s+```', '', results)
    result = json.loads(json_content)
    return result


@app.get("/ocr")
def ocr_agent(image_url:str):
    
    completion = client.chat.completions.create(
        model="llama-3.2-90b-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """Extract the key details from this restaurant bill and return **only** a valid JSON object.  
                                    Strictly follow this format and do not include any explanations, additional text, or comments.  

                                    Return **only** this JSON format:  
                                    ```json
                                    {{
                                        "Restaurant": "restaurant name",
                                        "Date": "date",
                                        "Time": "time",
                                        "Items": [
                                            {
                                                "Name": "item name",
                                                "Price": "price",
                                                "Quantity": "quantity"
                                            }
                                        ],
                                        "Total": "Total Billing amount"
                                    }}
                                    ```
                                    If any field is missing in the bill, leave it as an empty string. Do not add extra information or modify the structure.
                                    <STRICT>RETURN ONLY THE JSON OBJECT.</STRICT>
                        """,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                        },
                    },
                ],
            }
        ],
        temperature=0,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )
    
    results = completion.choices[0].message.content
    
    return extract_json(results)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)