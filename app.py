from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()
import json
import re
import os

from groq import Groq

app = Flask(__name__)
CORS(app)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)


def extract_json(results):
    json_content = re.sub(r"```json\s+|\s+```", "", results)
    result = json.loads(json_content)
    return result


@app.route("/")
def index():
    return "Welcome to the OCR"


@app.route("/ocr", methods=["GET"])
def ocr_agent():
    image_url = request.args.get("image_url")

    completion = client.chat.completions.create(
        model="llama-3.2-90b-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """
                                    Analyze this receipt image and extract the information in JSON format:
                                    Strictly follow this format and do not include any explanations, additional text, or comments.  
                                    Return **only** this JSON format:
                                    ```json
                                    {{
                                        "amount": number,
                                        "date": "ISO date string",
                                        "description": "string",
                                        "merchantName": "string",
                                        "category": "string"
                                    }}
                        
                                    If its not a recipt, return an empty object.
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

    return jsonify(extract_json(results))


pp = """
Extract the key details from this restaurant bill and return **only** a valid JSON object.  
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
"""

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8000)
