from firecrawl import FirecrawlApp
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
import json
import os
import time
from langchain_groq import ChatGroq

def scrape_data(url, retries=3, delay=5):
    load_dotenv()
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        raise EnvironmentError("FIRECRAWL_API_KEY not found in environment variables.")

    app = FirecrawlApp(api_key=api_key)

    for attempt in range(1, retries + 1):
        print(f"Scraping attempt {attempt} for URL: {url}")
        try:
            response = app.scrape_url(url)
            data = response.model_dump()
            if 'markdown' in data:
                print("Scraping successful.") 
                return data['markdown']
            else:
                raise KeyError("No 'markdown' key found in response.")
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt < retries:
                time.sleep(delay)
            else:
                raise

def save_raw_data(raw_data, timestamp, output_folder='output'):
    os.makedirs(output_folder, exist_ok=True)
    raw_path = os.path.join(output_folder, f"raw_data_{timestamp}.md")
    with open(raw_path, 'w', encoding='utf-8') as f:
        f.write(raw_data)
    print(f"Raw data saved to {raw_path}")

def format_data(data, fields=None, max_length=3000):
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found.")

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama3-8b-8192", 
        temperature=0.3
    )

    if fields is None:
        fields = ["title","type","release_year","genre","rating",
                  "cast", "synopsis"
            
        ]

    truncated_data = data[:max_length]

    system_msg = (
        "You are an intelligent data extraction assistant. Your task is to extract structured JSON data from real estate listings text. "
        "Only return pure JSON — no extra comments or explanations."
    )

    user_msg = (
        f"Extract the following fields:\n{fields}\n\n"
        f"Text:\n{truncated_data}"
    )

    response = llm.invoke([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ])

    try:
        parsed_json = json.loads(response.content.strip())
        return parsed_json
    except json.JSONDecodeError as e:
        print(f"JSON error: {e}")
        raise ValueError("Could not parse JSON from LLM output.")

def save_formatted_data(formatted_data, timestamp, output_folder="output"):
    os.makedirs(output_folder, exist_ok=True)

    json_path = os.path.join(output_folder, f"formatted_data_{timestamp}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, indent=4)
    print(f"Formatted data saved to {json_path}")

    if isinstance(formatted_data, dict):
        formatted_data = [formatted_data]

    if isinstance(formatted_data, list):
        df = pd.DataFrame(formatted_data)
        excel_path = os.path.join(output_folder, f"formatted_data_{timestamp}.xlsx")
        df.to_excel(excel_path, index=False)
        print(f"Formatted data saved to Excel at {excel_path}")

if __name__ == "__main__":
    url = "https://www.imdb.com/imdbpicks/summer-watch-guide/?ref_=hm_edcft_csegswg_ft_1_i"

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_data = scrape_data(url)
        save_raw_data(raw_data, timestamp)
        formatted_data = format_data(raw_data)
        save_formatted_data(formatted_data, timestamp)
    except Exception as e:
        print(f"An error occurred: {e}")