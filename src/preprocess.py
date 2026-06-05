import re
import pandas as pd

def clean_text(text):
    if pd.isna(text):
        return ""
    
    text = str(text) 
    text = text.lower()

    #remove url
    text = re.sub(r"http\s+", "", text)

    #remove mention
    text = re.sub(r"@\w+", "", text)
    text = text.replace("#", "")
    text = text.replace("/n", " ")

    text = re.sub(r"\s+", " ", text)

    return text.strip()




