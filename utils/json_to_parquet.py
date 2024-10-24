import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import json

# Get the directory where the script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Create full path to input and output files
input_file = os.path.join(current_dir, 'steam_reviews.json')
output_file = os.path.join(current_dir, 'steam_reviews_sample.parquet')

def clean_json_line(line):
    # Remove trailing comma and whitespace
    line = line.strip()
    if line.endswith(','):
        line = line[:-1]
    return line

try:
    valid_rows = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        # Skip the first line containing '['
        first_line = f.readline()
        print("First line:", first_line)
        
        # Read and process the second line to see what we're dealing with
        second_line = f.readline()
        print("Second line:", second_line)
        
        # Reset file pointer
        f.seek(0)
        
        # Skip the opening bracket
        f.readline()
        
        count = 0
        while len(valid_rows) < 1000 and count < 2000:  # Limit attempts
            line = f.readline()
            if not line or line.strip() == ']':
                break
                
            cleaned_line = clean_json_line(line)
            
            try:
                if cleaned_line:
                    row = json.loads(cleaned_line)
                    valid_rows.append(row)
                    if len(valid_rows) % 10 == 0:
                        print(f"Processed {len(valid_rows)} valid records")
            except json.JSONDecodeError as je:
                print(f"Failed to parse line: {cleaned_line[:100]}...")
                print(f"Error: {str(je)}")
                continue
            
            count += 1
    
    print(f"Total valid records: {len(valid_rows)}")
    
    if valid_rows:
        # Convert to DataFrame and sample
        df = pd.DataFrame(valid_rows)
        df_sample = df.sample(n=min(200, len(df)), random_state=42)
        
        # Save to parquet
        table_sample = pa.Table.from_pandas(df_sample)
        pq.write_table(table_sample, output_file)
        print(f"Successfully created sample dataset with {len(df_sample)} random samples: {output_file}")
    else:
        print("No valid records found to process!")
        
except Exception as e:
    print(f"Error: {str(e)}")