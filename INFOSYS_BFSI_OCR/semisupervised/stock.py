import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
import time
from pymongo import MongoClient  # Import pymongo for MongoDB integration

# Start the timer to measure performance
start_time = time.time()

# Create a list that contains the URL for each page.
pages = []

# The total number of pages in the entire set is 5
for page_number in range(1, 5):  # Change the range for more pages
    url_start = 'https://www.centralcharts.com/en/price-list-ranking/'
    url_end = 'ALL/asc/ts_19-us-nasdaq-stocks--qc_1-alphabetical-order?p='
    url = url_start + url_end + str(page_number)
    pages.append(url)

# Initialize a list to hold the stock data
values_list = []

# Cycle through each page.
for page in pages:
    try:
        webpage = requests.get(page)
        webpage.raise_for_status()  # Check if request was successful
        soup = bs(webpage.text, 'html.parser')

        # Find the table containing stock data
        stock_table = soup.find('table', class_='tabMini tabQuotes')
        
        if stock_table:  # Check if the table was found
            tr_tag_list = stock_table.find_all('tr')

            # Loop through each row in the table, skipping the header row
            for each_tr_tag in tr_tag_list[1:]:
                td_tag_list = each_tr_tag.find_all('td')

                # Collect data from the first 7 <td> elements
                row_values = [td.text.strip() for td in td_tag_list[:7]]
                values_list.append(row_values)
        else:
            print(f"No table found on page: {page}")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching page {page}: {e}")
        continue

# Convert the list of values into a DataFrame
df = pd.DataFrame(values_list, columns=['Company', 'Change', 'Open', 'High', 'Low', 'Volume', 'Market Cap'])

# Save the DataFrame to a CSV file
df.to_csv('stock_data.csv', index=False)

# MongoDB connection setup
try:
    # Replace the following URI with your MongoDB connection string
    client = MongoClient("mongodb://localhost:27017/")  # Adjust the URI for your MongoDB setup
    db = client['stock_data_db']  # Database name
    collection = db['stocks']  # Collection name

    # Convert DataFrame to a list of dictionaries and insert into MongoDB
    stock_data_dict = df.to_dict(orient='records')
    collection.insert_many(stock_data_dict)

    print("Stock data successfully inserted into MongoDB!")

except Exception as e:
    print(f"Error connecting to MongoDB: {e}")

# Create HTML file to visualize the stock data
html_table = df.to_html(index=False, escape=False)

html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Data Visualization</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f4f4f4;
        }}
        h1 {{
            text-align: center;
        }}
        table {{
            width: 100%;
            margin-top: 20px;
            border-collapse: collapse;
            border: 1px solid #ddd;
        }}
        th, td {{
            padding: 12px;
            text-align: center;
            border: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        tr:hover {{
            background-color: #ddd;
        }}
    </style>
</head>
<body>
    <h1>Stock Data Overview</h1>
    {html_table}
</body>
</html>
"""

# Write the HTML content to a file
with open('stock_data_visualization.html', 'w') as file:
    file.write(html_content)

# Print the elapsed time
print(f"--- {time.time() - start_time} seconds ---")
