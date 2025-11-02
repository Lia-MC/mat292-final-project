import logging

with open('my_app.log', 'r') as f:
    for line in f:
        print(line.strip()) # Process each log line