import boto3

# Create a DynamoDB client
dynamodb = boto3.client('dynamodb', region_name='ap-south-1')  # Change region if needed

# Create the Employee table
table_name = 'Users_table'

try:
    response = dynamodb.create_table(
        TableName=table_name,
        KeySchema=[
            {
                'AttributeName': 'user_id',  # Partition Key
                'KeyType': 'HASH'
            }
        ],
        AttributeDefinitions=[
            {
                'AttributeName': 'user_id',
                'AttributeType': 'S'  # S = String (can also use N for Number, B for Binary)
            }
        ],
        ProvisionedThroughput={
            'ReadCapacityUnits': 5,
            'WriteCapacityUnits': 5
        }
    )

    print(f"Creating table {table_name}...")
    print("Status:", response['TableDescription']['TableStatus'])

except dynamodb.exceptions.ResourceInUseException:
    print(f"Table {table_name} already exists.")

