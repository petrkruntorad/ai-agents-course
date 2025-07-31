import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from pathlib import Path
import psycopg2

# Load environment variables
if Path('.env.local').exists():
    load_dotenv('.env.local')
else:
    load_dotenv()

# Database connection
db_conn = psycopg2.connect("dbname=%s user=%s host=%s port=%d password=%s" % (
    os.environ.get("DB_NAME"),
    os.environ.get("DB_USER"),
    os.environ.get("DB_HOST"),
    int(os.environ.get("DB_PORT")),
    os.environ.get("DB_PASS")
))

# Model
llm = ChatOpenAI(model="gpt-4.1-nano", api_key=os.environ.get("OPENAI_API_KEY"))

# Open a cursor to perform database operations
db_cur = db_conn.cursor()

# Create table 'clients' if not exists
db_cur.execute("""
CREATE TABLE IF NOT EXISTS clients (
    id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    email VARCHAR(255) NOT NULL,
    firstname VARCHAR(255) NOT NULL,
    lastname VARCHAR(255) NOT NULL,
    requested_service VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

db_conn.commit()

@tool
def get_all_rows_by_client_email(email: str) -> list:
    """Select all rows for client with specified email."""
    db_cur.execute('SELECT * FROM clients WHERE clients.email = %s', (email,))
    return db_cur.fetchall()

@tool
def get_all_rows_by_client_email_and_service(email: str, service_name: str) -> list:
    """Select rows from a table that match email and service."""
    db_cur.execute('SELECT * FROM clients WHERE clients.email = %s AND clients.requested_service = %s', (email, service_name))
    return db_cur.fetchall()

@tool
def insert_new_request(email: str, firstname: str, lastname: str, service_name: str) -> str:
    """Insert rows in table clients based on extracted information from user input."""
    db_cur.execute('INSERT INTO clients (email, firstname, lastname, requested_service) VALUES (%s, %s, %s, %s);', (email, firstname, lastname, service_name))
    db_conn.commit()
    return 'Request for service %s inserted for user %s %s with email %s' % (
        service_name,
        firstname,
        lastname,
        service_name,
    )

tools = [get_all_rows_by_client_email, get_all_rows_by_client_email_and_service, insert_new_request]

# Agent
graph = create_react_agent(
    model=llm,
    tools=tools,
    prompt="You are a tool for extracting user data and requested services from a subject. You can also return and evaluate whether the data is already stored in the database based on the email and requested service. If the data is already registered, do not re-enter it into the database."
)

# ---------------------------
# Run the graph
# MESSAGES are stored ONLY within the graph state !!!!
# EACH USER INPUT IS A NEW STATE !!!!
# =>  NO HISTORY for chat interaction !!!!!!
# ---------------------------
while True:
    print("Zadejte text (pro ukončení napište na nový řádek [[exit]]):")
    lines = []
    while True:
        line = input()
        if line == "[[exit]]":
            break
        lines.append(line)
    user_input = "\n".join(lines)

    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    for event in graph.stream({"messages": ("user", user_input)}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)