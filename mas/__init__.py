from dotenv import load_dotenv

# Credentials are read by mas.settings when they are first needed, not here.
# This used to be `os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE")`,
# which raised TypeError rather than a readable message when .env was absent, and
# made import order load-bearing: mas.llm read the variables at import, so
# importing it before mas failed.
load_dotenv()
