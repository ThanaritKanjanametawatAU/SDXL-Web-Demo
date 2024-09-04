from fasthtml.common import *
from dotenv import load_dotenv

load_dotenv()
TESTVAR = os.environ.get("TESTVAR")

app, rt = fast_app()
# This is test Comment

@rt('/')

def get():
    return Div(P(f"Hello World! {TESTVAR}"), hx_get="/change")



serve()