from fasthtml.common import *
from dotenv import load_dotenv

load_dotenv()
TESVAR = os.getenv("TESTVAR")
app, rt = fast_app()

@rt('/')

def get():
    return Div(P(f"Hello World! {TESVAR}"), hx_get="/change")



serve()