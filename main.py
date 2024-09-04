from fasthtml.common import *
from dotenv import load_dotenv

load_dotenv()
TESTVAR = os.environ.get("TESTVAR")

app, rt = fast_app()
# This is test Comment

@rt('/')
def generation_preview(id):
    if os.path.exists(f"gens/{id}.png"):
        return Div(Img(src=f"/gens/{id}.png"), id=f'gen-{id}')
    else:
        return Div("Generating...", id=f'gen-{id}',
                   hx_post=f"/generations/{id}",
                   hx_trigger='every 1s', hx_swap='outerHTML')


@app.post("/generations/{id}")
def get(id: int): return generation_preview(id)


@app.post("/")
def post(prompt: str):
    id = len(generations)
    generate_and_save(prompt, id)
    generations.append(prompt)
    clear_input = Input(id="new-prompt", name="prompt", placeholder="Enter a prompt", hx_swap_oob='true')
    return generation_preview(id), clear_input


@app.get("/")
def get():
    inp = Input(id="new-prompt", name="prompt", placeholder="Enter a prompt")
    add = Form(Group(inp, Button("Generate")), hx_post="/", target_id='gen-list', hx_swap="afterbegin")
    gen_list = Div(id='gen-list')
    return Title('Image Generation Demo'), Main(H1('Magic Image Generation'), add, gen_list, cls='container')



serve()