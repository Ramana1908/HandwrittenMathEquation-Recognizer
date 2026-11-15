"""Web demo: upload a photo or draw an equation, and see how it is read and solved.

    pip install -r requirements.txt
    python app.py            # then open http://127.0.0.1:7860
"""
import os

import cv2
import gradio as gr
import numpy as np
import sympy as sp

from recognizer import load_model, recognize

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.join(HERE, 'examples')
MODELS = {'CNN': load_model('cnn'), 'ViT': load_model('vit')}

PRETTY = {'div': '/', 'int': '∫', 'lt': '<', 'gt': '>', 'leq': '≤', 'geq': '≥',
          'neq': '≠', '*': '·'}
LATEX = [{'left': '$$', 'right': '$$', 'display': True}]


def format_result(result):
    """Solver output (number, list of solutions, relation, bool or LaTeX string) -> LaTeX."""
    if isinstance(result, str):
        return result
    if isinstance(result, bool):
        return r'\text{%s}' % result
    if isinstance(result, (list, tuple)):
        if not result:
            return r'\text{no solution}'
        return 'x = ' + ',\\ '.join(sp.latex(v) for v in result)
    try:
        return sp.latex(result)
    except Exception:
        return r'\text{%s}' % result


def draw_boxes(gray, boxes):
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    t = max(2, gray.shape[1] // 400)
    for x, y, w, h in boxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (230, 60, 60), t)
    return img


def solve(image, model_name):
    empty = ('', None, [])
    if image is None:
        return ('',) + empty
    if isinstance(image, dict):                      # Sketchpad value
        image = image.get('composite')
        if image is None or image.min() > 200:      # nothing drawn on the white page
            return ('Draw an equation on the canvas first.',) + empty
    from recognizer import to_gray
    gray = to_gray(image)
    out = recognize(gray, MODELS[model_name])
    if not out['symbols']:
        return ('No symbols found. Use dark ink on a light background.',) + empty

    read_as = (out['latex'] or out['equation']).replace('*', r' \cdot ')
    lines = [f"**Read as**\n\n$${read_as}$$"]
    if out['error']:
        lines.append(f"**Result:** {out['error']}")
    else:
        lines.append(f"**Result**\n\n$${format_result(out['result'])}$$")
    symbols = ' '.join(PRETTY.get(s, s) for s in out['symbols'])
    patches = [(cv2.resize(p, (90, 90), interpolation=cv2.INTER_NEAREST), PRETTY.get(s, s))
               for p, s in zip(out['patches'], out['symbols'])]
    return ('\n\n'.join(lines), f"`{out['equation']}`  ·  symbols: {symbols}",
            draw_boxes(gray, out['boxes']), patches)


BLANK_PAGE = np.full((300, 900, 3), 255, np.uint8)   # white paper for the canvas

EXAMPLES = [os.path.join(EXAMPLES_DIR, f) for f in sorted(os.listdir(EXAMPLES_DIR))
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

INTRO = """# Handwritten Math Recognizer
Upload a photo of a handwritten equation, or draw one, and it is read and solved.

Works with one line of math in dark ink on light paper, using digits, `+ − ( ) = x d / ∫ < > ≤ ≥ ≠`, \
exponents and a dot for multiplication. It can't read `×`, `÷` or decimals.
"""

with gr.Blocks(title='Handwritten Math Recognizer', analytics_enabled=False) as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.Tab('Upload'):
                    upload = gr.Image(type='numpy', image_mode='RGB', label='Equation image',
                                      sources=['upload', 'webcam', 'clipboard'], height=300)
                    upload_btn = gr.Button('Solve', variant='primary')
                    gr.Examples(EXAMPLES, inputs=upload, label='Or try an example')
                with gr.Tab('Draw'):
                    sketch = gr.Sketchpad(value=BLANK_PAGE, type='numpy', image_mode='RGB',
                                          canvas_size=(900, 300), fixed_canvas=True,
                                          brush=gr.Brush(default_size=6, colors=['#000000'],
                                                         color_mode='fixed'),
                                          layers=False, transforms=(), height=340,
                                          label='Draw an equation (one line)')
                    sketch_btn = gr.Button('Solve', variant='primary')
            model = gr.Radio(list(MODELS), value='CNN', label='Classifier',
                             info='The CNN is more accurate on real handwriting.')
        with gr.Column(scale=2):
            answer = gr.Markdown(latex_delimiters=LATEX)
            details = gr.Markdown()
            boxes = gr.Image(label='Detected symbols', interactive=False)
            gallery = gr.Gallery(label='What the classifier saw', columns=10, height=150,
                                 object_fit='contain', allow_preview=False)

    outputs = [answer, details, boxes, gallery]
    upload_btn.click(solve, [upload, model], outputs)
    upload.change(solve, [upload, model], outputs)   # also runs when an example is picked
    sketch_btn.click(solve, [sketch, model], outputs)

if __name__ == '__main__':
    demo.launch(server_name=os.environ.get('HOST', '127.0.0.1'))
