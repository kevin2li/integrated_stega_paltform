from h2o_wave import Q, main, app, ui, on, handle_on
from icecream import ic
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import bwm
import imutils
import os

@app('/demo')
async def serve(q:Q):
    links = q.args.user_files
    if links:
        items = [ui.text_xl('Files uploaded!')]
        for link in links:
            local_path = await q.site.download(link, './upload')
            ic(local_path)
            #
            # The file is now available locally; process the file.
            # To keep this example simple, we just read the file size.
            #
            size = os.path.getsize(local_path)

            items.append(ui.link(label=f'{os.path.basename(link)} ({size} bytes)', download=True, path=link))
            # Clean up
            # os.remove(local_path)

        items.append(ui.button(name='back', label='Back', primary=True))
        q.page['example'].items = items
    else:
        q.page['example'] = ui.form_card(box='1 1 4 10', items=[
            ui.text_xl('Upload some files'),
            ui.file_upload(name='user_files', label='Upload', multiple=True),
        ])
    await q.page.save()

