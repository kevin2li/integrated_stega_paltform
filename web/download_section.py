import os
import sys
sys.path.append(os.path.abspath('..'))

from h2o_wave import Q, app, handle_on, main, on, ui
from icecream import ic
from web.utils import layout1, layout2, empty

#================================================================
# 下载专区
#================================================================
@on(arg='dataset_text')
async def serve(q:Q):
    empty(q)
    q.page['meta'] = layout1
    q.page['content'] = ui.form_card(box='content', items=[
        ui.text('正在开发中...')
    ])
    # q.page['chat'] = ui.chat_card(box='3 3 5 8', title='chat')
    await q.page.save()

@on(arg='dataset_img')
async def serve(q:Q):
    empty(q)
    cover_md = '''
    [BOSSBASE_v1.01](http://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip)'''
    spatial_md = """
    - WOW
    - HUGO
    - S-UNIWARD
    - HILL
    - MiPOD
    """
    freq_md = """
    - J-UNIWARD
    - UERD
    - J-MiPOD
    """
    q.page['content'] = ui.form_card(box='content', items=[
        ui.text('# 1. Cover'),
        ui.link('BOSSbase_v1.01', 'http://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip'),
        ui.text('# 2. 空域自适应隐写'),
        ui.text(spatial_md, size='l'),
        ui.text('# 3. 频域自适应隐写'),
        ui.text(freq_md),
        ui.text('# 4. 神经网络隐写')
    ])
    await q.page.save()