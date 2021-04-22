import os
import sys
sys.path.append(os.path.abspath('..'))

from h2o_wave import Q, app, handle_on, main, on, ui
from icecream import ic
from web.utils import layout1, layout2, empty

#================================================================
# 文本隐写分析
#================================================================
@on()
async def text_steganalysis(q:Q):
    del q.page['content_left1']
    del q.page['content_left2']
    q.page['meta'] = layout2
    q.page['v_nav'].value = '#menu/steganalysis'
    q.page['content_left'] = ui.form_card(box='content_left', items=[
        ui.text('马上弄')
    ])
    q.page['content_right'] = ui.form_card(box='content_right', items=[
        ui.text('马上弄2')
    ])
    await q.page.save()