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

@on()
async def text_dataset_level(q:Q):
    q.page['content_left'] = ui.form_card(box=ui.box('content_left', order=2, height='550px'), title='Inputs', items=[
        ui.checklist(name='checklist', label='数据集', choices=[ui.choice(name=x, label=x) for x in ['A', 'B', 'C']]),
        ui.checklist(name='checklist', label='隐写分析模型', choices=[ui.choice(name=x, label=x) for x in ['model1', 'model2', 'model3', 'model4']]),
        ui.button('start_dataset_analysis', label='开始检测', primary=True)
    ])
    await q.page.save()

@on()
async def text_instance_level(q:Q):
    q.page['meta'] = layout2
    q.page['content_left'] = ui.form_card(box=ui.box('content_left', order=2, height='550px'), title='Inputs', items=[
        ui.textbox(name='suspect_text', label='输入可疑文本', required=True),
        ui.checklist(name='checklist', label='隐写分析模型',
                        choices=[ui.choice(name=x, label=x) for x in ['CNN', 'RNN', ]]),

        ui.button('text_start_analysis', label='开始检测', primary=True)
    ])
    q.page['content_right'] = ui.form_card(box=ui.box('content_right', order=2), title='Outputs', items=[
        ui.text('1111'),
    ])
    await q.page.save()