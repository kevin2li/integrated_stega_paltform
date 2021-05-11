'''
Author: your name
Date: 2021-04-22 11:29:33
LastEditTime: 2021-05-05 13:39:08
LastEditors: your name
Description: In User Settings Edit
FilePath: /myapps/web/text_steganography.py
'''
import os
import sys
sys.path.append(os.path.abspath('..'))

from h2o_wave import Q, app, handle_on, main, on, ui
from icecream import ic
from web.utils import layout1, layout2, empty

#================================================================
# 文本隐写
#================================================================
@on()
async def text_embed(q:Q):
    q.page['meta'] = layout2
    q.page['content_left'] = ui.form_card(box='content_left', title='Inputs', items=[
        ui.textbox(name='prefix_words', label='句子开头:', required=True, placeholder='e.g. I have a ...'),
        ui.textbox(name='secret', label='秘密信息比特流:', required=True, placeholder='e.g. 0101011011'),
        ui.dropdown('option', label='语言模型:', value='rnn', choices=[
            ui.choice('rnn', 'RNN-Stega'),
            ui.choice('vae', 'VAE-Stega'),
            ui.choice('gpt', 'GPT-Stega'),
        ]),
        ui.button('start_gen', label='开始生成', primary=True)
    ])
    q.page['content_right'] = ui.form_card(box='content_right', title='Outputs', items=[
        ui.text('blalala....'),
    ])
    await q.page.save()

@on()
async def text_extract(q:Q):
    q.page['meta'] = layout2
    # q.page['content'] = ui.form_card(box='content', items=[
    #     ui.text('正在开发中...')
    # ])
    q.page['content_left'] = ui.form_card(box='content_left', title='Inputs', items=[
        ui.textbox(name='secret_text', label='秘密文本:', required=True, placeholder='e.g. I have a ...'),
        ui.dropdown('option', label='语言模型:', value='rnn', choices=[
            ui.choice('rnn', 'RNN-Stega'),
            ui.choice('vae', 'VAE-Stega'),
            ui.choice('gpt', 'GPT-Stega'),
        ]),
        ui.button('text_start_extract', label='开始提取', primary=True)
    ])
    q.page['content_right'] = ui.form_card(box='content_right', title='Outputs', items=[
        ui.text('blalala....'),
    ])
    await q.page.save()