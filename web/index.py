import os
import sys
sys.path.append(os.path.abspath('..'))

from h2o_wave import Q, app, handle_on, main, on, ui
from icecream import ic
from web.utils import *
from web.image_steganography import *
from web.imgae_steganalysis import *
from web.text_steganography import *
from web.text_steganalysis import *
from web.download_section import *


first_level_menu = {
    'menu/index': '首页',
    'menu/image_stega': '图像隐写',
    'menu/image_steganalysis': '图像隐写分析',
    'menu/text_stega': '文本隐写',
    'menu/text_steganalysis': '文本隐写分析',
    'menu/download_section': '文本隐写分析',
}
first_level_help = {
    'help/about': '关于',
    'help/support': '支持',
}

#================================================================
# 首页
#================================================================
async def reset(q:Q):
    if not q.client.initialized:
        await menu_index(q)

@app('/')
async def serve(q:Q):
    ic(q.args)
    # ic(vars(q.page['tab_bar'].tab_bar))
    ic(q.client.initialized)
    location = q.args['#']
    ic(location)
    if location:
        location = location.replace('/', '_')
        await eval(location)(q)
    elif not str(q.args):
        location = 'menu_index'
        await eval(location)(q)
    else:
        await handle_on(q)
#================================================================
# 菜单项
#================================================================
@on(arg='#menu/index')
async def menu_index(q:Q):
    if not q.client.initialized:
        q.page['meta'] = layout1
        q.page['header'] = ui.header_card(
            # Place card in the header zone, regardless of viewport size.
            box='header',
            title='集成隐写/分析平台',
            subtitle='Integrated Steganography/Steganalysis platform',
        )
        q.page['tab_bar'] = ui.tab_card(box='tab_bar', items=[
            # ui.tab('image_stega', '图像隐写'),
            # ui.tab('image_steganalysis', '图像隐写分析'),
            # ui.tab('text_stega', '文本隐写'),
            # ui.tab('text_steganalysis', '文本隐写分析'),
            # ui.tab('#menu/download_section', '下载专区'),
        ])
        q.page['v_nav'] = ui.nav_card(
            box=ui.box('sidebar', height='100%'),
            value='#menu/index',
            items=[
                ui.nav_group('Menu', items=[
                    ui.nav_item(name=f'#{k}', label=v) for k, v in first_level_menu.items()
                ]),
                ui.nav_group('帮助', items=[
                    ui.nav_item(name=f'#{k}', label=v) for k, v in first_level_help.items()
                ])
            ],
        )
        q.page['content'] = ui.form_card(box='content', items=[
            ui.text('正在开发中...')
        ])
        q.client.initialized = True
        await q.page.save()
    else:
        q.page['v_nav'].value = '#menu/index'
        q.page['tab_bar'] = ui.tab_card(box='tab_bar', items=[
            # ui.tab('image_stega', '图像隐写'),
            # ui.tab('image_steganalysis', '图像隐写分析'),
            # ui.tab('text_stega', '文本隐写'),
            # ui.tab('text_steganalysis', '文本隐写分析'),
            # ui.tab('#menu/download_section', '下载专区'),
        ])
        q.page['content'] = ui.form_card(box='content', items=[
            ui.text('正在开发中...')
        ])
        await q.page.save()

@on(arg='#menu/image_stega')
async def menu_image_stega(q:Q):
    await reset(q)
    q.page['meta'] = layout1
    q.page['v_nav'].value = '#menu/image_stega'
    q.page['tab_bar'] = ui.tab_card(box='tab_bar', items=[
        ui.tab('image_embed', '嵌入'),
        ui.tab('image_extract', '提取'),
        ui.tab('image_watermark', '数字水印'),
    ])
    q.page['content'] = ui.form_card(box='content', items=[
        ui.text('正在开发中...')
    ])
    await q.page.save()

@on(arg='#menu/image_steganalysis')
async def menu_image_steganalysis(q:Q):
    await reset(q)
    empty(q)
    q.page['meta'] = layout1
    q.page['v_nav'].value = '#menu/image_steganalysis'
    q.page['tab_bar'] = ui.tab_card(box='tab_bar', items=[
        ui.tab('image_instance_level', '单样本分析'),
        ui.tab('image_dataset_level', '数据集分析'),
    ])
    q.args['image_instance_level'] = True
    await image_instance_level(q)

@on(arg='#menu/text_stega')
async def menu_text_stega(q:Q):
    await reset(q)
    q.page['meta'] = layout1
    q.page['v_nav'].value = '#menu/text_stega'
    q.page['tab_bar'] = ui.tab_card(box='tab_bar', items=[
        ui.tab('text_embed', '嵌入'),
        ui.tab('text_extract', '提取'),
    ])
    q.page['content'] = ui.form_card(box='content', items=[
        ui.text('正在开发中...')
    ])
    await q.page.save()

@on(arg='#menu/text_steganalysis')
async def menu_text_steganalysis(q:Q):
    await reset(q)
    empty(q)
    q.page['meta'] = layout1
    q.page['v_nav'].value = '#menu/text_steganalysis'
    q.page['tab_bar'] = ui.tab_card(box='tab_bar', items=[
        ui.tab('text_instance_level', '单样本分析'),
        ui.tab('text_dataset_level', '数据集分析'),
    ])
    q.page['content'] = ui.form_card(box='content', items=[
        ui.text('正在开发中...')
    ])
    await q.page.save()

@on(arg='#menu/download_section')
async def menu_download_section(q:Q):
    await reset(q)
    q.page['meta'] = layout1
    q.page['v_nav'].value = '#menu/download_section'
    q.page['tab_bar'] = ui.tab_card(box='tab_bar', items=[
        ui.tab('dataset', '数据集'),
        ui.tab('code', '代码'),
    ])
    q.page['content'] = ui.form_card(box='content', items=[
        ui.text('正在开发中...')
    ])
    await q.page.save()

@on(arg='#help/about')
async def help_about(q:Q):
    await reset(q)
    q.page['meta'] = layout1
    q.page['v_nav'].value = '#help/about'
    q.page['tab_bar'] = ui.tab_card(box='tab_bar', items=[
        
    ])
    q.page['content'] = ui.form_card(box='content', items=[
        ui.text('正在开发中...')
    ])
    await q.page.save()

@on(arg='#help/support')
async def help_support(q:Q):
    await reset(q)
    q.page['meta'] = layout1
    q.page['v_nav'].value = '#help/support'
    q.page['tab_bar'] = ui.tab_card(box='tab_bar', items=[

    ])
    q.page['content'] = ui.form_card(box='content', items=[
        ui.text('正在开发中...')
    ])
    await q.page.save()



