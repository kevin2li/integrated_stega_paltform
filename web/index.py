import os
import sys
sys.path.append(os.path.abspath('..'))

from h2o_wave import Q, app, handle_on, main, on, ui, graphics
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
    'menu/download_section': '下载专区',
}
first_level_help = {
    'help/about': '关于',
    'help/support': '支持',
}

#================================================================
# 首页
#================================================================
@app('/')
async def serve(q:Q):
    ic(q.args)
    # ic(vars(q.page['tab_bar'].tab_bar))
    ic(q.client.initialized)
    if not q.client.initialized:
        await menu_index(q)
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
            # ui.text('正在开发中...'),
        ])
        q.client.initialized = True
        await q.page.save()
    else:
        q.page['meta'] = layout1
        del q.page['content_left']
        del q.page['content_right']
        q.page['v_nav'].value = '#menu/index'
        q.page['tab_bar'] = ui.tab_card(box='tab_bar', items=[
            ui.tab('12', '待定1'),
            ui.tab('1223', '待定2'),
            # ui.tab('image_steganalysis', '图像隐写分析'),
            # ui.tab('text_stega', '文本隐写'),
            # ui.tab('text_steganalysis', '文本隐写分析'),
            # ui.tab('#menu/download_section', '下载专区'),

        ])
        image_path, = await q.site.upload(['/home/kevin2li/wave/myapps/web/upload/2.png'])
        q.page['content'] = ui.markdown_card(box='content', title='首页', content=f'![plot]({image_path})')
        await q.page.save()

@on(arg='#menu/image_stega')
async def menu_image_stega(q:Q):
    q.page['meta'] = layout1
    q.page['v_nav'].value = '#menu/image_stega'
    q.page['tab_bar'] = ui.tab_card(box='tab_bar', items=[
        ui.tab('image_embed', '嵌入'),
        ui.tab('image_extract', '提取'),
        # ui.tab('image_watermark', '数字水印'),
    ])
    await image_embed(q)

@on(arg='#menu/image_steganalysis')
async def menu_image_steganalysis(q:Q):
    q.page['meta'] = layout1
    q.page['v_nav'].value = '#menu/image_steganalysis'
    q.page['tab_bar'] = ui.tab_card(box='tab_bar', items=[
        ui.tab('image_instance_level', '单样本分析'),
        ui.tab('image_dataset_level', '数据集分析'),
    ])
    await image_instance_level(q)

@on(arg='#menu/text_stega')
async def menu_text_stega(q:Q):
    q.page['meta'] = layout1
    q.page['v_nav'].value = '#menu/text_stega'
    q.page['tab_bar'] = ui.tab_card(box='tab_bar', items=[
        ui.tab('text_embed', '嵌入'),
        ui.tab('text_extract', '提取'),
    ])
    await text_embed(q)

@on(arg='#menu/text_steganalysis')
async def menu_text_steganalysis(q:Q):
    q.page['meta'] = layout1
    q.page['v_nav'].value = '#menu/text_steganalysis'
    q.page['tab_bar'] = ui.tab_card(box='tab_bar', items=[
        ui.tab('text_instance_level', '单样本分析'),
        ui.tab('text_dataset_level', '数据集分析'),
    ])
    await text_instance_level(q)

@on(arg='#menu/download_section')
async def menu_download_section(q:Q):
    q.page['meta'] = layout1
    q.page['v_nav'].value = '#menu/download_section'
    q.page['tab_bar'] = ui.tab_card(box='tab_bar', items=[
        ui.tab('dataset', '数据集'),
        ui.tab('code', '代码'),
        ui.tab('paper', '论文'),
    ])
    await dataset(q)

@on(arg='#help/about')
async def help_about(q:Q):
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
    q.page['meta'] = layout1
    q.page['v_nav'].value = '#help/support'
    q.page['tab_bar'] = ui.tab_card(box='tab_bar', items=[

    ])
    q.page['content'] = ui.form_card(box='content', items=[
        ui.text('正在开发中...')
    ])
    await q.page.save()



