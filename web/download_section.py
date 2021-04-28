'''
Author: your name
Date: 2021-04-22 11:16:16
LastEditTime: 2021-04-27 20:25:08
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /myapps/web/download_section.py
'''
import os
import sys
sys.path.append(os.path.abspath('..'))

from h2o_wave import Q, app, handle_on, main, on, ui
from icecream import ic
from web.utils import layout1, layout2, empty

#================================================================
# 下载专区
#================================================================

#TODO
@on()
async def dataset(q: Q):
    q.page['content'] = ui.form_card(box='content', title='', items=[
        ui.text('# 原始载体数据集'),
        ui.link('BOSSBase v1.01', path='http://lab.cb301.icu/wp-content/uploads/2020/11/BOSSbase_1.01.zip', download=True),
        ui.link('BOWS2', path='http://lab.cb301.icu/wp-content/uploads/2020/11/BOWS2.zip', download=True),
        ui.link('ALASKA', path='https://www.kaggle.com/c/alaska2-image-steganalysis', download=True),
        ui.link('tiny-imagenet-200', path='http://lab.cb301.icu/wp-content/uploads/2020/12/tiny-imagenet-200.zip', download=True),
        ui.text('# 隐写数据集'),
        ui.link('BOSSBase-HILL-0.4bpp', path='http://lab.cb301.icu/wp-content/uploads/2020/11/bb_H_0.4.zip', download=True),
        ui.link('BOSSBase-SUNIWARD-0.4bpp', path='http://lab.cb301.icu/wp-content/uploads/2020/12/bb_S_0.4.zip', download=True),
    ])
    await q.page.save()

#TODO
@on()
async def code(q: Q):
    q.page['content'] = ui.form_card(box='content', title='', items=[
        ui.text('# 自适应隐写算法(matlab)'),
        ui.link('WOW', path='http://lab.cb301.icu/wp-content/uploads/2020/11/WOW_matlab.zip', download=True),
        ui.link('S-UNIWARD', path='http://lab.cb301.icu/wp-content/uploads/2020/11/S-UNIWARD_matlab.zip', download=True),
        ui.link('MiPOD', path='http://lab.cb301.icu/wp-content/uploads/2020/11/MiPOD_matlab.zip', download=True),
        ui.link('HUGO', path='http://lab.cb301.icu/wp-content/uploads/2020/11/HUGO_bounding_matlab.zip', download=True),
        ui.link('HILL', path='http://lab.cb301.icu/wp-content/uploads/2021/04/HILL_matlab.zip', download=True),
        ui.text('# 隐写分析'),
        ui.link('ZhuNet', path='https://github.com/kevin2li/steganalysis_pl/blob/main/src/models/zhunet.py', target='_blank'),
        ui.link('SRNet', path='https://github.com/kevin2li/steganalysis_pl/blob/main/src/models/srnet.py', target='_blank'),
        ui.link('YedNet', path='https://github.com/kevin2li/steganalysis_pl/blob/main/src/models/yednet.py', target='_blank'),
        ui.link('YeNet', path='https://github.com/kevin2li/steganalysis_pl/blob/main/src/models/yenet.py', target='_blank'),
        ui.link('XuNet', path='https://github.com/kevin2li/steganalysis_pl/blob/main/src/models/xunet.py', target='_blank'),
    ])
    await q.page.save()

@on()
async def paper(q: Q):
    q.page['content'] = ui.form_card(box='content', title='', items=[
        ui.text('# 隐写'),
        ui.link('付章杰,李恩露,程旭,黄永峰,胡雨婷.基于深度学习的图像隐写研究进展[J].计算机研究与发展,2021,58(03):548-568.', path='http://lab.cb301.icu/wp-content/uploads/2021/04/%E5%9F%BA%E4%BA%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%9A%84%E5%9B%BE%E5%83%8F%E9%9A%90%E5%86%99%E7%A0%94%E7%A9%B6%E8%BF%9B%E5%B1%95.zip', download=True),
        ui.link('付章杰,王帆,孙星明,王彦.基于深度学习的图像隐写方法研究[J].计算机学报,2020,43(09):1656-1672.', path='http://lab.cb301.icu/wp-content/uploads/2021/04/%E5%9F%BA%E4%BA%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%9A%84%E5%9B%BE%E5%83%8F%E9%9A%90%E5%86%99%E6%96%B9%E6%B3%95%E7%A0%94%E7%A9%B6.zip', download=True),
        ui.text('# 隐写分析'),
        ui.link('陈君夫,付章杰,张卫明,程旭,孙星明.基于深度学习的图像隐写分析综述[J].软件学报,2021,32(02):551-578.', path='http://lab.cb301.icu/wp-content/uploads/2021/04/%E5%9F%BA%E4%BA%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%9A%84%E5%9B%BE%E5%83%8F%E9%9A%90%E5%86%99%E5%88%86%E6%9E%90%E7%BB%BC%E8%BF%B0.zip', download=True),
    ])
    await q.page.save()

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