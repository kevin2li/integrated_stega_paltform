'''
Author: your name
Date: 2021-04-22 11:08:05
LastEditTime: 2021-04-23 18:43:15
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /myapps/web/imgae_steganalysis.py
'''
import os
import sys
sys.path.append(os.path.abspath('..'))
from pathlib import Path
import torch
from h2o_wave import Q, app, handle_on, main, on, ui
from icecream import ic
from web.utils import *
import torch.nn.functional as F
from project.sa import YedNet, ZhuNet
models = {'YedNet': YedNet, 'ZhuNet': ZhuNet}
#================================================================
# 图像隐写分析
#================================================================
@on()
async def image_instance_level(q:Q):
    q.page['meta'] = layout2
    q.page['content_left'] = ui.form_card(box=ui.box('content_left', order=2, height='550px'), title='Inputs', items=[
        ui.text('**上传可疑图片:**'),
        ui.file_upload(name='suspect_img', label='上传', multiple=False, file_extensions=['png', 'jpg', 'jpeg'], max_file_size=10, max_size=15, height='3'),
        # ui.dropdown('option', label='隐写分析模型:', value='SRNet', choices=[
        #     ui.choice('SRNet', 'SRNet'),
        #     ui.choice('ZhuNet', 'ZhuNet'),
        #     ui.choice('YedNet', 'Yedroudj-Net'),
        #     ui.choice('srm', 'SRM'),
        # ]),
        ui.checklist(name='checklist', label='隐写分析模型',
                        choices=[ui.choice(name=x, label=x) for x in ['SRNet', 'ZhuNet', 'YedNet', 'SRM']]),

        ui.button('image_start_analysis', label='开始检测', primary=True)
    ])
    q.page['content_right'] = ui.form_card(box=ui.box('content_right', order=2), title='Outputs', items=[
        ui.text('1111'),
    ])
    await q.page.save()


@on()
async def image_dataset_level(q:Q):
    q.page['content_left'] = ui.form_card(box=ui.box('content_left', order=2, height='550px'), title='Inputs', items=[
        ui.checklist(name='checklist', label='数据集', choices=[ui.choice(name=x, label=x) for x in ['S-UNIWARD 0.4bpp', 'WOW 0.4bpp', 'HUGO 0.4bpp']]),
        ui.checklist(name='checklist', label='隐写分析模型', choices=[ui.choice(name=x, label=x) for x in ['SRNet', 'ZhuNet', 'YedNet', 'SRM']]),
        ui.button('start_dataset_analysis', label='开始检测', primary=True)
    ])
    await q.page.save()

@on()
async def suspect_img(q:Q):
    path = q.args['suspect_img']
    if path:
        save_dir = Path('./upload')
        save_dir.mkdir(parents=True, exist_ok=True)
        local_path = await q.site.download(path[0], save_dir)
        q.client.suspect_img_path = local_path
        ic(local_path)
        q.page['content_left'].items[1].file_upload.label = '已上传'
        ic(vars(q.page))
    await q.page.save()

@on()
async def image_start_analysis(q:Q):
    suspect_img_path = q.client.suspect_img_path
    checklist = q.args['checklist']
    ic(suspect_img_path)
    ic(checklist)
    if suspect_img_path and checklist:
        FLAG = False
        for i in checklist:
            if i not in ('ZhuNet', 'YedNet'):
                FLAG = True
                break
        if FLAG:
            q.page['meta'].dialog = ui.dialog(title='error', items=[
                ui.text('对不起，暂不支持全部所选模型, 目前仅支持ZhuNet和YedNet!'),
                ui.buttons([ui.button(name='sure', label='确定', primary=True)])
            ])
        else:
            img = img_preprocess(suspect_img_path)
            ic(img.shape)
            result = {}
            for i in checklist:
                model = models[i]()
                if i == 'ZhuNet':
                    params = torch.load('/home/wave/myapp/project/sa/zhunet/zhunet_wow.ptparams', map_location='cpu')
                    model.load_state_dict(params)
                out = model(img)
                out = F.softmax(out, dim=-1).squeeze()
                result[i] = out.tolist()
            ic(result)
            
            q.page['content_right'] = ui.form_card(box=ui.box('content_right'), title='Outputs', items=[
                ui.text(f'检测模型:{checklist[0]}'),
                ui.text(f"结果:{str(result)}"),
            ])
            q.page['content_left'].items[1].file_upload.label = '上传'
            # del q.client.suspect_img_path
    else:
        q.page['meta'].dialog = ui.dialog(title='error', items=[
            ui.text('对不起，请检查是否输入完整!'),
            ui.buttons([ui.button(name='sure', label='确定', primary=True)])
        ])
    await q.page.save()
