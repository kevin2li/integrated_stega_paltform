'''
Author: Kevin Li
Date: 2021-04-22 11:08:05
LastEditTime: 2021-05-06 19:34:13
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /myapps/web/imgae_steganalysis.py
'''
import os
import sys

import yaml
sys.path.append(os.path.abspath('..'))
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from h2o_wave import Q, app, data, handle_on, main, on, ui
from icecream import ic
from project import root_dir
from project.steganalysis.tf.models import XuNet
from project.steganalysis.torch.models import YedNet, ZhuNet

from web.utils import *


#================================================================
# 图像隐写分析
#================================================================
@on()
async def image_instance_level(q:Q):
    q.page['meta'] = layout2
    q.page['content_left'] = ui.form_card(box=ui.boxes(
                ui.box('content_left', height='100%'),
                ui.box('content_left', height='100%'),
                ui.box('content_left', height='600px'),
                ui.box('content_left', height='850px'),
            ), title='Inputs', items=[
        ui.text('**1. 上传可疑图片:**'),
        ui.file_upload(name='suspect_img', label='上传', multiple=False, file_extensions=['png', 'jpg', 'jpeg'], max_file_size=10, max_size=15, height='200px'),
        # ui.expander(name='expander', label='高级选项', items=[
        #     ui.dropdown('framework', label='框架:', value='pytorch', required=True, choices=[
        #         ui.choice(name=x, label=x) for x in ['pytorch', 'tensorflow']
        #     ]),
        #     ui.dropdown('source', label='数据集:', value='WOW', required=True, choices=[
        #         ui.choice(name=x, label=x) for x in ['WOW', 'S-UNIWARD', 'HILL', 'HUGO', 'MG', 'MVG', 'UT-GAN', 'MiPOD']
        #     ]),
        #     ui.dropdown('embedding_rate', label='嵌入率:', value='0.4 bpp', required=True, choices=[
        #         ui.choice(name=x, label=x) for x in ['0.2 bpp', '0.4 bpp', '0.6 bpp', '0.8 bpp']
        #     ]),
        # ]),
        ui.text('**2. 选择模型:**'),
        ui.dropdown('framework', label='框架:', value='pytorch', required=True, choices=[
            ui.choice(name=x, label=x) for x in ['pytorch', 'tensorflow']
        ]),
        ui.dropdown('dataset', label='数据集:', value='WOW', required=True, choices=[
            ui.choice(name=x, label=x) for x in ['WOW', 'S-UNIWARD', 'HILL', 'HUGO', 'MG', 'MVG', 'UT-GAN', 'MiPOD']
        ]),
        ui.dropdown('embedding_rate', label='嵌入率:', value='0.4 bpp', required=True, choices=[
            ui.choice(name=x, label=x) for x in ['0.2 bpp', '0.4 bpp', '0.6 bpp', '0.8 bpp']
        ]),
        ui.dropdown('models', label='隐写分析模型:', values=['SRNet'], required=True, choices=[
            ui.choice(name=x, label=x) for x in ['SRNet', 'ZhuNet', 'YedNet', 'XuNet', 'YeNet']
        ]),

        ui.button('image_start_analysis', label='开始检测', primary=True, disabled=False)
    ])
    q.page['content_right'] = ui.form_card(box=ui.box('content_right', order=2), title='Outputs', items=[
        # ui.text('1111'),
    ])
    
    await q.page.save()


@on()
async def image_dataset_level(q:Q):
    q.page['content_left'] = ui.form_card(box=ui.box('content_left', order=2, height='100%'), title='Inputs', items=[
        ui.dropdown(name='options2', label='数据集', required=True, values=['S-UNIWARD 0.4bpp'], choices=[ui.choice(name=x, label=x) for x in ['S-UNIWARD 0.4bpp', 'WOW 0.4bpp', 'HUGO 0.4bpp']]),
        ui.dropdown(name='options3', label='隐写分析模型', required=True, values=['SRNet'], choices=[ui.choice(name=x, label=x) for x in ['SRNet', 'ZhuNet', 'YedNet', 'SRM']]),
        ui.button('start_dataset_analysis', label='开始检测', primary=True)
    ])
    await q.page.save()

@on()
async def suspect_img(q:Q):
    path = q.args['suspect_img']
    if path:
        save_dir = Path('upload')
        save_dir.mkdir(parents=True, exist_ok=True)
        local_path = await q.site.download(path[0], str(save_dir))
        q.client.suspect_img_path = local_path
        ic(local_path)
        q.page['content_left'].items[1].file_upload.label = '已上传'
        # q.page['content_left'].items[6].button.disabled = False
        q.client.upload = True
    await q.page.save()

@on()
async def image_start_analysis(q:Q):
    ic('erhsjfksdf')
    # get inputs
    suspect_img_path = q.client.suspect_img_path
    models = q.args['models']
    framework = q.args['framework']
    dataset = q.args['dataset']
    embedding_rate = q.args['embedding_rate']
    ic(models)
    # keep state
    q.page['content_left'].items[3].dropdown.values=[framework]
    q.page['content_left'].items[4].dropdown.values=[dataset]
    q.page['content_left'].items[5].dropdown.values=[embedding_rate]
    q.page['content_left'].items[6].dropdown.values=models

    ic(suspect_img_path)
    if not q.client.upload:
        q.page['meta'].dialog = ui.dialog(title='error', items=[
            ui.text('对不起，请先上传图片!'),
            ui.buttons([ui.button(name='sure', label='确定', primary=True)])
        ])
    if suspect_img_path and models and framework and dataset and embedding_rate:
        ic('ok')
        def predict(suspect_img_path, models):
            result = {}
            for model_name in models:
                # read map.yml
                path = '/home/likai/my_repos/integrated_stega_paltform/project/steganalysis/res/map.yml'
                with open(path) as f:
                    args = yaml.safe_load(f)
                # get checkpoint_path
                embedding_rate = embedding_rate.split()[0]
                checkpoint_path = args['framework'][framework]['dataset'][dataset]['embedding_rate'][embedding_rate]['model'][model_name]
                ic(checkpoint_path)
                img = img_preprocess(suspect_img_path)
                # load model and weights
                model = eval(model_name)()
                if framework == 'torch':
                    if checkpoint_path:
                        model = model.load_from_checkpoint(checkpoint_path)
                    model.eval()
                    out = model(img)
                    out = F.softmax(out, dim=-1).squeeze()
                elif framework == 'tensorflow':
                    img = np.array(Image.open(suspect_img_path))
                    if checkpoint_path:
                        model.load_weights(checkpoint_path)
                    out = model(img).squeeze()
                result[model_name] = out.tolist()
            return result
        q.page['content_right'] = ui.form_card(box='content_right', title='Outputs', items=[ui.progress('检测中...')])
        await q.page.save()
        result = await q.run(predict, suspect_img_path, models)
        ic(result)
        # df = pd.DataFrame(result)
        # df['type'] = ['cover', 'stego']
        # df['prob'] = result[model_name]
        # q.page['content_right'] = ui.form_card(box=ui.box('content_right'), title='Outputs', items=[
        #     ui.text(f"结果:{str(result)}"),
        #     ui.visualization(ui.plot([ui.mark(type='interval', x='=type', y='=prob', x_title='类型', y_title='概率')]), data=data(fields=df.columns.tolist(), rows=df.values.tolist(), pack=True))
        # ])
        fig = plot_group_bars(result)
        fig.savefig('bar.png')
        suspect_img_path2,  = await q.site.upload([suspect_img_path])
        result_img_path,  = await q.site.upload(['bar.png'])
        ic(result_img_path)
        os.remove('bar.png')
                
        q.page['content_right'] = ui.markdown_card(box=ui.boxes(
            ui.box('content_right', height='100%'),
            ui.box('content_right', height='100%'),
            ui.box('content_right', height='600px'),
            ui.box('content_right', height='850px'),
        ), title='Outputs',
            content=f"**image:** {suspect_img_path.split('/')[-1]}\n\n ![plot]({suspect_img_path2})\n\n **result:**\n\n ![plot]({result_img_path})"
        )
        q.page['content_left'].items[1].file_upload.label = '上传'

    await q.page.save()
