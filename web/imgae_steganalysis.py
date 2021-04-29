'''
Author: your name
Date: 2021-04-22 11:08:05
LastEditTime: 2021-04-29 17:52:57
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /myapps/web/imgae_steganalysis.py
'''
import os
import sys
sys.path.append(os.path.abspath('..'))

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from h2o_wave import Q, app, handle_on, main, on, ui, data
from icecream import ic
from project.sa import YedNet, ZhuNet, XuNet
from pathlib import Path
import matplotlib.pyplot as plt
from web.utils import *
root_dir = Path('/home/kevin2li/wave/myapps/')  # wsl
# root_dir = Path('/root/wave/myapp')  # aliyun
# root_dir = Path('/home/likai/integrated_stega_paltform/') # lab
#================================================================
# 图像隐写分析
#================================================================
@on()
async def image_instance_level(q:Q):
    q.page['meta'] = layout2
    q.page['content_left'] = ui.form_card(box=ui.box('content_left', order=2, height='550px'), title='Inputs', items=[
        ui.text('**上传可疑图片:**'),
        ui.file_upload(name='suspect_img', label='上传', multiple=False, file_extensions=['png', 'jpg', 'jpeg'], max_file_size=10, max_size=15, height='200px'),
        ui.expander(name='expander', label='高级选项', items=[
            ui.dropdown('source', label='数据集:', value='WOW', required=True, choices=[
                ui.choice(name=x, label=x) for x in ['WOW', 'S-UNIWARD', 'HILL', 'HUGO', 'MG', 'MVG', 'UT-GAN', 'SUI']
            ]),
            ui.dropdown('embedding_rate', label='嵌入率:', value='0.4 bpp', required=True, choices=[
                ui.choice(name=x, label=x) for x in ['0.2 bpp', '0.4 bpp', '0.6 bpp', '0.8 bpp']
            ]),
        ]),
        ui.dropdown('options', label='隐写分析模型:', values=['SRNet'], required=True, choices=[
            ui.choice(name=x, label=x) for x in ['SRNet', 'ZhuNet', 'YedNet', 'XuNet', 'YeNet']
        ]),
        # ui.checklist(name='checklist', label='隐写分析模型',
        #                 choices=[ui.choice(name=x, label=x) for x in ['SRNet', 'ZhuNet', 'YedNet', 'XuNet', 'SRM', 'A', 'B', 'C', 'D', 'E', 'F']]),

        ui.button('image_start_analysis', label='开始检测', primary=True)
    ])
    q.page['content_right'] = ui.form_card(box=ui.box('content_right', order=2), title='Outputs', items=[
        # ui.text('1111'),
    ])
    await q.page.save()


@on()
async def image_dataset_level(q:Q):
    q.page['content_left'] = ui.form_card(box=ui.box('content_left', order=2, height='550px'), title='Inputs', items=[
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
        ic(vars(q.page))
    await q.page.save()

@on()
async def image_start_analysis(q:Q):
    suspect_img_path = q.client.suspect_img_path
    options = q.args['options']
    ic(suspect_img_path)
    ic(options)
    if suspect_img_path and options:
        FLAG = False
        for i in options:
            if i not in ('ZhuNet', 'YedNet', 'XuNet'):
                FLAG = True
                break
        if FLAG:
            q.page['meta'].dialog = ui.dialog(title='error', items=[
                ui.text('对不起，暂不支持全部所选模型, 目前仅支持ZhuNet和YedNet!'),
                ui.buttons([ui.button(name='sure', label='确定', primary=True)])
            ])
        else:
            result = {}
            for model_name in options:
                version = 'torch'
                img = img_preprocess(suspect_img_path)
                model = eval(model_name)()
                
                # load weights
                if model_name == 'ZhuNet':
                    model = model.load_from_checkpoint(str(root_dir / 'project/sa/zhunet/zhunet-epoch=210-val_loss=0.44-val_acc=0.85.ckpt'))
                elif model_name == 'YedNet':
                    model = model.load_from_checkpoint(str(root_dir / 'project/sa/yednet/epoch=247-val_loss=0.48-val_acc=0.81.ckpt'))
                elif model_name == 'XuNet': # tf implemented currently
                    version = 'tf'
                    img = np.array(Image.open(suspect_img_path))
                    model.load_weights(str(root_dir / 'project/sa/xunet/saved-model-117-0.85.hdf5'))
                elif model_name == 'SRNet':
                    pass
                elif model_name == 'YeNet':
                    pass
                
                # predict
                if version == 'torch':
                    model.eval()
                    out = model(img)
                    out = F.softmax(out, dim=-1).squeeze()
                elif version == 'tf':
                    out = model(img).squeeze()

                result[model_name] = out.tolist()
                
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
            image_path,  = await q.site.upload(['bar.png'])
            ic(image_path)
            os.remove('bar.png')
            q.page['content_right'] = ui.markdown_card(box=ui.box('content_right'), title='Outputs',
                content=f'![plot]({image_path})'
            )
            q.page['content_left'].items[1].file_upload.label = '上传'
            # del q.client.suspect_img_path
    else:
        q.page['meta'].dialog = ui.dialog(title='error', items=[
            ui.text('对不起，请检查是否输入完整!'),
            ui.buttons([ui.button(name='sure', label='确定', primary=True)])
        ])
    await q.page.save()
