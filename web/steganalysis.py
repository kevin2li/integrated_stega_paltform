import os
import sys
sys.path.append(os.path.abspath('..'))

from h2o_wave import Q, app, handle_on, main, on, ui
from icecream import ic
from web.utils import layout1, layout2, empty

#================================================================
# 图像隐写分析
#================================================================
@on(arg='image_steganalysis')
async def serve(q:Q):
    del q.page['content_left']
    q.page['meta'] = layout2
    q.page['v_nav'].value = '#menu/steganalysis'
    q.page['content_left1'] = ui.tab_card(box=ui.box('content_left', order=1, height='50px'), link=True, items=[
        ui.tab('image_steganalysis', '单张图片检测'),
        ui.tab('dataset_level', '数据集检测'),
    ])
    q.page['content_right1'] = ui.tab_card(box=ui.box('content_right', order=1, height='50px'), link=True, items=[

    ])
    ic(q.client.instance_level)
    if not q.client.instance_level:
        q.page['content_left2'] = ui.form_card(box=ui.box('content_left', order=2, height='550px'), title='Inputs', items=[
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

            ui.button('start_analysis', label='开始检测', primary=True)
        ])
        q.page['content_right2'] = ui.form_card(box=ui.box('content_right', order=2), title='Outputs', items=[
            ui.text('1111'),
        ])
        q.client.instance_level = True
    await q.page.save()


@on(arg='dataset_level')
async def serve(q:Q):
    q.client.instance_level = False
    del q.page['content_left2']
    q.page['content_left'] = ui.form_card(box=ui.box('content_left', order=2, height='550px'), title='Inputs', items=[
        ui.checklist(name='checklist', label='数据集', choices=[ui.choice(name=x, label=x) for x in ['S-UNIWARD 0.4bpp', 'WOW 0.4bpp', 'HUGO 0.4bpp']]),
        ui.checklist(name='checklist', label='隐写分析模型', choices=[ui.choice(name=x, label=x) for x in ['SRNet', 'ZhuNet', 'YedNet', 'SRM']]),
        ui.button('start_dataset_analysis', label='开始检测', primary=True)
    ])
    await q.page.save()

@on(arg='suspect_img')
async def serve(q:Q):
    path = q.args['suspect_img']
    if path:
        local_path = await q.site.download(path[0], './upload')
        q.client.suspect_img_path = local_path
        ic(local_path)
        q.page['content_left'].items[1].file_upload.label = '已上传'
        ic(vars(q.page))
    await q.page.save()

@on(arg='start_analysis')
async def serve(q:Q):
    suspect_img_path = q.client.suspect_img_path
    checklist = q.args['checklist']
    ic(suspect_img_path)
    ic(checklist)
    if suspect_img_path and checklist:
        FLAG = False
        for i in checklist:
            if i not in ('ZhuNet', 'YedNet'):
                FLAG = True
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

                out = model(img)

                out = F.softmax(out, dim=-1).squeeze()
                result[i] = out.tolist()
            ic(result)
            
            q.page['content_right2'] = ui.form_card(box=ui.box('content_right', order=2), title='Outputs', items=[
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


#================================================================
# 文本隐写分析
#================================================================
@on(arg='text_steganalysis')
async def serve(q:Q):
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