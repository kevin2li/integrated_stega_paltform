from h2o_wave import Q, main, app, ui, on, handle_on
from icecream import ic
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import bwm
import imutils
from src.models import YedNet, ZhuNet
import torchvision.transforms as T
import torch.nn.functional as F
from src.datasetmgr import getDataLoader
from src.config import args
trainloader, testloader = getDataLoader(args)

#================================================================
# 变量声明
#================================================================
layout1 = ui.meta_card(box='', title='集成隐写/分析平台', layouts=[
              ui.layout(
                  breakpoint='xs',
                  width='100%',
                  zones=[
                      ui.zone('header'),
                      ui.zone('body', direction=ui.ZoneDirection.ROW, zones=[
                          ui.zone('sidebar', size='25%'),
                          ui.zone('body', size='75%', direction=ui.ZoneDirection.COLUMN, zones=[
                              ui.zone('tab_bar', size='15%'),
                              ui.zone('content', size='85%', direction=ui.ZoneDirection.COLUMN)
                          ]),
                      ]),
                      ui.zone('footer'),
                  ]
              ),
              ui.layout(
                  breakpoint='m',
                  width='100%',
                  zones=[
                      ui.zone('header'),
                      ui.zone('body', direction=ui.ZoneDirection.ROW, zones=[
                          ui.zone('sidebar', size='25%'),
                          ui.zone('body', size='75%', direction=ui.ZoneDirection.COLUMN, zones=[
                              ui.zone('tab_bar', size='15%'),
                              ui.zone('content', size='85%', direction=ui.ZoneDirection.COLUMN)
                          ]),
                      ]),
                      ui.zone('footer'),
                  ]
              ),
              ui.layout(
                  breakpoint='xl',
                  width='1200px',
                  zones=[
                      ui.zone('header'),
                      ui.zone('body', direction=ui.ZoneDirection.ROW, zones=[
                          ui.zone('sidebar', size='25%'),
                          ui.zone('body', size='75%', direction=ui.ZoneDirection.COLUMN, zones=[
                              ui.zone('tab_bar', size='15%'),
                              ui.zone('content', size='85%', direction=ui.ZoneDirection.COLUMN)
                          ]),
                      ]),
                      ui.zone('footer'),
                  ]
              )
            ])

layout2 = ui.meta_card(box='', title='集成隐写/分析平台', layouts=[
                ui.layout(
                    breakpoint='xs',
                    zones=[
                        ui.zone('header'),
                        ui.zone('body', direction=ui.ZoneDirection.ROW, zones=[
                            ui.zone('sidebar', size='25%'),
                            ui.zone('body', size='75%', direction=ui.ZoneDirection.COLUMN, zones=[
                                ui.zone('tab_bar', size='15%'),
                                ui.zone('content', size='85%', direction=ui.ZoneDirection.ROW, zones=[
                                    # ui.zone('content_upper', size="10%"),
                                    # ui.zone('content_lower', size="90%", direction=ui.ZoneDirection.ROW, zones=[
                                    ui.zone('content_left', size="50%"),
                                    ui.zone('content_right', size="50%"),
                                    # ]),
                                ])
                            ]),
                        ]),
                        ui.zone('footer'),
                    ]
                ),
                ui.layout(
                    breakpoint='m',
                    zones=[
                        ui.zone('header'),
                        ui.zone('body', direction=ui.ZoneDirection.ROW, zones=[
                            ui.zone('sidebar', size='25%'),
                            ui.zone('body', size='75%', direction=ui.ZoneDirection.COLUMN, zones=[
                                ui.zone('tab_bar', size='15%'),
                                ui.zone('content', size='85%', direction=ui.ZoneDirection.ROW, zones=[
                                    # ui.zone('content_upper', size="10%"),
                                    # ui.zone('content_lower', size="90%", direction=ui.ZoneDirection.ROW, zones=[
                                    ui.zone('content_left', size="50%"),
                                    ui.zone('content_right', size="50%"),
                                    # ]),
                                ])
                            ]),
                        ]),
                        ui.zone('footer'),
                    ]
                ),
                ui.layout(
                    breakpoint='xl',
                    width='1200px',
                    zones=[
                        ui.zone('header'),
                        ui.zone('body', direction=ui.ZoneDirection.ROW, zones=[
                            ui.zone('sidebar', size='25%'),
                            ui.zone('body', size='75%', direction=ui.ZoneDirection.COLUMN, zones=[
                                ui.zone('tab_bar', size='15%'),
                                ui.zone('content', size='85%', direction=ui.ZoneDirection.ROW, zones=[
                                    # ui.zone('content_upper', size="10%"),
                                    # ui.zone('content_lower', size="90%", direction=ui.ZoneDirection.ROW, zones=[
                                    ui.zone('content_left', size="50%"),
                                    ui.zone('content_right', size="50%"),
                                    # ]),
                                ])
                            ]),
                        ]),
                        ui.zone('footer'),
                    ]
                )
            ])

eval_transforms = T.Compose([
    T.ToTensor()
])

models = {'YedNet': YedNet, 'ZhuNet': ZhuNet}
#================================================================
# 菜单项
#================================================================
@on(arg='#menu/index')
async def serve(q:Q):
    empty(q)
    q.page['meta'] = layout1
    q.page['v_nav'].value = '#menu/index'
    q.page['tab_bar'] = ui.tab_card(box='tab_bar', items=[
        ui.tab('image_stega', '图像隐写'),
        ui.tab('image_steganalysis', '图像隐写分析'),
        ui.tab('text_stega', '文本隐写'),
        ui.tab('text_steganalysis', '文本隐写分析'),
        ui.tab('#menu/download_section', '下载专区'),
    ])
    q.page['content'] = ui.form_card(box='content', items=[
        ui.text('正在开发中...')
    ])
    await q.page.save()

@on(arg='#menu/steganography')
async def serve(q:Q):
    empty(q)
    q.page['meta'] = layout1
    q.page['v_nav'].value = '#menu/steganography'
    q.page['tab_bar'] = ui.tab_card(box='tab_bar', items=[
        ui.tab('text_stega', '文本隐写'),
        ui.tab('image_stega', '图像隐写'),
        ui.tab('image_watermark', '数字水印'),
    ])
    q.page['content'] = ui.form_card(box='content', items=[
        ui.text('正在开发中...')
    ])
    await q.page.save()

@on(arg='#menu/steganalysis')
async def serve(q:Q):
    empty(q)
    q.page['meta'] = layout1
    q.page['v_nav'].value = '#menu/steganalysis'
    q.page['tab_bar'] = ui.tab_card(box='tab_bar', items=[
        ui.tab('#menu/steganalysis', '文本隐写分析'),
        ui.tab('image_steganalysis', '图像隐写分析'),
    ])
    q.page['content'] = ui.form_card(box='content', items=[
        ui.text('正在开发中...')
    ])
    await q.page.save()

@on(arg='#menu/download_section')
async def serve(q:Q):
    empty(q)
    q.page['meta'] = layout1
    q.page['v_nav'].value = '#menu/download_section'
    q.page['tab_bar'] = ui.tab_card(box='tab_bar', items=[
        ui.tab('dataset_img', '图像隐写'),
        ui.tab('dataset_text', '文本隐写'),
    ])
    q.page['content'] = ui.form_card(box='content', items=[
        ui.text('正在开发中...')
    ])
    await q.page.save()

@on(arg='#help/about')
async def serve(q:Q):
    empty(q)
    q.page['meta'] = layout1
    q.page['v_nav'].value = '#help/about'
    q.page['content'] = ui.form_card(box='content', items=[
        ui.text('正在开发中...')
    ])
    await q.page.save()

@on(arg='#help/support')
async def serve(q:Q):
    empty(q)
    q.page['meta'] = layout1
    q.page['v_nav'].value = '#help/support'
    q.page['content'] = ui.form_card(box='content', items=[
        ui.text('正在开发中...')
    ])
    await q.page.save()
#================================================================
# 数字水印
#================================================================
@on(arg='image_watermark')
async def serve(q:Q):
    q.page['meta'] = layout1
    del q.page['content']
    ic(q.args)
    q.page['wm_option'] = ui.tab_card(box=ui.box('content', height='10%'), items=[
        ui.tab('image_watermark', '嵌入水印'),
        ui.tab('extract', '提取水印')
    ], link=True)
    q.page['upload'] = ui.form_card(box=ui.box('content', order=2, height='400px'), title='', items=[
        ui.stepper(name='stepper', items=[
            ui.step(label='Step 1', icon='上传原图'),
            ui.step(label='Step 2', icon='上传水印'),
            ui.step(label='Step 3', icon='开始嵌入'),
            ui.step(label='Step 4', icon='下载图片'),
        ]),
        ui.text_m('步骤一: 上传原图'),
        ui.file_upload(name='origin', label='上传原图', multiple=False, file_extensions=['png', 'jpg', 'jpeg'], max_file_size=10, max_size=15, height='3'),
        ui.button('start_embed', '开始嵌入', primary=False, disabled=True)
    ])
    await q.page.save()

@on(arg='origin')
async def serve(q:Q):
    q.page['meta'] = layout1
    origin_path = q.args['origin']
    if origin_path:
        origin_local_path = await q.site.download(origin_path[0], './upload')
        ic(origin_local_path)
        q.client.origin_path = origin_local_path
        q.page['upload'] = ui.form_card(box=ui.box('content', order=2, height='400px'), title='', items=[
            ui.stepper(name='stepper', items=[
                ui.step(label='Step 1', icon='', done=True),
                ui.step(label='Step 2', icon='', done=False),
                ui.step(label='Step 3', icon='', done=False),
                ui.step(label='Step 4', icon='', done=False),
            ]),
            ui.text_m('步骤二: 上传水印'),
            ui.file_upload(name='watermark', label='上传水印', multiple=False, file_extensions=['png', 'jpg', 'jpeg'], max_file_size=10, max_size=15, height='3'),
            ui.button('start_embed', '开始嵌入', primary=False, disabled=True)
        ])

    await q.page.save()

@on(arg='watermark')
async def serve(q:Q):
    q.page['meta'] = layout1
    watermark_path = q.args['watermark']
    if watermark_path:
        watermark_local_path = await q.site.download(watermark_path[0], './upload')
        q.client.watermark_path = watermark_local_path
        q.page['upload'] = ui.form_card(box=ui.box('content', order=2, height='400px'), title='', items=[
            ui.stepper(name='stepper', items=[
                ui.step(label='Step 1', icon='', done=True),
                ui.step(label='Step 2', icon=''),
                ui.step(label='Step 3', icon=''),
                ui.step(label='Step 4', icon=''),
            ]),
            ui.text_m('步骤三: 开始嵌入'),
            ui.file_upload(name='watermark', label='上传水印', multiple=False, file_extensions=['png', 'jpg', 'jpeg'], max_file_size=10, max_size=15, height='3'),
            ui.button('start_embed', '开始嵌入', primary=True)
        ])
    await q.page.save()

@on(arg='start_embed')
async def serve(q:Q):
    q.page['meta'] = layout1
    ic(q.client.origin_path, q.client.watermark_path)
    if q.client.origin_path and q.client.watermark_path:
        bwm1 = bwm.watermark(4399,2333,36,20)
        bwm1.read_ori_img(q.client.origin_path)
        bwm1.read_wm(q.client.watermark_path)
        embed_path = './upload/watermarked.png'
        bwm1.embed(embed_path)
        # image = image_to_base64(embed_path).decode()
        # q.page['embedding'] = ui.image_card(box='content', title='嵌入水印后',image=image, type='png')
        download_path, = await q.site.upload(['/home/kevin2li/wave/myapps/upload/watermarked.png'])
        del q.client.origin_path
        del q.client.watermark_path
        ic(download_path)
        q.page['upload'] = ui.form_card(box=ui.box('content', order=2, height='400px'), title='', items=[
            ui.stepper(name='stepper', items=[
                ui.step(label='Step 1', icon='', done=True),
                ui.step(label='Step 2', icon='', done=True),
                ui.step(label='Step 3', icon='', done=True),
                ui.step(label='Step 4', icon='', done=False),
            ]),
            ui.text_m('步骤四: 完成，点击下载'),
            ui.file_upload(name='none', label='上传水印', multiple=False, file_extensions=['png', 'jpg', 'jpeg'], max_file_size=10, max_size=15, height='3'),
            ui.link('下载', path=download_path, button=True)
        ])
    else:
        q.page['meta'].dialog = ui.dialog(title='error', items=[
            ui.text('请继续上传图片'),
            ui.buttons([ui.button(name='sure', label='确定', primary=True)])
        ])
    await q.page.save()

@on(arg='sure')
async def serve(q:Q):
    q.page['meta'].dialog = None
    await q.page.save()


@on(arg='extract')
async def serve(q:Q):
    q.page['meta'] = layout1
    q.page['upload'] = ui.form_card(box=ui.box('content', order=2, height='500px'), title='', items=[
        ui.text('1.输入水印大小:'),
        ui.textbox(name='width', label='宽:', value='64'),
        ui.textbox(name='height', label='高:', value='64'),
        ui.text('2.上传带水印图片:'),
        ui.file_upload(name='watermarked', label='上传', multiple=False,
                            file_extensions=['png', 'jpg'], max_file_size=10, max_size=15, height='2'),
        ui.button('ex_wm', '下载', primary=False, disabled=True)
    ])
    await q.page.save()

@on(arg='watermarked')
async def serve(q:Q):
    q.page['meta'] = layout1
    w, h, watermarked_path = q.args['width'], q.args['height'], q.args['watermarked']
    watermarked_path = q.site.download(watermarked_path[0], './upload')
    if w and h and watermarked_path:
        bwm1 = bwm.watermark(4399,2333,36,20, wm_shape=(w, h))
        out_path = ".upload/extract_wm.png"
        watermark_img = bwm1.extract(watermarked_path, out_path)
        q.page['upload'] = ui.form_card(box=ui.box('content', order=2, height='400px'), title='上传带水印图片', items=[
            ui.textbox(name='width', label='宽', value='64'),
            ui.textbox(name='height', label='高', value='64'),
            ui.file_upload(name='watermarked', label='上传', multiple=False,
                                file_extensions=['png', 'jpg'], max_file_size=10, max_size=15, height='2'),
            ui.link('下载', path=out_path, button=True)
        ])
    await q.page.save()

#================================================================
# 隐写术
#================================================================
@on(arg='image_stega')
async def serve(q:Q):
    empty(q)
    q.page['meta'] = layout1
    q.page['v_nav'].value = '#menu/steganography'
    q.page['content'] = ui.markdown_card(box='content', title='Steganography', content='''
    # 隐写算法
    # 代码
    # 使用示例
    # 预训练模型下载
    ''')
    await q.page.save()

@on(arg='text_stega')
async def serve(q:Q):
    del q.page['content_left1']
    del q.page['content_left2']
    empty(q)
    q.page['meta'] = layout2
    q.page['v_nav'].value = '#menu/steganography'
    # q.page['content'] = ui.form_card(box='content', items=[
    #     ui.text('正在开发中...')
    # ])
    q.page['content_upper'] = ui.tab_card(box='content_upper', items=[])
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

#================================================================
# 隐写分析
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

#================================================================
# 下载专区
#================================================================
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
#================================================================
# 首页
#================================================================
@app('/')
async def serve(q:Q):
    ic(q.args)
    # ic(vars(q.page))
    if not q.client.initialized:
        q.page['meta'] = layout1
        q.page['header'] = ui.header_card(
            # Place card in the header zone, regardless of viewport size.
            box='header',
            title='集成隐写/分析平台',
            subtitle='Integrated Steganography/Steganalysis platform',
        )
        q.page['tab_bar'] = ui.tab_card(box='tab_bar', items=[
            ui.tab('image_stega', '图像隐写'),
            ui.tab('image_steganalysis', '图像隐写分析'),
            ui.tab('text_stega', '文本隐写'),
            ui.tab('text_steganalysis', '文本隐写分析'),
            ui.tab('#menu/download_section', '下载专区'),
        ])
        q.page['v_nav'] = ui.nav_card(
            box=ui.box('sidebar', height='500px'),
            value='#menu/index',
            items=[
                ui.nav_group('Menu', items=[
                    ui.nav_item(name='#menu/index', label='首页'),
                    ui.nav_item(name='#menu/steganography', label='隐写术'),
                    ui.nav_item(name='#menu/steganalysis', label='隐写分析'),
                    ui.nav_item(name='#menu/download_section', label='下载专区'),
                ]),
                ui.nav_group('帮助', items=[
                    ui.nav_item(name='#help/about', label='关于'),
                    ui.nav_item(name='#help/support', label='支持'),
                ])
            ],
        )
        q.page['content'] = ui.form_card(box='content', items=[
            ui.text('正在开发中...')
        ])
        q.page['footer'] = ui.footer_card(box='footer', caption='江苏省南京市宁六路219号')
        q.client.initialized = True
        await q.page.save()
    else:
        await handle_on(q)


#================================================================
# 工具函数
#================================================================
def image_to_base64(image_path:str):
    ext = image_path.split('.')[-1].upper()
    if ext == 'JPG':
        ext = 'jpeg'
    img = Image.open(image_path)
    output_buffer = BytesIO()
    img.save(output_buffer, format=ext)
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    return base64_str

def empty(q: Q):
    del q.page['content']
    del q.page['wm_option']
    del q.page['upload']

# class ImageStega():
#     def __init__(self, q:Q):
#         self.q = q

#     @app('/ddd')
#     async def serve(self, q:Q):
#         q.page['content'] = ui.form_card(box='content', items=[
#             ui.text('skdjfhksdhfj')
#         ])
#         await q.page.save()

# ImageStega()


def img_preprocess(img_path, eval_transforms=eval_transforms):
    img = Image.open(img_path)
    img = eval_transforms(img)
    img = img.unsqueeze(0)
    return img