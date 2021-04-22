import os
import sys
sys.path.append(os.path.abspath('..'))

from h2o_wave import Q, app, handle_on, main, on, ui
from icecream import ic
from web.utils import layout1, layout2, empty

#================================================================
# 图像隐写
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
# 文本隐写
#================================================================
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