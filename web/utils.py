import os
import sys

sys.path.append(os.path.abspath('..'))

import base64
from io import BytesIO

import torchvision.transforms as T
from h2o_wave import Q, app, handle_on, main, on, ui
from icecream import ic
from PIL import Image

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
                  width='80%',
                  height='100%',
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
                    width='80%',
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

def empty(q: Q):
    del q.page['content']
    del q.page['wm_option']
    del q.page['upload']


def img_preprocess(img_path, eval_transforms=eval_transforms):
    img = Image.open(img_path)
    img = eval_transforms(img)
    img = img.unsqueeze(0)
    return img

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

@on()
async def sure(q:Q):
    q.page['meta'].dialog = None
    await q.page.save()

