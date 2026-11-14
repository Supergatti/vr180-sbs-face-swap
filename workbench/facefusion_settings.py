from __future__ import annotations

import facefusion_runtime


MODELS = facefusion_runtime.HOME / '.assets' / 'models'
DEFAULTS = {
    'face_swapper_model': 'inswapper_128_fp16',
    'face_swapper_pixel_boost': '128x128',
    'face_enhancer_model': 'gfpgan_1.4',
    'face_enhancer_blend': 80,
    'face_enhancer_weight': 1.0,
    'face_mask_types': ['box'],
    'face_mask_regions': ['skin', 'left-eyebrow', 'right-eyebrow', 'left-eye', 'right-eye',
                          'glasses', 'nose', 'mouth', 'upper-lip', 'lower-lip'],
    'face_mask_blur': .3,
    'face_mask_padding': (0, 0, 0, 0),
    'face_detector_angles': [0],
    'face_detector_score': .35,
    'face_landmarker_score': .5,
    'vr_detector_score': .25,
    'reference_face_distance': .48,
}

CONVERTERS = {
    **{f'ghost_{i}_256': 'arcface_converter_ghost' for i in (1, 2, 3)},
    'hififace_unofficial_256': 'arcface_converter_hififace',
    'simswap_256': 'arcface_converter_simswap',
    'simswap_unofficial_512': 'arcface_converter_simswap',
}
MASK_MODELS = {'occlusion': 'xseg_1', 'region': 'bisenet_resnet_34'}
DETECTOR_ANGLES = (0, 90, 180, 270)


def detector_angles(values):
    if not isinstance(values, list) or not values or any(type(angle) is not int or angle not in DETECTOR_ANGLES for angle in values):
        raise ValueError('至少选择一个检测角度：0°、90°、180° 或 270°。')
    return sorted(set(values))


def installed(*models):
    return all((MODELS / f'{model}.onnx').is_file() for model in models)


def catalog():
    facefusion_runtime.configure()
    from facefusion.processors import choices

    return {
        'defaults': DEFAULTS,
        'swappers': [
            {'name': name, 'sizes': sizes,
             'available': installed(name, CONVERTERS[name]) if name in CONVERTERS else installed(name)}
            for name, sizes in choices.face_swapper_set.items()
        ],
        'enhancers': [{'name': name, 'available': installed(name)} for name in choices.face_enhancer_models],
        'masks': {kind: installed(model) for kind, model in MASK_MODELS.items()},
    }


def bounded(value, low, high, label, integer=False):
    types = (int,) if integer else (int, float)
    if type(value) not in types or not low <= value <= high:
        raise ValueError(f'{label}范围：{low}–{high}。')
    return value


def selected(values, allowed, label):
    if not isinstance(values, list) or not values or any(value not in allowed for value in values):
        raise ValueError(f'请选择有效的{label}。')
    return list(dict.fromkeys(values))


def validate(options, enhance):
    if not isinstance(options, dict) or options.keys() - DEFAULTS.keys():
        raise ValueError('无效的 FaceFusion 参数。')
    result = {**DEFAULTS, **options}
    models = catalog()
    swapper = next((m for m in models['swappers'] if m['name'] == result['face_swapper_model']), None)
    enhancer = next((m for m in models['enhancers'] if m['name'] == result['face_enhancer_model']), None)
    if not swapper or not swapper['available']:
        raise ValueError('换脸模型或其转换权重未安装。')
    if result['face_swapper_pixel_boost'] not in swapper['sizes']:
        raise ValueError('当前换脸模型不支持所选像素增强尺寸。')
    if not enhancer or (enhance and not enhancer['available']):
        raise ValueError('所选人脸增强模型未安装。')
    masks = selected(result['face_mask_types'], ('box', *MASK_MODELS), '遮罩类型')
    if any(kind != 'box' and not models['masks'][kind] for kind in masks):
        raise ValueError('所选遮罩模型未安装。')
    result['face_mask_types'] = masks
    result['face_mask_regions'] = selected(result['face_mask_regions'], DEFAULTS['face_mask_regions'], '遮罩区域')
    margins = result['face_mask_padding']
    if not isinstance(margins, (list, tuple)) or len(margins) != 4:
        raise ValueError('请提供上、右、下、左四个遮罩内缩值。')
    margins = tuple(bounded(v, 0, 99, '遮罩内缩', True) for v in margins)
    if margins[0] + margins[2] >= 100 or margins[1] + margins[3] >= 100:
        raise ValueError('相对两边的遮罩内缩之和必须小于 100%。')
    result['face_mask_padding'] = margins
    result['face_detector_angles'] = detector_angles(result['face_detector_angles'])
    for key, low, high, label in (
        ('face_mask_blur', 0, 1, '遮罩模糊'),
        ('face_enhancer_weight', 0, 1, '增强权重'),
        ('face_detector_score', .05, .95, '透视精检阈值'),
        ('face_landmarker_score', 0, 1, '关键点阈值'),
        ('vr_detector_score', .05, .95, 'YOLO180 粗检阈值'),
        ('reference_face_distance', .01, 1.5, '参考脸距离'),
    ):
        bounded(result[key], low, high, label)
    bounded(result['face_enhancer_blend'], 0, 100, '增强混合比例', True)
    return result
