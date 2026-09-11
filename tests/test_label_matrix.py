"""标签命中矩阵与界面判据测试。

设计要点
--------
* **直接驱动真实 `FeatureSet`**：不自己复刻模板匹配/缩放逻辑，
  测的就是运行期那条路径（`ensure_feature` + `find_one_feature`）。
* **不依赖截图进仓库**：截图目录用 `OK_DNA_SCREENSHOTS` 指定
  （默认工作区 `../新版本截图`）。找不到就整组 skip，不让 CI 红。
* **不判断场景名**：只断言"哪些判据命中了"，这正是脚本的判定方式。

三层
----
1. `TestLabelMatrix`  —— 素材完整性 + 9 判据 × 全部截图 的命中矩阵
2. `TestUiCoords`     —— 搜索框缩放 / 坐标换算 / 命名一致性
3. `TestScreenFlow`   —— 截图序列：每张图只断言"哪些判据命中了"

运行：
    .venv\\Scripts\\python.exe -m unittest tests.test_label_matrix -v
"""
import json
import os
import sys
import unittest

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ok.feature.FeatureSet import FeatureSet, adjust_coordinates
from src.ui.Defs import DISCRIMINATORS, REF_WIDTH, REF_HEIGHT, Ui, COORD

# 判据模板按 2560x1440 标注（理由见《重构方案-代码篇.md》§3.2）。
# `Defs.py` / `标注方案.md` 里的坐标是 1600x900 基准，这里按 ANNO_SCALE 换算。
ANNO_W, ANNO_H = 2560, 1440
ANNO_SCALE = ANNO_W / REF_WIDTH          # 1.6

# 原始截图目录（37 张 1600x900，工作区里，不进仓库）
_DEFAULT_SHOTS = os.path.join(os.path.dirname(REPO), '新版本截图')
SCREENSHOT_DIR = os.environ.get('OK_DNA_SCREENSHOTS', _DEFAULT_SHOTS)
COCO_JSON = os.path.join(REPO, 'assets', 'coco_annotations.json')
TEMPLATE_DIR = os.path.join(REPO, 'ok_templates')
ANNO_SHOT_DIR = os.path.join(TEMPLATE_DIR, 'ok_dna_1600')

# label -> 该判据应该命中的截图（ASCII 名）
SCREENS = {
    'start_screen_start_btn': [
        'start_screen_explore_attr', 'start_screen_survey', 'start_screen_defence',
        'start_screen_hedge', 'start_screen_expel', 'start_screen_letter'],
    'manual_select_not_use': [
        'manual_select_from_start', 'manual_select_after_result',
        'manual_select_after_comm', 'manual_select_next_round'],
    'action_dialog_retreat': [
        'action_dialog_explore', 'action_dialog_defence', 'action_dialog_letter'],
    'action_dialog_continue': [
        'action_dialog_explore', 'action_dialog_defence', 'action_dialog_letter'],
    'result_quit_btn': [
        'result_explore', 'result_defence', 'result_expel', 'result_commission', 'result_letter'],
    'letter_select_abandon': ['letter_select_from_start', 'letter_select_after_result'],
    'letter_select_ingame_confirm': ['letter_select_from_ingame'],
    'letter_reward_confirm': ['letter_reward'],
    'esc_menu_settings': ['esc_menu'],
}

# ASCII 名 -> 原始中文名（工作区截图目录里存的是中文名）
CN_NAME = {
    'start_screen_explore_attr': '探险开始i界面(包含属性选择).png',
    'start_screen_survey': '勘察开始界面.png',
    'start_screen_defence': '扼守开始界面.png',
    'start_screen_hedge': '避险开始界面.png',
    'start_screen_expel': '驱离开始界面.png',
    'start_screen_letter': '委托密函角色类开始界面.png',
    'manual_select_from_start': '开始界面委托手册选择.png',
    'manual_select_after_result': '探险结算界面再次进行委托书选择.png',
    'manual_select_after_comm': '委托完成再次进行委托手册选择界面.png',
    'manual_select_next_round': '探险继续轮次委托选择.png',
    'action_dialog_explore': '探险一轮结束.png',
    'action_dialog_defence': '扼守局内一轮完成.png',
    'action_dialog_letter': '委托密函角色类选择完奖励继续轮次询问界面.png',
    'result_explore': '探险结算.png',
    'result_defence': '扼守结算.png',
    'result_expel': '驱离完成结算.png',
    'result_commission': '委托完成结算.png',
    'result_letter': '委托密函角色类结算界面.png',
    'letter_select_from_start': '委托密函角色类开始界面选择密函界面.png',
    'letter_select_from_ingame': '委托密函角色类局内继续选择密函界面.png',
    'letter_select_after_result': '委托密函结算界面再次进行密函选择界面.png',
    'letter_reward': '委托密函角色类选择密函奖励界面.png',
    'esc_menu': '局内ESC界面.png',
}

ALL_KEYS = sorted({k for v in SCREENS.values() for k in v})

# 每个判据的模板框（**1600x900 基准**，人类可读）。
# 必须与 assets 里的 bbox 一致 —— `test_00` 会按 ANNO_SCALE 换算后逐条比对。
TEMPLATE_BOX = {
    'start_screen_start_btn': (1277, 805, 29, 30),
    'manual_select_not_use': (512, 377, 63, 64),
    'action_dialog_retreat': (501, 608, 29, 22),
    'action_dialog_continue': (1013, 610, 30, 24),
    'result_quit_btn': (1318, 783, 22, 26),
    'letter_select_abandon': (925, 568, 34, 21),
    'letter_select_ingame_confirm': (1083, 570, 50, 20),
    'letter_reward_confirm': (680, 735, 31, 25),
    'esc_menu_settings': (1144, 795, 52, 39),
}


def _to_anno(box):
    """1600x900 基准 -> assets 里的 2560x1440 原图坐标。"""
    return tuple(round(v * ANNO_SCALE) for v in box)


def _imread(path):
    """读图，兼容中文路径。"""
    return cv2.imdecode(np.frombuffer(open(path, 'rb').read(), np.uint8), cv2.IMREAD_COLOR)


def _locate(key):
    """找截图路径。

    优先 `ok_templates/okdna_<key>.png`（标注分辨率 2560x1440，
    和 assets 的 bbox 完全对齐）；没有则退回工作区原始 1600x900 截图。
    """
    for cand in (os.path.join(TEMPLATE_DIR, 'okdna_' + key + '.png'),
                 os.path.join(SCREENSHOT_DIR, key + '.png'),
                 os.path.join(SCREENSHOT_DIR, CN_NAME.get(key, '')),
                 os.path.join(TEMPLATE_DIR, key + '.png')):
        if cand and os.path.exists(cand):
            return cand
    return None


def _load_shots():
    """载入全部截图（保持各自原始分辨率，不再缩放）。

    标注分辨率的截图会原样使用，保证模板与 frame 同尺度 ——
    这样测的是"标注坐标 == 检测坐标"，而不是插值误差。
    """
    out = {}
    for k in ALL_KEYS:
        p = _locate(k)
        if not p:
            continue
        img = _imread(p)
        if img is not None:
            out[k] = img
    return out


def _load_coco():
    with open(COCO_JSON, encoding='utf-8') as f:
        return json.load(f)


SHOTS_AVAILABLE = os.path.isdir(SCREENSHOT_DIR) or os.path.isdir(ANNO_SHOT_DIR)
_skip_reason = '未找到截图目录（用环境变量 OK_DNA_SCREENSHOTS 指定）'


class _FeatureSetMixin:
    """薄封装：直接驱动真实 FeatureSet（每个标签独立加载，避免互相清缓存）。"""

    @classmethod
    def setUpClass(cls):
        cls.coco = _load_coco()
        cls.cat_by_name = {c['name']: c['id'] for c in cls.coco['categories']}
        cls.bbox = {}
        for ann in cls.coco['annotations']:
            for name, cid in cls.cat_by_name.items():
                if cid == ann['category_id']:
                    cls.bbox[name] = [round(v) for v in ann['bbox']]
                    break

    def _new_fs(self):
        return FeatureSet(False, COCO_JSON,
                          default_horizontal_variance=0.004,
                          default_vertical_variance=0.004,
                          default_threshold=0.8)

    def find(self, label, frame, use_search_box=True):
        """在一个 frame 上跑真实的 find_one。

        `use_search_box=True` 时显式传 `Defs.py` 里声明的搜索框 ——
        因为框架默认只按 `default_*_variance`(0.004) 在模板周围扩 ±6px，
        对"开始按钮有 10px 位移""同一按钮有多种布局"这类情况不够用。
        运行期 `BaseDNATask.find_ui()` 也是这么传的。
        """
        fs = self._new_fs()
        fs.check_size(frame)
        fs.ensure_feature(label)
        if label not in fs.feature_dict:
            return None
        box = None
        if use_search_box:
            box = self._search_box(label, frame.shape[1], frame.shape[0])
        res = fs.find_one_feature(frame, label, threshold=0.8, box=box)
        return res[0] if res else None

    @staticmethod
    def _search_box(label, W, H):
        """把 Defs 的搜索框（1600x900 基准）换算成本帧坐标，并保证能放下模板。

        必须用 `adjust_coordinates` 换算 —— 框架对"屏幕右半 / 下半"的元素是按
        **右/下边距**缩放的（`scale_by_anchor`），不是简单乘比例。自己乘比例
        会在大分辨率下把搜索框算到屏幕外，导致匹配到别处。
        """
        from ok import Box
        sx0, sy0, sx1, sy1 = DISCRIMINATORS[label][0]
        bx0, by0, _, _, _ = adjust_coordinates(sx0, sy0, 0, 0, W, H, REF_WIDTH, REF_HEIGHT,
                                               hcenter=True)
        bx1, by1, _, _, _ = adjust_coordinates(sx1, sy1, 0, 0, W, H, REF_WIDTH, REF_HEIGHT,
                                               hcenter=True)
        bx0, by0 = max(0, min(bx0, bx1)), max(0, min(by0, by1))
        bx1, by1 = min(W, max(bx0 + 1, bx1)), min(H, max(by0 + 1, by1))

        # 模板换算同样用 adjust_coordinates
        tx, ty, tw, th = _to_anno(TEMPLATE_BOX[label])
        nx, ny, nw, nh, _ = adjust_coordinates(tx, ty, tw, th, W, H, ANNO_W, ANNO_H, hcenter=True)
        # 兜底：确保搜索区至少比模板大一圈（框架要求模板 < 搜索区）
        bx0 = min(bx0, max(0, nx - 5))
        by0 = min(by0, max(0, ny - 5))
        bx1 = max(bx1, min(W, nx + nw + 5))
        by1 = max(by1, min(H, ny + nh + 5))
        return Box(bx0, by0, bx1 - bx0, by1 - by0, name='search_' + label)


@unittest.skipUnless(SHOTS_AVAILABLE, _skip_reason)
class TestLabelMatrix(_FeatureSetMixin, unittest.TestCase):
    """第 1 层：素材完整性 + 命中矩阵。"""

    def test_00_coco_matches_spec(self):
        """assets 里的 bbox 必须与《标注方案》的模板框逐条一致。

        这是"标注"与"运行期素材"之间的守门测试：谁改了没同步，这里立刻红。
        """
        for label, expect in TEMPLATE_BOX.items():
            with self.subTest(label=label):
                self.assertIn(label, self.cat_by_name, 'coco 里缺少 label %s' % label)
                self.assertIn(label, self.bbox, 'coco 里缺少 %s 的标注' % label)
                want = list(_to_anno(expect))
                self.assertEqual(want, self.bbox[label],
                                 '%s bbox 与规格不一致(按 x%.1f): %s (期望 %s)'
                                 % (label, ANNO_SCALE, self.bbox[label], want))

    def test_01_template_loads_and_is_small(self):
        """模板要能从 assets 加载出来，尺寸与规格一致，且足够小（省开销）。"""
        for label, box1600 in TEMPLATE_BOX.items():
            with self.subTest(label=label):
                fs = self._new_fs()
                fs.check_size(np.zeros((ANNO_H, ANNO_W, 3), np.uint8))
                fs.ensure_feature(label)
                self.assertIn(label, fs.feature_dict, '加载不到 %s' % label)
                feat = fs.feature_dict[label]
                ex, ey, ew, eh = _to_anno(box1600)
                self.assertEqual((ew, eh), (feat.width, feat.height),
                                 '%s 加载出的模板尺寸 %dx%d，期望 %dx%d'
                                 % (label, feat.width, feat.height, ew, eh))
                self.assertLessEqual(box1600[2] * box1600[3], 4200,
                                     '%s 模板像素过多（1600 基准）' % label)

    def test_02_discriminator_count(self):
        """判据正好 9 个 —— 多了说明又把"点击目标"标成 label 了。"""
        self.assertEqual(9, len(DISCRIMINATORS))

    # ------------------------------------------------------------------ 命中
    def test_10_hits_its_own_screens(self):
        """判据在它该命中的截图上必须 conf >= 0.95。

        位置容差：开始界面的 ◯ 在 6 张截图里有 **10px(1600 基准) 的 y 位移**
        （多一行"角色"/属性面板的界面整体下移），搜索框就是为覆盖它而设的。
        """
        shots = _load_shots()
        self.assertTrue(shots, '一张截图都没找到')
        for label, keys in SCREENS.items():
            for key in keys:
                if key not in shots:
                    continue
                frame = shots[key]
                with self.subTest(label=label, screen=key):
                    box = self.find(label, frame)
                    self.assertIsNotNone(box, '%s 在 %s 上没找到' % (label, key))
                    self.assertGreaterEqual(box.confidence, 0.95,
                                            '%s@%s conf 过低 %.4f' % (label, key, box.confidence))
                    # 期望位置：按 frame 与标注分辨率的比例换算
                    ex, ey = _to_anno(TEMPLATE_BOX[label])[:2]
                    ex = round(ex * frame.shape[1] / ANNO_W)
                    ey = round(ey * frame.shape[0] / ANNO_H)
                    self.assertLessEqual(abs(box.x - ex), 16,
                                         '%s@%s x 偏移 %d (期望 %d, 得到 %d)'
                                         % (label, key, box.x - ex, ex, box.x))
                    self.assertLessEqual(abs(box.y - ey), 16,
                                         '%s@%s y 偏移 %d (期望 %d, 得到 %d)'
                                         % (label, key, box.y - ey, ey, box.y))

    def test_11_never_matches_wrong_screens(self):
        """判据在不该命中的截图上不得 conf >= 0.8（防串门）。"""
        shots = _load_shots()
        for label, keys in SCREENS.items():
            for key, frame in shots.items():
                if key in keys:
                    continue
                with self.subTest(label=label, screen=key):
                    box = self.find(label, frame)
                    if box is not None:
                        self.assertLess(box.confidence, 0.8,
                                        '%s 误命中 %s (conf=%.4f)' % (label, key, box.confidence))

    def test_12_search_boxes_are_distinct(self):
        boxes = [tuple(DISCRIMINATORS[k][0]) for k in DISCRIMINATORS]
        self.assertEqual(len(boxes), len(set(boxes)), '存在重复的搜索框')

    # ------------------------------------------------------------------ 分辨率
    def test_20_works_at_other_resolutions(self):
        """同一套模板在别的分辨率下也必须命中。

        注意：这里是把 2560x1440 截图**缩小**到目标分辨率再匹配，属于真实场景
        （游戏以低分辨率渲染）。阈值放宽到 0.75 —— 缩图本身会损失细节，
        但"能不能找到、位置对不对"仍然必须成立。
        """
        shots = _load_shots()
        if not shots:
            self.skipTest('没有截图')
        base_key = 'start_screen_letter'
        for W, H in ((1920, 1080), (1280, 720)):
            for label, keys in SCREENS.items():
                key = keys[0]
                if key not in shots:
                    continue
                frame = cv2.resize(shots[key], (W, H), interpolation=cv2.INTER_AREA)
                with self.subTest(label=label, res='%dx%d' % (W, H)):
                    box = self.find(label, frame)
                    self.assertIsNotNone(box, '%s 在 %dx%d 下没找到' % (label, W, H))
                    self.assertGreaterEqual(box.confidence, 0.75,
                                            '%s@%dx%d conf %.4f' % (label, W, H, box.confidence))
                    ex, ey = _to_anno(TEMPLATE_BOX[label])[:2]
                    ex = round(ex * W / ANNO_W)
                    ey = round(ey * H / ANNO_H)
                    self.assertLessEqual(abs(box.x - ex), 12,
                                         '%s@%dx%d x 偏移 %d (期望 %d, 得到 %d)'
                                         % (label, W, H, box.x - ex, ex, box.x))
                    self.assertLessEqual(abs(box.y - ey), 12,
                                         '%s@%dx%d y 偏移 %d (期望 %d, 得到 %d)'
                                         % (label, W, H, box.y - ey, ey, box.y))


class TestUiCoords(unittest.TestCase):
    """第 2 层：搜索框缩放 / 坐标换算 / 命名一致性。"""

    TEMPLATE = TEMPLATE_BOX

    def test_01_template_fits_in_search_box(self):
        """框架要求：模板必须小于搜索框，且模板要落在搜索框内（所有分辨率）。"""
        for label, (sb, _iface, _note) in DISCRIMINATORS.items():
            sx0, sy0, sx1, sy1 = sb
            tx, ty, tw, th = self.TEMPLATE[label]
            for W, H in ((1600, 900), (1920, 1080), (2560, 1440), (1280, 720)):
                with self.subTest(label=label, res='%dx%d' % (W, H)):
                    nx, ny, nw, nh, _ = adjust_coordinates(tx, ty, tw, th, W, H, REF_WIDTH, REF_HEIGHT,
                                                           hcenter=True)
                    bx0, by0, _, _, _ = adjust_coordinates(sx0, sy0, 0, 0, W, H, REF_WIDTH, REF_HEIGHT,
                                                           hcenter=True)
                    bx1, by1, _, _, _ = adjust_coordinates(sx1, sy1, 0, 0, W, H, REF_WIDTH, REF_HEIGHT,
                                                           hcenter=True)
                    self.assertLessEqual(nw, bx1 - bx0, '%s 模板比搜索框宽' % label)
                    self.assertLessEqual(nh, by1 - by0, '%s 模板比搜索框高' % label)
                    self.assertLessEqual(bx0, nx, '%s 模板超出搜索框左边' % label)
                    self.assertLessEqual(nx + nw, bx1, '%s 模板超出搜索框右边' % label)
                    self.assertLessEqual(by0, ny, '%s 模板超出搜索框上边' % label)
                    self.assertLessEqual(ny + nh, by1, '%s 模板超出搜索框下边' % label)

    def test_02_search_box_has_margin(self):
        """搜索框四周都要留出余量（模板要尽量小，但不能贴边）。"""
        for label, (sb, _iface, _note) in DISCRIMINATORS.items():
            sx0, sy0, sx1, sy1 = sb
            tx, ty, tw, th = self.TEMPLATE[label]
            with self.subTest(label=label):
                self.assertGreaterEqual(tx - sx0, 6, '%s 左边距太小' % label)
                self.assertGreaterEqual(sx1 - (tx + tw), 6, '%s 右边距太小' % label)
                self.assertGreaterEqual(ty - sy0, 6, '%s 上边距太小' % label)
                self.assertGreaterEqual(sy1 - (ty + th), 6, '%s 下边距太小' % label)

    def test_03_coords_inside_ref_frame(self):
        """所有点击坐标都必须落在 1600x900 内。"""
        def check(name, c):
            self.assertTrue(0 <= c[0] <= REF_WIDTH, '%s x 越界: %s' % (name, c))
            self.assertTrue(0 <= c[1] <= REF_HEIGHT, '%s y 越界: %s' % (name, c))

        for name in dir(COORD):
            if name.startswith('_'):
                continue
            val = getattr(COORD, name)
            if isinstance(val, tuple) and len(val) == 2 and all(isinstance(v, int) for v in val):
                check(name, val)
            elif isinstance(val, dict):
                for k, v in val.items():
                    check('%s[%s]' % (name, k), v)

    def test_04_coords_scale(self):
        """点击坐标在 2 倍分辨率下必须等比放大。"""
        c = COORD.LETTER_REWARD_CONFIRM
        nx, ny, _, _, _ = adjust_coordinates(c[0], c[1], 0, 0, 3200, 1800, REF_WIDTH, REF_HEIGHT,
                                             hcenter=True, vcenter=True)
        self.assertAlmostEqual(nx, c[0] * 2, delta=3)
        self.assertAlmostEqual(ny, c[1] * 2, delta=3)

    def test_05_interface_names(self):
        expected = {'start_screen', 'manual_select', 'action_dialog', 'result',
                    'letter_select', 'letter_select_ingame', 'letter_reward', 'esc_menu'}
        actual = {iface for _b, iface, _n in DISCRIMINATORS.values()}
        self.assertEqual(expected, actual)

    def test_06_labels_in_label_enum(self):
        from src.LabelEnum import LabelEnum
        names = {e.value for e in LabelEnum}
        for label in DISCRIMINATORS:
            with self.subTest(label=label):
                self.assertIn(label, names)

    # ------------------------------------------------- 判据 == 点得中的地方
    LABEL_CLICK = {
        'start_screen_start_btn': 'START_SCREEN_BTN',
        'manual_select_not_use': 'MANUAL_NOT_USE',
        'action_dialog_retreat': 'ACTION_RETREAT',
        'action_dialog_continue': 'ACTION_CONTINUE',
        'result_quit_btn': 'RESULT_QUIT',
        'letter_select_abandon': 'LETTER_ABANDON',
        'letter_select_ingame_confirm': 'LETTER_INGAME_CONFIRM',
        'letter_reward_confirm': 'LETTER_REWARD_CONFIRM',
        'esc_menu_settings': 'ESC_SETTINGS',
    }

    def test_13_click_coord_lands_on_matched_element(self):
        """点击坐标必须落在判据检测到的元素所属按钮范围内（所有分辨率）。

        判据是"图标"而不是"整按钮"，所以用 `PAD` 表示图标周围多大范围仍算同一个按钮。
        """
        PAD = {
            'start_screen_start_btn': (60, 40),
            'manual_select_not_use': (40, 40),
            'action_dialog_retreat': (60, 30),
            'action_dialog_continue': (60, 30),
            'result_quit_btn': (90, 30),
            'letter_select_abandon': (50, 30),
            'letter_select_ingame_confirm': (60, 30),
            'letter_reward_confirm': (110, 40),
            'esc_menu_settings': (25, 20),
        }
        for label, coord_name in self.LABEL_CLICK.items():
            cx, cy = getattr(COORD, coord_name)
            tx, ty, tw, th = TEMPLATE_BOX[label]
            px, py = PAD[label]
            for W, H in ((1600, 900), (1920, 1080), (2560, 1440)):
                with self.subTest(label=label, res='%dx%d' % (W, H)):
                    nx, ny, nw, nh, _ = adjust_coordinates(tx, ty, tw, th, W, H,
                                                           REF_WIDTH, REF_HEIGHT, hcenter=True)
                    kx, ky, _, _, _ = adjust_coordinates(cx, cy, 0, 0, W, H,
                                                         REF_WIDTH, REF_HEIGHT,
                                                         hcenter=True, vcenter=True)
                    sx = W / REF_WIDTH
                    self.assertTrue(nx - px * sx <= kx <= nx + nw + px * sx,
                                    '%s 点击 x=%d 落在图标(%d..%d)±%d 之外'
                                    % (label, kx, nx, nx + nw, int(px * sx)))
                    self.assertTrue(ny - py * sx <= ky <= ny + nh + py * sx,
                                    '%s 点击 y=%d 落在图标(%d..%d)±%d 之外'
                                    % (label, ky, ny, ny + nh, int(py * sx)))

    def test_14_search_box_contains_template_at_anno_scale(self):
        """搜索框换算到标注分辨率后，仍必须包住标注的模板框。"""
        for label, (sb, _iface, _note) in DISCRIMINATORS.items():
            sx0, sy0, sx1, sy1 = _to_anno(sb)
            tx, ty, tw, th = _to_anno(TEMPLATE_BOX[label])
            with self.subTest(label=label):
                self.assertLessEqual(sx0, tx, '%s 搜索框左边界超了' % label)
                self.assertLessEqual(tx + tw, sx1, '%s 搜索框右边界超了' % label)
                self.assertLessEqual(sy0, ty, '%s 搜索框上边界超了' % label)
                self.assertLessEqual(ty + th, sy1, '%s 搜索框下边界超了' % label)

    def test_15_template_reference_matches_search_reference(self):
        """结构性防线：模板与搜索框在**任意分辨率**下都必须落在同一块区域。

        模板按 `ANNO_W x ANNO_H`（2560x1440）标注，搜索框按 `REF_WIDTH x REF_HEIGHT`
        （1600x900）声明。框架对屏幕右/下半的元素是按**边距**缩放的，所以两套参考
        尺寸一旦不一致，大分辨率下搜索框会漂到屏幕另一侧 —— 这个测试就是盯这个。
        """
        for label in DISCRIMINATORS:
            tx, ty, tw, th = _to_anno(TEMPLATE_BOX[label])
            sb = DISCRIMINATORS[label][0]
            for W, H in ((1280, 720), (1600, 900), (1920, 1080), (2560, 1440), (3840, 2160)):
                with self.subTest(label=label, res='%dx%d' % (W, H)):
                    nx, ny, nw, nh, _ = adjust_coordinates(tx, ty, tw, th, W, H,
                                                           ANNO_W, ANNO_H, hcenter=True)
                    bx0, by0, _, _, _ = adjust_coordinates(sb[0], sb[1], 0, 0, W, H,
                                                           REF_WIDTH, REF_HEIGHT, hcenter=True)
                    bx1, by1, _, _, _ = adjust_coordinates(sb[2], sb[3], 0, 0, W, H,
                                                           REF_WIDTH, REF_HEIGHT, hcenter=True)
                    # 搜索框要包含模板（留 1px 容差给取整）
                    self.assertLessEqual(bx0 - 1, nx,
                                         '%s@%dx%d 搜索框左边界 %d 在模板 %d 右侧（参考尺寸不一致）'
                                         % (label, W, H, bx0, nx))
                    self.assertLessEqual(nx + nw, bx1 + 1,
                                         '%s@%dx%d 搜索框右边界 %d 在模板右缘 %d 左侧（参考尺寸不一致）'
                                         % (label, W, H, bx1, nx + nw))
                    self.assertLessEqual(by0 - 1, ny,
                                         '%s@%dx%d 搜索框上边界 %d 在模板 %d 下方（参考尺寸不一致）'
                                         % (label, W, H, by0, ny))
                    self.assertLessEqual(ny + nh, by1 + 1,
                                         '%s@%dx%d 搜索框下边界 %d 在模板下缘 %d 上方（参考尺寸不一致）'
                                         % (label, W, H, by1, ny + nh))


@unittest.skipUnless(SHOTS_AVAILABLE, _skip_reason)
class TestScreenFlow(_FeatureSetMixin, unittest.TestCase):
    """第 3 层：截图序列 —— 每张图只断言"哪些判据命中了"。"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.shots = _load_shots()
        cls._cache = {}

    def hits(self, key):
        if key in self._cache:
            return self._cache[key]
        frame = self.shots.get(key)
        found = set()
        if frame is not None:
            for label in DISCRIMINATORS:
                box = self.find(label, frame)
                if box is not None and box.confidence >= 0.8:
                    found.add(label)
        self._cache[key] = found
        return found

    FLOW = [
        ('start_screen_letter', ['start_screen_start_btn']),
        ('manual_select_from_start', ['manual_select_not_use']),
        ('action_dialog_explore', ['action_dialog_retreat', 'action_dialog_continue']),
        ('result_letter', ['result_quit_btn']),
    ]

    def test_01_flow_steps(self):
        for key, expected in self.FLOW:
            if key not in self.shots:
                continue
            with self.subTest(screen=key):
                hits = self.hits(key)
                for label in expected:
                    self.assertIn(label, hits,
                                  '%s 应命中 %s，实际命中 %s' % (key, label, sorted(hits)))

    def test_02_manual_dialog_both_layouts(self):
        """手册弹窗两种布局（双按钮 / 单按钮）都靠同一个 ⊘ 判据识别。"""
        for key in ('manual_select_from_start', 'manual_select_next_round'):
            if key not in self.shots:
                continue
            with self.subTest(screen=key):
                self.assertIn('manual_select_not_use', self.hits(key))

    def test_03_letter_select_variants_differ(self):
        """局外版命中 [Esc] 判据；局内版命中自己的 [Space] 判据。"""
        if 'letter_select_from_start' in self.shots:
            h = self.hits('letter_select_from_start')
            self.assertIn('letter_select_abandon', h)
            self.assertNotIn('letter_select_ingame_confirm', h)
        if 'letter_select_from_ingame' in self.shots:
            h = self.hits('letter_select_from_ingame')
            self.assertIn('letter_select_ingame_confirm', h)
            self.assertNotIn('letter_select_abandon', h)

    def test_04_all_start_screens_share_one_label(self):
        """6 张开始界面（含属性面板版 / 角色行版）必须都命中同一个 ◯。"""
        keys = ['start_screen_explore_attr', 'start_screen_letter', 'start_screen_survey',
                'start_screen_defence', 'start_screen_hedge', 'start_screen_expel']
        seen = 0
        for key in keys:
            if key not in self.shots:
                continue
            seen += 1
            with self.subTest(screen=key):
                self.assertIn('start_screen_start_btn', self.hits(key))
        self.assertGreaterEqual(seen, 1, '一张开始界面截图都没有')

    def test_05_esc_menu_detected(self):
        if 'esc_menu' in self.shots:
            self.assertIn('esc_menu_settings', self.hits('esc_menu'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
