"""自动苏乙终结技：Q 图标 ON/OFF 判别测试。

这两个 label（`suyi_q_on` / `suyi_q_off`）在本轮素材重构中被保留下来
（见《素材重构文档/重构方案-代码篇.md》§4.3），所以对应的测试也保留。

判定规则（与 `CommissionsTask.create_skill_ticker` 一致）：
    两个都匹配到时比置信度；只匹配到一个时，那个胜出。
"""
import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ok.test.TaskTestCase import TaskTestCase
from src.config import config
from src.tasks.CommissionsTask import CommissionsTask

IMAGES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')


@unittest.skipUnless(os.path.exists(os.path.join(IMAGES, 'suyi_q_on.png')),
                     '缺少 tests/images/suyi_q_on.png')
class TestSuyiQ(TaskTestCase):
    task_class = CommissionsTask
    config = config

    def _decide(self, image):
        """返回 (on_conf, off_conf, 判定结果)。"""
        self.set_image(os.path.join('tests', 'images', image))
        on_res = self.task.find_one('suyi_q_on')
        off_res = self.task.find_one('suyi_q_off')
        on_conf = on_res.confidence if on_res else 0.0
        off_conf = off_res.confidence if off_res else 0.0
        # 与 CommissionsTask 里的逻辑一致：off_conf > on_conf 才按 Q
        should_press = off_conf > on_conf
        return on_conf, off_conf, should_press

    def test_01_on_state_does_not_press(self):
        """ON 状态：技能已开 → 不该再按 Q。"""
        on_conf, off_conf, should_press = self._decide('suyi_q_on.png')
        self.assertGreater(on_conf, 0.0, 'ON 图标应能匹配到')
        self.assertFalse(should_press,
                         'ON 状态下不该按 Q (on=%.4f off=%.4f)' % (on_conf, off_conf))

    def test_02_off_state_presses(self):
        """OFF 状态：技能未开 → 应按 Q。"""
        on_conf, off_conf, should_press = self._decide('suyi_q_off.png')
        self.assertGreater(off_conf, 0.0, 'OFF 图标应能匹配到')
        self.assertTrue(should_press,
                        'OFF 状态下应按 Q (on=%.4f off=%.4f)' % (on_conf, off_conf))


if __name__ == '__main__':
    unittest.main(verbosity=2)
