import re
import time
import numpy as np
import cv2
from enum import Enum
from functools import cached_property

from ok import find_boxes_by_name, TaskDisabledException
from src.tasks.BaseDNATask import BaseDNATask, isolate_white_text_to_black, color_filter
from src.ui.Defs import Ui, COORD
from src.tasks.config.CommissionConfig import (
    CommissionConfig,
    LETTER_HANDLE_AUTO_SELECT_FIRST,
    LETTER_HANDLE_START_DIRECTLY,
    LETTER_HANDLE_WAIT_USER,
    LETTER_HANDLE_MODES,
    LETTER_REWARD_DEFAULT,
    LETTER_REWARD_COUNT_ZERO,
    LETTER_REWARD_COUNT_MIN,
    LETTER_REWARD_COUNT_MAX,
    LETTER_REWARD_WAIT_USER,
)
from src.tasks.config.CommissionSkillConfig import CommissionSkillConfig


class Mission(Enum):
    START = 1
    CONTINUE = 2
    STOP = 3
    GIVE_UP = 4


class CommissionsTask(BaseDNATask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_round = 0
        self.current_wave = -1
        self.mission_status = None
        self.action_timeout = 15
        self.wave_future = None

    @cached_property
    def commission_config(self):
        return self.get_task_by_class(CommissionConfig).config
    
    @cached_property
    def commission_skill_config(self):
        return self.get_task_by_class(CommissionSkillConfig).config

    def get_letter_handle_mode(self):
        mode = self.commission_config.get("自动处理密函", LETTER_HANDLE_AUTO_SELECT_FIRST)
        if mode in LETTER_HANDLE_MODES:
            return mode
        return LETTER_HANDLE_AUTO_SELECT_FIRST

    def setup_commission_config(self):
        self.default_config.update({
            "轮次": 5,
            "超时时间": 90,
        })
        self.config_description.update({
            "轮次": "打几个轮次",
            "超时时间": "超时后将重启任务",
        })

    # ------------------------------------------------------------------
    # 界面判据（唯一出处 src/ui/Defs.py）
    #   命名规则：find_<界面>_<元素>；判据命中 == 当前在这个界面上
    # ------------------------------------------------------------------
    def find_action_dialog_retreat(self, threshold=0):
        """行动抉择弹窗的「撤离」（原 find_ingame_quit_btn）。"""
        return self.find_ui(Ui.ACTION_DIALOG_RETREAT, threshold=threshold)

    def find_action_dialog_continue(self, threshold=0):
        """行动抉择弹窗的「继续挑战」（原 find_ingame_continue_btn）。"""
        return self.find_ui(Ui.ACTION_DIALOG_CONTINUE, threshold=threshold)

    def find_start_btn(self, threshold=0, box=None, template=None):
        """开始界面的「开始」按钮（新版只有一个位置，不再分 bottom/big）。"""
        return self.find_ui(Ui.START_SCREEN_START, threshold=threshold, box=box, template=template)

    def find_manual_select_btn(self, threshold=0):
        """委托手册弹窗（用 ⊘ 不使用当判据）。"""
        return self.find_ui(Ui.MANUAL_SELECT_NOT_USE, threshold=threshold)

    def find_letter_btn(self, threshold=0):
        """密函选择弹窗（局外版）。"""
        return self.find_ui(Ui.LETTER_SELECT_ABANDON, threshold=threshold)

    def find_letter_ingame_btn(self, threshold=0):
        """密函选择弹窗（局内继续轮次版，布局不同）。"""
        return self.find_ui(Ui.LETTER_SELECT_INGAME_CONFIRM, threshold=threshold)

    def find_letter_reward_btn(self, threshold=0):
        """密函奖励弹窗。"""
        return self.find_ui(Ui.LETTER_REWARD_CONFIRM, threshold=threshold)

    def find_esc_menu(self, threshold=0):
        """局内 ESC 菜单（用「设置」齿轮当判据）。"""
        return self.find_ui(Ui.ESC_MENU_SETTINGS, threshold=threshold)


    def open_in_mission_menu(self, time_out=20, raise_if_not_found=True):
        if self.find_esc_menu():
            return True
        found = False
        start = time.time()
        while time.time() - start < time_out:
            self.send_key("esc")
            if self.wait_until(self.find_esc_menu, time_out=2, raise_if_not_found=False):
                found = True
                break
        else:
            if raise_if_not_found:
                raise Exception("未找到任务菜单")
        self.sleep(0.2)
        return found

    def start_mission(self, timeout=0):
        """点「开始」（判据：开始界面的 ◯），然后等进入下一步（手册弹窗或密函弹窗）。"""
        action_timeout = self.action_timeout if timeout == 0 else timeout
        box = self.box_of_screen_scaled(2560, 1440, 69, 969, 2498, 1331, name="reward_drag_area", hcenter=True)
        deadline = time.time() + action_timeout
        clicked = False

        while time.time() < deadline:
            if self.find_start_btn():
                # 点检测到的按钮本身（不再用写死坐标）
                self.click_ui_coord(COORD.START_SCREEN_BTN, name="start_mission",
                                    after_sleep=0.2, use_safe_move=True, safe_move_box=box)
                clicked = True
            elif self.find_manual_select_btn() or self.find_letter_interface():
                return              # 已经进入下一步
            else:
                self.next_frame()
                continue

            if self.wait_until(condition=lambda: self.find_manual_select_btn() or self.find_letter_interface(),
                               time_out=2):
                return

        if clicked:
            raise Exception("点击开始后未进入委托手册/密函界面")
        raise Exception("等待开始任务超时")

    def quit_mission(self, timeout=0):
        action_timeout = self.action_timeout if timeout == 0 else timeout
        self.wait_until(self.find_action_dialog_retreat, time_out=action_timeout, raise_if_not_found=True)
        self.wait_until(
            condition=lambda: not self.find_action_dialog_retreat(),
            post_action=lambda: self.click_ui_coord(COORD.ACTION_RETREAT, name="quit_mission", after_sleep=0.25),
            time_out=action_timeout,
            raise_if_not_found=True,
        )
        self.sleep(1)
        self.wait_until(lambda: not self.in_team(), time_out=action_timeout, raise_if_not_found=True)

    def give_up_mission(self, timeout=0):
        def is_mission_start_iface():
            return self.find_start_btn() or self.find_action_dialog_continue() or self.find_esc_menu()

        action_timeout = self.action_timeout if timeout == 0 else timeout

        if self.open_in_mission_menu(time_out=10, raise_if_not_found=False):
            # ESC 菜单里「放弃挑战」-> 再确认一次
            self.wait_until(
                condition=lambda: not self.find_esc_menu(),
                post_action=lambda: self.click_ui_coord(COORD.ESC_GIVEUP, name="esc_giveup", after_sleep=0.25),
                time_out=action_timeout,
                raise_if_not_found=True,
            )
            self.sleep(0.5)

        self.wait_until(condition=is_mission_start_iface, time_out=60, raise_if_not_found=False)

    def continue_mission(self, timeout=0):
        if self.in_team():
            return False
        action_timeout = self.action_timeout if timeout == 0 else timeout
        self.wait_until(
            condition=lambda: not self.find_action_dialog_continue() and not self.find_action_dialog_retreat(),
            post_action=lambda: self.click_ui_coord(COORD.ACTION_CONTINUE, name="continue_mission", after_sleep=0.25),
            time_out=action_timeout,
            raise_if_not_found=True,
        )
        self.sleep(0.5)
        return True

    def choose_drop_rate(self, timeout=0):
        """选委托手册：先按配置点格子，再点「开始挑战」/「确认选择」直到弹窗关闭。"""
        def click_confirm():
            if self.find_manual_select_btn():
                self.click_ui_coord(COORD.MANUAL_CONFIRM, name="manual_confirm", after_sleep=0.25)

        action_timeout = self.action_timeout if timeout == 0 else timeout
        self.sleep(0.5)
        self.choose_drop_rate_item()
        self.wait_until(
            condition=lambda: not self.find_manual_select_btn(),
            post_action=click_confirm,
            time_out=action_timeout,
            raise_if_not_found=True,
        )

    def choose_drop_rate_item(self):
        if not hasattr(self, "config"):
            return
        drop_rate = self.commission_config.get("委托手册", "不使用")
        if drop_rate == "不使用":
            return
        round_to_use = [int(num) for num in re.findall(r'\d+', self.commission_config.get("委托手册指定轮次", ""))]
        if len(round_to_use) != 0:
            if self.mission_status != Mission.CONTINUE:
                if 1 not in round_to_use:
                    return
            elif self.current_round == 0 or (self.current_round + 1) not in round_to_use:
                return
        coord = COORD.MANUAL_ITEM.get(drop_rate)
        if coord is not None:
            self.click_ui_coord(coord, name="manual_%s" % drop_rate)
        self.log_info(f"使用委托手册: {drop_rate}")
        self.sleep(0.25)

    def choose_letter(self, timeout=0):
        if not hasattr(self, "config"):
            return
        action_timeout = self.action_timeout if timeout == 0 else timeout
        mode = self.get_letter_handle_mode()
        if mode == LETTER_HANDLE_WAIT_USER:
            self.log_info_notify("需自行选择密函")
            self.soundBeep()
            self.wait_until(
                lambda: not self.find_letter_interface(),
                time_out=300,
                raise_if_not_found=True,
            )
            return
        if mode in (LETTER_HANDLE_AUTO_SELECT_FIRST, LETTER_HANDLE_START_DIRECTLY) and self.find_letter_interface():
            box = self.box_of_screen(0.4432, 0.3556, 0.9750, 0.6037, name="letter_drag_area", hcenter=True)
            if mode == LETTER_HANDLE_AUTO_SELECT_FIRST:
                not_use_edge = self.box_of_screen(0.4443, 0.3630, 0.4526, 0.4991, name="not_use_edge", hcenter=True)
                self.sleep(0.1)
                for _ in range(2):
                    self.click_ui_coord(
                        COORD.MANUAL_NOT_USE,
                        name="letter_not_use",
                        use_safe_move=True,
                        safe_move_box=box,
                        down_time=0.02,
                        after_sleep=0.1,
                    )
                    if self.wait_until(lambda: self.calculate_color_percentage(white_color, not_use_edge) < 0.05, time_out=1):
                        break
                else:
                    self.log_info_notify("密函已耗尽")
                    self.soundBeep()
                    raise TaskDisabledException
                
            deadline = time.time() + action_timeout
            while time.time() < deadline:
                letter_btn = self.find_letter_btn()
                if letter_btn:
                    self.move_back_from_safe_position()
                    break
                else:
                    self.move_mouse_to_safe_position()
                    self.next_frame()
            else:
                self.log_info_notify("未找到密函确认按钮")
                self.soundBeep()
                raise TaskDisabledException

            self.wait_until(
                condition=lambda: not self.find_letter_interface(),
                post_action=lambda: self.click_ui_coord(COORD.LETTER_CONFIRM, name="letter_confirm",
                                                        after_sleep=1, use_safe_move=True, safe_move_box=box),
                time_out=action_timeout,
                raise_if_not_found=True,
            )

    def choose_target_letter_reward(self):
        reward_pattern = re.compile(r'[:：]\s*([0-9]+)')
        def get_rewards():
            box = self.box_of_screen(0.328, 0.643, 0.678, 0.672, hcenter=True, name="letter_reward")
            return self.ocr(box=box, match=reward_pattern)
        
        start = time.time()
        while time.time() - start < 10:
            rewards = get_rewards()
            if len(rewards) == 3:
                break
            self.sleep(0.1)
        else:
            self.log_info("超时：未识别到3个奖励选项，使用默认奖励")
            return

        self.sleep(0.3)
        rewards = get_rewards()

        if len(rewards) != 3:
            self.log_info(f"异常：稳定后识别数量不符 (识别到 {len(rewards)} 个)，使用默认奖励")
            return

        rewards.sort(key=lambda reward: reward.x)

        parsed_items = []
        for idx, reward in enumerate(rewards):
            match = reward_pattern.search(reward.name)
            if not match:
                self.log_info(f"第 {idx + 1} 个奖励数量识别失败，使用默认奖励")
                return
            count = int(match.group(1))
            parsed_items.append({
                'index': idx + 1,
                'count': count,
                'reward_obj': reward,
                'name': reward.name
            })

        strategy = self.commission_config.get("密函奖励偏好")
        target_item = None

        self.log_info(f"当前识别到的奖励持有数: {[item['count'] for item in parsed_items]}")

        if strategy == LETTER_REWARD_COUNT_ZERO:
            for item in parsed_items:
                if item['count'] == 0:
                    target_item = item
                    break
            if not target_item:
                self.log_info("未识别到持有数为0的奖励，使用默认奖励")
                return

        elif strategy == LETTER_REWARD_COUNT_MIN:
            target_item = min(parsed_items, key=lambda x: x['count'])

        elif strategy == LETTER_REWARD_COUNT_MAX:
            target_item = max(parsed_items, key=lambda x: x['count'])

        if target_item:
            self.log_info(f"策略[{strategy}] -> 选择第 {target_item['index']} 个奖励，持有数: {target_item['count']}")
            # 奖励卡内容是随机的 -> 不建 label，按固定坐标点第 index 张
            card = (COORD.LETTER_REWARD_CARD_1, COORD.LETTER_REWARD_CARD_2,
                    COORD.LETTER_REWARD_CARD_3)[target_item['index'] - 1]
            self.click_ui_coord(card, name="reward_card_%d" % target_item['index'],
                                down_time=0.02, after_sleep=0.5)

    def choose_letter_reward(self, timeout=0):
        action_timeout = self.action_timeout if timeout == 0 else timeout
        reward_strategy = self.commission_config.get("密函奖励偏好", LETTER_REWARD_DEFAULT)
        if reward_strategy == LETTER_REWARD_WAIT_USER:
            self.log_info_notify("需自行选择密函奖励")
            self.soundBeep()
            self.wait_until(
                lambda: not self.find_letter_reward_btn(),
                time_out=300,
                raise_if_not_found=True,
            )
        else:
            if reward_strategy in (LETTER_REWARD_COUNT_ZERO, LETTER_REWARD_COUNT_MIN, LETTER_REWARD_COUNT_MAX):
                self.choose_target_letter_reward()
            self.wait_until(
                condition=lambda: not self.find_letter_reward_btn(),
                post_action=lambda: self.click_ui_coord(COORD.LETTER_REWARD_CONFIRM,
                                                        name="letter_reward_confirm",
                                                        down_time=0.02, after_sleep=0.25),
                time_out=action_timeout,
                raise_if_not_found=True,
            )
        self.sleep(0.1)
        self.wait_until(lambda: not self.in_team(), time_out=3, settle_time=0.5)

    def create_skill_ticker(self):
        skills = []
        def create_ticker(local_n):
            def action():
                self.log_onetime_info("全局技能设定: " + str(self.commission_skill_config), "全局技能设定")
                skill = self.commission_skill_config.get(f"技能{local_n}", "不使用")
                if skill == "不使用":
                    return
                after_sleep = self.commission_skill_config.get(f"技能{local_n}_释放后等待", 0.0)
                if skill == "战技":
                    self.get_current_char().send_combat_key()
                elif skill == "Ctrl+战技（赛琪专属）":
                    self.get_current_char().send_combat_key_with_ctrl()
                elif skill == "终结技":
                    self.get_current_char().send_ultimate_key()
                elif skill == "自动苏乙终结技":
                    on_result = self.find_one('suyi_q_on')
                    off_result = self.find_one('suyi_q_off')
                    on_conf = on_result.confidence if on_result else 0
                    off_conf = off_result.confidence if off_result else 0
                    if off_conf > on_conf:
                        self.get_current_char().send_ultimate_key()
                    else:
                        self.get_current_char().click()
                elif skill == "魔灵支援":
                    self.get_current_char().send_geniemon_key()
                elif skill == "普攻":
                    self.get_current_char().click()
                if after_sleep > 10:
                    self.log_onetime_info(f"检测到长延时：释放技能 {local_n} 后将等待 {after_sleep} 秒，可能影响脚本运行，请确认是否符合预期")
                self.sleep(after_sleep)

            return self.create_ticker(
                action, 
                interval=lambda: self.commission_skill_config.get(f"技能{local_n}_释放频率", 5.0), 
                interval_random_range=(0.8, 1.2)
            )
        
        for n in range(1, 5):
            skills.append(create_ticker(n))

        return self.create_ticker_group(skills)

    def get_round_info(self):
        """获取并更新当前轮次信息。"""
        if self.in_team():
            return
        box = self.box_of_screen(0.241, 0.361, 0.259, 0.394, name="green_mark", hcenter=True)
        self.wait_until(lambda: self.calculate_color_percentage(green_mark_color, box) > 0.135, time_out=1)
        round_info_box = self.box_of_screen_scaled(2560, 1440, 500, 500, 620, 620, name="round_info", hcenter=True)
        texts = self.ocr(box=round_info_box, frame_processor=ocr_normalize, name="round_info")
        # img = ocr_normalize(round_info_box.crop_frame(self.frame))
        # self.screenshot(name=f"round_info_ocr_{texts}", frame=img)

        prev_round = self.current_round
        new_round_from_ocr = None
        if texts and texts[0].name.isdigit():
            new_round_from_ocr = int(texts[0].name)
            self.log_debug(f"get_round_info ocr 轮次 {new_round_from_ocr}")

        if new_round_from_ocr is not None:
            self.current_round = new_round_from_ocr
        elif self.current_round != 0:  # OCR失败，但之前已有轮次记录，则递增
            self.current_round += 1

        if prev_round != self.current_round:
            self.info_set("当前轮次", self.current_round)

    def get_wave_info(self):
        if not self.in_team():
            return
        if self.wave_future and self.wave_future.done():
            texts = self.wave_future.result()
            self.wave_future = None
            if texts and len(texts) >= 1:
                prev_wave = self.current_wave
                if (m := re.match(r"(\d)/\d", texts[0].name)):
                    self.current_wave = int(m.group(1))
                else:
                    return
                if prev_wave != self.current_wave:
                    self.info_set("当前波次", self.current_wave)
            return
        if self.wave_future is None:
            mission_info_box = self.box_of_screen(0.107, 0.343, 0.174, 0.386, name="mission_info", hcenter=True)
            frame = self.frame.copy()
            self.wave_future = self.thread_pool_executor.submit(self.ocr, frame=frame,
                                                                box=mission_info_box,
                                                                frame_processor=isolate_white_text_to_black,
                                                                match=re.compile(r"\d/\d"))

    def reset_wave_info(self):
        if self.wave_future is not None:
            self.wave_future.cancel()
            self.wave_future = None
        self.current_wave = -1
        self.info_set("当前波次", self.current_wave)

    def wait_until_get_wave_info(self):
        self.log_info("等待波次信息...")
        while self.current_wave == -1:
            self.get_wave_info()
            self.sleep(0.2)

    def handle_mission_interface(self, stop_func=lambda: False):
        """每步操作后重新看画面：匹配到哪个元素，就执行哪一段逻辑。

        骨架保持原有优先级顺序（见《素材重构文档/重构方案-代码篇.md》§1.2）；
        这里只把"识别机制"换成界面判据、把"坐标来源"换成 Defs。
        """
        if self.in_team():
            return False

        self.check_for_monthly_card()

        # 优先级 1：密函奖励（只在奖励弹窗出现时有意义）
        if self.find_letter_reward_btn():
            self.log_info("处理任务界面: 选择密函奖励")
            self.choose_letter_reward()
            return

        # 优先级 2：密函选择 / 委托手册
        if self.find_letter_interface():
            self.log_info("处理任务界面: 选择密函")
            self.choose_letter()
            return self.get_return_status()
        elif self.find_manual_select_btn():
            self.log_info("处理任务界面: 选择委托手册")
            self.choose_drop_rate()
            return self.get_return_status()

        # 优先级 3：开始 / 继续 / 放弃
        if self.find_start_btn():
            self.log_info("处理任务界面: 开始任务")
            self.start_mission()
            self.mission_status = Mission.START
            return
        elif self.find_action_dialog_continue() or self.find_action_dialog_retreat():
            if stop_func():
                self.log_info("处理任务界面: 终止任务")
                return Mission.STOP
            self.log_info("处理任务界面: 继续任务")
            self.continue_mission()
            self.mission_status = Mission.CONTINUE
            return
        elif self.find_esc_menu():
            self.log_info("处理任务界面: 放弃任务")
            self.give_up_mission()
            return Mission.GIVE_UP
        return False

    def get_return_status(self):
        ret = self.mission_status if self.mission_status else Mission.START
        self.mission_status = None
        return ret

    def reset_and_transport(self):
        # 1) 打开局内菜单(ESC 菜单)
        self.open_in_mission_menu()
        self.wait_until(
            condition=lambda: not self.find_esc_menu(),
            # 2) 点击"设置"入口, 点击后应关闭 ESC 菜单
            post_action=lambda: self.click_ui_coord(COORD.ESC_SETTINGS, name="esc_settings"),
            time_out=10,
        )
        setting_box = self.box_of_screen_scaled(2560, 1440, 738, 4, 1123, 79, name="other_section", hcenter=True)
        setting_other = self.wait_until(lambda: self.find_one("setting_other", box=setting_box), time_out=10,
                                        raise_if_not_found=True)
        # 云游戏局内打开菜单必然卡一下
        self.sleep(0.5)
        self.wait_until(
            condition=lambda: self.calculate_color_percentage(setting_menu_selected_color, setting_other) > 0.24,
            # 3) 点击“其他设置”页签(setting_other), 直到该页签呈选中状态(通过颜色占比判断)，点击不要那么频繁
            post_action=lambda: self.click_box_random(setting_other, after_sleep=0.5),
            time_out=10,
        )
        self.sleep(0.5)
        safe_box = self.box_of_screen_scaled(2560, 1440, 125, 207, 1811, 1234, name="safe_box", hcenter=True)
        # 4) 点“重置角色/复位并传送”
        #    NOTE: 这个二次确认弹窗本轮没截到图，暂时沿用原来的相对坐标（见《标注方案》§8 待补）。
        self.click_ui_coord(COORD.RESET_TRANSPORT_CONFIRM, name="reset_transport",
                            after_sleep=0.5, use_safe_move=True, safe_move_box=safe_box)
        self.sleep(0.5)
        safe_box2 = self.box_of_screen_scaled(2560, 1440, 1298, 772, 1735, 846, name="safe_box", hcenter=True)
        # 5) 点二次确认弹框的“确认”（同上，待补截图后换成判据）
        self.click_ui_coord(COORD.RESET_TRANSPORT_OK, name="reset_transport_ok",
                            after_sleep=0.5, use_safe_move=True, safe_move_box=safe_box2)
        if not self.wait_until(self.in_team, time_out=10):
            self.ensure_main()
            return False
        return True

    def find_letter_interface(self):
        """密函选择弹窗（局外版或局内版都算）。"""
        return self.find_letter_btn() or self.find_letter_ingame_btn()


class QuickAssistTask:

    def __init__(self, owner: "CommissionsTask"):
        self._owner = owner
        self._move_task = None
        self._aim_task = None

    def run(self):
        if self._owner.commission_config.get("自动穿引共鸣", False):
            if not self._move_task:
                from src.tasks.trigger.AutoMoveTask import AutoMoveTask

                self._move_task = self._owner.get_task_by_class(AutoMoveTask)

            if self._move_task:
                self._move_task.try_connect_listener()
                self._move_task.run()
        
        if self._owner.commission_config.get("自动花弓", False):
            if not self._aim_task:
                from src.tasks.trigger.AutoAimTask import AutoAimTask

                self._aim_task = self._owner.get_task_by_class(AutoAimTask)

            if self._aim_task:
                self._aim_task.try_connect_listener()
                self._aim_task.run()

    def reset(self):
        if self._move_task:
            self._move_task.reset()
            self._move_task.try_disconnect_listener()
        if self._aim_task:
            self._aim_task.reset()
            self._aim_task.try_disconnect_listener()

def ocr_normalize(cv_image):
    cv_image = color_filter(cv_image, round_info_color)
    _, cv_image = cv2.threshold(cv_image, 127, 255, cv2.THRESH_BINARY)
    cv_image = cv2.resize(cv_image, None, fx=1, fy=0.75, interpolation=cv2.INTER_AREA)
    cv_image = cv2.erode(cv_image, np.ones((2, 2), np.uint8) , iterations=1)
    cv_image = cv2.bitwise_not(cv_image)
    return cv_image

setting_menu_selected_color = {
    'r': (220, 255),  # Red range
    'g': (200, 255),  # Green range
    'b': (125, 250)  # Blue range
}

retry_btn_color = {
    'r': (220, 230),  # Red range
    'g': (175, 185),  # Green range
    'b': (79, 89)  # Blue range
}

green_mark_color = {
    'r': (40, 55),  # Red range
    'g': (165, 170),  # Green range
    'b': (120, 130)  # Blue range
}

round_info_color = {
    'r': (200, 255),  # Red range
    'g': (200, 255),  # Green range
    'b': (200, 255)  # Blue range
}

white_color = {
    'r': (245, 255),  # Red range
    'g': (245, 255),  # Green range
    'b': (245, 255)  # Blue range
}

def _default_movement():
    pass
