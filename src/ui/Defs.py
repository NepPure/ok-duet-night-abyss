"""ui/Defs.py  —  界面判据与点击坐标的唯一出处。

设计原则（详见《素材重构文档/标注方案.md》）
--------------------------------------------
1. **label 的唯一职责是回答"我在哪个界面"** —— 所以这里只定义 9 个"判据"。
2. 判据一旦命中，那个界面要做什么、点哪里，都是**固定坐标**，不需要再做 label。
3. 模板里**不含中文、不含用户可变数字、不含随机内容**（奖励卡/道具格一律不标）。
4. 坐标一律以 **1600x900** 为基准；运行期由框架按实际分辨率等比缩放
   （`ok.feature.FeatureSet.adjust_coordinates`）。

用法
----
    from src.ui.Defs import DISCRIMINATORS, Ui, COORD

    box = self.find_ui(Ui.RESULT_QUIT_BTN)      # 找判据（返回 Box 或 None）
    self.click_ui_coord(COORD.RESULT_QUIT)      # 按坐标点
"""

REF_WIDTH = 1600
REF_HEIGHT = 900


class Ui:
    """9 个界面判据 label 名（与 LabelEnum / assets 里的一致）。"""
    START_SCREEN_START = 'start_screen_start_btn'
    MANUAL_SELECT_NOT_USE = 'manual_select_not_use'
    ACTION_DIALOG_RETREAT = 'action_dialog_retreat'
    ACTION_DIALOG_CONTINUE = 'action_dialog_continue'
    RESULT_QUIT_BTN = 'result_quit_btn'
    LETTER_SELECT_ABANDON = 'letter_select_abandon'
    LETTER_SELECT_INGAME_CONFIRM = 'letter_select_ingame_confirm'
    LETTER_REWARD_CONFIRM = 'letter_reward_confirm'
    ESC_MENU_SETTINGS = 'esc_menu_settings'


# label -> (搜索框 (x0,y0,x1,y1), 判据界面名, 说明)
DISCRIMINATORS = {
    Ui.START_SCREEN_START: (
        (1262, 788, 1322, 848), 'start_screen',
        '◯ 开始按钮 -> 开始界面（6 个开始界面共用）'),
    Ui.MANUAL_SELECT_NOT_USE: (
        (497, 362, 590, 456), 'manual_select',
        '⊘ 不使用 图标 -> 委托手册弹窗（4 个入口共用）'),
    Ui.ACTION_DIALOG_RETREAT: (
        (486, 592, 560, 645), 'action_dialog',
        '撤离 门形图标 -> 行动抉择弹窗'),
    Ui.ACTION_DIALOG_CONTINUE: (
        (998, 594, 1075, 650), 'action_dialog',
        '继续挑战 ◯ 图标 -> 行动抉择弹窗'),
    Ui.RESULT_QUIT_BTN: (
        (1296, 768, 1358, 815), 'result',
        '退出委托 蓝色门形图标 -> 任务结算（5 张共用）'),
    Ui.LETTER_SELECT_ABANDON: (
        (910, 553, 985, 605), 'letter_select',
        '[Esc] 键位芯片 -> 密函选择（局外：开始界面/结算后进入）'),
    Ui.LETTER_SELECT_INGAME_CONFIRM: (
        (1063, 555, 1145, 605), 'letter_select_ingame',
        '[Space] 键位芯片 -> 密函选择（局内：继续轮次）'),
    Ui.LETTER_REWARD_CONFIRM: (
        (665, 720, 730, 775), 'letter_reward',
        '确认选择 ◯ 图标 -> 密函奖励'),
    Ui.ESC_MENU_SETTINGS: (
        (1129, 780, 1215, 850), 'esc_menu',
        '设置 齿轮图标 -> 局内 ESC 菜单'),
}


class COORD:
    """点击坐标（1600x900 基准）。判据负责"认界面"，这里负责"点哪里"。"""

    # ---- 开始界面 ----
    START_SCREEN_BTN = (1291, 820)

    # ---- 委托手册弹窗（点哪一格由配置决定）----
    MANUAL_NOT_USE = (543, 409)
    MANUAL_ITEM = {
        '100%': (674, 394),
        '200%': (804, 394),
        '800%': (934, 394),
        '2000%': (1064, 394),
    }
    MANUAL_CANCEL = (617, 609)            # 双按钮版 [Esc] 取消
    MANUAL_CONFIRM = (925, 609)           # 双按钮版 [Space] 开始挑战
    MANUAL_CONFIRM_NEXT = (759, 611)      # 单按钮版 [Space] 确认选择（局内继续轮次）

    # ---- 行动抉择弹窗 ----
    ACTION_CONTINUE = (1028, 622)
    ACTION_RETREAT = (515, 619)

    # ---- 任务结算 ----
    RESULT_AGAIN = (1122, 796)
    RESULT_QUIT = (1427, 796)

    # ---- 密函选择 ----
    LETTER_ABANDON = (942, 578)
    LETTER_CONFIRM = (1292, 583)              # 局外版
    LETTER_INGAME_CONFIRM = (1108, 580)       # 局内版

    # ---- 密函奖励（奖励卡内容是随机的 -> 只按固定坐标点，不做 label）----
    LETTER_REWARD_CARD_1 = (578, 466)
    LETTER_REWARD_CARD_2 = (807, 466)
    LETTER_REWARD_CARD_3 = (1037, 466)
    LETTER_REWARD_CONFIRM = (808, 755)

    # ---- ESC 菜单 ----
    ESC_CONTINUE = (1013, 817)
    ESC_SETTINGS = (1170, 814)
    ESC_CHAR_SKILL = (1326, 811)
    ESC_GIVEUP = (1479, 812)

    # ---- 月卡（暂未重做，沿用旧坐标）----
    MONTHLY_CARD_CLOSE = (800, 801)

    # ---- 重置角色/复位并传送 二次确认弹窗 ----
    #   NOTE: 本轮没截到这个弹窗，坐标是旧代码里那两处相对坐标换算来的（1600x900 基准）。
    #         待补截图后应改成"判据 label + 坐标"。
    RESET_TRANSPORT_CONFIRM = (943, 542)      # 旧 0.51,0.5866 ~ 0.66875,0.6189 的中心
    RESET_TRANSPORT_OK = (962, 506)           # 旧 0.531,0.547 ~ 0.671,0.578 的中心
