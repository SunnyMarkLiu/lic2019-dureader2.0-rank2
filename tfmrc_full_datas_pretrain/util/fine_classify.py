import re


class FineClassify(object):
    def __init__(self):
        pattern_who = re.compile('什么人|谁')
        pattern_long = re.compile('多久|多长时间')
        pattern_how = re.compile('怎样|如何|怎么')
        pattern_when = re.compile(
            '何时|什么时候|哪[\s\S]{0,3}(秒|分钟|小时|天|星期|月|年|世纪|期)|时间$')
        pattern_num = re.compile(
            '多少|多高|多远|多重|多大|多长|几|费用|比例|通过率|额度|手续费|速度|电话|金额|价格|税费|分数线|概率|尺寸')
        pattern_where = re.compile('哪里|在哪|什么[\s\S]*?地方|到哪|去哪|从哪')
        pattern_why = re.compile(
            '为什么|为啥|什么原因|由来|原因|因素|[\s\S]*?不足|意思|含义|没反应|打不开|无法|找不到|异常|不显示|停止工作|关不掉|错误$|没声音|收不到|失败')
        pattern_what = re.compile('什么|哪[个|些|种|集|国|家]')
        pattern_rank = re.compile(
            '排名|排行|前[二|三|四|五|六|七|八|九|十]|[二|三|四|五|六|七|八|九|十]大')
        pattern_diff = re.compile('区别|不同|差异|差别')
        pattern_solution = re.compile('方法|办法|做法|算法|流程|技巧|攻略|规则|操作|规律|原理')
        pattern_result = re.compile('作用|影响|功效|疗效|后果|功能')
        self.pattern_list = [pattern_who, pattern_long, pattern_how, pattern_when, pattern_num, pattern_where,
                             pattern_why, pattern_what, pattern_rank, pattern_diff, pattern_solution, pattern_result]
        self.pattern_names = ['who', 'long', 'how', 'when', 'num',
                              'where', 'why', 'what', 'rank', 'diff', 'solution', 'result']
        assert len(self.pattern_list) == len(self.pattern_names)
        self.label_distrib = [0] * (len(self.pattern_list) + 1)

    def get_classify_label(self, text):
        """
        返回text对应的label_id以及label_name
        """
        for pid, (pattern, name) in enumerate(zip(self.pattern_list, self.pattern_names)):
            if re.findall(pattern, text):
                self.label_distrib[pid] += 1
                return pid, name
        # 未匹配
        self.label_distrib[-1] += 1
        return len(self.pattern_list), 'unk'

    def show_distrib(self):
        """
        打印各个标签的分布情况
        """
        all_cnt = sum(self.label_distrib)
        if all_cnt == 0:
            print('all fine classify label is zero!')
            return
        for name, cnt in zip(self.pattern_names + ['unk'], self.label_distrib):
            print('{}: {} -> {}'.format(name, cnt, cnt / all_cnt))
