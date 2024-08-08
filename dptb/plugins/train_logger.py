from dptb.plugins.base_plugin import Plugin
from collections import defaultdict
import logging

log = logging.getLogger(__name__)

class Logger(Plugin):
    alignment = 4
    # 不同字段之间的分隔符
    separator = '-' * 81 
    
    def __init__(self, fields, interval=None):
        if interval is None:
            interval = [(1, 'iteration'), (1, 'epoch')]
        super(Logger, self).__init__(interval)

        # filed 指定打印 trainer.stats 字典中的key。比如 loss。 
        # 对于打印key1对应的字典中的key2 trainer.stats['key1']['key2'], 
        # 输入的fields wei 'key1.key2'
        self.field_widths = defaultdict(lambda: defaultdict(int))
        self.fields = list(map(lambda f: f.split('.'), fields))

    def _join_results(self, results):
        # results： list [...,('name',['value1','value2',...]),..]
        joined_out = map(lambda i: (i[0], ' '.join(i[1])), results)
        joined_fields = map(lambda i: '{}: {}'.format(i[0], i[1]), joined_out)
        return '\t'.join(joined_fields)

    def log(self, msg):
        log.info(msg)

    def register(self, trainer):
        self.trainer = trainer

    def gather_stats(self):
        result = {}
        return result

    def _align_output(self, field_idx, output):
        #当 output 中存在多个条目的时候 ，通过给out 加空格的方式对齐输出。
        for output_idx, o in enumerate(output):
            if len(o) < self.field_widths[field_idx][output_idx]:
                num_spaces = self.field_widths[field_idx][output_idx] - len(o)
                output[output_idx] += ' ' * num_spaces
            else:
                self.field_widths[field_idx][output_idx] = len(o)

    def _gather_outputs(self, field, log_fields, stat_parent, stat, require_dict=False):
        # 如果 field 有两种可能 [key1], 或者 [key1, key2] 比如：['loss'] or ['loss','last']
        output = []
        name = ''
        if isinstance(stat, dict):
            # log_fields 打印输出的格式化设置。 
            # 例如：log_epoch_fields = ['{epoch_mean' + number_format + '}' + unit]
            # 会将字典 stat 中的 key epoch_mean 对应的值输出到输出中。
            log_fields = stat.get(log_fields, [])
            name = stat.get('log_name', '.'.join(field))
            # fileds=[['accuracy','last']]。所以这里的'.'.join(fields) 。
            # 起到一个还原名称 'accuracy.last' 。
            for f in log_fields:
                output.append(f.format(**stat))
        elif not require_dict:
            # 在这里的话，如果子模块stat不是字典且require_dict=False
            # 那么他就会以父模块的打印格式和打印单位作为输出结果的方式。
            name = '.'.join(field)
            number_format = stat_parent.get('log_format', '')
            unit = stat_parent.get('log_unit', '')
            fmt = '{' + number_format + '}' + unit
            output.append(fmt.format(stat))
        
        # output ： list = [ 一个格式化输出的条目的字符串]
        # name: filed 对应的名称,
        return name, output

    def _log_all(self, log_fields, prefix=None, suffix=None, require_dict=False):
        results = []
        for field_idx, field in enumerate(self.fields):
            parent, stat = None, self.trainer.stats
            
            for f in field:
                # 递归地去获取trainer.stats[key1][key2][...]的值
                parent, stat = stat, stat[f]
                # stat 指向子字典
                name, output = self._gather_outputs(field, log_fields,
                                                parent, stat, require_dict)
                if not output:
                    continue
                self._align_output(field_idx, output)
                # results: ('name',['value']) 字符串  
                results.append((name, output))
        if not results:
            return
        output = self._join_results(results)
        loginfo = []

        if prefix is not None:
            loginfo.append(prefix)
            loginfo.append("\t")

        loginfo.append(output)
        if suffix is not None:
            loginfo.append("\t")
            loginfo.append(suffix)
        self.log("".join(loginfo))

    def iteration(self, **kwargs):

        self._log_all('log_iter_fields',prefix="iteration:{}".format(kwargs.get('time')))

    def epoch(self, **kwargs):
        self._log_all('log_epoch_fields',
                      prefix='Epoch {} summary:'.format(kwargs.get('time')),
                      suffix='\n'+ self.separator,
                      require_dict=True)
