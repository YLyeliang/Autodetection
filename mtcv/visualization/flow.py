from graphviz import Digraph
from dropblock import DropBlock2D

def flow_chart():
    dot = Digraph(comment='The test Table', format='png')
    dot.node('a', '原始实例')
    dot.node('b', 'TB抠图')
    dot.node('c', 'TB图像增强')
    dot.node('d', '训练样本生成')
    dot.node('e', '模型训练')
    dot.node('f', '实际测试')
    dot.edges(['ab', 'bc', 'cd', 'de', 'ef'])
    dot.view()


# flow_chart()

def framework_chart():
    dot = Digraph(comment='tbdetection_framework', format='png')
    dot.node('a', '检测框架')
    dot.node('b', '数据部分')
    dot.node('c', '检测部分')
    dot.node('d', '特效库')
    dot.node('e', '融合方法')
    dot.node('f', '批量生成api')
    dot.node('g', '训练格式转换')
    dot.node('h', 'backbone')
    dot.node('i', 'neck')
    dot.node('j', 'head')
    dot.node('k', 'detector')
    dot.edges(['ab', 'ac', 'bd', 'be', 'bf', 'bg', 'ch', 'ci', 'cj', 'ck'])
    dot.view()


framework_chart()
