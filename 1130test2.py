import numpy as np
import pandas as pd
from pyecharts.charts import Bar

# 数据x轴，y轴
x = list(range(1, 8))
y = [145, 176, 150, 186, 179, 142, 165]

# 调用柱状图对象
bar = Bar()
# 添加x轴，y轴
bar.add_xaxis(x)
bar.add_yaxis("身高", y)

# 显示自动生成html,默认保存在安装路径
# bar.render()

# 保存到指定的位置
bar.render('height.html')

# 在notebook中显示图形
bar.render_notebook()