代码入门
========
  * [Python基础](##Python基础)
  * [基础数据知识，基本的 python 包和其他工具](##基础数据知识，基本的python包和其他工具)
  * [基础ML库](##基础ML库)
  * [ML封装库](##ML封装库)
  

Python基础
----------

-   了解基本的python语法。推荐书籍：**Python编程：从入门到实践**,
    **流畅的Python**。

-   熟悉使用pip包管理，尝试使用venv、virtualenv、conda等进行环境管理
    - <https://pypi.org/>
    - <https://virtualenv.pypa.io/en/latest/>
    - <https://docs.conda.io/en/latest/> 
    - <https://www.anaconda.com/> 

基础数据知识，基本的 python 包 和 其他工具
------------------------------------------

几乎所有的库/工具都在文档里有tutorial，跟着走一遍就可以了

-   数据格式: csv, json, xml, sql

-   数据采集
    - requests: <https://requests.readthedocs.io/en/master/> 
    - scrapy: <https://scrapy.org/> 
    - bs4: <https://www.crummy.com/software/BeautifulSoup/bs4/doc/>

-   数据处理 
    - pandas: <https://pandas.pydata.org/> 
    - numpy: <https://numpy.org/>

-   数据可视化 
    - matplotlib: <https://matplotlib.org/> 
    - plotly: <https://plotly.com/>

-   数据存储 
    - sqlite: <https://docs.python.org/3/library/sqlite3.html> 
    - mysql: <https://dev.mysql.com/doc/> 
    - MongoDB: <https://www.mongodb.com/>

基础ML库
--------

-   sklearn: <https://scikit-learn.org/>。传统机器学习库。

-   tensorflow:
    <https://www.tensorflow.org/>。谷歌的文档一贯质量很高，直接看官方文档就好。tensorflow2.0
    tutorial: <https://github.com/dragen1860/TensorFlow-2.x-Tutorials>

-   Pytorch: <https://pytorch.org/>。目前主流的趋势是转向 torch。
    torch有一个推荐的教程，[d2l](https://d2l.ai/)，可以选择使用里面的torch版代码进行学习。 
    Pytorch的一个入门教程：<https://github.com/ShusenTang/Dive-into-DL-PyTorch>

ML封装库
--------

本质是对于上面提到的库的封装，但是使用起来非常高效

-   PyCaret:<https://pycaret.org/>。封装了sklearn和一些其他的传统机器学习库，并且整合了AutoML自动调参。

-   pytorch-lightning:
    <https://pytorchlightning.ai>。封装了pytorch，使得代码的可读性更好。个人感觉称为best
    practice更好。

-   keras:
    <https://keras.io/>。主要封装了tensorflow，使用起来比较简单，易于上手。

-   ML.NET:
    <https://dotnet.microsoft.com/apps/machinelearning-ai/ml-dotnet>。可以一行代码不写进炼丹，全程GUI。底层是tf，不过还不是很成熟，熟悉C#可以玩玩。