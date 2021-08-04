# 基于多负荷特征的FHMM负荷分解  
## 数据集
[AMpds](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/MXB7VO)
⚠数据集转换
* 由于python版本的差异，需要预先将AMpds的csv文件进行格式处理，并以Dataframe格式进行存储。

* csv_handle.py文件内将各用电器数据汇总成一个总数据文件，运行main函数之前需预先运行此文件。文件内input_path及out_path需视数据集所在位置进行修改。

* 转换得到的总数据文件：列一级索引为各用电器简称，如B1E,FGE等；二级索引为各电器物理量，如V,I,P等。行索引为转换后的时间戳。
