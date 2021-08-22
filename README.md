# 基于多负荷特征的FHMM负荷分解  
## 数据集
[AMpds](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/MXB7VO)
## ⚠数据集转换
* 由于python版本的差异，需要预先将AMpds的csv文件进行格式处理，并以Dataframe格式进行存储。

* csv_handle.py文件内将各用电器数据汇总成一个总数据文件，运行main函数之前需预先运行此文件。文件内input_path及out_path需视数据集所在位置进行修改。

* 转换得到的总数据文件：列一级索引为各用电器简称，如B1E,FGE等；二级索引为各电器物理量，如V,I,P等。行索引为转换后的时间戳。
## F1指标修改
[公式来源](https://www.sciencedirect.com/science/article/abs/pii/S0306261917312369)  
进行对比实验时，在NIMLTK工具包loss.py文件添加如下代码段：
<pre><code>def energy_f1score(app_gt, app_pred):  
    gt_temp = np.array(app_gt).reshape(-1,1)  
    pred_temp = np.array(app_pred).reshape(-1,1)  
    temp=np.hstack((gt_temp,pred_temp))  
    molecule=temp.min(axis=1).sum()  
    denominator= pred_temp.sum()  
    P=molecule/denominator  
    denominator= gt_temp.sum()  
    R=molecule/denominator  
    return (2*(P*R)/(P+R))
</code></pre>
## 后续
后续将对基于多负荷特征的FHMM负荷分解算法进行函数形式上的修改，使其能够被NILMTK工具包调用
