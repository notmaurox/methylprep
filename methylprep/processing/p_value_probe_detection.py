try:
    from methylcheck import mean_beta_compare
except ImportError:
    mean_beta_compare = None
import types
from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


def _pval_sesame_preprocess(data_container):
    """Performs p-value detection of low signal/noise probes. This ONE SAMPLE version uses meth/unmeth before it is contructed into a _SampleDataContainer__data_frame.
    - returns a dataframe of probes and their detected p-value levels.
    - this will be saved to the csv output, so it can be used to drop probes at later step.
    - output: index are probes (IlmnID or illumina_id); one column [poobah_pval] contains the sample p-values.
    - called by pipeline CLI --poobah option.
    - confirmed that this version produces identical results to the pre-v1.5.0 version on 2021-06-16
    """
    # 2021-03-22 assumed 'mean_value' for red and green MEANT meth and unmeth (OOBS), respectively.
    funcG = ECDF(data_container.oobG['Unmeth'].values)
    data_container.data_dict["oobG_mean"] = float(data_container.oobG['Unmeth'].values.mean())
    data_container.data_dict["oobG_std"] = float(data_container.oobG['Unmeth'].values.std())
    funcR = ECDF(data_container.oobR['Meth'].values)
    data_container.data_dict["oobR_mean"] = float(data_container.oobR['Meth'].values.mean())
    data_container.data_dict["oobR_std"] = float(data_container.oobR['Meth'].values.std())
    # sns.histplot(data=data_container.oobG['Unmeth'].values)
    # sns.histplot(data=data_container.oobR['Meth'].values)
    # plt.show()
    pIR = pd.DataFrame(
        index=data_container.IR.index,
        data=1-np.maximum(funcR(data_container.IR['Meth']), funcR(data_container.IR['Unmeth'])),
        columns=['poobah_pval'])
    pIG = pd.DataFrame(
        index=data_container.IG.index,
        data=1-np.maximum(funcG(data_container.IG['Meth']), funcG(data_container.IG['Unmeth'])),
        columns=['poobah_pval'])
    pII = pd.DataFrame(
        index=data_container.II.index,
        data=1-np.maximum(funcG(data_container.II['Meth']), funcR(data_container.II['Unmeth'])),
        columns=['poobah_pval'])
    # pval output: index is IlmnID; and threre's one column, 'poobah_pval' with p-values
    pval = pd.concat([pIR,pIG,pII])
    return pval

def _pval_minfi_preprocess(data_container):
    # # negative control p-value
    # # Pull M and U values
    # meth = pd.DataFrame(data_containers[0]._SampleDataContainer__data_frame.index)
    # unmeth = pd.DataFrame(data_containers[0]._SampleDataContainer__data_frame.index)

    # for i,c in enumerate(data_containers):
    #     sample = data_containers[i].sample
    #     m = c._SampleDataContainer__data_frame.rename(columns={'meth':sample})
    #     u = c._SampleDataContainer__data_frame.rename(columns={'unmeth':sample})
    #     meth = pd.merge(left=meth,right=m[sample],left_on='IlmnID',right_on='IlmnID',)
    #     unmeth = pd.merge(left=unmeth,right=u[sample],left_on='IlmnID',right_on='IlmnID')

    # # Create empty dataframes for red and green negative controls
    # negctlsR = pd.DataFrame(data_containers[0].ctrl_red['Extended_Type'])
    # negctlsG = pd.DataFrame(data_containers[0].ctrl_green['Extended_Type'])

    # # Fill red and green dataframes
    dfR = data_container.ctrl_red
    dfR = dfR[dfR['Control_Type']=='NEGATIVE']
    dfR = dfR[['Extended_Type','mean_value']]
    dfG = data_container.ctrl_green
    dfG = dfG[dfG['Control_Type']=='NEGATIVE']
    dfG = dfG[['Extended_Type','mean_value']]
    negctlsR = dfR
    negctlsG = dfG

    # Reset index on dataframes
    negctlsG = negctlsG.set_index('Extended_Type')
    negctlsR = negctlsR.set_index('Extended_Type')

    # # first pull out sections of manifest (will be used to identify which probes belong to each IG, IR, II)
    # manifest = data_container.manifest.data_frame[['Infinium_Design_Type','Color_Channel']]
    # IG = manifest[(manifest['Color_Channel']=='Grn') & (manifest['Infinium_Design_Type']=='I')]
    # IR = manifest[(manifest['Color_Channel']=='Red') & (manifest['Infinium_Design_Type']=='I')]
    # II = manifest[manifest['Infinium_Design_Type']=='II']

    # # Get M and U values for IG, IR and II
    # IG_meth = pd.merge(left=IG,right=meth,on='IlmnID').drop(columns=['Infinium_Design_Type','Color_Channel']).set_index('IlmnID')
    # IG_unmeth = pd.merge(left=IG,right=unmeth,on='IlmnID').drop(columns=['Infinium_Design_Type','Color_Channel']).set_index('IlmnID')
    # IR_meth = pd.merge(left=IR,right=meth,on='IlmnID').drop(columns=['Infinium_Design_Type','Color_Channel']).set_index('IlmnID')
    # IR_unmeth = pd.merge(left=IR,right=unmeth,on='IlmnID').drop(columns=['Infinium_Design_Type','Color_Channel']).set_index('IlmnID')
    # II_meth = pd.merge(left=II,right=meth,on='IlmnID').drop(columns=['Infinium_Design_Type','Color_Channel']).set_index('IlmnID')
    # II_unmeth = pd.merge(left=II,right=unmeth,on='IlmnID').drop(columns=['Infinium_Design_Type','Color_Channel']).set_index('IlmnID')

    # Calcuate parameters
    sdG = stats.median_absolute_deviation(negctlsG)
    muG = np.median(negctlsG,axis=0)
    sdR = stats.median_absolute_deviation(negctlsR)
    muR = np.median(negctlsR,axis=0)

    # calculate p values for type 1 Red
    pIR = pd.DataFrame(
        index=data_container.IR.index,
        data=1-stats.norm.cdf(
            data_container.IR['Meth']+data_container.IR['Unmeth'],
            2*muR, 2*sdR
        ),
        columns=['pNegECDF_pval']
    )
    # calculate p values for type 1 Green
    pIG = pd.DataFrame(
        index=data_container.IG.index,
        data=1-stats.norm.cdf(
            data_container.IG['Meth']+data_container.IG['Unmeth'],
            2*muG, 2*sdG
        ),
        columns=['pNegECDF_pval']
    )
    # calculat4e p values for type II
    pII = pd.DataFrame(
        index=data_container.II.index,
        data=1-stats.norm.cdf(
            data_container.II['Meth']+data_container.II['Unmeth'],
            muR+muG,sdR+sdG
        ),
        columns=['pNegECDF_pval']
    )
    # concat and sort
    pval = pd.concat([pIR, pIG, pII])
    return pval

def _pval_minfi(data_containers):
    # negative control p-value
    # Pull M and U values
    meth = pd.DataFrame(data_containers[0]._SampleDataContainer__data_frame.index)
    unmeth = pd.DataFrame(data_containers[0]._SampleDataContainer__data_frame.index)

    for i,c in enumerate(data_containers):
        sample = data_containers[i].sample
        m = c._SampleDataContainer__data_frame.rename(columns={'meth':sample})
        u = c._SampleDataContainer__data_frame.rename(columns={'unmeth':sample})
        meth = pd.merge(left=meth,right=m[sample],left_on='IlmnID',right_on='IlmnID',)
        unmeth = pd.merge(left=unmeth,right=u[sample],left_on='IlmnID',right_on='IlmnID')

    # Create empty dataframes for red and green negative controls
    negctlsR = pd.DataFrame(data_containers[0].ctrl_red['Extended_Type'])
    negctlsG = pd.DataFrame(data_containers[0].ctrl_green['Extended_Type'])

    # Fill red and green dataframes
    for i,c in enumerate(data_containers):
        sample = str(data_containers[i].sample)
        dfR = c.ctrl_red
        dfR = dfR[dfR['Control_Type']=='NEGATIVE']
        dfR = dfR[['Extended_Type','mean_value']].rename(columns={'mean_value':sample})
        dfG = c.ctrl_green
        dfG = dfG[dfG['Control_Type']=='NEGATIVE']
        dfG = dfG[['Extended_Type','mean_value']].rename(columns={'mean_value':sample})
        negctlsR = pd.merge(left=negctlsR,right=dfR,on='Extended_Type')
        negctlsG = pd.merge(left=negctlsG,right=dfG,on='Extended_Type')

    # Reset index on dataframes
    negctlsG = negctlsG.set_index('Extended_Type')
    negctlsR = negctlsR.set_index('Extended_Type')

    # Get M and U values for IG, IR and II

    # first pull out sections of manifest (will be used to identify which probes belong to each IG, IR, II)
    manifest = data_containers[0].manifest.data_frame[['Infinium_Design_Type','Color_Channel']]
    IG = manifest[(manifest['Color_Channel']=='Grn') & (manifest['Infinium_Design_Type']=='I')]
    IR = manifest[(manifest['Color_Channel']=='Red') & (manifest['Infinium_Design_Type']=='I')]
    II = manifest[manifest['Infinium_Design_Type']=='II']

    # second merge with meth and unmeth dataframes
    IG_meth = pd.merge(left=IG,right=meth,on='IlmnID').drop(columns=['Infinium_Design_Type','Color_Channel']).set_index('IlmnID')
    IG_unmeth = pd.merge(left=IG,right=unmeth,on='IlmnID').drop(columns=['Infinium_Design_Type','Color_Channel']).set_index('IlmnID')
    IR_meth = pd.merge(left=IR,right=meth,on='IlmnID').drop(columns=['Infinium_Design_Type','Color_Channel']).set_index('IlmnID')
    IR_unmeth = pd.merge(left=IR,right=unmeth,on='IlmnID').drop(columns=['Infinium_Design_Type','Color_Channel']).set_index('IlmnID')
    II_meth = pd.merge(left=II,right=meth,on='IlmnID').drop(columns=['Infinium_Design_Type','Color_Channel']).set_index('IlmnID')
    II_unmeth = pd.merge(left=II,right=unmeth,on='IlmnID').drop(columns=['Infinium_Design_Type','Color_Channel']).set_index('IlmnID')

    # Calcuate parameters
    sdG = stats.median_absolute_deviation(negctlsG)
    muG = np.median(negctlsG,axis=0)
    sdR = stats.median_absolute_deviation(negctlsR)
    muR = np.median(negctlsR,axis=0)

    # calculate p values for type 1 Red
    pIR = pd.DataFrame(index=IR_meth.index,
                   data=1 - stats.norm.cdf(IR_meth+IR_unmeth,2*muR,2*sdR),
                   columns=IR_meth.columns)

    # calculate p values for type 1 Green
    pIG = pd.DataFrame(index=IG_meth.index,
                   data=1 - stats.norm.cdf(IG_meth+IG_unmeth,2*muG,2*sdG),
                   columns=IG_meth.columns)

    # calculat4e p values for type II
    pII = pd.DataFrame(index=II_meth.index,
                  data=1-stats.norm.cdf(II_meth+II_unmeth,muR+muG,sdR+sdG),
                  columns=II_meth.columns)
    # concat and sort
    pval = pd.concat([pIR, pIG, pII])
    pval = pval.sort_values(by='IlmnID')
    return pval

def _pval_sesame(data_containers):
    # pOOHBah, called using ___ in sesame
    # Pull M and U values
    meth = pd.DataFrame(data_containers[0]._SampleDataContainer__data_frame.index)
    unmeth = pd.DataFrame(data_containers[0]._SampleDataContainer__data_frame.index)

    for i,c in enumerate(data_containers):
        sample = data_containers[i].sample
        m = c._SampleDataContainer__data_frame.rename(columns={'meth':sample})
        u = c._SampleDataContainer__data_frame.rename(columns={'unmeth':sample})
        meth = pd.merge(left=meth, right=m[sample], left_on='IlmnID', right_on='IlmnID',)
        unmeth = pd.merge(left=unmeth, right=u[sample], left_on='IlmnID', right_on='IlmnID')

    # Separate M and U values for IG, IR and II
    # first pull out sections of manifest (will be used to identify which probes belong to each IG, IR, II)
    manifest = data_containers[0].manifest.data_frame[['Infinium_Design_Type','Color_Channel']]
    IG = manifest[(manifest['Color_Channel']=='Grn') & (manifest['Infinium_Design_Type']=='I')]
    IR = manifest[(manifest['Color_Channel']=='Red') & (manifest['Infinium_Design_Type']=='I')]
    II = manifest[manifest['Infinium_Design_Type']=='II']

    # second merge with meth and unmeth dataframes
    IG_meth = pd.merge(left=IG,right=meth,on='IlmnID').drop(columns=['Infinium_Design_Type','Color_Channel']).set_index('IlmnID')
    IG_unmeth = pd.merge(left=IG,right=unmeth,on='IlmnID').drop(columns=['Infinium_Design_Type','Color_Channel']).set_index('IlmnID')
    IR_meth = pd.merge(left=IR,right=meth,on='IlmnID').drop(columns=['Infinium_Design_Type','Color_Channel']).set_index('IlmnID')
    IR_unmeth = pd.merge(left=IR,right=unmeth,on='IlmnID').drop(columns=['Infinium_Design_Type','Color_Channel']).set_index('IlmnID')
    II_meth = pd.merge(left=II,right=meth,on='IlmnID').drop(columns=['Infinium_Design_Type','Color_Channel']).set_index('IlmnID')
    II_unmeth = pd.merge(left=II,right=unmeth,on='IlmnID').drop(columns=['Infinium_Design_Type','Color_Channel']).set_index('IlmnID')

    pval = pd.DataFrame(data=manifest.index, columns=['IlmnID'])
    for i,c in enumerate(data_containers):
        # 2021-03-22 assumed 'mean_value' for red and green MEANT meth and unmeth (OOBS), respectively.
        funcG = ECDF(data_containers[i].oobG['unmeth'].values)
        funcR = ECDF(data_containers[i].oobR['meth'].values)
        sample = data_containers[i].sample
        pIR = pd.DataFrame(index=IR_meth.index,data=1-np.maximum(funcR(IR_meth[sample]), funcR(IR_unmeth[sample])),columns=[sample])
        pIG = pd.DataFrame(index=IG_meth.index,data=1-np.maximum(funcG(IG_meth[sample]), funcG(IG_unmeth[sample])),columns=[sample])
        pII = pd.DataFrame(index=II_meth.index,data=1-np.maximum(funcG(II_meth[sample]), funcR(II_unmeth[sample])),columns=[sample])
        p = pd.concat([pIR,pIG,pII]).reset_index()
        pval = pd.merge(pval,p)
    return pval
