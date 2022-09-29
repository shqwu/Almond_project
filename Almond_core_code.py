
import math 
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import netCDF4 as nc
from numpy import genfromtxt
import scipy.stats as st
from numpy import savetxt
import matplotlib.patches as mpatches
import matplotlib
import seaborn as sns
from scipy import signal
from scipy import stats
from matplotlib.lines import Line2D
import matplotlib as mpl
##multiply coef and aci
aci_num = 13
for i in range(1,11):
    locals()['coef'+str(i)] = np.zeros((100,aci_num*2+32))
coef_sum = np.zeros((0,aci_num*2+32))
for i in range(1,1001):
    locals()['coef'+str(((i-1)//100)+1)][i%100-1] = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/data_6_19/coef_'+str(i)+'.csv', delimiter = ',')
for i in range(1,11):
    coef_sum = np.row_stack((coef_sum, locals()['coef'+str(i)]))
#coef_sum = np.zeros((82))
#for i in range(1,11):
#   locals()['coef'+str(i)] = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/data_5_11/coef_'+str(i)+'.csv', delimiter = ',')

ACI_list = ['MarMay_$T_{ave}$', 'JuneAug_$T_{ave}$', 'SepNov_$T_{ave}$', 'NovJan_$T_{ave}$', 'Feb_$T_{min}$', 'Mar$T_{min}$', 'MarJuly_GDD4', 'MarJuly_KDD25', 'MarJuly_KDD30', 'NovJan_Freeze', 'NovJan_Chill', 'Jan_P', 'Feb_P'
            , 'MarMay_P', 'JuneAug_P', 'SepNov_P', 'NovJan_P', 'WYppt', 'MarMay_ETo', 'JuneAug_ETo', 'SepNov_ETo', 'NovJan_ETo', 'AprOct_ETo', 'JanFeb_T4.4', 'JanFeb_T12.8', 'JanFeb_T10_21', 'JanFeb_T21.1-30.6'
            , 'JanFeb_SpH', 'JanFeb_Wind']
ACI_list = ['NovJan_Freeze', 'NovJan_ETo', 'NovJan_P', 'NovJan_$T_{mean}$', 'Jan_P', 'Feb_P', 'Feb_$T_{min}$', 'FebMar_T12.8', 'FebMar_ETo', 'FebMar_GDD4', 'FebMar_KDD30', 'FebMar_P', 'FebMar_$T_{mean}$','Mar_$T_{min}$', 'MarJul_ETo', 'MarJul_GDD4', 'MarJul_KDD30', 'MarJul_P', 'MarJul_$T_{mean}$', 'AugOct_ETo', 'AugOct_P', 'AugOct_$T_{mean}$'
            , 'FebMar_SpH', 'FebMar_Windpeed']

ACI_list = ['DormancyFreeze', 'DormancyTmean','DormancyETo', 'JanPpt', 'BloomPpt', 'BloomETo', 'BloomCDD15', 'BloomKDD30','SpH','windspeed','MarTmin', 'GrowingETo', 'GrowingGDD4', 'GrowingKDD30','harvest_Ppt']
ACI_list = ['DormancyChill', 'DormancyFreeze', 'DormancyETo', 'DormancyPpt', 'DormancyTmean', 'JanPpt', 'FebPpt', 'FebTmin', 'CDD15','BloomETo', 'BloomGDD4', 'BloomKDD30', 'BloomPpt', 'BloomTmean','MarTmin', 'GrowingETo', 'GrowingGDD4', 'GrowingKDD30', 'GrowingPpt', 'GrowingTmean', 'harvest_ETo', 'harvest_Ppt', 'harvest_Tmean'
            , 'SpH', 'WndSpd']    
ACI_list= ['Dormancy_Freeze', 'Dormancy_$T_{avg}$','Dormancy_ETo', 'Jan_P','Feb_$T_{min}$','Bloom_P', 'Bloom_ETo', 'Bloom_GDD4','Specific Humidity','Wind Speed','Mar_$T_{min}$', 'Growing_ETo', 'Growing_GDD4', 'Growing_KDD30','harvest_P']
ACI_list= ['Dormancy_Freeze', 'Dormancy_$T_{avg}$','Dormancy_ETo', 'Jan_P','Bloom_P', 'Bloom_ETo', 'Bloom_GDD4','Bloom_$T_{min}$','Specific Humidity','Wind Speed', 'Growing_ETo', 'Growing_GDD4', 'Growing_KDD30','harvest_P']

ACI_list= ['Dormancy_Freeze', 'Dormancy_$T_{avg}$','Dormancy_ETo', 'Jan_P','Feb_$T_{min}$','Bloom_P', 'Bloom_ETo', 'Bloom_GDD4','Specific Humidity','Wind Speed','Mar_$T_{min}$', 'Growing_ETo', 'Growing_T_${mean}$', 'Growing_KDD30','harvest_P']
ACI_list= ['Dormancy_Freeze', 'Dormancy_$T_{avg}$','Dormancy_ETo', 'Jan_P','Bloom_P', 'Bloom_ETo', 'Bloom_GDD4','Bloom_$T_{min}$','Specific Humidity','Wind Speed', 'Growing_ETo', 'Growing_T_${mean}$', 'Growing_KDD30','harvest_P']
ACI_list = ['Dormancy_Freeze','Dormancy_ETo','Jan_Ppt','Bloom_Ppt','Bloom_Tmin' ,'Bloom_ETo', 'Bloom_GDD4','Specific_Humidity','Windy_days','Growing_ETo', 'Growing_KDD30','harvest_Ppt']
ACI_list = ['Dormancy_Freeze','Dormancy_ETo','Jan_Ppt','Bloom_Ppt','Bloom_Tmin' ,'Bloom_ETo', 'Bloom_GDD4','Bloom_Humidity','Windy_days','Growing_ETo','GrowingGDD4', 'Growing_KDD30','harvest_Ppt']

county_list = ['Butte', 'Colusa', 'Fresno', 'Glenn', 'Kern', 'Kings', 'Madera', 'Merced', 'San Joaquin', 'Solano', 'Stanislaus', 'Sutter', 'Tehama', 'Tulare', 'Yolo', 'Yuba']                      

model_list = ['bcc-csm1-1-m', 'bcc-csm1-1','BNU-ESM', 'CanESM2', 'CSIRO-Mk3-6-0', 'GFDL-ESM2G', 'GFDL-ESM2M', 'inmcm4', 'IPSL-CM5A-LR', 'IPSL-CM5A-MR','CNRM-CM5', 'HadGEM2-CC365','HadGEM2-ES365', 'IPSL-CM5B-LR', 'MIROC5', 'MIROC-ESM', 'MIROC-ESM-CHEM']

## obtain production from excel file (* 'yield' should be 'production' in the below codes)
csv_year_name = ['198008','198108','198208', '198308', '198408', '198508', '198608', '198708', '198808', '198908', '199008',
            '199108', '199208', '199308', '199408', '199508', '199608', '199708', '199808', '199908', '200008', '200108', '200208',
            '200308', '200410', '200508', '200608', '200708', '200810', '200910', '201010','201112', '201212', '201708']
csv_year = ['1980','1981','1982','1983','1984', '1985', '1986', '1987', '1988', '1989', '1990', '1991', '1992', '1993',  '1994', '1995', '1996', '1997', '1998', '1999','2000','2001', '2002', '2003', '2004','2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012']
csv_year1_name = ['2013', '2014', '2015', '2016','2017','2018', '2019', '2020']



county_list = ['Butte', 'Colusa', 'Fresno', 'Glenn', 'Kern', 'Kings', 'Madera', 'Merced', 'SanJ', 'Solano', 'Stanislaus', 'Sutter', 'Tehama', 'Tulare', 'Yolo', 'Yuba']                      
county_code = [7,11,19,21,29,31,39,47,77,95,99,101,103,107,113,115]
yield_csv = np.zeros((41,17))
for i in range(0,33):
    year_per = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/california-county/'+str(csv_year_name[i])+'cactb00.csv', delimiter = ',')
    yield_csv[i,0] = int(csv_year[i])
    for j in range(0,16):
        county_id = county_code[j]
        k = 0
        while year_per[np.array(np.where(year_per[:,3] == county_id))[0,k], 1] != 261999:
            k = k+1
        yield_csv[i,j+1] = year_per[np.array(np.where(year_per[:,3] == county_id))[0,k], 6]

for i in range(0,8):
    year_per = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/california-county/'+str(csv_year1_name[i])+'cropyear.csv', delimiter = ',')
    yield_csv[i+33,0] = int(csv_year1_name[i])
    for j in range(0,16):
        county_id = county_code[j]
        k = 0
        while year_per[np.array(np.where(year_per[:,3] == county_id))[0,k], 1] != 261999:
            k = k+1
        yield_csv[i+33,j+1] = year_per[np.array(np.where(year_per[:,3] == county_id))[0,k], 6]


area_csv = np.zeros((41,17))
for i in range(0,33):
    year_per = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/california-county/'+str(csv_year_name[i])+'cactb00.csv', delimiter = ',')
    area_csv[i,0] = int(csv_year[i])
    for j in range(0,16):
        county_id = county_code[j]
        k = 0
        while year_per[np.array(np.where(year_per[:,3] == county_id))[0,k], 1] != 261999:
            k = k+1
        area_csv[i,j+1] = year_per[np.array(np.where(year_per[:,3] == county_id))[0,k], 5]

for i in range(0,8):
    year_per = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/california-county/'+str(csv_year1_name[i])+'cropyear.csv', delimiter = ',')
    area_csv[i+33,0] = int(csv_year1_name[i])
    for j in range(0,16):
        county_id = county_code[j]
        k = 0
        while year_per[np.array(np.where(year_per[:,3] == county_id))[0,k], 1] != 261999:
            k = k+1
        area_csv[i+33,j+1] = year_per[np.array(np.where(year_per[:,3] == county_id))[0,k], 5]

production_csv = np.zeros((41,17))
for i in range(0,33):
    year_per = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/california-county/'+str(csv_year_name[i])+'cactb00.csv', delimiter = ',')
    production_csv[i,0] = int(csv_year[i])
    for j in range(0,16):
        county_id = county_code[j]
        k = 0
        while year_per[np.array(np.where(year_per[:,3] == county_id))[0,k], 1] != 261999:
            k = k+1
        production_csv[i,j+1] = year_per[np.array(np.where(year_per[:,3] == county_id))[0,k], 7]

for i in range(0,8):
    year_per = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/california-county/'+str(csv_year1_name[i])+'cropyear.csv', delimiter = ',')
    production_csv[i+33,0] = int(csv_year1_name[i])
    for j in range(0,16):
        county_id = county_code[j]
        k = 0
        while year_per[np.array(np.where(year_per[:,3] == county_id))[0,k], 1] != 261999:
            k = k+1
        production_csv[i+33,j+1] = year_per[np.array(np.where(year_per[:,3] == county_id))[0,k], 7]

##
area = area_csv[0:41,1:]
gridmet = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/gugu_aci_csv_6_19.csv', delimiter = ',')
simulation_gridmet = np.zeros((656, 1000))
production_gridmet = np.zeros((656, 1000))
for trial in range(1,11):
    for i in range(0,656):
        for j in range(0,100):
            simulation_gridmet[i,j+((trial-1)*100)] = np.nansum(gridmet[i,:]*locals()['coef'+str(trial)][j,:])
for index in range(0,16):
    for year in range(0,41):
        production_gridmet[index*41+year,:] = simulation_gridmet[index*41+year,:]*area[year,index%16]
production_gridmet_split = np.split(production_gridmet,16) 
production_gridmet_all = np.zeros((41,1000)) 
for county_id in range(0,16):
    for year in range(0,41):
        production_gridmet_all[year,:] = production_gridmet_all[year,:]+production_gridmet_split[county_id][year,:]
yield_gridmet_state = np.zeros((41,1000))
for year in range(0,41):
    yield_gridmet_state[year,:] = production_gridmet_all[year,:]/np.sum(area[year])

production = production_csv[0:41,1:]
production_observed_all = np.sum(production, axis = 1)
yield_observed_state = np.zeros((41))
for year in range(0,41):
    yield_observed_state[year] = production_observed_all[year]/np.sum(area[year])

median_yield_1980_rcp45 = np.median(yield_all_hist_rcp45[0])
median_yield_1980_rcp85 = np.median(yield_all_hist_rcp85[0])
tech_trend = np.zeros((1000))
for county_id in range(0,16):
    tech_trend = tech_trend+coef_sum[:,aci_num*2+county_id]*area[-1,county_id]/np.sum(area[-1])
tech_trend_sum = np.zeros((120,1000))
for i in range(1,121):
    tech_trend_sum[i-1,:] = tech_trend*i
future_tech_trend_rcp45 = tech_trend_sum + median_yield_1980_rcp45
future_tech_trend_rcp85 = tech_trend_sum + median_yield_1980_rcp85

tech_trend_county = np.zeros((16,120))
for county_id in range(0,16):
    tech_trend_county[county_id, :] = np.median(coef_sum[:,county_id + aci_num*2], axis = 0)
for year in range(1,121):
    tech_trend_county[:, year-1] = tech_trend_county[:, year-1] * year
future_tech_trend_county_rcp45 = np.zeros((120,16))
future_tech_trend_county_rcp85 = np.zeros((120,16))
for year in range(0,120):
    future_tech_trend_county_rcp45[year,:] = tech_trend_county[:,year] + yield_1980_rcp45
    future_tech_trend_county_rcp85[year,:] = tech_trend_county[:,year] + yield_1980_rcp85

##plot gridmet aci vs aci contribution
#R2_total_sum = np.zeros((1))
#for i in range(1,1001):
#    locals()['score'+str(i)] = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/data_5_11/score_'+str(i)+'.csv', delimiter = ',')
#    R2_total_sum = np.column_stack((R2_total_sum,locals()['score'+str(i)] ))
#R2_total_sum = R2_total_sum[:,1:]
#median_R2_index = np.where(R2_total_sum == np.nanpercentile(R2_total_sum, 50,interpolation='higher'))[1]
#gridmet_aci_contribution = np.zeros((656,aci_num))
#for i in range(0,656):
 #   gridmet_aci_contribution[i] = (gridmet[i,:]*coef_sum[median_R2_index,:])[0,0:aci_num] +  (gridmet[i,:]*coef_sum[median_R2_index,:])[0,aci_num:aci_num*2]

gridmet_aci_contribution = np.zeros((656,aci_num,1000))
for i in range(0,656):
    for trial in range(0,1000):
        gridmet_aci_contribution[i,:,trial] = (gridmet[i,:]*coef_sum[trial,:])[0:aci_num] +  (gridmet[i,:]*coef_sum[trial,:])[aci_num:aci_num*2]
gridmet_aci_contribution = np.mean(gridmet_aci_contribution, axis=2)
plt.figure(figsize = (40,210))
for i in range(0,aci_num):
    plt.subplot((aci_num+1)//2,2,i+1)
    plt.scatter(x = gridmet[:,i], y = gridmet_aci_contribution[:,i], color = 'k')
    plt.title(str(ACI_list[i]), fontsize = 30)
    plt.xticks(fontsize = 30)
    plt.yticks(fontsize = 30)
plt.savefig('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/Almond_plots/plot_hnrg/almond-land-avg/Growth_stage_ACI_6_19/scatter_gridmet_aci_acicontribution.pdf', dpi = 300)

gridmet_aci_contribution_ca = np.zeros((41,14))
for index in range(0,16):
    for year in range(0,41):
        gridmet_aci_contribution_ca[year,:] = gridmet_aci_contribution_ca[year,:] + (np.split(gridmet_aci_contribution,16)[index])[year]*area[year,index]/np.sum(area[year])

plt.figure(figsize = (40,160))
for year in range(0,41):
    plt.subplot(41,1,year+1)
    plt.bar(x = ACI_list, height = gridmet_aci_contribution_ca[year,:], color = bar_color)
    plt.xticks(rotation = 90, fontsize = 30)
    plt.yticks(fontsize = 30)
    plt.ylim(-0.1,0.1)
    if year <40:
        plt.xticks('')
plt.savefig('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/Almond_plots/plot_hnrg/almond-land-avg/Growth_stage_ACI_6_19/bar_gridmet_aci_acicontribution_year.pdf', dpi = 300)

plt.figure(figsize = (50,100))
for i in range(0,14):
    plt.subplot(14,1,i+1)
    plt.bar(x = np.arange(1980,2021), height = gridmet_aci_contribution_ca[:,i], color = bar_color[i])
    plt.xticks(fontsize = 30)
    plt.yticks(fontsize = 30)
    plt.title(str(ACI_list[i]), fontsize = 35)
    plt.axline((1980,0),(2020,0), color = 'r', linestyle = 'dashed')
plt.savefig('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/Almond_plots/plot_hnrg/almond-land-avg/Growth_stage_ACI_6_19/bar_gridmet_aci_acicontribution_aci.pdf', dpi = 300)



##hist for new model
for trial in range(1,11):
    for model in range(0,17):
        simulation_rcp45 = np.zeros((656,100))
        simulation_rcp45_s = np.zeros((656,100))
        simulation_rcp85 = np.zeros((656,100))
        simulation_rcp85_s = np.zeros((656,100))
        aci_rcp45 = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_6_19/to_2020/'+str(model_list[model])+'hist_rcp45_ACI.csv', delimiter = ',')
        aci_rcp45_s = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_6_19/to_2020/'+str(model_list[model])+'hist_rcp45_s_ACI.csv', delimiter = ',')
        aci_rcp85 = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_6_19/to_2020/'+str(model_list[model])+'hist_rcp85_ACI.csv', delimiter = ',')
        aci_rcp85_s = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_6_19/to_2020/'+str(model_list[model])+'hist_rcp85_s_ACI.csv', delimiter = ',')
        for i in range(0,656):
            for j in range(0,100):
                simulation_rcp45[i,j] = np.nansum(aci_rcp45[i,:]*locals()['coef'+str(trial)][j,:])
                simulation_rcp45_s[i,j] = np.nansum(aci_rcp45_s[i,:]*locals()['coef'+str(trial)][j,:])
                simulation_rcp85[i,j] = np.nansum(aci_rcp85[i,:]*locals()['coef'+str(trial)][j,:])
                simulation_rcp85_s[i,j] = np.nansum(aci_rcp85_s[i,:]*locals()['coef'+str(trial)][j,:])
        savetxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_6_19/to_2020/'+str(model_list[model])+'_hist_prediction_'+str(trial)+'_rcp45.csv', simulation_rcp45, delimiter = ',')
        savetxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_6_19/to_2020/'+str(model_list[model])+'_hist_prediction_'+str(trial)+'_rcp45_s.csv', simulation_rcp45_s, delimiter = ',')
        savetxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_6_19/to_2020/'+str(model_list[model])+'_hist_prediction_'+str(trial)+'_rcp85.csv', simulation_rcp85, delimiter = ',')
        savetxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_6_19/to_2020/'+str(model_list[model])+'_hist_prediction_'+str(trial)+'_rcp85_s.csv', simulation_rcp85_s, delimiter = ',')

##future
for trial in range(1,11):
    for model in range(0,17):
        simulation_rcp45 = np.zeros((1264,100))
        simulation_rcp45_s = np.zeros((1264,100))
        simulation_rcp85 = np.zeros((1264,100))
        simulation_rcp85_s = np.zeros((1264,100))
        aci_rcp45 = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_6_19/to_2020/'+str(model_list[model])+'future_rcp45_ACI.csv', delimiter = ',')
        aci_rcp45_s = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_6_19/to_2020/'+str(model_list[model])+'future_rcp45_s_ACI.csv', delimiter = ',')
        aci_rcp85 = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_6_19/to_2020/'+str(model_list[model])+'future_rcp85_ACI.csv', delimiter = ',')
        aci_rcp85_s = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_6_19/to_2020/'+str(model_list[model])+'future_rcp85_s_ACI.csv', delimiter = ',')
        for i in range(0,1264):
            for j in range(0,100):
                simulation_rcp45[i,j] = np.nansum(aci_rcp45[i,:]*locals()['coef'+str(trial)][j,:])
                simulation_rcp45_s[i,j] = np.nansum(aci_rcp45_s[i,:]*locals()['coef'+str(trial)][j,:])
                simulation_rcp85[i,j] = np.nansum(aci_rcp85[i,:]*locals()['coef'+str(trial)][j,:])
                simulation_rcp85_s[i,j] = np.nansum(aci_rcp85_s[i,:]*locals()['coef'+str(trial)][j,:])
        savetxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_6_19/to_2020/'+str(model_list[model])+'_future_prediction_'+str(trial)+'_rcp45.csv', simulation_rcp45, delimiter = ',')
        savetxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_6_19/to_2020/'+str(model_list[model])+'_future_prediction_'+str(trial)+'_rcp45_s.csv', simulation_rcp45_s, delimiter = ',')
        savetxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_6_19/to_2020/'+str(model_list[model])+'_future_prediction_'+str(trial)+'_rcp85.csv', simulation_rcp85, delimiter = ',')
        savetxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_6_19/to_2020/'+str(model_list[model])+'_future_prediction_'+str(trial)+'_rcp85_s.csv', simulation_rcp85_s, delimiter = ',')

## obtain aci*coef area-weighted CA
for model in range(0,17):
    locals()['aci_contribution_model_rcp45_'+str(model)] = np.zeros((624,1000,30))
    locals()['aci_contribution_model_rcp85_'+str(model)] = np.zeros((624,1000,30))
    for trial in range(1,11):
        simulation_rcp45 = np.zeros((624,101))
        simulation_rcp85 = np.zeros((624,101))
        aci_rcp45 = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_6_19/'+str(model_list[model])+'hist_rcp45_ACI.csv', delimiter = ',')[:,0:30]
        aci_rcp85 = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_6_19/'+str(model_list[model])+'hist_rcp85_ACI.csv', delimiter = ',')[:,0:30]
        for i in range(0,624):
            for j in range(0,101):
                locals()['aci_contribution_model_rcp45_'+str(model)][i,(trial-1)*101+j,:] = aci_rcp45[i,:]*locals()['coef'+str(trial)][j,0:30]
                locals()['aci_contribution_model_rcp85_'+str(model)][i,(trial-1)*101+j,:] = aci_rcp85[i,:]*locals()['coef'+str(trial)][j,0:30]
area = area_csv[0:39,1:]
for model in range(0,17):
    for index in range(0,16):
        for year in range(0,39):
            locals()['aci_contribution_model_rcp45_'+str(model)][index*39+year,:,:] = locals()['aci_contribution_model_rcp45_'+str(model)][index*39+year,:,:]*area[year,index%16]/np.sum(area[year])             
            locals()['aci_contribution_model_rcp85_'+str(model)][index*39+year,:,:] = locals()['aci_contribution_model_rcp85_'+str(model)][index*39+year,:,:]*area[year,index%16]/np.sum(area[year])
for model in range(0,17):
    locals()['aci_contribution_model_rcp45_sum'+str(model)] = np.zeros((39,1000,30))
    locals()['aci_contribution_model_rcp85_sum'+str(model)] = np.zeros((39,1000,30))
    for county_id in range(0,16):
        for year in range(0,39):
            locals()['aci_contribution_model_rcp45_sum'+str(model)][year,:,:] = locals()['aci_contribution_model_rcp45_sum'+str(model)][year,:,:] + np.split(locals()['aci_contribution_model_rcp45_'+str(model)],16,axis=0)[county_id][year,:,:]      
            locals()['aci_contribution_model_rcp85_sum'+str(model)][year,:,:] = locals()['aci_contribution_model_rcp85_sum'+str(model)][year,:,:] + np.split(locals()['aci_contribution_model_rcp85_'+str(model)],16,axis=0)[county_id][year,:,:]      
aci_contribution_model_rcp45 = np.zeros((17,39,1000,30))
aci_contribution_model_rcp85 = np.zeros((17,39,1000,30))
for model in range(0,17):
    aci_contribution_model_rcp45[model, :,:,:] = locals()['aci_contribution_model_rcp45_sum'+str(model)]
    aci_contribution_model_rcp85[model, :,:,:] = locals()['aci_contribution_model_rcp85_sum'+str(model)]
aci_contribution_model_rcp45 = np.mean(aci_contribution_model_rcp45, axis=0)
aci_contribution_model_rcp85 = np.mean(aci_contribution_model_rcp85, axis=0)

for model in range(0,17):
    locals()['aci_contribution_model_rcp45_future_'+str(model)] = np.zeros((1296,1000,30))
    locals()['aci_contribution_model_rcp85_future_'+str(model)] = np.zeros((1296,1000,30))
    for trial in range(1,11):
        simulation_rcp45 = np.zeros((1296,101))
        simulation_rcp85 = np.zeros((1296,101))
        aci_rcp45 = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_6_19/'+str(model_list[model])+'future_rcp45_ACI.csv', delimiter = ',')[:,0:30]
        aci_rcp85 = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_6_19/'+str(model_list[model])+'future_rcp85_ACI.csv', delimiter = ',')[:,0:30]
        for i in range(0,1296):
            for j in range(0,101):
                locals()['aci_contribution_model_rcp45_future_'+str(model)][i,(trial-1)*101+j,:] = aci_rcp45[i,:]*locals()['coef'+str(trial)][j,0:30]
                locals()['aci_contribution_model_rcp85_future_'+str(model)][i,(trial-1)*101+j,:] = aci_rcp85[i,:]*locals()['coef'+str(trial)][j,0:30]
for model in range(0,17):
    for index in range(0,16):
        locals()['aci_contribution_model_rcp45_future_'+str(model)][index*81:(index+1)*81,:,:] = locals()['aci_contribution_model_rcp45_future_'+str(model)][index*81:(index+1)*81,:,:]*area[-1,index]/np.sum(area[-1])             
        locals()['aci_contribution_model_rcp85_future_'+str(model)][index*81:(index+1)*81,:,:] = locals()['aci_contribution_model_rcp85_future_'+str(model)][index*81:(index+1)*81,:,:]*area[-1,index]/np.sum(area[-1])
for model in range(0,17):
    locals()['aci_contribution_model_rcp45_sum_future_'+str(model)] = np.zeros((81,1000,30))
    locals()['aci_contribution_model_rcp85_sum_future_'+str(model)] = np.zeros((81,1000,30))
    for county_id in range(0,16):
        for year in range(0,81):
            locals()['aci_contribution_model_rcp45_sum_future_'+str(model)][year,:,:] = locals()['aci_contribution_model_rcp45_sum_future_'+str(model)][year,:,:] + np.split(locals()['aci_contribution_model_rcp45_future_'+str(model)],16,axis=0)[county_id][year,:,:]      
            locals()['aci_contribution_model_rcp85_sum_future_'+str(model)][year,:,:] = locals()['aci_contribution_model_rcp85_sum_future_'+str(model)][year,:,:] + np.split(locals()['aci_contribution_model_rcp85_future_'+str(model)],16,axis=0)[county_id][year,:,:]      
aci_contribution_model_rcp45_future = np.zeros((17,81,1000,30))
aci_contribution_model_rcp85_future = np.zeros((17,81,1000,30))
for model in range(0,17):
    aci_contribution_model_rcp45_future[model, :,:,:] = locals()['aci_contribution_model_rcp45_sum_future_'+str(model)]
    aci_contribution_model_rcp85_future[model, :,:,:] = locals()['aci_contribution_model_rcp85_sum_future_'+str(model)]
aci_contribution_model_rcp45_future = np.mean(aci_contribution_model_rcp45_future, axis=0)
aci_contribution_model_rcp85_future = np.mean(aci_contribution_model_rcp85_future, axis=0)

aci_contribution_rcp45_total = np.row_stack((aci_contribution_model_rcp45, aci_contribution_model_rcp45_future))
aci_contribution_rcp85_total = np.row_stack((aci_contribution_model_rcp85, aci_contribution_model_rcp85_future))

for i in range(0,15):
    aci_contribution_rcp45_total[:,:,i] = aci_contribution_rcp45_total[:,:,i]+aci_contribution_rcp45_total[:,:,i+15]
    aci_contribution_rcp85_total[:,:,i] = aci_contribution_rcp85_total[:,:,i]+aci_contribution_rcp85_total[:,:,i+15]

aci_contribution_rcp45_total = aci_contribution_rcp45_total[:,:,0:15]
aci_contribution_rcp85_total = aci_contribution_rcp85_total[:,:,0:15]

aci_contribution_rcp45_total = np.load('C:/Users/Pancake/Box/aci_contribution_5_20/aci_contribution_rcp45_total.npy')
aci_contribution_rcp85_total = np.load('C:/Users/Pancake/Box/aci_contribution_5_20/aci_contribution_rcp85_total.npy')

aci_contribution_rcp45_2001_2020 = np.mean(aci_contribution_rcp45_total[21:41,:,:], axis=0)
aci_contribution_rcp85_2001_2020 = np.mean(aci_contribution_rcp85_total[21:41,:,:], axis=0)

aci_contribution_rcp45_2041_2060 = np.mean(aci_contribution_rcp45_total[61:81,:,:], axis=0)
aci_contribution_rcp85_2041_2060 = np.mean(aci_contribution_rcp85_total[61:81,:,:], axis=0)

aci_contribution_rcp45_2080_2099 = np.mean(aci_contribution_rcp45_total[100:120,:,:], axis=0)
aci_contribution_rcp85_2080_2099 = np.mean(aci_contribution_rcp85_total[100:120,:,:], axis=0)

plt.figure(figsize = (20,30))
plt.subplot(2,1,1)
plt.bar(x = ACI_list, height = np.median(aci_contribution_rcp45_2080_2099, axis=0))
plt.title('RCP4.5', fontsize = 30)
plt.yticks(fontsize = 30)
plt.ylim(-0.8,0.6)
plt.xticks('')
plt.subplot(2,1,2)
plt.bar(x = ACI_list, height = np.median(aci_contribution_rcp85_2080_2099, axis=0))
plt.xticks(fontsize = 30, rotation = 90)
plt.title('RCP8.5', fontsize = 30)
plt.yticks(fontsize= 30)
plt.ylim(-0.8,0.6)
plt.savefig('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/Almond_plots/plot_hnrg/almond-land-avg/Growth_stage_ACI_5_15/aci_contribution_value.png', dpi = 200)
##calculate contribution change and plot
aci_contribution_rcp45_change_percent_2041_2060 = np.zeros((1000,aci_num))
aci_contribution_rcp85_change_percent_2041_2060 = np.zeros((1000,aci_num))
aci_contribution_rcp45_change_percent_2080_2099 = np.zeros((1000,aci_num))
aci_contribution_rcp85_change_percent_2080_2099 = np.zeros((1000,aci_num))

aci_contribution_rcp45_change_total_2041_2060 = np.zeros((1000))
aci_contribution_rcp85_change_total_2041_2060 = np.zeros((1000))
aci_contribution_rcp45_change_total_2080_2099 = np.zeros((1000))
aci_contribution_rcp85_change_total_2080_2099 = np.zeros((1000))
for i in range(0,1000):
    aci_contribution_rcp45_change_total_2041_2060[i] = np.sum(aci_contribution_rcp45_2041_2060[i,:])-np.sum(aci_contribution_rcp45_2001_2020[i,:])
    aci_contribution_rcp85_change_total_2041_2060[i] = np.sum(aci_contribution_rcp85_2041_2060[i,:])-np.sum(aci_contribution_rcp85_2001_2020[i,:])
    aci_contribution_rcp45_change_total_2080_2099[i] = np.sum(aci_contribution_rcp45_2080_2099[i,:])-np.sum(aci_contribution_rcp45_2001_2020[i,:])
    aci_contribution_rcp85_change_total_2080_2099[i] = np.sum(aci_contribution_rcp85_2080_2099[i,:])-np.sum(aci_contribution_rcp85_2001_2020[i,:])
    aci_contribution_rcp45_change_percent_2041_2060[i,:] = 100*(aci_contribution_rcp45_2041_2060[i,:]-aci_contribution_rcp45_2001_2020[i,:])/np.absolute(aci_contribution_rcp45_change_total_2041_2060[i])
    aci_contribution_rcp85_change_percent_2041_2060[i,:] = 100*(aci_contribution_rcp85_2041_2060[i,:]-aci_contribution_rcp85_2001_2020[i,:])/np.absolute(aci_contribution_rcp85_change_total_2041_2060[i])
    aci_contribution_rcp45_change_percent_2080_2099[i,:] = 100*(aci_contribution_rcp45_2080_2099[i,:]-aci_contribution_rcp45_2001_2020[i,:])/np.absolute(aci_contribution_rcp45_change_total_2080_2099[i])
    aci_contribution_rcp85_change_percent_2080_2099[i,:] = 100*(aci_contribution_rcp85_2080_2099[i,:]-aci_contribution_rcp85_2001_2020[i,:])/np.absolute(aci_contribution_rcp85_change_total_2080_2099[i])


mean_rcp45_2041_2060 = np.nanmean(aci_contribution_rcp45_change_percent_2041_2060, axis=0)
mean_rcp85_2041_2060 = np.nanmean(aci_contribution_rcp85_change_percent_2041_2060, axis=0)
mean_rcp45_2080_2099 = np.nanmean(aci_contribution_rcp45_change_percent_2080_2099, axis=0)
mean_rcp85_2080_2099 = np.nanmean(aci_contribution_rcp85_change_percent_2080_2099, axis=0)

median_rcp45_2041_2060 = np.nanmedian(aci_contribution_rcp45_change_percent_2041_2060, axis=0)
median_rcp85_2041_2060 = np.nanmedian(aci_contribution_rcp85_change_percent_2041_2060, axis=0)
median_rcp45_2080_2099 = np.nanmedian(aci_contribution_rcp45_change_percent_2080_2099, axis=0)
median_rcp85_2080_2099 = np.nanmedian(aci_contribution_rcp85_change_percent_2080_2099, axis=0)

### get index of median 
higher_index_rcp85_2080_2099 = np.where(aci_contribution_rcp85_change_total_2080_2099 == np.percentile(aci_contribution_rcp85_change_total_2080_2099, 50, interpolation='higher'))
lower_index_rcp85_2080_2099 = np.where(aci_contribution_rcp85_change_total_2080_2099 == np.percentile(aci_contribution_rcp85_change_total_2080_2099, 50, interpolation='lower'))
median_aci_contribution_rcp85_2080_2099 = (aci_contribution_rcp85_change_percent_2080_2099[higher_index_rcp85_2080_2099]+aci_contribution_rcp85_change_percent_2080_2099[lower_index_rcp85_2080_2099])/2
higher_index_rcp45_2080_2099 = np.where(aci_contribution_rcp45_change_total_2080_2099 == np.percentile(aci_contribution_rcp45_change_total_2080_2099, 50, interpolation='higher'))
lower_index_rcp45_2080_2099 = np.where(aci_contribution_rcp45_change_total_2080_2099 == np.percentile(aci_contribution_rcp45_change_total_2080_2099, 50, interpolation='lower'))
median_aci_contribution_rcp45_2080_2099 = (aci_contribution_rcp45_change_percent_2080_2099[higher_index_rcp45_2080_2099]+aci_contribution_rcp45_change_percent_2080_2099[lower_index_rcp45_2080_2099])/2

higher_index_rcp85_2041_2060 = np.where(aci_contribution_rcp85_change_total_2041_2060 == np.percentile(aci_contribution_rcp85_change_total_2041_2060, 50, interpolation='higher'))
lower_index_rcp85_2041_2060 = np.where(aci_contribution_rcp85_change_total_2041_2060 == np.percentile(aci_contribution_rcp85_change_total_2041_2060, 50, interpolation='lower'))
median_aci_contribution_rcp85_2041_2060 = (aci_contribution_rcp85_change_percent_2041_2060[higher_index_rcp85_2041_2060]+aci_contribution_rcp85_change_percent_2041_2060[lower_index_rcp85_2041_2060])/2
higher_index_rcp45_2041_2060 = np.where(aci_contribution_rcp45_change_total_2041_2060 == np.percentile(aci_contribution_rcp45_change_total_2041_2060, 50, interpolation='higher'))
lower_index_rcp45_2041_2060 = np.where(aci_contribution_rcp45_change_total_2041_2060 == np.percentile(aci_contribution_rcp45_change_total_2041_2060, 50, interpolation='lower'))
median_aci_contribution_rcp45_2041_2060 = (aci_contribution_rcp45_change_percent_2041_2060[higher_index_rcp45_2041_2060]+aci_contribution_rcp45_change_percent_2041_2060[lower_index_rcp45_2041_2060])/2


## calculate and plot contribution magnitude
aci_contribution_rcp45_change_2041_2060 = np.zeros((1000,29))
aci_contribution_rcp85_change_2041_2060 = np.zeros((1000,29))
aci_contribution_rcp45_change_2080_2099 = np.zeros((1000,29))
aci_contribution_rcp85_change_2080_2099 = np.zeros((1000,29))
for i in range(0,1000):
    aci_contribution_rcp45_change_2041_2060[i,:] = (aci_contribution_rcp45_2041_2060[i,:]-aci_contribution_rcp45_2001_2020[i,:])
    aci_contribution_rcp85_change_2041_2060[i,:] = (aci_contribution_rcp85_2041_2060[i,:]-aci_contribution_rcp85_2001_2020[i,:])
    aci_contribution_rcp45_change_2080_2099[i,:] = (aci_contribution_rcp45_2080_2099[i,:]-aci_contribution_rcp45_2001_2020[i,:])
    aci_contribution_rcp85_change_2080_2099[i,:] = (aci_contribution_rcp85_2080_2099[i,:]-aci_contribution_rcp85_2001_2020[i,:])
median_rcp45_2041_2060 = np.nanmedian(aci_contribution_rcp45_change_2041_2060, axis=0)
median_rcp85_2041_2060 = np.nanmedian(aci_contribution_rcp85_change_2041_2060, axis=0)
median_rcp45_2080_2099 = np.nanmedian(aci_contribution_rcp45_change_2080_2099, axis=0)
median_rcp85_2080_2099 = np.nanmedian(aci_contribution_rcp85_change_2080_2099, axis=0)

aci_delete_index = np.where(median_rcp45_2041_2060 ==0)
ci_rcp45_2041_2060 = np.zeros((29,2))
ci_rcp45_2041_2060[:,0] = np.percentile(aci_contribution_rcp45_change_2041_2060, 97.5, axis=0)
ci_rcp45_2041_2060[:,1] = np.percentile(aci_contribution_rcp45_change_2041_2060, 2.5, axis=0)
yerr_rcp45_2041_2060 = np.c_[median_rcp45_2041_2060-ci_rcp45_2041_2060[:,0], ci_rcp45_2041_2060[:,1]-median_rcp45_2041_2060].T

ci_rcp45_2080_2099 = np.zeros((29,2))
ci_rcp45_2080_2099[:,0] = np.percentile(aci_contribution_rcp45_change_2080_2099, 97.5, axis=0)
ci_rcp45_2080_2099[:,1] = np.percentile(aci_contribution_rcp45_change_2080_2099, 2.5, axis=0)
yerr_rcp45_2080_2099 = np.c_[median_rcp45_2080_2099-ci_rcp45_2080_2099[:,0], ci_rcp45_2080_2099[:,1]-median_rcp45_2080_2099].T

ci_rcp85_2041_2060 = np.zeros((29,2))
ci_rcp85_2041_2060[:,0] = np.percentile(aci_contribution_rcp85_change_2041_2060, 97.5, axis=0)
ci_rcp85_2041_2060[:,1] = np.percentile(aci_contribution_rcp85_change_2041_2060, 2.5, axis=0)
yerr_rcp85_2041_2060 = np.c_[median_rcp85_2041_2060-ci_rcp85_2041_2060[:,0], ci_rcp85_2041_2060[:,1]-median_rcp85_2041_2060].T

ci_rcp85_2080_2099 = np.zeros((29,2))
ci_rcp85_2080_2099[:,0] = np.percentile(aci_contribution_rcp85_change_2080_2099, 97.5, axis=0)
ci_rcp85_2080_2099[:,1] = np.percentile(aci_contribution_rcp85_change_2080_2099, 2.5, axis=0)
yerr_rcp85_2080_2099 = np.c_[median_rcp85_2080_2099-ci_rcp85_2080_2099[:,0], ci_rcp85_2080_2099[:,1]-median_rcp85_2080_2099].T


plt.figure(figsize = (30,20))
plt.bar(np.arange(0,np.delete(ACI_list, aci_delete_index).shape[0]), np.delete(median_rcp45_2041_2060, aci_delete_index),  yerr = np.delete(yerr_rcp45_2041_2060, aci_delete_index,axis = 1), capsize = 5, color = 'blue', width = 0.25)
plt.bar(np.arange(0,np.delete(ACI_list, aci_delete_index).shape[0])+0.25, np.delete(median_rcp45_2080_2099, aci_delete_index),  yerr = np.delete(yerr_rcp45_2080_2099, aci_delete_index,axis = 1), capsize = 5, color = 'red', width = 0.25)
plt.bar(np.arange(0,np.delete(ACI_list, aci_delete_index).shape[0])+0.5, np.delete(median_rcp85_2041_2060, aci_delete_index),  yerr = np.delete(yerr_rcp45_2041_2060, aci_delete_index,axis = 1), capsize = 5, color = 'blue', width = 0.25)
plt.bar(np.arange(0,np.delete(ACI_list, aci_delete_index).shape[0])+0.75, np.delete(median_rcp85_2080_2099, aci_delete_index),  yerr = np.delete(yerr_rcp45_2080_2099, aci_delete_index,axis = 1), capsize = 5, color = 'red', width = 0.25)

bar_color = sns.color_palette("tab20")
plt.figure(figsize = (30,20))
ax1 = plt.subplot(2,1,1)
plt.bar(np.arange(0,np.delete(ACI_list, aci_delete_index).shape[0]), np.delete(median_rcp45_2041_2060, aci_delete_index),  yerr = np.delete(yerr_rcp45_2041_2060, aci_delete_index,axis = 1), error_kw=dict(lw=4, capsize=7, capthick=3), color = bar_color, width = 0.5, label = '2050', edgecolor = 'black')
plt.bar(np.arange(0,np.delete(ACI_list, aci_delete_index).shape[0])+0.5, np.delete(median_rcp45_2080_2099, aci_delete_index),  yerr = np.delete(yerr_rcp45_2080_2099, aci_delete_index,axis = 1),  error_kw=dict(lw=4, capsize=7, capthick=3), color = bar_color, width = 0.5, label = '2099', edgecolor = 'black', hatch = '//')
plt.yticks(fontsize = 35)
plt.ylim(-1.6,0.5)
plt.ylabel('ACI contribution change (ton/acre)', y = 0, labelpad = 20, fontsize = 35)
ax1.axes.xaxis.set_visible(False)
plt.text(-1,-1.5, 'RCP 4.5', fontsize = 35)
ax2 = plt.subplot(2,1,2)
plt.bar(np.arange(0,np.delete(ACI_list, aci_delete_index).shape[0]), np.delete(median_rcp85_2041_2060, aci_delete_index),  yerr = np.delete(yerr_rcp85_2041_2060, aci_delete_index,axis = 1),  error_kw=dict(lw=4, capsize=7, capthick=3), color = bar_color, width = 0.5, edgecolor = 'black')
plt.bar(np.arange(0,np.delete(ACI_list, aci_delete_index).shape[0])+0.5, np.delete(median_rcp85_2080_2099, aci_delete_index),  yerr = np.delete(yerr_rcp85_2080_2099, aci_delete_index,axis = 1), error_kw=dict(lw=4, capsize=7, capthick=3), color = bar_color, width = 0.5, edgecolor = 'black',hatch = '//')
plt.xticks(np.arange(0,np.delete(ACI_list, aci_delete_index).shape[0])+0.2, np.delete(ACI_list, aci_delete_index), fontsize = 35, rotation = 60)
plt.yticks(fontsize = 35)
plt.ylim(-1.6,0.5)
plt.text(-1,-1.5, 'RCP 8.5', fontsize = 35)
plt.subplots_adjust(bottom=0.2)
for xtick, color in zip(ax2.get_xticklabels(), bar_color):
    xtick.set_color(color)
mpl.rcParams['hatch.linewidth'] = 2
white_patch = mpatches.Patch( facecolor = 'white', label = 'By 2050', edgecolor = 'black', linewidth = 3)
hatch_patch = mpatches.Patch(facecolor = 'w', hatch = '///', label = 'By 2099', edgecolor = 'black')
plt.legend(handles = [white_patch, hatch_patch], fontsize =35, loc = 'lower right')
plt.savefig('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/Almond_plots/plot_hnrg/aci_contribution.png', dpi = 200)


## county-level aci*coef
aci_contribution_rcp45_county_hist = np.zeros((16,17,39,1000,15))
aci_contribution_rcp85_county_hist = np.zeros((16,17,39,1000,15))
aci_contribution_rcp45_county_future = np.zeros((16,17,81,1000,15))
aci_contribution_rcp85_county_future = np.zeros((16,17,81,1000,15))
for county in range(0,16):
    for model in range(0,17):
        aci_contribution_rcp45_county_hist[county,model,:,:,:] = np.split(locals()['aci_contribution_model_rcp45_'+str(model)],16)[county][:,:,0:15]+np.split(locals()['aci_contribution_model_rcp45_'+str(model)],16)[county][:,:,15:30]
        aci_contribution_rcp85_county_hist[county,model,:,:,:] = np.split(locals()['aci_contribution_model_rcp85_'+str(model)],16)[county][:,:,0:15]+np.split(locals()['aci_contribution_model_rcp85_'+str(model)],16)[county][:,:,15:30]
        aci_contribution_rcp45_county_future[county,model,:,:,:] = np.split(locals()['aci_contribution_model_rcp45_future_'+str(model)],16)[county][:,:,0:15]+np.split(locals()['aci_contribution_model_rcp45_future_'+str(model)],16)[county][:,:,15:30]
        aci_contribution_rcp85_county_future[county,model,:,:,:] = np.split(locals()['aci_contribution_model_rcp85_future_'+str(model)],16)[county][:,:,0:15]+np.split(locals()['aci_contribution_model_rcp85_future_'+str(model)],16)[county][:,:,15:30]
aci_contribution_rcp45_county = np.concatenate((aci_contribution_rcp45_county_hist, aci_contribution_rcp45_county_future),axis=2)
aci_contribution_rcp85_county = np.concatenate((aci_contribution_rcp85_county_hist, aci_contribution_rcp85_county_future),axis=2)

aci_contribution_rcp45_county_2050_change =  np.mean((np.mean(aci_contribution_rcp45_county[:,:,61:81,:,:], axis=2)-np.mean(aci_contribution_rcp45_county[:,:,21:41,:,:],axis=2)), axis=1)
aci_contribution_rcp45_county_2090_change =  np.mean((np.mean(aci_contribution_rcp45_county[:,:,100:120,:,:], axis=2)-np.mean(aci_contribution_rcp45_county[:,:,21:41,:,:],axis=2)), axis=1)
aci_contribution_rcp45_county_2050_change_percent = np.zeros((16,1000,15))
aci_contribution_rcp45_county_2090_change_percent = np.zeros((16,1000,15))

aci_contribution_rcp85_county_2050_change =  np.mean((np.mean(aci_contribution_rcp85_county[:,:,61:81,:,:], axis=2)-np.mean(aci_contribution_rcp85_county[:,:,21:41,:,:],axis=2)), axis=1)
aci_contribution_rcp85_county_2090_change =  np.mean((np.mean(aci_contribution_rcp85_county[:,:,100:120,:,:], axis=2)-np.mean(aci_contribution_rcp85_county[:,:,21:41,:,:],axis=2)), axis=1)
aci_contribution_rcp85_county_2050_change_percent = np.zeros((16,1000,15))
aci_contribution_rcp85_county_2090_change_percent = np.zeros((16,1000,15))

for county in range(0,16):
    for trial in range(0,1000):
        aci_contribution_rcp45_county_2050_change_percent[county,trial,:] = 100*aci_contribution_rcp45_county_2050_change[county,trial,:]/np.abs(np.sum(aci_contribution_rcp45_county_2050_change[county,trial,:]))
        aci_contribution_rcp45_county_2090_change_percent[county,trial,:] = 100*aci_contribution_rcp45_county_2090_change[county,trial,:]/np.abs(np.sum(aci_contribution_rcp45_county_2090_change[county,trial,:]))
        aci_contribution_rcp85_county_2050_change_percent[county,trial,:] = 100*aci_contribution_rcp85_county_2050_change[county,trial,:]/np.abs(np.sum(aci_contribution_rcp85_county_2050_change[county,trial,:]))
        aci_contribution_rcp85_county_2090_change_percent[county,trial,:] = 100*aci_contribution_rcp85_county_2090_change[county,trial,:]/np.abs(np.sum(aci_contribution_rcp85_county_2090_change[county,trial,:]))

aci_delete_index = np.where(aci_contribution_rcp85_county_2090_change_percent[0,0,0,0]==0)
ACI = np.delete(ACI_list, aci_delete_index)
bar_color = sns.color_palette("tab20")                                                                                            
plt.figure(figsize = (30,30))
for county in range(0,16):
    ax = plt.subplot(16,1,county+1)
    plt.bar(ACI[aci_order],np.delete(np.median(np.mean(np.mean(aci_contribution_rcp85_county_2090_change_percent, axis=1),axis=1),axis=1)[county],aci_delete_index)[aci_order], color = bar_color)
    plt.ylim(-100,100)
    plt.text(x = -1, y = 65, s = str(county_list[county]), fontsize = 20)
    if county < 15:
        plt.xticks('')
for xtick, color in zip(ax.get_xticklabels(), bar_color):
    xtick.set_color(color)
plt.xticks(fontsize = 25, rotation = 45)
plt.annotate('', xy = (0.045, -2), xycoords = 'axes fraction', xytext = (0.26,-2), arrowprops = dict(arrowstyle = '<|-|>, head_width = 2, head_length = 2', linewidth = 5, color = 'k'))
plt.annotate('Dormancy', xy = (0.1, -2.5), xycoords = 'axes fraction', fontsize = 35)
plt.annotate('', xy = (0.295, -2), xycoords = 'axes fraction', xytext = (0.64,-2), arrowprops = dict(arrowstyle = '<|-|>, head_width = 2, head_length = 2', linewidth = 5, color = 'k'))
plt.annotate('Bloom (Pollination)', xy = (0.37,-2.5), xycoords = 'axes fraction', fontsize = 35)
plt.annotate('', xy = (0.67, -2), xycoords = 'axes fraction', xytext = (0.89,-2), arrowprops = dict(arrowstyle = '<|-|>, head_width = 2, head_length = 2', linewidth = 5, color = 'k'))
plt.annotate('Growth', xy = (0.75, -2.5), xycoords = 'axes fraction', fontsize = 35)
plt.annotate('', xy = (0.905, -2), xycoords = 'axes fraction', xytext = (0.965,-2), arrowprops = dict(arrowstyle = '<|-|>, head_width = 2, head_length = 2', linewidth = 5, color = 'k'))
plt.annotate('Harvest', xy = (0.89, -2.5), xycoords = 'axes fraction', fontsize = 35)
plt.ylabel('Contribution % to Climate-induced Yield Change', fontsize = 30, y =10)
plt.savefig('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/Almond_plots/plot_hnrg/aci_contribution_by_county_croplandfilterred.png', dpi = 200)
plt.show()



aci_contribution_rcp45_county_2050_change_percent = np.load('C:/Users/Pancake/Box/aci_contribution_5_20/aci_contribution_rcp45_county_2050_change_percent.npy')
aci_contribution_rcp45_county_2090_change_percent = np.load('C:/Users/Pancake/Box/aci_contribution_5_20/aci_contribution_rcp45_county_2090_change_percent.npy')
aci_contribution_rcp85_county_2050_change_percent = np.load('C:/Users/Pancake/Box/aci_contribution_5_20/aci_contribution_rcp85_county_2050_change_percent.npy')
aci_contribution_rcp85_county_2090_change_percent = np.load('C:/Users/Pancake/Box/aci_contribution_5_20/aci_contribution_rcp85_county_2090_change_percent.npy')

aci_contribution_rcp45_county_2050_change_percent_median = np.median(aci_contribution_rcp45_county_2050_change_percent, axis=1)
aci_contribution_rcp45_county_2090_change_percent_median = np.median(aci_contribution_rcp45_county_2090_change_percent, axis=1)
aci_contribution_rcp85_county_2050_change_percent_median = np.median(aci_contribution_rcp85_county_2050_change_percent, axis=1)
aci_contribution_rcp85_county_2090_change_percent_median = np.median(aci_contribution_rcp85_county_2090_change_percent, axis=1)

aci_contribution_rcp45_county_2050_change = np.load('C:/Users/Pancake/Box/aci_contribution_5_20/aci_contribution_rcp45_county_2050_change.npy')
aci_contribution_rcp45_county_2090_change = np.load('C:/Users/Pancake/Box/aci_contribution_5_20/aci_contribution_rcp45_county_2090_change.npy')
aci_contribution_rcp85_county_2050_change = np.load('C:/Users/Pancake/Box/aci_contribution_5_20/aci_contribution_rcp85_county_2050_change.npy')
aci_contribution_rcp85_county_2090_change = np.load('C:/Users/Pancake/Box/aci_contribution_5_20/aci_contribution_rcp85_county_2090_change.npy')

aci_contribution_rcp45_county_2050_change_toal = np.sum(aci_contribution_rcp45_county_2050_change, axis = 2)
aci_contribution_rcp45_county_2090_change_toal = np.sum(aci_contribution_rcp45_county_2050_change, axis = 2)
aci_contribution_rcp85_county_2050_change_toal = np.sum(aci_contribution_rcp85_county_2090_change, axis = 2)
aci_contribution_rcp85_county_2090_change_toal = np.sum(aci_contribution_rcp85_county_2090_change, axis = 2)




aci_order = [0,1,2,3,4,5,6,22,23,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
aci_order = [3, 8, 14, 9,16, 17,18, 4, 10, 5, 0, 11,12,1,6, 7, 15, 13,2]
aci_delete_index = 12
aci_order = [9,10,3,16,21,11,12,23,24,25,26,27,28,4,5,13,0,18,6,7,8,22,14,1,19,2,20,15,17]

fig,ax=plt.subplots(16,1, figsize = (30,30))
for county in range(0,16):     
    formatting = "{:,.1f}"
    from matplotlib.ticker import FuncFormatter
    index = np.array(ACI_list)[:]
    data = aci_contribution_rcp85_county_2050_change_percent_median[county]
    index = np.delete(index, aci_delete_index)
    data = np.delete(data, aci_delete_index)
    index=np.array(index)
    data=np.array(data)
    changes = {'amount' : data}
    def money(x, pos):
        return formatting.format(x)
    formatter = FuncFormatter(money)    
    trans = pd.DataFrame(data=changes,index=index)
    blank = trans.amount.cumsum().shift(1).fillna(0)
    trans['positive'] = trans['amount'] > 0
    total = trans.sum().amount
    step = blank.reset_index(drop=True).repeat(3).shift(-1)
    step[1::3] = np.nan
    trans.loc[trans['positive'] > 1, 'positive'] = 99
    trans.loc[trans['positive'] < 0, 'positive'] = 99
    trans.loc[(trans['positive'] > 0) & (trans['positive'] < 1), 'positive'] = 99
    trans['color'] = trans['positive']
    trans.loc[trans['positive'] == 1, 'color'] = '#29EA38' #green_color
    trans.loc[trans['positive'] == 0, 'color'] = '#FB3C62' #red_color
    trans.loc[trans['positive'] == 99, 'color'] = '#24CAFF' #blue_color
    my_colors = list(trans.color)
    plt.subplot(16,1,county+1)
    plt.text(s = str(county_list[county]), x = -3, y = -100, fontsize = 35)
    plt.plot(step.index, step.values, 'k', linewidth = 2)
    ax[county].bar(range(0,len(trans.index)), trans.amount, width=0.6, edgecolor = 'black',linewidth = 2,
             bottom=blank, color=my_colors)       
    plt.yticks(fontsize = 25)
    y_height = trans.amount.cumsum().shift(1).fillna(0)
    temp = list(trans.amount) 
    for i in range(len(temp)):
        if (i > 0) & (i < (len(temp) - 1)):
            temp[i] = temp[i] + temp[i-1]
    trans['temp'] = temp
    plot_max = trans['temp'].max()
    plot_min = trans['temp'].min()
    if all(i >= 0 for i in temp):
        plot_min = 0
    if all(i < 0 for i in temp):
        plot_max = 0
    if abs(plot_max) >= abs(plot_min):
        maxmax = abs(plot_max)   
    else:
        maxmax = abs(plot_min)
    pos_offset = maxmax / 40
    plot_offset = maxmax / 15 
    loop = 0
    for index, row in trans.iterrows():
        if row['amount'] == total:
            y = y_height[loop]
        else:
            y = y_height[loop] + row['amount']
        if row['amount'] > 0:
            y += (pos_offset*2)
            plt.annotate(formatting.format(row['amount']),(loop,y),ha="center", color = 'g', fontsize=20)
        else:
            y -= (pos_offset*4)
            plt.annotate(formatting.format(row['amount']),(loop,y-25),ha="center", color = 'r', fontsize=20)
        loop+=1
        plt.ylim(-190,20)
    plt.axhline(0, color='black', linewidth = 0.6, linestyle="dashed")
    if county == 15:
        plt.xticks(np.arange(0,len(trans)), trans.index, rotation=90, fontsize = 30)
    else:
        plt.tick_params(axis = 'x' , which = 'both', bottom = False, top = False, labelbottom = False)
plt.suptitle('% of ACI Contribution to Yield Change by Mid of the Century under RCP8.5' , fontsize = 40, y = 0.91)
for i in (0,2,4,6,8,10):
    rect=mpatches.Rectangle([-0.5+i,-190], 1, 4000, ec='white', fc='grey', alpha=0.2, clip_on=False)
    ax[15].add_patch(rect)

plt.annotate('', xy = (0.055, -3.5), xycoords = 'axes fraction', xytext = (0.24,-3.5), arrowprops = dict(arrowstyle = '<|-|>, head_width = 2, head_length = 2', linewidth = 5, color = 'k'))
plt.annotate('Dormancy', xy = (0.095, -4.3), xycoords = 'axes fraction', fontsize = 35)
plt.annotate('', xy = (0.295, -3.5), xycoords = 'axes fraction', xytext = (0.705,-3.5), arrowprops = dict(arrowstyle = '<|-|>, head_width = 2, head_length = 2', linewidth = 5, color = 'k'))
plt.annotate('Bloom (Pollination)', xy = (0.39,-4.3), xycoords = 'axes fraction', fontsize = 35)
plt.annotate('', xy = (0.75, -3.5), xycoords = 'axes fraction', xytext = (0.87,-3.5), arrowprops = dict(arrowstyle = '<|-|>, head_width = 2, head_length = 2', linewidth = 5, color = 'k'))
plt.annotate('Growth', xy = (0.77, -4.3), xycoords = 'axes fraction', fontsize = 35)
plt.annotate('', xy = (0.905, -3.5), xycoords = 'axes fraction', xytext = (0.965,-3.5), arrowprops = dict(arrowstyle = '<|-|>, head_width = 2, head_length = 2', linewidth = 5, color = 'k'))
plt.annotate('Harvest', xy = (0.89, -4.3), xycoords = 'axes fraction', fontsize = 35)
#plt.tight_layout()
plt.savefig('C:C:/Users/Pancake/Box/UCDGlobalChange/shqwu/Almond_plots/plot_hnrg/almond-land-avg/Growth_stage_ACI_6_19/aci_contribution_by_county_rcp85_2050.png', bbox_inches='tight', dpi = 200)


area = area_csv[0:41,1:]

yield_all_hist = np.zeros((656,0))
##hist rcp45
average_model = np.zeros((656,1000))
for model_id in range(0,17):
    locals()['model_'+str(model_id)] = np.zeros((656,0))
    for trial in range(1,11):
        locals()[str(model_list[model_id])+str(trial)+'_hist'] = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_6_19/to_2020/'+str(model_list[model_id])+'_hist_prediction_'+str(trial)+'_rcp45.csv', delimiter=',')
        locals()[str(model_list[model_id])+str(trial)+'_hist'][locals()[str(model_list[model_id])+str(trial)+'_hist']<0] = 0
        yield_all_hist = np.column_stack((yield_all_hist,locals()[str(model_list[model_id])+str(trial)+'_hist']))
        locals()['model_'+str(model_id)] = np.column_stack((locals()['model_'+str(model_id)], locals()[str(model_list[model_id])+str(trial)+'_hist']))
    locals()['model_'+str(model_id)+'_average'] = np.nanmedian(locals()['model_'+str(model_id)], axis = 1) ## change to median 
    average_model = average_model+locals()['model_'+str(model_id)]
average_model = average_model/17
average_model_hist = average_model
production_average_model = np.zeros((656,1000))
yield_all_model = np.column_stack((model_0_average, model_1_average,model_2_average,model_3_average,model_4_average,model_5_average,model_6_average,model_7_average,model_8_average, model_9_average, model_10_average, model_11_average, model_12_average, model_13_average, model_14_average, model_15_average, model_16_average))
yield_all_model_hist_rcp45 = yield_all_hist
yield_all_model_hist_rcp45_average_model = yield_all_model
production_all_model = np.zeros((656,17))
production_all_hist = np.zeros((656,17000))
yield_1980_rcp45 = np.zeros((16))
for index in range(0,16):
    yield_1980_rcp45[index] = np.mean(yield_all_model[index * 41,:])
for index in range(0,16):
    for year in range(0,41):
        production_all_model[index*41+year,:] = yield_all_model[index*41+year,:]*area[year,index]
        production_average_model[index*41+year,:] = average_model[index*41+year,:]*area[year,index]
        production_all_hist[index*41+year,:] = yield_all_hist[index*41+year,:]*area[year,index]
production_all_hist_split = np.split(production_all_hist, 16)
production_model_split = np.split(production_all_model,16) 
production_model_split_hist = production_model_split
production_across_state_hist = np.zeros((41,17))
production_average_model_split = np.split(production_average_model,16) 
production_average_model_across_state_hist = np.zeros((41,1000))
production_all_hist_split_across_state = np.zeros((41,17000)) 
for county_id in range(0,16):
    for year in range(0,41):
        production_across_state_hist[year,:] = production_across_state_hist[year,:]+production_model_split[county_id][year,:]
        production_average_model_across_state_hist[year,:] = production_average_model_across_state_hist[year,:]+production_average_model_split[county_id][year,:]
        production_all_hist_split_across_state[year, :] = production_all_hist_split_across_state[year,:]+production_all_hist_split[county_id][year,:]
production_all_rcp45_hist = production_all_hist_split_across_state
yield_across_state_hist_rcp45 =np.zeros((41,17))
yield_average_model_hist_rcp45 = np.zeros((41,1000))
yield_all_hist_rcp45 = np.zeros((41,17000))
for year in range(0,41):
    yield_across_state_hist_rcp45[year,:] = production_across_state_hist[year,:]/np.sum(area[year])
    yield_average_model_hist_rcp45[year,:] = production_average_model_across_state_hist[year,:]/np.sum(area[year])
    yield_all_hist_rcp45[year,:] = production_all_hist_split_across_state[year,:]/np.sum(area[year])

yield_all_hist = np.zeros((656,0))
##hist rcp45
average_model = np.zeros((656,1000))
for model_id in range(0,17):
    locals()['model_'+str(model_id)] = np.zeros((656,0))
    for trial in range(1,11):
        locals()[str(model_list[model_id])+str(trial)+'_hist'] = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_6_19/to_2020/'+str(model_list[model_id])+'_hist_prediction_'+str(trial)+'_rcp45_s.csv', delimiter=',')
        locals()[str(model_list[model_id])+str(trial)+'_hist'][locals()[str(model_list[model_id])+str(trial)+'_hist']<0] = 0
        yield_all_hist = np.column_stack((yield_all_hist,locals()[str(model_list[model_id])+str(trial)+'_hist']))
        locals()['model_'+str(model_id)] = np.column_stack((locals()['model_'+str(model_id)], locals()[str(model_list[model_id])+str(trial)+'_hist']))
    locals()['model_'+str(model_id)+'_average'] = np.median(locals()['model_'+str(model_id)], axis = 1) ## change to median 
    average_model = average_model+locals()['model_'+str(model_id)]
average_model = average_model/17
average_model_hist = average_model
production_average_model = np.zeros((656,1000))
yield_all_model = np.column_stack((model_0_average, model_1_average,model_2_average,model_3_average,model_4_average,model_5_average,model_6_average,model_7_average,model_8_average, model_9_average, model_10_average, model_11_average, model_12_average, model_13_average, model_14_average, model_15_average, model_16_average))
yield_all_model_hist_rcp45_s = yield_all_hist
yield_all_model_hist_rcp45_s_average_model = yield_all_model
production_all_model = np.zeros((656,17))
production_all_hist = np.zeros((656,17000))
for index in range(0,16):
    for year in range(0,41):
        production_all_model[index*41+year,:] = yield_all_model[index*41+year,:]*area[year,index]
        production_average_model[index*41+year,:] = average_model[index*41+year,:]*area[year,index]
        production_all_hist[index*41+year,:] = yield_all_hist[index*41+year,:]*area[year,index]

production_all_hist_split = np.split(production_all_hist, 16)
production_model_split = np.split(production_all_model,16) 
production_model_split_hist = production_model_split
production_across_state_hist = np.zeros((41,17))
production_average_model_split = np.split(production_average_model,16) 
production_average_model_across_state_hist = np.zeros((41,1000))
production_all_hist_split_across_state = np.zeros((41,17000))
for county_id in range(0,16):
    for year in range(0,41):
        production_across_state_hist[year,:] = production_across_state_hist[year,:]+production_model_split[county_id][year,:]
        production_average_model_across_state_hist[year,:] = production_average_model_across_state_hist[year,:]+production_average_model_split[county_id][year,:]
        production_all_hist_split_across_state[year, :] = production_all_hist_split_across_state[year,:]+production_all_hist_split[county_id][year,:]
production_all_rcp45_s_hist = production_all_hist_split_across_state
yield_across_state_hist_rcp45_s =np.zeros((41,17))
yield_average_model_hist_rcp45_s = np.zeros((41,1000))
yield_all_hist_rcp45_s = np.zeros((41,17000))
for year in range(0,41):
    yield_across_state_hist_rcp45_s[year,:] = production_across_state_hist[year,:]/np.sum(area[year])
    yield_average_model_hist_rcp45_s[year,:] = production_average_model_across_state_hist[year,:]/np.sum(area[year])
    yield_all_hist_rcp45_s[year,:] = production_all_hist_split_across_state[year,:]/np.sum(area[year])

##hist rcp85
yield_all_hist = np.zeros((656,0))
average_model = np.zeros((656,1000))
for model_id in range(0,17):
    locals()['model_'+str(model_id)] = np.zeros((656,0))
    for trial in range(1,11):
        locals()[str(model_list[model_id])+str(trial)+'_hist'] = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_6_19/to_2020/'+str(model_list[model_id])+'_hist_prediction_'+str(trial)+'_rcp85.csv', delimiter=',')
        locals()[str(model_list[model_id])+str(trial)+'_hist'][locals()[str(model_list[model_id])+str(trial)+'_hist']<0] = 0
        yield_all_hist = np.column_stack((yield_all_hist,locals()[str(model_list[model_id])+str(trial)+'_hist']))
        locals()['model_'+str(model_id)] = np.column_stack((locals()['model_'+str(model_id)], locals()[str(model_list[model_id])+str(trial)+'_hist']))
    locals()['model_'+str(model_id)+'_average'] = np.median(locals()['model_'+str(model_id)], axis = 1) ## change to median 
    average_model = average_model+locals()['model_'+str(model_id)]
average_model = average_model/17
average_model_hist = average_model
production_average_model = np.zeros((656,1000))
yield_all_model = np.column_stack((model_0_average, model_1_average,model_2_average,model_3_average,model_4_average,model_5_average,model_6_average,model_7_average,model_8_average, model_9_average, model_10_average, model_11_average, model_12_average, model_13_average, model_14_average, model_15_average, model_16_average))
yield_all_model_hist_rcp85 = yield_all_hist
yield_all_model_hist_rcp85_average_model = yield_all_model
production_all_model = np.zeros((656,17))
production_all_hist = np.zeros((656,17000))
yield_1980_rcp85 = np.zeros((16))
for index in range(0,16):
    yield_1980_rcp85[index] = np.mean(yield_all_model[index * 41,:])
for index in range(0,16):
    for year in range(0,41):
        production_all_model[index*41+year,:] = yield_all_model[index*41+year,:]*area[year,index%16]
        production_average_model[index*41+year,:] = average_model[index*41+year,:]*area[year,index%16]
        production_all_hist[index*41+year,:] = yield_all_hist[index*41+year,:]*area[year,index%16]
production_all_hist_split = np.split(production_all_hist, 16)
production_model_split = np.split(production_all_model,16) 
production_model_split_hist = production_model_split
production_across_state_hist = np.zeros((41,17))
production_average_model_split = np.split(production_average_model,16) 
production_average_model_across_state_hist = np.zeros((41,1000))
production_all_hist_split_across_state = np.zeros((41,17000))
for county_id in range(0,16):
    for year in range(0,41):
        production_across_state_hist[year,:] = production_across_state_hist[year,:]+production_model_split[county_id][year,:]
        production_average_model_across_state_hist[year,:] = production_average_model_across_state_hist[year,:]+production_average_model_split[county_id][year,:]
        production_all_hist_split_across_state[year, :] = production_all_hist_split_across_state[year,:]+production_all_hist_split[county_id][year,:]
production_all_rcp85_hist = production_all_hist_split_across_state
yield_across_state_hist_rcp85 =np.zeros((41,17))
yield_average_model_hist_rcp85 = np.zeros((41,1000))
yield_all_hist_rcp85 = np.zeros((41,17000))
for year in range(0,41):
    yield_across_state_hist_rcp85[year,:] = production_across_state_hist[year,:]/np.sum(area[year])
    yield_average_model_hist_rcp85[year,:] = production_average_model_across_state_hist[year,:]/np.sum(area[year])
    yield_all_hist_rcp85[year,:] = production_all_hist_split_across_state[year,:]/np.sum(area[year])

yield_all_hist = np.zeros((656,0))
##hist rcp85_s
average_model = np.zeros((656,1000))
for model_id in range(0,17):
    locals()['model_'+str(model_id)] = np.zeros((656,0))
    for trial in range(1,11):
        locals()[str(model_list[model_id])+str(trial)+'_hist'] = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_6_19/to_2020/'+str(model_list[model_id])+'_hist_prediction_'+str(trial)+'_rcp85_s.csv', delimiter=',')
        locals()[str(model_list[model_id])+str(trial)+'_hist'][locals()[str(model_list[model_id])+str(trial)+'_hist']<0] = 0
        yield_all_hist = np.column_stack((yield_all_hist,locals()[str(model_list[model_id])+str(trial)+'_hist']))
        locals()['model_'+str(model_id)] = np.column_stack((locals()['model_'+str(model_id)], locals()[str(model_list[model_id])+str(trial)+'_hist']))
    locals()['model_'+str(model_id)+'_average'] = np.median(locals()['model_'+str(model_id)], axis = 1) ## change to median 
    average_model = average_model+locals()['model_'+str(model_id)]
average_model = average_model/17
average_model_hist = average_model
production_average_model = np.zeros((656,1000))
yield_all_model = np.column_stack((model_0_average, model_1_average,model_2_average,model_3_average,model_4_average,model_5_average,model_6_average,model_7_average,model_8_average, model_9_average, model_10_average, model_11_average, model_12_average, model_13_average, model_14_average, model_15_average, model_16_average))
yield_all_model_hist_rcp85_s = yield_all_hist
yield_all_model_hist_rcp85_s_average_model = yield_all_model
production_all_model = np.zeros((656,17))
production_all_hist = np.zeros((656,17000))
for index in range(0,16):
    for year in range(0,41):
        production_all_model[index*41+year,:] = yield_all_model[index*41+year,:]*area[year,index]
        production_average_model[index*41+year,:] = average_model[index*41+year,:]*area[year,index]
        production_all_hist[index*41+year,:] = yield_all_hist[index*41+year,:]*area[year,index]

production_all_hist_split = np.split(production_all_hist, 16)
production_model_split = np.split(production_all_model,16) 
production_model_split_hist = production_model_split
production_across_state_hist = np.zeros((41,17))
production_average_model_split = np.split(production_average_model,16) 
production_average_model_across_state_hist = np.zeros((41,1000))
production_all_hist_split_across_state = np.zeros((41,17000))
for county_id in range(0,16):
    for year in range(0,41):
        production_across_state_hist[year,:] = production_across_state_hist[year,:]+production_model_split[county_id][year,:]
        production_average_model_across_state_hist[year,:] = production_average_model_across_state_hist[year,:]+production_average_model_split[county_id][year,:]
        production_all_hist_split_across_state[year, :] = production_all_hist_split_across_state[year,:]+production_all_hist_split[county_id][year,:]
production_all_rcp85_s_hist = production_all_hist_split_across_state
yield_across_state_hist_rcp85_s =np.zeros((41,17))
yield_average_model_hist_rcp85_s = np.zeros((41,1000))
yield_all_hist_rcp85_s = np.zeros((41,17000))
for year in range(0,41):
    yield_across_state_hist_rcp85_s[year,:] = production_across_state_hist[year,:]/np.sum(area[year])
    yield_average_model_hist_rcp85_s[year,:] = production_average_model_across_state_hist[year,:]/np.sum(area[year])
    yield_all_hist_rcp85_s[year,:] = production_all_hist_split_across_state[year,:]/np.sum(area[year])
    

area = area[-1]
#future rcp45
yield_all = np.zeros((1264,0))
average_model = np.zeros((1264,1000))
for model_id in range(0,17):
    locals()['model_'+str(model_id)] = np.zeros((1264,0))
    for trial in range(1,11):
        locals()[str(model_list[model_id])+str(trial)+'_rcp45'] = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_6_19/to_2020/'+str(model_list[model_id])+'_future_prediction_'+str(trial)+'_rcp45.csv', delimiter=',')
        locals()[str(model_list[model_id])+str(trial)+'_rcp45'][locals()[str(model_list[model_id])+str(trial)+'_rcp45']<0] = 0
        yield_all = np.column_stack((yield_all,locals()[str(model_list[model_id])+str(trial)+'_rcp45']))
        locals()['model_'+str(model_id)] = np.column_stack((locals()['model_'+str(model_id)], locals()[str(model_list[model_id])+str(trial)+'_rcp45']))
    locals()['model_'+str(model_id)+'_average'] = np.median(locals()['model_'+str(model_id)], axis = 1)
    average_model = average_model+locals()['model_'+str(model_id)]
average_model = average_model/17
average_model_rcp45 = average_model
production_average_model = np.zeros((1264,1000))
yield_all_model = np.column_stack((model_0_average, model_1_average,model_2_average,model_3_average,model_4_average,model_5_average,model_6_average,model_7_average,model_8_average, model_9_average, model_10_average, model_11_average, model_12_average, model_13_average, model_14_average, model_15_average, model_16_average))
yield_all_model_future_rcp45 = yield_all
production_all_model = np.zeros((1264,17))
production_all = np.zeros((1264,17000))
for index in range(0,16):
    for year in range(0,79):
        production_all_model[index*79+year,:] = yield_all_model[index*79+year,:]*area[index]
        production_average_model[index*79+year,:] = average_model[index*79+year,:]*area[index]
        production_all[index*79+year,:] = yield_all[index*79+year,:]*area[index]
production_all_split = np.split(production_all, 16)
production_model_split = np.split(production_all_model,16) 
production_model_split_45 = production_model_split
production_across_state_rcp45 = np.zeros((79,17))
production_average_model_split = np.split(production_average_model,16) 
production_average_model_across_state_rcp45 = np.zeros((79,1000))
production_all_split_across_state = np.zeros((79,17000))
for county_id in range(0,16):
    for year in range(0,79):
        production_across_state_rcp45[year,:] = production_across_state_rcp45[year,:]+production_model_split[county_id][year,:]
        production_average_model_across_state_rcp45[year,:] = production_average_model_across_state_rcp45[year,:]+production_average_model_split[county_id][year,:]
        production_all_split_across_state[year, :] = production_all_split_across_state[year,:]+production_all_split[county_id][year,:]
production_all_rcp45_future = production_all_split_across_state
yield_across_state_future_rcp45 =np.zeros((79,17))
yield_average_model_future_rcp45 = np.zeros((79,1000))
yield_all_future_rcp45 = np.zeros((79,17000))

for year in range(0,79):
    yield_across_state_future_rcp45[year,:] = production_across_state_rcp45[year,:]/np.sum(area)
    yield_average_model_future_rcp45[year,:] = production_average_model_across_state_rcp45[year,:]/np.sum(area)
    yield_all_future_rcp45[year,:] = production_all_split_across_state[year,:]/np.sum(area)


# future rcp45_s
yield_all = np.zeros((1264,0))
average_model = np.zeros((1264,1000))
for model_id in range(0,17):
    locals()['model_'+str(model_id)] = np.zeros((1264,0))
    for trial in range(1,11):
        locals()[str(model_list[model_id])+str(trial)+'_rcp45'] = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_6_19/to_2020/'+str(model_list[model_id])+'_future_prediction_'+str(trial)+'_rcp45_s.csv', delimiter=',')
        locals()[str(model_list[model_id])+str(trial)+'_rcp45'][locals()[str(model_list[model_id])+str(trial)+'_rcp45']<0] = 0
        yield_all = np.column_stack((yield_all,locals()[str(model_list[model_id])+str(trial)+'_rcp45']))
        locals()['model_'+str(model_id)] = np.column_stack((locals()['model_'+str(model_id)], locals()[str(model_list[model_id])+str(trial)+'_rcp45']))
    locals()['model_'+str(model_id)+'_average'] = np.median(locals()['model_'+str(model_id)], axis = 1)
    average_model = average_model+locals()['model_'+str(model_id)]
average_model = average_model/17
average_model_rcp45 = average_model
production_average_model = np.zeros((1264,1000))
yield_all_model = np.column_stack((model_0_average, model_1_average,model_2_average,model_3_average,model_4_average,model_5_average,model_6_average,model_7_average,model_8_average, model_9_average, model_10_average, model_11_average, model_12_average, model_13_average, model_14_average, model_15_average, model_16_average))
yield_all_model_future_rcp45_s = yield_all
production_all_model = np.zeros((1264,17))
production_all = np.zeros((1264,17000))
for index in range(0,16):
    for year in range(0,79):
        production_all_model[index*79+year,:] = yield_all_model[index*79+year,:]*area[index]
        production_average_model[index*79+year,:] = average_model[index*79+year,:]*area[index]
        production_all[index*79+year,:] = yield_all[index*79+year,:]*area[index]
production_all_split = np.split(production_all, 16)
production_model_split = np.split(production_all_model,16) 
production_model_split_45 = production_model_split
production_across_state_rcp45 = np.zeros((79,17))
production_average_model_split = np.split(production_average_model,16) 
production_average_model_across_state_rcp45 = np.zeros((79,1000))
production_all_split_across_state = np.zeros((79,17000))
for county_id in range(0,16):
    for year in range(0,79):
        production_across_state_rcp45[year,:] = production_across_state_rcp45[year,:]+production_model_split[county_id][year,:]
        production_average_model_across_state_rcp45[year,:] = production_average_model_across_state_rcp45[year,:]+production_average_model_split[county_id][year,:]
        production_all_split_across_state[year, :] = production_all_split_across_state[year,:]+production_all_split[county_id][year,:]
production_all_rcp45_s_future = production_all_split_across_state
yield_across_state_future_rcp45_s =np.zeros((79,17))
yield_average_model_future_rcp45_s = np.zeros((79,1000))
yield_all_future_rcp45_s = np.zeros((79,17000))

for year in range(0,79):
    yield_across_state_future_rcp45_s[year,:] = production_across_state_rcp45[year,:]/np.sum(area)
    yield_average_model_future_rcp45_s[year,:] = production_average_model_across_state_rcp45[year,:]/np.sum(area)
    yield_all_future_rcp45_s[year,:] = production_all_split_across_state[year,:]/np.sum(area)

#rcp85
yield_all = np.zeros((1264,0))
average_model = np.zeros((1264,1000))
for model_id in range(0,17):
    locals()['model_'+str(model_id)] = np.zeros((1264,0))
    for trial in range(1,11):
        locals()[str(model_list[model_id])+str(trial)+'_rcp85'] = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_6_19/to_2020/'+str(model_list[model_id])+'_future_prediction_'+str(trial)+'_rcp85.csv', delimiter=',')
        locals()[str(model_list[model_id])+str(trial)+'_rcp85'][locals()[str(model_list[model_id])+str(trial)+'_rcp85']<0] = 0
        yield_all = np.column_stack((yield_all,locals()[str(model_list[model_id])+str(trial)+'_rcp85']))
        locals()['model_'+str(model_id)] = np.column_stack((locals()['model_'+str(model_id)], locals()[str(model_list[model_id])+str(trial)+'_rcp85']))
    locals()['model_'+str(model_id)+'_average'] = np.median(locals()['model_'+str(model_id)], axis = 1)
    average_model = average_model+locals()['model_'+str(model_id)]
average_model = average_model/17
average_model_rcp85 = average_model
production_average_model = np.zeros((1264,1000))
yield_all_model = np.column_stack((model_0_average, model_1_average,model_2_average,model_3_average,model_4_average,model_5_average,model_6_average,model_7_average,model_8_average, model_9_average, model_10_average, model_11_average, model_12_average, model_13_average, model_14_average, model_15_average, model_16_average))
yield_all_model_future_rcp85 = yield_all
production_all_model = np.zeros((1264,17))
production_all = np.zeros((1264,17000))
for index in range(0,16):
    for year in range(0,79):
        production_all_model[index*79+year,:] = yield_all_model[index*79+year,:]*area[index]
        production_average_model[index*79+year,:] = average_model[index*79+year,:]*area[index]
        production_all[index*79+year,:] = yield_all[index*79+year,:]*area[index]
production_all_split = np.split(production_all, 16)
production_model_split = np.split(production_all_model,16) 
production_model_split_85 = production_model_split
production_across_state_rcp85 = np.zeros((79,17))
production_average_model_split = np.split(production_average_model,16) 
production_average_model_across_state_rcp85 = np.zeros((79,1000))
production_all_split_across_state = np.zeros((79,17000))
for county_id in range(0,16):
    for year in range(0,79):
        production_across_state_rcp85[year,:] = production_across_state_rcp85[year,:]+production_model_split[county_id][year,:]
        production_average_model_across_state_rcp85[year,:] = production_average_model_across_state_rcp85[year,:]+production_average_model_split[county_id][year,:]
        production_all_split_across_state[year, :] = production_all_split_across_state[year,:]+production_all_split[county_id][year,:]
production_all_rcp85_future = production_all_split_across_state
yield_across_state_future_rcp85 =np.zeros((79,17))
yield_average_model_future_rcp85 = np.zeros((79,1000))
yield_all_future_rcp85 = np.zeros((79,17000))

for year in range(0,79):
    yield_across_state_future_rcp85[year,:] = production_across_state_rcp85[year,:]/np.sum(area)
    yield_average_model_future_rcp85[year,:] = production_average_model_across_state_rcp85[year,:]/np.sum(area)
    yield_all_future_rcp85[year,:] = production_all_split_across_state[year,:]/np.sum(area)

#rcp85_stop_tech
yield_all = np.zeros((1264,0))
average_model = np.zeros((1264,1000))
for model_id in range(0,17):
    locals()['model_'+str(model_id)] = np.zeros((1264,0))
    for trial in range(1,11):
        locals()[str(model_list[model_id])+str(trial)+'_rcp85'] = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_6_19/to_2020/'+str(model_list[model_id])+'_future_prediction_'+str(trial)+'_rcp85_s.csv', delimiter=',')
        locals()[str(model_list[model_id])+str(trial)+'_rcp85'][locals()[str(model_list[model_id])+str(trial)+'_rcp85']<0] = 0
        yield_all = np.column_stack((yield_all,locals()[str(model_list[model_id])+str(trial)+'_rcp85']))
        locals()['model_'+str(model_id)] = np.column_stack((locals()['model_'+str(model_id)], locals()[str(model_list[model_id])+str(trial)+'_rcp85']))
    locals()['model_'+str(model_id)+'_average'] = np.median(locals()['model_'+str(model_id)], axis = 1)
    average_model = average_model+locals()['model_'+str(model_id)]
average_model = average_model/17
average_model_rcp85 = average_model
production_average_model = np.zeros((1264,1000))
yield_all_model = np.column_stack((model_0_average, model_1_average,model_2_average,model_3_average,model_4_average,model_5_average,model_6_average,model_7_average,model_8_average, model_9_average, model_10_average, model_11_average, model_12_average, model_13_average, model_14_average, model_15_average, model_16_average))
yield_all_model_future_rcp85_s = yield_all
production_all_model = np.zeros((1264,17))
production_all = np.zeros((1264,17000))
for index in range(0,16):
    for year in range(0,79):
        production_all_model[index*79+year,:] = yield_all_model[index*79+year,:]*area[index]
        production_average_model[index*79+year,:] = average_model[index*79+year,:]*area[index]
        production_all[index*79+year,:] = yield_all[index*79+year,:]*area[index]
production_all_split = np.split(production_all, 16)
production_model_split = np.split(production_all_model,16) 
production_model_split_85 = production_model_split
production_across_state_rcp85 = np.zeros((79,17))
production_average_model_split = np.split(production_average_model,16) 
production_average_model_across_state_rcp85 = np.zeros((79,1000))
production_all_split_across_state = np.zeros((79,17000))
for county_id in range(0,16):
    for year in range(0,79):
        production_across_state_rcp85[year,:] = production_across_state_rcp85[year,:]+production_model_split[county_id][year,:]
        production_average_model_across_state_rcp85[year,:] = production_average_model_across_state_rcp85[year,:]+production_average_model_split[county_id][year,:]
        production_all_split_across_state[year, :] = production_all_split_across_state[year,:]+production_all_split[county_id][year,:]
production_all_rcp85_s_future = production_all_split_across_state
yield_across_state_future_rcp85_s =np.zeros((79,17))
yield_average_model_future_rcp85_s = np.zeros((79,1000))
yield_all_future_rcp85_s = np.zeros((79,17000))

for year in range(0,79):
    yield_across_state_future_rcp85_s[year,:] = production_across_state_rcp85[year,:]/np.sum(area)
    yield_average_model_future_rcp85_s[year,:] = production_average_model_across_state_rcp85[year,:]/np.sum(area)
    yield_all_future_rcp85_s[year,:] = production_all_split_across_state[year,:]/np.sum(area)




##plot production change in 2050/2099
import geopandas
state = geopandas.read_file('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/CA_Counties/CA_Counties_TIGER2016.shp')
county_list = ['Butte', 'Colusa', 'Fresno', 'Glenn', 'Kern', 'Kings', 'Madera', 'Merced', 'San Joaquin', 'Solano', 'Stanislaus', 'Sutter', 'Tehama', 'Tulare', 'Yolo', 'Yuba']                      

rcp45_50 = np.zeros((58))
rcp45_s_50 = np.zeros((58))
rcp85_50 = np.zeros((58))
rcp85_s_50 = np.zeros((58))
rcp45_99 = np.zeros((58))
rcp45_s_99 = np.zeros((58))
rcp85_99 = np.zeros((58))
rcp85_s_99 = np.zeros((58))

for i in range(0,58):
    for j in range(0,16):
        if state.NAME[i] == county_list[j]:
            rcp45_50[i] = ((np.mean(production_model_split_45[j][43,:])/np.mean(production_model_split_45[j][0,:]))-1)*100
            rcp45_s_50[i] = ((np.mean(production_model_split_45_s[j][43,:])/np.mean(production_model_split_45_s[j][0,:]))-1)*100
            rcp85_50[i] = ((np.mean(production_model_split_85[j][43,:])/np.mean(production_model_split_85[j][0,:]))-1)*100
            rcp85_s_50[i] = ((np.mean(production_model_split_85_s[j][43,:])/np.mean(production_model_split_85_s[j][0,:]))-1)*100
          
            rcp45_99[i] = ((np.mean(production_model_split_45[j][92,:])/np.mean(production_model_split_45[j][0,:]))-1)*100
            rcp45_s_99[i] = ((np.mean(production_model_split_45_s[j][92,:])/np.mean(production_model_split_45_s[j][0,:]))-1)*100
            rcp85_99[i] = ((np.mean(production_model_split_85[j][92,:])/np.mean(production_model_split_85[j][0,:]))-1)*100
            rcp85_s_99[i] = ((np.mean(production_model_split_85_s[j][92,:])/np.mean(production_model_split_85_s[j][0,:]))-1)*100

rcp45_50[rcp45_50 == 0] = np.nan
rcp45_s_50[rcp45_s_50 == 0] = np.nan
rcp85_50[rcp85_50 == 0] = np.nan
rcp85_s_50[rcp85_s_50 == 0] = np.nan
rcp45_99[rcp45_99 == 0] = np.nan
rcp45_s_99[rcp45_s_99 == 0] = np.nan
rcp85_99[rcp85_99 == 0] = np.nan
rcp85_s_99[rcp85_s_99 == 0] = np.nan

state['rcp45_50'] = rcp45_50
state['rcp45_s_50'] = rcp45_s_50
state['rcp85_50'] = rcp85_50
state['rcp85_s_50'] = rcp85_s_50
state['rcp45_99'] = rcp45_99
state['rcp45_s_99'] = rcp45_s_99
state['rcp85_99'] = rcp85_99
state['rcp85_s_99'] = rcp85_s_99

state.plot(column = 'rcp45_50', legend = True, cmap = 'rainbow',missing_kwds = {"color": "lightgrey"}, figsize = (15,15), edgecolor = 'k', vmin = 0, vmax = 100)
plt.title('RCP4.5 in 2050', fontsize = 15)
plt.xticks([])
plt.yticks([])

state.plot(column = 'rcp45_s_50', legend = True, cmap = 'rainbow',missing_kwds = {"color": "lightgrey"}, figsize = (15,15), edgecolor = 'k',vmin = 0, vmax = 100)
plt.title('RCP4.5 w/o tech in 2050', fontsize = 15)
plt.xticks([])
plt.yticks([])

state.plot(column = 'rcp85_50', legend = True, cmap = 'rainbow',missing_kwds = {"color": "lightgrey"}, figsize = (15,15), edgecolor = 'k', vmin = 0, vmax = 100)
plt.title('RCP8.5 in 2050', fontsize = 15)
plt.xticks([])
plt.yticks([])

state.plot(column = 'rcp85_s_50', legend = True, cmap = 'rainbow',missing_kwds = {"color": "lightgrey"}, figsize = (15,15), edgecolor = 'k', vmin = 0, vmax = 100)
plt.title('RCP8.5 w/o tech in 2050', fontsize = 15)
plt.xticks([])
plt.yticks([])


state.plot(column = 'rcp45_99', legend = True, cmap = 'rainbow',missing_kwds = {"color": "lightgrey"}, figsize = (15,15), edgecolor = 'k', vmin = 0, vmax = 200)
plt.title('RCP4.5 in 2050', fontsize = 15)
plt.xticks([])
plt.yticks([])

state.plot(column = 'rcp45_s_99', legend = True, cmap = 'rainbow',missing_kwds = {"color": "lightgrey"}, figsize = (15,15), edgecolor = 'k',vmin = 0, vmax = 200)
plt.title('RCP4.5 w/o tech in 2050', fontsize = 15)
plt.xticks([])
plt.yticks([])

state.plot(column = 'rcp85_99', legend = True, cmap = 'rainbow',missing_kwds = {"color": "lightgrey"}, figsize = (15,15), edgecolor = 'k', vmin = 0, vmax = 200)
plt.title('RCP8.5 in 2050', fontsize = 15)
plt.xticks([])
plt.yticks([])

state.plot(column = 'rcp85_s_99', legend = True, cmap = 'rainbow',missing_kwds = {"color": "lightgrey"}, figsize = (15,15), edgecolor = 'k', vmin = 0, vmax = 200)
plt.title('RCP8.5 w/o tech in 2050', fontsize = 15)
plt.xticks([])
plt.yticks([])

##code to calculate yearly tech improvement with 1/year decrease per year
tech_d = np.zeros((93))
tech_d[0]=28
for i in range(1,93):
    tech_d[i] = i+28-(i+1)*(i)/2/93

##calculate average tech improvement coef by region
tech_c = np.zeros((16))
for trail in range(1,11):
    
    

##create space between box
from matplotlib.patches import PathPatch

def adjust_box_widths(g, fac):
    """
    Adjust the withs of a seaborn-generated boxplot.
    """

    # iterating through Axes instances
    for ax in g.axes:

        # iterating through axes artists:
        for c in ax.get_children():

            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5*(xmin+xmax)
                xhalf = 0.5*(xmax - xmin)

                # setting new width of box
                xmin_new = xmid-fac*xhalf
                xmax_new = xmid+fac*xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new]) 
## For meeting 10/5/2021
## calculate average total production (2048-2052) and (2097-2099) for tech_improve, rcp45 w/o tech, rcp85 w/o tech 
## plot boxplot
import seaborn as sns
import pandas as pd

yield_across_state_total_rcp45 = np.row_stack((yield_across_state_hist_rcp45, yield_across_state_future_rcp45))
yield_across_state_total_rcp45_s = np.row_stack((yield_across_state_hist_rcp45_s, yield_across_state_future_rcp45_s))
yield_across_state_total_rcp85 = np.row_stack((yield_across_state_hist_rcp85, yield_across_state_future_rcp85))
yield_across_state_total_rcp85_s = np.row_stack((yield_across_state_hist_rcp85_s, yield_across_state_future_rcp85_s))

yield_average_model_total_rcp45 = np.row_stack((yield_average_model_hist_rcp45, yield_average_model_future_rcp45))
yield_average_model_total_rcp45_s = np.row_stack((yield_average_model_hist_rcp45_s, yield_average_model_future_rcp45_s))
yield_average_model_total_rcp85 = np.row_stack((yield_average_model_hist_rcp85, yield_average_model_future_rcp85))
yield_average_model_total_rcp85_s = np.row_stack((yield_average_model_hist_rcp85, yield_average_model_future_rcp85_s))

yield_all_sum_rcp45 = np.row_stack((yield_all_hist_rcp45, yield_all_future_rcp45))
yield_all_sum_rcp45_s = np.row_stack((yield_all_hist_rcp45_s, yield_all_future_rcp45_s))
yield_all_sum_rcp85 = np.row_stack((yield_all_hist_rcp85, yield_all_future_rcp85))
yield_all_sum_rcp85_s = np.row_stack((yield_all_hist_rcp85_s, yield_all_future_rcp85_s))


yield_change_2000 = np.zeros((17,4))
yield_change_2050 = np.zeros((17,4))
yield_change_2099 = np.zeros((17,4))

yield_change_2000[:,0] = (np.nanmean(yield_across_state_total_rcp45[20:41,:],axis = 0))
yield_change_2000[:,1] = (np.nanmean(yield_across_state_total_rcp45_s[20:41], axis = 0))
yield_change_2000[:,2] = (np.nanmean(yield_across_state_total_rcp85[20:41], axis= 0))
yield_change_2000[:,3] = (np.nanmean(yield_across_state_total_rcp85_s[20:41], axis= 0 ))

yield_change_2050[:,0] = ((np.nanmean(yield_across_state_total_rcp45[60:81,:],axis = 0))-yield_change_2000[:,0])*100/yield_change_2000[:,0]
yield_change_2050[:,1] = ((np.nanmean(yield_across_state_total_rcp45_s[60:81,:], axis = 0))-yield_change_2000[:,1])*100/yield_change_2000[:,1]
yield_change_2050[:,2] = ((np.nanmean(yield_across_state_total_rcp85[60:81,:], axis= 0))-yield_change_2000[:,2])*100/yield_change_2000[:,2]
yield_change_2050[:,3] = ((np.nanmean(yield_across_state_total_rcp85_s[60:81,:], axis= 0 ))-yield_change_2000[:,3])*100/yield_change_2000[:,3]

yield_change_2099[:,0] = ((np.nanmean(yield_across_state_total_rcp45[100:120,:],axis = 0))-yield_change_2000[:,0])*100/yield_change_2000[:,0]
yield_change_2099[:,1] = ((np.nanmean(yield_across_state_total_rcp45_s[100:120,:], axis = 0))-yield_change_2000[:,1])*100/yield_change_2000[:,1]
yield_change_2099[:,2] = ((np.nanmean(yield_across_state_total_rcp85[100:120,:], axis= 0))-yield_change_2000[:,2])*100/yield_change_2000[:,2]
yield_change_2099[:,3] = ((np.nanmean(yield_across_state_total_rcp85_s[100:120,:], axis= 0))-yield_change_2000[:,3])*100/yield_change_2000[:,3]

yield_change_2000_ave_model = np.zeros((1000,4))
yield_change_2050_ave_model = np.zeros((1000,4))
yield_change_2099_ave_model = np.zeros((1000,4))

yield_change_2000_ave_model[:,0] = (np.nanmean(yield_average_model_total_rcp45[20:41,:],axis = 0))
yield_change_2000_ave_model[:,1] = (np.nanmean(yield_average_model_total_rcp45_s[20:41], axis = 0))
yield_change_2000_ave_model[:,2] = (np.nanmean(yield_average_model_total_rcp85[20:41], axis= 0))
yield_change_2000_ave_model[:,3] = (np.nanmean(yield_average_model_total_rcp85_s[20:41], axis= 0 ))

yield_change_2050_ave_model[:,0] = ((np.nanmean(yield_average_model_total_rcp45[60:81,:],axis = 0))-yield_change_2000_ave_model[:,0])*100/yield_change_2000_ave_model[:,0]
yield_change_2050_ave_model[:,1] = ((np.nanmean(yield_average_model_total_rcp45_s[60:81,:], axis = 0))-yield_change_2000_ave_model[:,1])*100/yield_change_2000_ave_model[:,1]
yield_change_2050_ave_model[:,2] = ((np.nanmean(yield_average_model_total_rcp85[60:81,:], axis= 0))-yield_change_2000_ave_model[:,2])*100/yield_change_2000_ave_model[:,2]
yield_change_2050_ave_model[:,3] = ((np.nanmean(yield_average_model_total_rcp85_s[60:81,:], axis= 0 ))-yield_change_2000_ave_model[:,3])*100/yield_change_2000_ave_model[:,3]

yield_change_2099_ave_model[:,0] = ((np.nanmean(yield_average_model_total_rcp45[100:120,:],axis = 0))-yield_change_2000_ave_model[:,0])*100/yield_change_2000_ave_model[:,0]
yield_change_2099_ave_model[:,1] = ((np.nanmean(yield_average_model_total_rcp45_s[100:120,:], axis = 0))-yield_change_2000_ave_model[:,1])*100/yield_change_2000_ave_model[:,1]
yield_change_2099_ave_model[:,2] = ((np.nanmean(yield_average_model_total_rcp85[100:120,:], axis= 0))-yield_change_2000_ave_model[:,2])*100/yield_change_2000_ave_model[:,2]
yield_change_2099_ave_model[:,3] = ((np.nanmean(yield_average_model_total_rcp85_s[100:120,:], axis= 0))-yield_change_2000_ave_model[:,3])*100/yield_change_2000_ave_model[:,3]

yield_change_all_2000 = np.zeros((17000,4))
yield_change_all_2050 = np.zeros((17000,4))
yield_change_all_2099 = np.zeros((17000,4))

yield_change_all_2000[:,0] = (np.nanmean(yield_all_sum_rcp45[20:41,:],axis = 0))
yield_change_all_2000[:,1] = (np.nanmean(yield_all_sum_rcp45_s[20:41], axis = 0))
yield_change_all_2000[:,2] = (np.nanmean(yield_all_sum_rcp85[20:41], axis= 0))
yield_change_all_2000[:,3] = (np.nanmean(yield_all_sum_rcp85_s[20:41], axis= 0 ))

yield_change_all_2050[:,0] = ((np.nanmean(yield_all_sum_rcp45[60:81,:],axis = 0))-yield_change_all_2000[:,0])*100/yield_change_all_2000[:,0]
yield_change_all_2050[:,1] = ((np.nanmean(yield_all_sum_rcp45_s[60:81,:], axis = 0))-yield_change_all_2000[:,1])*100/yield_change_all_2000[:,1]
yield_change_all_2050[:,2] = ((np.nanmean(yield_all_sum_rcp85[60:81,:], axis= 0))-yield_change_all_2000[:,2])*100/yield_change_all_2000[:,2]
yield_change_all_2050[:,3] = ((np.nanmean(yield_all_sum_rcp85_s[60:81,:], axis= 0 ))-yield_change_all_2000[:,3])*100/yield_change_all_2000[:,3]

yield_change_all_2099[:,0] = ((np.nanmean(yield_all_sum_rcp45[100:120,:],axis = 0))-yield_change_all_2000[:,0])*100/yield_change_all_2000[:,0]
yield_change_all_2099[:,1] = ((np.nanmean(yield_all_sum_rcp45_s[100:120,:], axis = 0))-yield_change_all_2000[:,1])*100/yield_change_all_2000[:,1]
yield_change_all_2099[:,2] = ((np.nanmean(yield_all_sum_rcp85[100:120,:], axis= 0))-yield_change_all_2000[:,2])*100/yield_change_all_2000[:,2]
yield_change_all_2099[:,3] = ((np.nanmean(yield_all_sum_rcp85_s[100:120,:], axis= 0))-yield_change_all_2000[:,3])*100/yield_change_all_2000[:,3]


##natural var


narutal_var_2050_rcp45 = np.mean(((yield_across_state_total_rcp45[60:81] - yield_across_state_total_rcp45[20:41])*100/yield_across_state_total_rcp45[20:41]), axis = 1)
narutal_var_2050_rcp45_s = np.mean(((yield_across_state_total_rcp45_s[60:81] - yield_across_state_total_rcp45_s[20:41])*100/yield_across_state_total_rcp45_s[20:41]), axis = 1)
narutal_var_2050_rcp85 = np.mean(((yield_across_state_total_rcp85[60:81] - yield_across_state_total_rcp85[20:41])*100/yield_across_state_total_rcp85[20:41]), axis = 1)
narutal_var_2050_rcp85_s = np.mean(((yield_across_state_total_rcp85_s[60:81] - yield_across_state_total_rcp85_s[20:41])*100/yield_across_state_total_rcp85_s[20:41]), axis = 1)

narutal_var_2099_rcp45 = np.mean(((yield_across_state_total_rcp45[99:120] - yield_across_state_total_rcp45[20:41])*100/yield_across_state_total_rcp45[20:41]), axis = 1)
narutal_var_2099_rcp45_s = np.mean(((yield_across_state_total_rcp45_s[99:120] - yield_across_state_total_rcp45_s[20:41])*100/yield_across_state_total_rcp45_s[20:41]), axis = 1)
narutal_var_2099_rcp85 = np.mean(((yield_across_state_total_rcp85[99:120] - yield_across_state_total_rcp85[20:41])*100/yield_across_state_total_rcp85[20:41]), axis = 1)
narutal_var_2099_rcp85_s = np.mean(((yield_across_state_total_rcp85_s[99:120] - yield_across_state_total_rcp85_s[20:41])*100/yield_across_state_total_rcp85_s[20:41]), axis = 1)

## all data
yield_change_all_20year_2000 = np.zeros((343400,4))
yield_change_all_20year_2050 = np.zeros((343400,4))
yield_change_all_20year_2099 = np.zeros((343400,4))

yield_change_all_20year_2000[:,0] = np.ndarray.flatten(yield_all_sum_rcp45[20:40])
yield_change_all_20year_2000[:,1] = np.ndarray.flatten(yield_all_sum_rcp45_s[20:40])
yield_change_all_20year_2000[:,2] = np.ndarray.flatten(yield_all_sum_rcp85[20:40])
yield_change_all_20year_2000[:,3] = np.ndarray.flatten(yield_all_sum_rcp85_s[20:40])

yield_change_all_20year_2050[:,0] = (np.ndarray.flatten(yield_all_sum_rcp45[60:80])-yield_change_all_20year_2000[:,0])*100/yield_change_all_20year_2000[:,0]
yield_change_all_20year_2050[:,1] = (np.ndarray.flatten(yield_all_sum_rcp45_s[60:80])-yield_change_all_20year_2000[:,1])*100/yield_change_all_20year_2000[:,1]
yield_change_all_20year_2050[:,2] = (np.ndarray.flatten(yield_all_sum_rcp85[60:80])-yield_change_all_20year_2000[:,2])*100/yield_change_all_20year_2000[:,2]
yield_change_all_20year_2050[:,3] = (np.ndarray.flatten(yield_all_sum_rcp85_s[60:80])-yield_change_all_20year_2000[:,3])*100/yield_change_all_20year_2000[:,3]

yield_change_all_20year_2099[:,0] = (np.ndarray.flatten(yield_all_sum_rcp45[100:120])-yield_change_all_20year_2000[:,0])*100/yield_change_all_20year_2000[:,0]
yield_change_all_20year_2099[:,1] = (np.ndarray.flatten(yield_all_sum_rcp45_s[100:120])-yield_change_all_20year_2000[:,1])*100/yield_change_all_20year_2000[:,1]
yield_change_all_20year_2099[:,2] = (np.ndarray.flatten(yield_all_sum_rcp85[100:120])-yield_change_all_20year_2000[:,2])*100/yield_change_all_20year_2000[:,2]
yield_change_all_20year_2099[:,3] = (np.ndarray.flatten(yield_all_sum_rcp85_s[100:120])-yield_change_all_20year_2000[:,3])*100/yield_change_all_20year_2000[:,3]





a = pd.DataFrame({'Year' : np.repeat('2050', 17), 'scenario' : np.repeat('Climate Var-RCP4.5',17) , 'Yield change %' : yield_change_2050[:,1]})
b = pd.DataFrame({'Year' : np.repeat('2050', 1000), 'scenario' : np.repeat('Stats. Var-RCP4.5',1000) , 'Yield change %' : yield_change_2050_ave_model[:,1]})
c = pd.DataFrame({'Year' : np.repeat('2050', 21), 'scenario' : np.repeat('Natural Var-RCP4.5',21) , 'Yield change %' : narutal_var_2050_rcp45_s})
d = pd.DataFrame({'Year' : np.repeat('2050', 17000), 'scenario' : np.repeat('20yr mean Var-RCP4.5',17000) , 'Yield change %' : yield_change_all_2050[:,1]})

e = pd.DataFrame({'Year' : np.repeat('2050', 17), 'scenario' : np.repeat('Climate Var-RCP8.5',17) , 'Yield change %' : yield_change_2050[:,3]})
f = pd.DataFrame({'Year' : np.repeat('2050', 1000), 'scenario' : np.repeat('Stats. Var-RCP8.5',1000) , 'Yield change %' : yield_change_2050_ave_model[:,3]})
g = pd.DataFrame({'Year' : np.repeat('2050', 21), 'scenario' : np.repeat('Natural Var-RCP8.5',21) , 'Yield change %' : narutal_var_2050_rcp85_s})
h = pd.DataFrame({'Year' : np.repeat('2050', 17000), 'scenario' : np.repeat('20yr mean Var-RCP8.5',17000) , 'Yield change %' : yield_change_all_2050[:,3]})

i = pd.DataFrame({'Year' : np.repeat('2099', 17), 'scenario' : np.repeat('Climate Var-RCP4.5',17) , 'Yield change %' : yield_change_2099[:,1]})
j = pd.DataFrame({'Year' : np.repeat('2099', 1000), 'scenario' : np.repeat('Stats. Var-RCP4.5',1000) , 'Yield change %' : yield_change_2099_ave_model[:,1]})
k = pd.DataFrame({'Year' : np.repeat('2099', 21), 'scenario' : np.repeat('Natural Var-RCP4.5',21) , 'Yield change %' : narutal_var_2099_rcp45_s})
l = pd.DataFrame({'Year' : np.repeat('2099', 17000), 'scenario' : np.repeat('20yr mean Var-RCP4.5',17000) , 'Yield change %' : yield_change_all_2099[:,1]})

m = pd.DataFrame({'Year' : np.repeat('2099', 17), 'scenario' : np.repeat('Climate Var-RCP8.5',17) , 'Yield change %' : yield_change_2099[:,3]})
n = pd.DataFrame({'Year' : np.repeat('2099', 1000), 'scenario' : np.repeat('Stats. Var-RCP8.5',1000) , 'Yield change %' : yield_change_2099_ave_model[:,3]})
o = pd.DataFrame({'Year' : np.repeat('2099', 21), 'scenario' : np.repeat('Natural Var-RCP8.5',21) , 'Yield change %' : narutal_var_2099_rcp85_s})
p = pd.DataFrame({'Year' : np.repeat('2099', 17000), 'scenario' : np.repeat('20yr mean Var-RCP8.5',17000) , 'Yield change %' : yield_change_all_2099[:,3]})

q = pd.DataFrame({'Year' : np.repeat('2050', 343400), 'scenario' : np.repeat('All Data-RCP4.5',343400) , 'Yield change %' : yield_change_all_20year_2050[:,1]})
r = pd.DataFrame({'Year' : np.repeat('2050', 343400), 'scenario' : np.repeat('All Data-RCP8.5',343400) , 'Yield change %' : yield_change_all_20year_2050[:,3]})
s = pd.DataFrame({'Year' : np.repeat('2099', 343400), 'scenario' : np.repeat('All Data-RCP4.5',343400) , 'Yield change %' : yield_change_all_20year_2099[:,1]})
t = pd.DataFrame({'Year' : np.repeat('2099', 343400), 'scenario' : np.repeat('All Data-RCP8.5',343400) , 'Yield change %' : yield_change_all_20year_2099[:,3]})

df = a.append(b).append(c).append(d).append(q).append(e).append(f).append(g).append(h).append(r).append(i).append(j).append(k).append(l).append(s).append(m).append(n).append(o).append(p).append(t)


fig = plt.figure(figsize = (22,15))
plt.suptitle('Simulated Almond Yield Change (without Tech advancement) in California since 2000', fontsize =28, y = 0.91)
my_pal = {'Climate Var-RCP4.5' : 'c', 'Stats. Var-RCP4.5': 'g', 'Natural Var-RCP4.5' : 'y','20yr mean Var-RCP4.5' : 'm','All Data-RCP4.5' : 'r', 'Climate Var-RCP8.5' : 'c', 'Stats. Var-RCP8.5': 'g','Natural Var-RCP8.5' : 'y', '20yr mean Var-RCP8.5' : 'm','All Data-RCP8.5' : 'r'}
ax = sns.boxplot(x = 'Year', y = 'Yield change %', hue = 'scenario', palette = my_pal, hue_order = ['Climate Var-RCP4.5', 'Stats. Var-RCP4.5', 'Natural Var-RCP4.5', '20yr mean Var-RCP4.5','All Data-RCP4.5', 'Climate Var-RCP8.5','Stats. Var-RCP8.5', 'Natural Var-RCP8.5' ,'20yr mean Var-RCP8.5','All Data-RCP8.5' ], data = df, linewidth = 2.5)
ax.set_ylim([-150, 250]) 
hatches = ['','','','','','','','','','','//','//','//','//','//','//','//','//','//','//','','','','','','//','//','//','//','//','//','//','//','//','//','//','//','//','//','//']
for hatch,patch in zip(hatches,ax.patches):
    patch.set_hatch(hatch)
adjust_box_widths(fig, 0.8)
blue_patch = mpatches.Patch(color='c', label='Climate Var (Data amount: 17)')
green_patch = mpatches.Patch(color='g', label='Stats. Var (Data amount: 1000)')
red_patch = mpatches.Patch(color='r', label='All Data Var (Data amount: 343400)')
yellow_patch = mpatches.Patch(color='y', label='Natural Var (Data amount: 20)')
purple_patch = mpatches.Patch(color='m', label='20-year mean Var (Data amount: 17000)')
hatch_patch = mpatches.Patch(facecolor = 'w', hatch = '///', label = 'RCP 8.5', edgecolor = 'k')
plt.legend(handles=[blue_patch, green_patch, yellow_patch, purple_patch, red_patch, hatch_patch], fontsize = 20,loc='upper center', bbox_to_anchor=(0.5, -0.07),
          fancybox=True, shadow=True, ncol=3)
plt.axvline(x = 0, linestyle='--', color = 'k')
plt.axvline(x = 1, linestyle='--', color = 'k')
plt.axhline(y = 0, linestyle = ':', color= 'r')
plt.xlabel('Year', fontsize = 27)
plt.ylabel('Yield Change %', fontsize = 27)
plt.xticks(fontsize = 24)
plt.yticks(fontsize = 24)
plt.savefig('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/Almond_plots/plot_hnrg/Yield_change_sum.png', dpi = 200)

a = pd.DataFrame({'Year' : np.repeat('2050', 17), 'scenario' : np.repeat('Climate Var-RCP4.5',17) , 'Yield change %' : yield_change_2050[:,0]})
b = pd.DataFrame({'Year' : np.repeat('2050', 1000), 'scenario' : np.repeat('Stats. Var-RCP4.5',1000) , 'Yield change %' : yield_change_2050_ave_model[:,0]})
c = pd.DataFrame({'Year' : np.repeat('2050', 21), 'scenario' : np.repeat('Natural Var-RCP4.5',21) , 'Yield change %' : narutal_var_2050_rcp45})
d = pd.DataFrame({'Year' : np.repeat('2050', 17000), 'scenario' : np.repeat('20yr mean Var-RCP4.5',17000) , 'Yield change %' : yield_change_all_2050[:,0]})

e = pd.DataFrame({'Year' : np.repeat('2050', 17), 'scenario' : np.repeat('Climate Var-RCP8.5',17) , 'Yield change %' : yield_change_2050[:,2]})
f = pd.DataFrame({'Year' : np.repeat('2050', 1000), 'scenario' : np.repeat('Stats. Var-RCP8.5',1000) , 'Yield change %' : yield_change_2050_ave_model[:,2]})
g = pd.DataFrame({'Year' : np.repeat('2050', 21), 'scenario' : np.repeat('Natural Var-RCP8.5',21) , 'Yield change %' : narutal_var_2050_rcp85})
h = pd.DataFrame({'Year' : np.repeat('2050', 17000), 'scenario' : np.repeat('20yr mean Var-RCP8.5',17000) , 'Yield change %' : yield_change_all_2050[:,2]})

i = pd.DataFrame({'Year' : np.repeat('2099', 17), 'scenario' : np.repeat('Climate Var-RCP4.5',17) , 'Yield change %' : yield_change_2099[:,0]})
j = pd.DataFrame({'Year' : np.repeat('2099', 1000), 'scenario' : np.repeat('Stats. Var-RCP4.5',1000) , 'Yield change %' : yield_change_2099_ave_model[:,0]})
k = pd.DataFrame({'Year' : np.repeat('2099', 21), 'scenario' : np.repeat('Natural Var-RCP4.5',21) , 'Yield change %' : narutal_var_2099_rcp45})
l = pd.DataFrame({'Year' : np.repeat('2099', 17000), 'scenario' : np.repeat('20yr mean Var-RCP4.5',17000) , 'Yield change %' : yield_change_all_2099[:,0]})

m = pd.DataFrame({'Year' : np.repeat('2099', 17), 'scenario' : np.repeat('Climate Var-RCP8.5',17) , 'Yield change %' : yield_change_2099[:,2]})
n = pd.DataFrame({'Year' : np.repeat('2099', 1000), 'scenario' : np.repeat('Stats. Var-RCP8.5',1000) , 'Yield change %' : yield_change_2099_ave_model[:,2]})
o = pd.DataFrame({'Year' : np.repeat('2099', 21), 'scenario' : np.repeat('Natural Var-RCP8.5',21) , 'Yield change %' : narutal_var_2099_rcp85})
p = pd.DataFrame({'Year' : np.repeat('2099', 17000), 'scenario' : np.repeat('20yr mean Var-RCP8.5',17000) , 'Yield change %' : yield_change_all_2099[:,2]})

q = pd.DataFrame({'Year' : np.repeat('2050', 343400), 'scenario' : np.repeat('All Data-RCP4.5',343400) , 'Yield change %' : yield_change_all_20year_2050[:,0]})
r = pd.DataFrame({'Year' : np.repeat('2050', 343400), 'scenario' : np.repeat('All Data-RCP8.5',343400) , 'Yield change %' : yield_change_all_20year_2050[:,2]})
s = pd.DataFrame({'Year' : np.repeat('2099', 343400), 'scenario' : np.repeat('All Data-RCP4.5',343400) , 'Yield change %' : yield_change_all_20year_2099[:,0]})
t = pd.DataFrame({'Year' : np.repeat('2099', 343400), 'scenario' : np.repeat('All Data-RCP8.5',343400) , 'Yield change %' : yield_change_all_20year_2099[:,2]})

df_tech = a.append(b).append(c).append(d).append(q).append(e).append(f).append(g).append(h).append(r).append(i).append(j).append(k).append(l).append(s).append(m).append(n).append(o).append(p).append(t)

fig = plt.figure(figsize = (22,15))
plt.suptitle('Simulated Almond Yield Change (with Tech advancement) in California since 2000', fontsize =28, y = 0.91)
my_pal = {'Climate Var-RCP4.5' : 'c', 'Stats. Var-RCP4.5': 'g', 'Natural Var-RCP4.5' : 'y','20yr mean Var-RCP4.5' : 'm','All Data-RCP4.5' : 'r', 'Climate Var-RCP8.5' : 'c', 'Stats. Var-RCP8.5': 'g','Natural Var-RCP8.5' : 'y', '20yr mean Var-RCP8.5' : 'm','All Data-RCP8.5' : 'r'}
ax = sns.boxplot(x = 'Year', y = 'Yield change %', hue = 'scenario', palette = my_pal, hue_order = ['Climate Var-RCP4.5', 'Stats. Var-RCP4.5', 'Natural Var-RCP4.5', '20yr mean Var-RCP4.5','All Data-RCP4.5', 'Climate Var-RCP8.5','Stats. Var-RCP8.5', 'Natural Var-RCP8.5' ,'20yr mean Var-RCP8.5','All Data-RCP8.5' ], data = df_tech, linewidth = 2.5)
ax.set_ylim([-150, 250]) 
hatches = ['','','','','','','','','','','//','//','//','//','//','//','//','//','//','//','','','','','','//','//','//','//','//','//','//','//','//','//','//','//','//','//','//']
for hatch,patch in zip(hatches,ax.patches):
    patch.set_hatch(hatch)
adjust_box_widths(fig, 0.8)
blue_patch = mpatches.Patch(color='c', label='Climate Var (Data amount: 17)')
green_patch = mpatches.Patch(color='g', label='Stats. Var (Data amount: 1000)')
red_patch = mpatches.Patch(color='r', label='All Data Var (Data amount: 343400)')
yellow_patch = mpatches.Patch(color='y', label='Natural Var (Data amount: 20)')
purple_patch = mpatches.Patch(color='m', label='20-year mean Var (Data amount: 17000)')
hatch_patch = mpatches.Patch(facecolor = 'w', hatch = '///', label = 'RCP 8.5', edgecolor = 'k')
plt.legend(handles=[blue_patch, green_patch, yellow_patch, purple_patch, red_patch, hatch_patch], fontsize = 20,loc='upper center', bbox_to_anchor=(0.5, -0.07),
          fancybox=True, shadow=True, ncol=3)
plt.axvline(x = 0, linestyle='--', color = 'k')
plt.axvline(x = 1, linestyle='--', color = 'k')
plt.axhline(y = 0, linestyle = ':', color= 'r')
plt.xlabel('Year', fontsize = 27)
plt.ylabel('Yield Change %', fontsize = 27)
plt.xticks(fontsize = 24)
plt.yticks(fontsize = 24)
plt.savefig('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/Almond_plots/plot_hnrg/Yield_change_sum_with_tech.png', dpi = 200)



##Yield STD change
std_yield_change_2000 = np.zeros((17,4))
std_yield_change_2050 = np.zeros((17,4))
std_yield_change_2099 = np.zeros((17,4))

std_yield_change_2000[:,0] = (np.nanstd(yield_across_state_total_rcp45[50:69,:],axis = 0))
std_yield_change_2000[:,1] = (np.nanstd(yield_across_state_total_rcp45_s[50:69], axis = 0))
std_yield_change_2000[:,2] = (np.nanstd(yield_across_state_total_rcp85[50:69], axis= 0))
std_yield_change_2000[:,3] = (np.nanstd(yield_across_state_total_rcp85_s[50:69], axis= 0 ))

std_yield_change_2050[:,0] = ((np.nanstd(yield_across_state_total_rcp45[89:109,:],axis = 0))-std_yield_change_2000[:,0])*100/std_yield_change_2000[:,0]
std_yield_change_2050[:,1] = ((np.nanstd(yield_across_state_total_rcp45_s[89:109,:], axis = 0))-std_yield_change_2000[:,1])*100/std_yield_change_2000[:,1]
std_yield_change_2050[:,2] = ((np.nanstd(yield_across_state_total_rcp85[89:109,:], axis= 0))-std_yield_change_2000[:,2])*100/std_yield_change_2000[:,2]
std_yield_change_2050[:,3] = ((np.nanstd(yield_across_state_total_rcp85_s[91:109,:], axis= 0 ))-std_yield_change_2000[:,3])*100/std_yield_change_2000[:,3]

std_yield_change_2099[:,0] = ((np.nanstd(yield_across_state_total_rcp45[128:148,:],axis = 0))-std_yield_change_2000[:,0])*100/std_yield_change_2000[:,0]
std_yield_change_2099[:,1] = ((np.nanstd(yield_across_state_total_rcp45_s[128:148,:], axis = 0))-std_yield_change_2000[:,1])*100/std_yield_change_2000[:,1]
std_yield_change_2099[:,2] = ((np.nanstd(yield_across_state_total_rcp85[128:148,:], axis= 0))-std_yield_change_2000[:,2])*100/std_yield_change_2000[:,2]
std_yield_change_2099[:,3] = ((np.nanstd(yield_across_state_total_rcp85_s[128:148,:], axis= 0))-std_yield_change_2000[:,3])*100/std_yield_change_2000[:,3]      

a = pd.DataFrame({'Year' : np.repeat('2050', 17), 'scenario' : np.repeat('RCP4.5',17) , 'Yield STD change %' : std_yield_change_2050[:,0]})
b = pd.DataFrame({'Year' : np.repeat('2050', 17), 'scenario' : np.repeat('RCP8.5',17) , 'Yield STD change %' : std_yield_change_2050[:,2]})

c = pd.DataFrame({'Year' : np.repeat('2050', 17), 'scenario' : np.repeat('RCP4.5 w/ tech',17) , 'Yield STD change %' : std_yield_change_2050[:,1]})
d = pd.DataFrame({'Year' : np.repeat('2050', 17), 'scenario' : np.repeat('RCP8.5 w/ tech',17) , 'Yield STD change %' : std_yield_change_2050[:,3]})

e = pd.DataFrame({'Year' : np.repeat('2099', 17), 'scenario' : np.repeat('RCP4.5',17) , 'Yield STD change %' : std_yield_change_2099[:,0]})
f = pd.DataFrame({'Year' : np.repeat('2099', 17), 'scenario' : np.repeat('RCP8.5',17) , 'Yield STD change %' : std_yield_change_2099[:,2]})

g = pd.DataFrame({'Year' : np.repeat('2099', 17), 'scenario' : np.repeat('RCP4.5 w/ tech',17) , 'Yield STD change %' : std_yield_change_2099[:,1]})
h = pd.DataFrame({'Year' : np.repeat('2099', 17), 'scenario' : np.repeat('RCP8.5 w/ tech',17) , 'Yield STD change %' : std_yield_change_2099[:,3]})
df = a.append(b).append(e).append(f)
df_tech = c.append(d).append(g).append(h)

plt.figure(figsize = (15,20))
plt.suptitle('Simulated Almond Yield STD Change in California since 2000', fontsize =22, y = 0.91)
plt.subplot(2,1,1)
sns.boxplot(x = 'Year', y = 'Yield STD change %', hue = 'scenario', data = df, linewidth = 2.5)
plt.xlabel('Year', fontsize = 17)
plt.ylabel('Yield STD Change %', fontsize = 17)
plt.legend(fontsize = 16)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.subplot(2,1,2)
sns.boxplot(x = 'Year', y = 'Yield STD change %', hue = 'scenario', data = df_tech, linewidth = 2.5)
plt.xlabel('Year', fontsize = 17)
plt.ylabel('Yield STD Change %', fontsize = 17)
plt.legend(fontsize = 16)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.savefig('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/Almond_plots/Yield_STD_change_boxplot.png', dpi = 200)


## Calculate tech trend coef. for different counties
coef = np.zeros((0,90))
for trial in range(1,11):
    coef_individual = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/lasso/coef_'+str(trial)+'.csv', delimiter = ',')
    coef = np.row_stack((coef,coef_individual))
for county in range(0,16):
    locals()[str(county_list[county])+'_tech_trend'] = pd.DataFrame({'County' : np.repeat(county_list[county],1000), 'Tech Coef' : coef[:,58+county]})
    if county == 0 :
        df = locals()[str(county_list[county])+'_tech_trend']
    else:
        df = df.append(locals()[str(county_list[county])+'_tech_trend'])

plt.figure(figsize = (15,10))
sns.boxplot(x = 'County', y = 'Tech Coef', data = df, linewidth = 2.5)
plt.xlabel('County', fontsize = 15)
plt.ylabel('Tech Improvement Trend Coef (ton/acre per year)', fontsize = 15)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.title('Tech Improvement Trend Coefficient', fontsize = 18)
plt.savefig('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/Almond_plots/summary_tech_trend.png', dpi = 200)



## plot ca map with % yield change
county_list = ['Butte', 'Colusa', 'Fresno', 'Glenn', 'Kern', 'Kings', 'Madera', 'Merced', 'San Joaquin', 'Solano', 'Stanislaus', 'Sutter', 'Tehama', 'Tulare', 'Yolo', 'Yuba']                      

ca = geopandas.read_file('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/CA_Counties/CA_Counties_TIGER2016.shp')
yield_all_model_rcp45_s = np.mean(yield_all_model_rcp45_s, axis = 1)
yield_all_model_rcp45_s = np.split(yield_all_model_rcp45_s, 16)
yield_all_model_rcp85_s = np.mean(yield_all_model_rcp85_s, axis = 1)
yield_all_model_rcp85_s = np.split(yield_all_model_rcp85_s, 16)

yield_change_for_shp_45_2050 = np.zeros((58))
yield_change_for_shp_45_2050[:] = np.nan
yield_change_for_shp_45_2099 = np.zeros((58))
yield_change_for_shp_45_2099[:] = np.nan
yield_change_for_shp_85_2050 = np.zeros((58))
yield_change_for_shp_85_2050[:] = np.nan
yield_change_for_shp_85_2099 = np.zeros((58))
yield_change_for_shp_85_2099[:] = np.nan


for i in range(0,58):
    for index in range(0,16):
        if county_list[index] == ca.NAME[i]:
            yield_change_for_shp_45_2050[i] = (np.mean(yield_all_model_rcp45_s[index][41:46])-yield_csv[38,index+1])*100/yield_csv[38,index+1]
            yield_change_for_shp_85_2050[i] = (np.mean(yield_all_model_rcp85_s[index][41:46])-yield_csv[38,index+1])*100/yield_csv[38,index+1]
            yield_change_for_shp_45_2099[i] = (np.mean(yield_all_model_rcp45_s[index][90:93])-yield_csv[38,index+1])*100/yield_csv[38,index+1]
            yield_change_for_shp_85_2099[i] = (np.mean(yield_all_model_rcp85_s[index][90:93])-yield_csv[38,index+1])*100/yield_csv[38,index+1]

           
yield_change_for_shp_45_2050_df = pd.DataFrame({'NAME' : ca.NAME, 'rcp45_2050' : yield_change_for_shp_45_2050})
yield_change_for_shp_45_2099_df = pd.DataFrame({'NAME' : ca.NAME, 'rcp45_2099' : yield_change_for_shp_45_2099})
yield_change_for_shp_85_2050_df = pd.DataFrame({'NAME' : ca.NAME, 'rcp85_2050' : yield_change_for_shp_85_2050})
yield_change_for_shp_85_2099_df = pd.DataFrame({'NAME' : ca.NAME, 'rcp85_2099' : yield_change_for_shp_85_2099})

ca_merge_rcp45_2050 =  ca.merge(yield_change_for_shp_45_2050_df, on = 'NAME')
ca_merge_rcp45_2099 =  ca.merge(yield_change_for_shp_45_2099_df, on = 'NAME')
ca_merge_rcp85_2050 =  ca.merge(yield_change_for_shp_85_2050_df, on = 'NAME')
ca_merge_rcp85_2099 =  ca.merge(yield_change_for_shp_85_2099_df, on = 'NAME')

plt.figure(figsize = (20,25))
ax = ca_merge_rcp45_2050.plot(ca_merge_rcp45_2050.rcp45_2050,edgecolor='black',missing_kwds={'color': 'lightgrey'}, legend = True, cmap = 'seismic',vmin = -30, vmax = 60)
plt.title('% Change in Almond Yield by 2050 since 2020 - rcp45 ')
ax.set_axis_off()
plt.savefig('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/Almond_plots/map_yield_change_rcp45_2050.png', dpi = 200)

plt.figure(figsize = (20,25))
ax = ca_merge_rcp45_2099.plot(ca_merge_rcp45_2099.rcp45_2099,edgecolor='black',missing_kwds={'color': 'lightgrey'}, legend = True, cmap = 'seismic',vmin = -30, vmax = 60)
plt.title('% Change in Almond Yield by 2099 since 2020 - rcp45')
ax.set_axis_off()
plt.savefig('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/Almond_plots/map_yield_change_rcp45_2099.png', dpi = 200)

plt.figure(figsize = (20,25))
ax = ca_merge_rcp85_2050.plot(ca_merge_rcp85_2050.rcp85_2050,edgecolor='black',missing_kwds={'color': 'lightgrey'}, legend = True, cmap = 'seismic',vmin = -30, vmax = 60)
plt.title('% Change in Almond Yield by 2050 since 2020 - rcp85')
ax.set_axis_off()
plt.savefig('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/Almond_plots/map_yield_change_rcp85_2050.png', dpi = 200)

plt.figure(figsize = (20,25))
ax = ca_merge_rcp85_2099.plot(ca_merge_rcp85_2099.rcp85_2099,edgecolor='black',missing_kwds={'color': 'lightgrey'}, legend = True, cmap = 'seismic',vmin = -30, vmax = 60)
plt.title('% Change in Almond Yield by 2099 since 2020 - rcp85')
ax.set_axis_off()
plt.savefig('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/Almond_plots/map_yield_change_rcp85_2099.png', dpi = 200)


##plot map of yield change for each county
county_list = ['Butte', 'Colusa', 'Fresno', 'Glenn', 'Kern', 'Kings', 'Madera', 'Merced', 'San Joaquin', 'Solano', 'Stanislaus', 'Sutter', 'Tehama', 'Tulare', 'Yolo', 'Yuba']
for i in range(0,16):
    locals()[str(county_list[i])+'yield_rcp45'] = np.row_stack((np.split(yield_all_model_hist_rcp45, 16)[i], np.split(yield_all_model_future_rcp45, 16)[i]))
    locals()[str(county_list[i])+'yield_rcp45_s'] = np.row_stack((np.split(yield_all_model_hist_rcp45_s, 16)[i], np.split(yield_all_model_future_rcp45_s, 16)[i]))
    locals()[str(county_list[i])+'yield_rcp85'] = np.row_stack((np.split(yield_all_model_hist_rcp85, 16)[i], np.split(yield_all_model_future_rcp85, 16)[i]))
    locals()[str(county_list[i])+'yield_rcp85_s'] = np.row_stack((np.split(yield_all_model_hist_rcp85_s, 16)[i], np.split(yield_all_model_future_rcp85_s, 16)[i]))

for i in range(0,16):
    locals()[str(county_list[i])+'county_yield_change_2000'] = np.zeros((17000,4))
    locals()[str(county_list[i])+'county_yield_change_2000'][:,0] = (np.nanmean(locals()[str(county_list[i])+'yield_rcp45'][20:40,:], axis = 0))
    locals()[str(county_list[i])+'county_yield_change_2000'][:,1] = (np.nanmean(locals()[str(county_list[i])+'yield_rcp45_s'][20:40,:], axis = 0))
    locals()[str(county_list[i])+'county_yield_change_2000'][:,2] = (np.nanmean(locals()[str(county_list[i])+'yield_rcp85'][20:40,:], axis = 0))
    locals()[str(county_list[i])+'county_yield_change_2000'][:,3] = (np.nanmean(locals()[str(county_list[i])+'yield_rcp85_s'][20:40,:], axis = 0))

    locals()[str(county_list[i])+'county_yield_change_2050'] = np.zeros((17000,4))
    locals()[str(county_list[i])+'county_yield_change_2050'][:,0] = ((np.nanmean(locals()[str(county_list[i])+'yield_rcp45'][60:80,:], axis=0))-locals()[str(county_list[i])+'county_yield_change_2000'][:,0])*100/locals()[str(county_list[i])+'county_yield_change_2000'][:,0]
    locals()[str(county_list[i])+'county_yield_change_2050'][:,1] = ((np.nanmean(locals()[str(county_list[i])+'yield_rcp45_s'][60:80,:], axis=0))-locals()[str(county_list[i])+'county_yield_change_2000'][:,1])*100/locals()[str(county_list[i])+'county_yield_change_2000'][:,1]
    locals()[str(county_list[i])+'county_yield_change_2050'][:,2] = ((np.nanmean(locals()[str(county_list[i])+'yield_rcp85'][60:80,:], axis=0))-locals()[str(county_list[i])+'county_yield_change_2000'][:,2])*100/locals()[str(county_list[i])+'county_yield_change_2000'][:,2]
    locals()[str(county_list[i])+'county_yield_change_2050'][:,3] = ((np.nanmean(locals()[str(county_list[i])+'yield_rcp85_s'][60:80,:], axis=0))-locals()[str(county_list[i])+'county_yield_change_2000'][:,3])*100/locals()[str(county_list[i])+'county_yield_change_2000'][:,3]

    locals()[str(county_list[i])+'county_yield_change_2099'] = np.zeros((17000,4))
    locals()[str(county_list[i])+'county_yield_change_2099'][:,0] = ((np.nanmean(locals()[str(county_list[i])+'yield_rcp45'][100:120,:], axis=0))-locals()[str(county_list[i])+'county_yield_change_2000'][:,0])*100/locals()[str(county_list[i])+'county_yield_change_2000'][:,0]
    locals()[str(county_list[i])+'county_yield_change_2099'][:,1] = ((np.nanmean(locals()[str(county_list[i])+'yield_rcp45_s'][100:120,:], axis=0))-locals()[str(county_list[i])+'county_yield_change_2000'][:,1])*100/locals()[str(county_list[i])+'county_yield_change_2000'][:,1]
    locals()[str(county_list[i])+'county_yield_change_2099'][:,2] = ((np.nanmean(locals()[str(county_list[i])+'yield_rcp85'][100:120,:], axis=0))-locals()[str(county_list[i])+'county_yield_change_2000'][:,2])*100/locals()[str(county_list[i])+'county_yield_change_2000'][:,2]
    locals()[str(county_list[i])+'county_yield_change_2099'][:,3] = ((np.nanmean(locals()[str(county_list[i])+'yield_rcp85_s'][100:120,:], axis=0))-locals()[str(county_list[i])+'county_yield_change_2000'][:,3])*100/locals()[str(county_list[i])+'county_yield_change_2000'][:,3]

median_yield_change_2050 = np.zeros((16,4))
median_yield_change_2099 = np.zeros((16,4))

for i in range(0,16):
    median_yield_change_2050[i,:] = np.nanmedian(locals()[str(county_list[i])+'county_yield_change_2050'], axis = 0)
    median_yield_change_2099[i,:] = np.nanmedian(locals()[str(county_list[i])+'county_yield_change_2099'], axis = 0)


yield_change_for_shp_45_2099 = np.zeros((58))
yield_change_for_shp_45_2099[:] = np.nan
yield_change_for_shp_85_2099 = np.zeros((58))
yield_change_for_shp_85_2099[:] = np.nan
N_S_order = np.zeros((16))

ca = geopandas.read_file('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/CA_Counties/CA_Counties_TIGER2016.shp')
ca_county_remove_shp = geopandas.read_file('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/CA_Counties/CA_Counties_TIGER2016.shp')
ca_county_remove = ['Sierra', 'Sacramento', 'Santa Barbara', 'Calaveras', 'Ventura','Los Angeles', 'Sonoma', 'San Diego', 'Placer', 'San Francisco', 'Marin', 'Mariposa', 'Lassen', 'Napa',
                    'Shasta', 'Monterey','Trinity', 'Mendocino', 'Inyo', 'Mono', 'Tuolumne', 'San Bernardino', 'Contra Costa', 'Alpine', 'El Dorado', 'San Benito', 'Humboldt','Riverside',
                    'Del Norte', 'Modoc', 'Santa Clara', 'Alameda', 'Nevada', 'Orange', 'Imperial', 'Amador', 'Lake', 'Plumas', 'San Mateo', 'Siskiyou', 'Santa Cruz','San Luis Obispo']
for i in range(0,len(ca_county_remove)):
    ca_county_remove_shp.drop(ca_county_remove_shp.index[ca_county_remove_shp['NAME']==ca_county_remove[i]], inplace=True)


for i in range(0,58):
    for index in range(0,16):
        if county_list[index] == ca.NAME[i]:
            yield_change_for_shp_45_2099[i] = median_yield_change_2099[index,1]
            yield_change_for_shp_85_2099[i] = median_yield_change_2099[index,3]
county_order_N_S = ['Tehama', 'Butte', 'Glenn', 'Yuba', 'Colusa', 'Sutter', 'Yolo', 'Solano', 'San Joaquin', 'Stanislaus', 'Madera', 'Merced', 'Fresno', 'Tulare', 'Kings', 'Kern']

for i in range(0,16):
    N_S_order[np.array(np.where(ca_county_remove_shp['NAME'] == county_order_N_S[i]))] = i+1
ca_county_remove_shp['N_S_order'] = N_S_order.astype(int)
#county-level hist yield average


yield_for_shp_obs_hist = np.zeros((58))
yield_for_shp_obs_hist[:] = np.nan
for i in range(0,58):
    for index in range(0,16):
        if county_list[index] == ca.NAME[i]:
            yield_for_shp_obs_hist[i] = np.mean(yield_csv[0:-2, 1:], axis = 0)[index]
            
yield_for_shp_gridmet_hist = np.zeros((58))
yield_for_shp_gridmet_hist[:] = np.nan
for i in range(0,58):
    for index in range(0,16):
        if county_list[index] == ca.NAME[i]:
            yield_for_shp_gridmet_hist[i] = (np.mean(np.split(np.median(simulation_gridmet, axis = 1),16)[index])-yield_for_shp_obs_hist[i])*100/yield_for_shp_obs_hist[i]

yield_for_shp_maca_rcp85_hist = np.zeros((58))
yield_for_shp_maca_rcp85_hist[:] = np.nan
for i in range(0,58):
    for index in range(0,16):
        if county_list[index] == ca.NAME[i]:
            yield_for_shp_maca_rcp85_hist[i] = (np.mean(np.split(yield_all_model_hist_rcp85_average_model,16)[index])-(np.mean(np.split(np.median(simulation_gridmet, axis = 1),16)[index])))*100/(np.mean(np.split(np.median(simulation_gridmet, axis = 1),16)[index]))

yield_for_shp_maca_rcp45_hist = np.zeros((58))
yield_for_shp_maca_rcp45_hist[:] = np.nan
for i in range(0,58):
    for index in range(0,16):
        if county_list[index] == ca.NAME[i]:
            yield_for_shp_maca_rcp45_hist[i] = (np.mean(np.split(yield_all_model_hist_rcp45_average_model,16)[index])-(np.mean(np.split(np.median(simulation_gridmet, axis = 1),16)[index])))*100/(np.mean(np.split(np.median(simulation_gridmet, axis = 1),16)[index]))
yield_for_shp_hist_df = pd.DataFrame({'NAME' : ca.NAME, 'Observation' : yield_for_shp_obs_hist, 'Gridmet' : yield_for_shp_gridmet_hist, 'RCP4.5' : yield_for_shp_maca_rcp45_hist, 'RCP8.5' : yield_for_shp_maca_rcp85_hist})
ca_merge_hist = ca.merge(yield_for_shp_hist_df, on = 'NAME')

fig, axes = plt.subplots(1,3, figsize=(40,12))
ax1 = ca_merge_hist.plot(ax = axes[0], column = ca_merge_hist.Observation,edgecolor='black',missing_kwds={'color': 'white'}, legend = True, cmap = 'OrRd')
ax1.set_axis_off()
ax1.set_title('Averaged Observed Almond \n Yield ton/acre over 1980-2018', fontsize = 35)
ax2 = ca_merge_hist.plot(ax = axes[1], column = ca_merge_hist.Gridmet,edgecolor='black',missing_kwds={'color': 'white'}, legend = True, cmap = 'RdBu_r')
ax2.set_axis_off()
ax2.set_title('Averaged GridMet-Observed Yield \n Difference % over 1980-2018', fontsize = 35)
ax4 = ca_merge_hist.plot(ax = axes[2], column = 'RCP8.5' ,edgecolor='black',missing_kwds={'color': 'white'}, legend = True, cmap = 'RdBu_r')
ax4.set_axis_off()
ax4.set_title('Averaged MACA(RCP8.5)-GridMet Yield \n Difference % over 1980-2018', fontsize = 35)
ca_county_remove_shp['coords'] = ca_county_remove_shp['geometry'].apply(lambda x: x.representative_point().coords[:])
ca_county_remove_shp['coords'] = [coords[0] for coords in ca_county_remove_shp['coords']]
for idx, row in ca_county_remove_shp.iterrows():
   ax1.annotate(row['NAME'], xy=row['coords'], horizontalalignment='center', color='black', fontsize =12)
   ax2.annotate(row['NAME'], xy=row['coords'], horizontalalignment='center', color='black', fontsize =12)
   ax4.annotate(row['NAME'], xy=row['coords'], horizontalalignment='center', color='black', fontsize =12)
fig4 = ax4.figure
fig4.axes[3].tick_params(labelsize = 35)
fig4.axes[4].tick_params(labelsize = 35)
fig4.axes[5].tick_params(labelsize = 35)
plt.savefig('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/Almond_plots/plot_hnrg/old/map_current_period_obs_gridmet_maca.png', dpi = 200)

yield_change_for_shp_45_2099_df = pd.DataFrame({'NAME' : ca.NAME, 'rcp45_2099' : yield_change_for_shp_45_2099})
yield_change_for_shp_85_2099_df = pd.DataFrame({'NAME' : ca.NAME, 'rcp85_2099' : yield_change_for_shp_85_2099})
ca_merge_rcp45_2099 =  ca.merge(yield_change_for_shp_45_2099_df, on = 'NAME')
ca_merge_rcp85_2099 =  ca.merge(yield_change_for_shp_85_2099_df, on = 'NAME')
df_county_yield_rcp45 = pd.DataFrame()
df_county_yield_rcp85 = pd.DataFrame()

for i in range(0,16):
    df_county_yield_rcp45_ind = pd.DataFrame({'County' : str(county_list[i]) , 'Yield Change % by 2099' : locals()[str(county_list[i])+'county_yield_change_2099'][:,1]})
    df_county_yield_rcp45 = df_county_yield_rcp45.append(df_county_yield_rcp45_ind)
    df_county_yield_rcp85_ind = pd.DataFrame({'County' : str(county_list[i]) , 'Yield Change % by 2099' : locals()[str(county_list[i])+'county_yield_change_2099'][:,3]})
    df_county_yield_rcp85 = df_county_yield_rcp85.append(df_county_yield_rcp85_ind)

from matplotlib import cm
norm = matplotlib.colors.Normalize(vmax = 0,vmin = -100)
fig, axes = plt.subplots(1,2, figsize=(30,12))
plt.subplot(1,2,1)
ax = ca_merge_rcp45_2099.plot(ax = axes[0], column = ca_merge_rcp45_2099.rcp45_2099,edgecolor='black',missing_kwds={'color': 'white'}, legend = True, cmap = 'OrRd_r', figsize = (15,15),vmin = -100, vmax = 0)
fig1 = ax.figure
fig1.axes[2].tick_params(labelsize = 30)
ax.set_axis_off()
fig = ax.figure
cb_ax = fig.axes[1]
cb_ax.tick_params(labelsize = 30)
ca_county_remove_shp['coords'] = ca_county_remove_shp['geometry'].apply(lambda x: x.representative_point().coords[:])
ca_county_remove_shp['coords'] = [coords[0] for coords in ca_county_remove_shp['coords']]
for idx, row in ca_county_remove_shp.iterrows():
   plt.annotate(row['N_S_order'], xy=row['coords'], horizontalalignment='center', color='black', fontsize =20)
county_order_N_S = ['Tehama', 'Butte', 'Glenn', 'Yuba', 'Colusa', 'Sutter', 'Yolo', 'Solano', 'San Joaquin', 'Stanislaus', 'Madera', 'Merced', 'Fresno', 'Tulare', 'Kings', 'Kern']
my_pal = {'Butte' : cm.OrRd_r(norm(median_yield_change_2099[0,1])), 'Colusa': cm.OrRd_r(norm(median_yield_change_2099[1,1])), 'Fresno' : cm.OrRd_r(norm(median_yield_change_2099[2,1])), 'Glenn' : cm.OrRd_r(norm(median_yield_change_2099[3,1])),
          'Kern' : cm.OrRd_r(norm(median_yield_change_2099[4,1])), 'Kings' : cm.OrRd_r(norm(median_yield_change_2099[5,1])), 'Madera' : cm.OrRd_r(norm(median_yield_change_2099[6,1])), 'Merced' : cm.OrRd_r(norm(median_yield_change_2099[7,1])),
          'San Joaquin' : cm.OrRd_r(norm(median_yield_change_2099[8,1])), 'Solano' : cm.OrRd_r(norm(median_yield_change_2099[9,1])), 'Stanislaus' : cm.OrRd_r(norm(median_yield_change_2099[10,1])), 'Sutter' : cm.OrRd_r(norm(median_yield_change_2099[11,1])),
          'Tehama' :cm.OrRd_r(norm(median_yield_change_2099[12,1])), 'Tulare' : cm.OrRd_r(norm(median_yield_change_2099[13,1])), 'Yolo' : cm.OrRd_r(norm(median_yield_change_2099[14,1])), 'Yuba': cm.OrRd_r(norm(median_yield_change_2099[15,1]))}
plt.subplot(1,2,2)
ax1 = sns.boxplot(ax = axes[1],x = 'Yield Change % by 2099', y = 'County', data = df_county_yield_rcp45,  order = county_order_N_S, palette = my_pal, showfliers = False)
plt.suptitle('County-level Yield Change % w/o Tech by 2099 under RCP 4.5', fontsize = 35)
plt.xlabel('Yield Change %', fontsize = 35)
plt.ylabel('')
plt.xticks(fontsize = 25)
ticks_county_order_N_S = ['[1]Tehama', '[2]Butte', '[3]Glenn', '[4]Yuba', '[5]Colusa', '[6]Sutter', '[7]Yolo', '[8]Solano', '[9]San Joaquin', '[10]Stanislaus', '[11]Madera', '[12]Merced', '[13]Fresno', '[14]Tulare', '[15]Kings', '[16]Kern']
plt.yticks(np.arange(0,16), ticks_county_order_N_S, fontsize = 25)
plt.tight_layout()
plt.savefig('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/Almond_plots/plot_hnrg/almond-land-avg/All_ACI/map_yield_change_rcp45_2099.png', dpi = 200)


fig, axes = plt.subplots(1,2, figsize=(30,12))
plt.subplot(1,2,1)
ax = ca_merge_rcp85_2099.plot(ax = axes[0], column = ca_merge_rcp85_2099.rcp85_2099,edgecolor='black',missing_kwds={'color': 'white'}, legend = True, cmap = 'OrRd_r',figsize = (15,15), vmin = -100, vmax = 0)
fig1 = ax.figure
fig1.axes[2].tick_params(labelsize = 30)
ax.set_axis_off()
fig = ax.figure
cb_ax = fig.axes[1]
cb_ax.tick_params(labelsize = 30)
ca_county_remove_shp['coords'] = ca_county_remove_shp['geometry'].apply(lambda x: x.representative_point().coords[:])
ca_county_remove_shp['coords'] = [coords[0] for coords in ca_county_remove_shp['coords']]
for idx, row in ca_county_remove_shp.iterrows():
   plt.annotate(row['N_S_order'], xy=row['coords'], horizontalalignment='center', color='black', fontsize =20)
my_pal = {'Butte' : cm.OrRd_r(norm(median_yield_change_2099[0,3])), 'Colusa': cm.OrRd_r(norm(median_yield_change_2099[1,3])), 'Fresno' : cm.OrRd_r(norm(median_yield_change_2099[2,3])), 'Glenn' : cm.OrRd_r(norm(median_yield_change_2099[3,3])),
          'Kern' : cm.OrRd_r(norm(median_yield_change_2099[4,3])), 'Kings' : cm.OrRd_r(norm(median_yield_change_2099[5,3])), 'Madera' : cm.OrRd_r(norm(median_yield_change_2099[6,3])), 'Merced' : cm.OrRd_r(norm(median_yield_change_2099[7,3])),
          'San Joaquin' : cm.OrRd_r(norm(median_yield_change_2099[8,3])), 'Solano' : cm.OrRd_r(norm(median_yield_change_2099[9,3])), 'Stanislaus' : cm.OrRd_r(norm(median_yield_change_2099[10,3])), 'Sutter' : cm.OrRd_r(norm(median_yield_change_2099[11,3])),
          'Tehama' :cm.OrRd_r(norm(median_yield_change_2099[12,3])), 'Tulare' : cm.OrRd_r(norm(median_yield_change_2099[13,3])), 'Yolo' : cm.OrRd_r(norm(median_yield_change_2099[14,3])), 'Yuba': cm.OrRd_r(norm(median_yield_change_2099[15,3]))}
plt.subplot(1,2,2)
ax1 = sns.boxplot(ax = axes[1], x = 'Yield Change % by 2099', y = 'County', data = df_county_yield_rcp85, order = county_order_N_S, palette = my_pal, showfliers = False)
plt.suptitle('County-level Yield Change % w/o Tech by 2099 under RCP 8.5', fontsize = 35)
plt.xlabel('Yield Change %', fontsize = 35)
plt.ylabel('')
plt.xticks(fontsize = 25)
ticks_county_order_N_S = ['[1]Tehama', '[2]Butte', '[3]Glenn', '[4]Yuba', '[5]Colusa', '[6]Sutter', '[7]Yolo', '[8]Solano', '[9]San Joaquin', '[10]Stanislaus', '[11]Madera', '[12]Merced', '[13]Fresno', '[14]Tulare', '[15]Kings', '[16]Kern']
plt.yticks(np.arange(0,16), ticks_county_order_N_S, fontsize = 25)
plt.tight_layout()
plt.savefig('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/Almond_plots/plot_hnrg/almond-land-avg/All_ACI/map_yield_change_rcp85_2099.png', dpi = 200)



    
    
##for 11/12 meeting

yield_2000_rcp45 = np.mean(yield_across_state_total_rcp45_s[50:69], axis = 0)
yield_2000_rcp85 = np.mean(yield_across_state_total_rcp85_s[50:69], axis = 0)
yield_2050_rcp45 = np.mean(yield_across_state_total_rcp45_s[89:109], axis = 0)
yield_2099_rcp45 = np.mean(yield_across_state_total_rcp45_s[129:149], axis = 0)
yield_2050_rcp85 = np.mean(yield_across_state_total_rcp85_s[89:109], axis = 0)
yield_2099_rcp85 = np.mean(yield_across_state_total_rcp85_s[129:149], axis = 0)
a = pd.DataFrame({'Year' : np.repeat('2000', 17), 'scenario' : np.repeat('RCP4.5',17) , 'Yield' : yield_2050_rcp45})
b = pd.DataFrame({'Year' : np.repeat('2000', 17), 'scenario' : np.repeat('RCP8.5',17) , 'Yield' : yield_2050_rcp85})
c = pd.DataFrame({'Year' : np.repeat('2050', 17), 'scenario' : np.repeat('RCP4.5',17) , 'Yield' : yield_2050_rcp45})
d = pd.DataFrame({'Year' : np.repeat('2099', 17), 'scenario' : np.repeat('RCP4.5',17) , 'Yield' : yield_2099_rcp45})
e = pd.DataFrame({'Year' : np.repeat('2050', 17), 'scenario' : np.repeat('RCP8.5',17) , 'Yield' : yield_2050_rcp85})
f = pd.DataFrame({'Year' : np.repeat('2099', 17), 'scenario' : np.repeat('RCP8.5',17) , 'Yield' : yield_2099_rcp85})
df_yield = a.append(b).append(c).append(d).append(e).append(f)
ax = sns.boxplot(x = 'Year', y = 'Yield', hue = 'scenario', data = df_yield, linewidth = 2.5)
plt.savefig('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/Almond_plots/boxplot_yield_only_aci.png', dpi = 200)


## hist of MACA ACI and  aci
gridmet = nc.Dataset('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lsso_Csv/MACA/Almond_data.nc')
maca_bcc_hist = nc.Dataset('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/bcc-csm1_hist_ACI.nc')
maca_bcc_rcp45 = nc.Dataset('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/bcc-csm1_rcp45_ACI.nc')
maca_bcc_rcp85 = nc.Dataset('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/bcc-csm1_rcp85_ACI.nc')

for i in range(0,29):
    if i == 27 or i == 4:
        pass
    else:
        present_gridmet = np.ndarray.flatten((gridmet.variables['ACI_value'][20:39,:,i]))
        present_maca_aci = np.ndarray.flatten((np.row_stack((maca_bcc_hist.variables['ACI_value'][:], maca_bcc_rcp45.variables['ACI_value'][:]))[50:69,:,i]))
        maca_aci_2050 = np.ndarray.flatten((np.row_stack((maca_bcc_hist.variables['ACI_value'][:], maca_bcc_rcp45.variables['ACI_value'][:]))[89:109,:,i]))
        maca_aci_2099 = np.ndarray.flatten((np.row_stack((maca_bcc_hist.variables['ACI_value'][:], maca_bcc_rcp45.variables['ACI_value'][:]))[128:148,:,i]))
        aci_total = np.concatenate((present_gridmet, present_maca_aci, maca_aci_2050, maca_aci_2099))
        scenario_a = np.repeat('Gridmet 2000-2020',304)
        scenario_b = np.repeat('Maca 2000-2020',304)
        scenario_c = np.repeat('Maca 2040-2060',320)
        scenario_d = np.repeat('Maca 2079-2099',320)
        scenario_total = np.concatenate((scenario_a, scenario_b, scenario_c, scenario_d))
        df = pd.DataFrame({'ACI value' : aci_total, 'Scenario' : scenario_total})
        sns.displot(df, x = 'ACI value', kind="kde", hue = 'Scenario', fill = True)
        plt.title(ACI_list[i])
        plt.savefig('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/Almond_plots/un_stand_ACI_dist_'+str(ACI_list[i])+'.png', dpi = 200)
        plt.show()



ACI_list = ['FallETo', 'JanPpt', 'SprGDD25', 'T10_21.1', 'SpringETo', 'WinterPpt', 'FallPpt', 'MarTmin', 'SpringPpt', 'T12.8'
            , 'T4.4', 'WinterTmean', 'FallTmean', 'SprGDD4', 'SpringTmean', 'SummerETo', 'WinterChill', 'wpd', 'FebPpt',
            'janfebSpH', 'SummerPrecp', 'FebTmin', 'A-O ETo', 'WinterETo', 'WYppt', 'SprGDD20','SummerTmean', 'T21.1-30.6', 'WinterFreeze']

gridmet = np.split(genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_csv_summary/gugu_aci_csv.csv', delimiter = ','),16)
maca_bcc_hist = np.split(genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_csv_summary_new/bcc-csm1hist_rcp45_ACI.csv', delimiter = ','),16)
maca_bcc_rcp45 = np.split(genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_csv_summary_new/bcc-csm1future_rcp45_ACI.csv', delimiter = ','),16)

present_gridmet_sum = np.zeros((0,90))
present_maca_bcc_hist_sum = np.zeros((0,90))
gridmet_sum_1980_2018 = np.zeros((0,90))
maca_bcc_hist_1980_2018_sum = np.zeros((0,90))
maca_aci_2050_sum = np.zeros((0,90))
maca_aci_2099_sum = np.zeros((0,90))

for j in range(0,16):
    gridmet_1980_2018_ind = gridmet[j]
    maca_bcc_hist_1980_2018_ind = maca_bcc_hist[j]
    present_gridmet_ind = gridmet[j][20:39]
    present_maca_bcc_hist_ind = maca_bcc_hist[j][20:39]
    maca_aci_2050_ind = maca_bcc_rcp45[j][21:42]
    maca_aci_2099_ind = maca_bcc_rcp45[j][60:81]
    gridmet_sum_1980_2018 = np.row_stack((gridmet_sum_1980_2018, gridmet_1980_2018_ind))
    maca_bcc_hist_1980_2018_sum = np.row_stack((maca_bcc_hist_1980_2018_sum, maca_bcc_hist_1980_2018_ind))
    present_gridmet_sum = np.row_stack((present_gridmet_sum,present_gridmet_ind))
    present_maca_bcc_hist_sum = np.row_stack((present_maca_bcc_hist_sum, present_maca_bcc_hist_ind))
    maca_aci_2050_sum = np.row_stack((maca_aci_2050_sum, maca_aci_2050_ind))
    maca_aci_2099_sum = np.row_stack((maca_aci_2099_sum, maca_aci_2099_ind))

time_present_gridmet_sum = np.zeros((0,91))
time_present_maca_bcc_hist_sum = np.zeros((0,91))
time_future_maca_sum = np.zeros((0,91))

for j in range(0,16):
    locals()['time_present_gridmet'+str(j)] = np.column_stack((np.arange(1980,2019,1), gridmet[j]))
    locals()['time_present_maca_bcc_hist'+str(j)] = np.column_stack((np.arange(1980,2019,1), maca_bcc_hist[j]))
    locals()['time_future_maca'+str(j)] = np.column_stack((np.arange(2019,2100,1), maca_bcc_rcp45[j]))


for i in range(0,29):
    if i == 27 or i == 4 or i == 3:
        pass
    else:
        fig, axes = plt.subplots(2, 1, figsize = (20,20))
        for j in range(0,16):
            sns.lineplot(x=locals()['time_present_gridmet'+str(j)][:,0], y=locals()['time_present_gridmet'+str(j)][:,i+1], ax = axes[0])
            sns.lineplot(x=locals()['time_present_maca_bcc_hist'+str(j)][:,0], y=locals()['time_present_maca_bcc_hist'+str(j)][:,i+1], ax = axes[0])
            sns.lineplot(x=locals()['time_future_maca'+str(j)][:,0], y=locals()['time_future_maca'+str(j)][:,i+1], ax = axes[0])
        present_gridmet = np.ndarray.flatten((present_gridmet_sum[:,i]))
        present_maca_aci = np.ndarray.flatten((present_maca_bcc_hist_sum[:,i]))
        maca_aci_2050 = np.ndarray.flatten((maca_aci_2050_sum[:,i]))
        maca_aci_2099 = np.ndarray.flatten((maca_aci_2099_sum[:,i]))
        gridmet_1980_2018 = np.ndarray.flatten((gridmet_sum_1980_2018[:,i]))
        maca_bcc_hist_1980_2018 = np.ndarray.flatten((maca_bcc_hist_1980_2018_sum[:,i]))
        aci_total = np.concatenate((present_gridmet, present_maca_aci, maca_aci_2050, maca_aci_2099, gridmet_1980_2018, maca_bcc_hist_1980_2018))
        scenario_a = np.repeat('Gridmet 2000-2020',304)
        scenario_b = np.repeat('Maca 2000-2020',304)
        scenario_c = np.repeat('Maca 2040-2060',336)
        scenario_d = np.repeat('Maca 2079-2099',336)
        scenario_e = np.repeat('Gridmet 1980-2018', 624)
        scenario_f = np.repeat('Maca 1980-2018', 624)
        scenario_total = np.concatenate((scenario_a, scenario_b, scenario_c, scenario_d, scenario_e, scenario_f))
        df = pd.DataFrame({'ACI value' : aci_total, 'Scenario' : scenario_total})
        sns.kdeplot(data = df, x = 'ACI value', hue = 'Scenario', fill = True,ax = axes[1])
        plt.suptitle(ACI_list[i])
        plt.savefig('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/Almond_plots/ACI_distribution/stand_ACI_dist_'+str(ACI_list[i])+'.png', dpi = 200)
        plt.show()

#plot aci hist
gridmet = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/gugu_aci_csv_12_3.csv', delimiter = ',')
aci_rcp45_hist_sum = np.zeros((0,90))
aci_rcp85_hist_sum = np.zeros((0,90))
aci_rcp45_future_sum = np.zeros((0,90))
aci_rcp85_future_sum = np.zeros((0,90))

for model in range(0,17):
    locals()['aci_rcp45_hist'+str(model)] = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_csv_summary_new_1_27_KDD/'+str(model_list[model])+'hist_rcp45_ACI.csv', delimiter = ',')
    locals()['aci_rcp85_hist'+str(model)] = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_csv_summary_new_1_27_KDD/'+str(model_list[model])+'hist_rcp85_ACI.csv', delimiter = ',')
    locals()['aci_rcp45'+str(model)] = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_csv_summary_new_1_27_KDD/'+str(model_list[model])+'future_rcp45_ACI.csv', delimiter = ',')
    locals()['aci_rcp85'+str(model)]  = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_csv_summary_new_1_27_KDD/'+str(model_list[model])+'future_rcp85_ACI.csv', delimiter = ',')
    aci_rcp45_hist_sum = np.row_stack((aci_rcp45_hist_sum, locals()['aci_rcp45_hist'+str(model)]))
    aci_rcp85_hist_sum = np.row_stack((aci_rcp85_hist_sum, locals()['aci_rcp85_hist'+str(model)]))
    aci_rcp45_future_sum = np.row_stack((aci_rcp45_future_sum, locals()['aci_rcp45'+str(model)]))
    aci_rcp85_future_sum = np.row_stack((aci_rcp85_future_sum, locals()['aci_rcp85'+str(model)]))
    
ACI_list = ['FallETo', 'JanPpt', 'KDD25', 'T10_21.1', 'SpringETo', 'WinterPpt', 'FallPpt', 'MarTmin', 'SpringPpt', 'T12.8'
            , 'T4.4', 'WinterTmean', 'FallTmean', 'SprGDD4', 'SpringTmean', 'SummerETo', 'WinterChill', 'wpd', 'FebPpt',
            'janfebSpH', 'SummerPrecp', 'FebTmin', 'A_O_ETo', 'WinterETo', 'WYppt', 'KDD30','SummerTmean', 'T21.1-30.6', 'WinterFreeze']
for i in range(0,29):  
    plt.figure(figsize = (20,20))
    plt.subplot(2,3,1)
    plt.title('GridMet RCP4.5 1980-2018', fontsize = 20)
    #plt.hist(gridmet[:,i], 100, edgecolor = 'b')    
    plt.subplot(2,3,2)
    plt.title('RCP4.5 1980-2018', fontsize = 20)
    plt.hist(aci_rcp45_hist_sum[:,i], 100, edgecolor = 'b')
    plt.subplot(2,3,3)
    plt.title('RCP8.5 1980-2018', fontsize = 20)
    plt.hist(aci_rcp85_hist_sum[:,i], 100, edgecolor = 'b')
    plt.subplot(2,3,4)
    plt.title('RCP4.5 2019-2099', fontsize = 20)
    plt.hist(aci_rcp45_future_sum[:,i], 100, edgecolor = 'b')
    plt.subplot(2,3,5)
    plt.title('RCP8.5 2019-2099', fontsize = 20)
    plt.hist(aci_rcp85_future_sum[:,i], 100, edgecolor = 'b')
    plt.suptitle(str(ACI_list[i]), fontsize = 25)
    #plt.savefig('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/Almond_plots/ACI_distribution/hist_ACI_'+str(ACI_list[i])+'.png',dpi = 200)
    plt.show()
    
##plot dist of 80-99 and scatter of gridmet vs yield
yield_csv = np.delete(yield_csv, 0,axis = 1) ## delete year col
yield_csv = np.ndarray.flatten(yield_csv,order = 'F')
year_list = np.ndarray.flatten(np.tile(np.arange(1980,2019,1),(16,1)))
cm = plt.cm.get_cmap('viridis')
##get index for year2080-2099
future_year_index = np.arange(61,81)
for i in range(1,272):
    future_year_index = np.concatenate((future_year_index, np.arange(61,81)+(i*81)))

for i in range(0,29):
    plt.figure(figsize = (25,20))
    plt.title(ACI_list[i])
    plt.scatter(x = gridmet[:,i], y = signal.detrend(yield_csv),c = year_list, cmap = cm, s = 100)
    plt.colorbar(orientation = 'horizontal')
    plt.xlabel('Standardized ACI', fontsize = 20)
    plt.ylabel('detrended Yield', fontsize =20)
    plt.yticks(fontsize = 18)
    plt.xticks(fontsize = 18)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2 = sns.kdeplot(aci_rcp85_future_sum[future_year_index,i], fill = True, color = 'r')
    plt.ylabel('RCP8.5 2080-2099 ACI Density.', fontsize = 20, color = 'r', rotation = 270, labelpad = 7)
    plt.yticks(fontsize = '18', color='r')
    plt.savefig('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/Almond_plots/ACI_distribution/yield_vs_gridmet_maca_'+str(ACI_list[i])+'.png',dpi = 200)
    plt.show()
## plot coef box plot
index_aci_0_median = np.column_stack((np.array(np.where(np.median(coef_sum,axis=0)[0:29]==0)),np.array(np.where(np.median(coef_sum,axis=0)[0:29]==0))+29))
coef_sum_remove_0_median = np.delete(coef_sum, index_aci_0_median, axis = 1)
ACI_list_remove_0_median = np.delete(ACI_list, np.where(np.median(coef_sum,axis=0)[0:29]==0))
df_coef = pd.DataFrame()
for i in range(0,ACI_list_remove_0_median.shape[0]):
    locals()['df_coef_'+str(i)] = pd.DataFrame({'aci' : np.repeat(ACI_list_remove_0_median[i], 1000), 'coef' : coef_sum_remove_0_median[:,i]})
    df_coef = df_coef.append(locals()['df_coef_'+str(i)])
df_coef_sq = pd.DataFrame()
for i in range(0,ACI_list_remove_0_median.shape[0]):
    locals()['df_coef_sq'+str(i)] = pd.DataFrame({'aci' : np.repeat(ACI_list_remove_0_median[i], 1000), 'coef' : coef_sum_remove_0_median[:,i+ACI_list_remove_0_median.shape[0]]})
    df_coef_sq = df_coef_sq.append(locals()['df_coef_sq'+str(i)])

df_coef_tech = pd.DataFrame()
for i in range(0,16):
    locals()['df_coef_tech'+str(i)] = pd.DataFrame({'county' : np.repeat(county_list[i], 1000), 'coef' : coef_sum_remove_0_median[:,i+58]})
    df_coef_tech = df_coef_tech.append(locals()['df_coef_tech'+str(i)])

df_coef_fixed = pd.DataFrame()
for i in range(0,16):
    locals()['df_coef_fixed'+str(i)] = pd.DataFrame({'county' : np.repeat(county_list[i], 1000), 'coef' : coef_sum_remove_0_median[:,i+74]})
    df_coef_fixed = df_coef_fixed.append(locals()['df_coef_fixed'+str(i)])

## get random color name
box_color = sns.color_palette('tab20', n_colors = ACI_list_remove_0_median.shape[0])
sns.set_palette(sns.color_palette('tab20', n_colors = ACI_list_remove_0_median.shape[0])) 

plt.figure(figsize = (25,20))
plt.suptitle('Lasso-modeled ACI Coefficients', fontsize = 28, y=0.94)
plt.subplot(2,1,1)
ax = sns.boxplot(x = 'aci', y = 'coef', data = df_coef)
plt.tick_params(axis = 'x', left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
plt.tick_params(axis = 'y', labelsize = 25)
plt.title('ACI Coef.', fontsize = 25)
plt.ylabel('value', fontsize = 25)
plt.subplot(2,1,2)
ax = sns.boxplot(x = 'aci', y = 'coef', data = df_coef_sq)
for (aci, color) in zip(ax.xaxis.get_ticklabels(), box_color):
    aci.set_color(color)
plt.tick_params(axis = 'x', labelrotation = 60, labelsize = 25)
plt.tick_params(axis = 'y', labelsize = 25)
plt.title(r'$ACI^2$ Coef.', fontsize = 25)
plt.xlabel('ACI list', fontsize = 25)
plt.ylabel('value', fontsize = 25)
plt.savefig('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/Almond_plots/plot_hnrg/ACI_coef_boxplot_2_17.png', dpi = 200)

plt.figure(figsize = (25,20))
plt.subplot(2,1,1)
ax = sns.boxplot(x = 'county', y = 'coef', data = df_coef_tech)
ax.tick_params(axis = 'x', left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
ax.tick_params(axis = 'y', labelsize = 18)
plt.title('Tech Trend Coef.', fontsize = 25)
plt.ylabel('value', fontsize = 20)
plt.subplot(2,1,2)
ax = sns.boxplot(x = 'county', y = 'coef', data = df_coef_fixed)
ax.tick_params(axis = 'x', labelrotation = 60, labelsize = 18)
ax.tick_params(axis = 'y', labelsize = 18)
plt.title('County Fixed Coef.', fontsize = 25)
plt.xlabel('County', fontsize = 20)
plt.ylabel('value', fontsize = 20)
plt.savefig('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/Almond_plots/County_coef_boxplot.png', dpi = 200)




##compare maca gridmet observed historical yield
gridmet = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_csv_summary/gugu_aci_csv.csv', delimiter = ',')
simulation_gridmet = np.zeros((624,1000))
for trial in range(1,11):
    for i in range(0,624):
        for j in range(0,101):
            simulation_gridmet[i,j+((trial-1)*101)] = np.nansum(gridmet[i,:]*locals()['coef'+str(trial)][j,:])


production_average_model = np.zeros((624,1000))
for index in range(0,16):
    for year in range(0,39):
        production_average_model[index*39+year,:] = simulation_gridmet[index*39+year,:]*area[year,index]

production_average_model_split = np.split(production_average_model,16) 
production_average_model_across_state_hist = np.zeros((39,1000))
for county_id in range(0,16):
    for year in range(0,39):
        production_average_model_across_state_hist[year,:] = production_average_model_across_state_hist[year,:]+production_average_model_split[county_id][year,:]
gridmet_yield = np.zeros((39,1000))
for year in range(0,39):
    gridmet_yield[year,:] = production_average_model_across_state_hist[year,:]/np.sum(area[year])


observed_yield = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_csv_summary/Almond_yieldTonacre_box.csv', delimiter = ',')
observed_yield = np.ndarray.flatten((observed_yield)[:,1:17])
production_average_model = np.zeros((624))
for index in range(0,16):
    for year in range(0,39):
        production_average_model[index*39+year] = observed_yield[index*39+year]*area[year,index]
production_average_model_split = np.split(production_average_model,16) 
production_average_model_across_state_hist = np.zeros((39))
for county_id in range(0,16):
    for year in range(0,39):
        production_average_model_across_state_hist[year] = production_average_model_across_state_hist[year]+production_average_model_split[county_id][year]
observed_yield = np.zeros((39))
for year in range(0,39):
    observed_yield[year] = production_average_model_across_state_hist[year]/np.sum(area[year])

year_list = np.arange(1980,2019)
plt.figure(figsize = (20,30))
plt.subplot(3,1,1)
plt.plot(observed_yield, color = 'b')
plt.title('Observed CA weighted yield 1980-2018', fontsize = 18)
plt.subplot(3,1,2)
plt.plot(np.mean(gridmet_yield,axis = 1), color = 'r')
plt.title('Gridmet CA weighted yield 1980-2018', fontsize = 18)
plt.subplot(3,1,3)
plt.plot(np.mean(yield_across_state_hist_rcp85_s,axis = 1), color = 'g')
plt.title('Gridmet CA weighted yield 1980-2018', fontsize = 18)


for i in range(0,58):
    for j in range(0,1000):
        if coef_sum[j,i] == 0:
            coef_sum[:,i] = 0


##re-order aci order
ACI_list_old = ['FallETo', 'JanPpt', 'KDD25', 'T10_21.1', 'SpringETo', 'WinterPpt', 'FallPpt', 'MarTmin', 'SpringPpt', 'T12.8'
            , 'T4.4', 'WinterTmean', 'FallTmean', 'SprGDD4', 'SpringTmean', 'SummerETo', 'WinterChill', 'wpd', 'FebPpt',
            'janfebSpH', 'SummerPpt', 'FebTmin', 'A_O_ETo', 'WinterETo', 'WYPrecp', 'KDD30','SummerTmean', 'T21.1-30.6', 'WinterFreeze']


ACI_list = ['SpringTmean', 'SummerTmean', 'FallTmean', 'WinterTmean', 'FebTmin', 'MarTmin', 'SprGDD4', 'KDD25', 'KDD30', 'WinterFreeze', 'WinterChill_new', 'JanPpt', 'FebPpt'
            , 'SpringPpt', 'SummerPrecp', 'FallPpt', 'WinterPpt', 'WYppt', 'SpringETo', 'SummerETo', 'FallETo', 'WinterETo', 'A_O_ETo', 'T4.4', 'T12.8', 'GDD10_21', 'T21.1-30.6'
            , 'janfebSpH', 'wpd']


gridmet_reorder = np.zeros((624,58))
for i in range(0,29):
    for j in range(0,29):
        if ACI_list_new[i] == ACI_list_old[j]:
            gridmet_reorder[:,i] = gridmet[:,j]
            gridmet_reorder[:,i+29] = gridmet[:,j+29]

##correlation matrix
cor_matrix = np.zeros((29,29))
for i in range(0,29):
    for j in range(0,29):
        cor_matrix[i,j] = np.corrcoef(gridmet_reorder[:,i], gridmet_reorder[:,j])[0,1]
plt.figure(figsize = (24,20))
ax = sns.heatmap(cor_matrix, cmap = 'coolwarm', linewidth = 0.3,vmin = -1,vmax = 1, annot = True)
ax.set_yticklabels(ACI_list_new, rotation = 0,fontsize = 18)
ax.set_xticklabels(ACI_list_new, rotation = 90,fontsize = 18)
plt.title('GridMet ACI Correlation Matrix', fontsize = 20)
plt.savefig('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/Almond_plots/cor_matrix_gridmet.png', dpi = 300)

##rank ACI based on square sum
aci_df = pd.DataFrame({'ACI' : ACI_list, 'square_sum':np.sum(coef_sum**2, axis = 0)[0:29], 'frequency of zero' : np.count_nonzero(coef_sum[:,0:29]==0, axis = 0)})
aci_df['square_sum'].rank(method = 'min')
aci_df.sort_values(by = 'square_sum', ascending = False)
aci_df.sort_values(by = 'frequency of zero')


## plot aci contribution
for model in range(0,17):
    np.split(yield_all_future_rcp85_s,17,1)[model]

area_share = area/np.sum(area) ## calculate area ratio of each county
coef_sum_median = np.median(coef_sum,axis=0)
aci_sum_mean_45 = np.mean(aci_rcp45_s_sum,axis = 0)
aci_sum_mean_85 = np.mean(aci_rcp85_s_sum,axis = 0)

aci_contribution_45 = np.zeros((1296,90))
aci_contribution_85 = np.zeros((1296,90))

for i in range(0,1296):
    aci_contribution_45[i,:] = coef_sum_median * aci_sum_mean_45[i,:]*area_share[np.int(i/81)]
    aci_contribution_85[i,:] = coef_sum_median * aci_sum_mean_85[i,:]*area_share[np.int(i/81)]

aci_contribution_45_split = np.split(aci_contribution_45,16)
aci_contribution_85_split = np.split(aci_contribution_85,16)

aci_contribution_sum_45 = np.zeros((81,29))
aci_contribution_sum_85 = np.zeros((81,29))
for i in range(0,16):
    aci_contribution_sum_45 = aci_contribution_sum_45 + aci_contribution_45_split[i][:,0:29]+aci_contribution_45_split[i][:,29:58]
    aci_contribution_sum_85 = aci_contribution_sum_85 + aci_contribution_85_split[i][:,0:29]+aci_contribution_85_split[i][:,29:58]

aci_contribution_sum_45 = np.transpose(aci_contribution_sum_45)
aci_contribution_sum_85 = np.transpose(aci_contribution_sum_85)

import matplotlib.colors as colors
import random

colors_list = list(colors._colors_full_map.values())

x = np.arange(2019,2100)
fig, ax = plt.subplots(1,1,figsize=(16, 9))
ax.stackplot(x , np.delete(aci_contribution_sum_45,  [6,10,27,17,24,26,23,19,18,16], axis = 0), labels = np.delete(ACI_list,  [6,10,27,17,24,26,23,19,18,16]), colors = random.sample(colors_list,20))
ax.legend(fontsize=10, ncol=4, loc = 'lower left')
plt.savefig('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/Almond_plots/ACI_stack_rcp45.png', dpi = 200)

fig, ax = plt.subplots(1,1,figsize=(16, 9))
ax.stackplot(x , np.delete(aci_contribution_sum_85,  [6,10,27,17,24,26,23,19,18,16], axis = 0), labels = np.delete(ACI_list,  [6,10,27,17,24,26,23,19,18,16]), colors = random.sample(colors_list,20))
ax.legend(fontsize=10, ncol=4, loc = 'lower left')
plt.savefig('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/Almond_plots/ACI_stack_rcp85.png', dpi = 200)

## plot scatter plot of gridmet aci vs yiield (only aci contributuion)
gridmet = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_csv_summary/gugu_aci_csv_2_5.csv', delimiter = ',')
yield_csv = np.delete(yield_csv, 0, axis = 1)
yield_csv = scipy.signal.detrend(yield_csv,axis = 1)
yield_csv = np.ndarray.flatten(yield_csv)
for i in range(0,29):
    plt.figure(figsize = (15,25))
    plt.subplot(2,1,1)
    plt.scatter(gridmet[:,i], yield_csv)
    plt.title('ACI v.s Yield')
    plt.subplot(2,1,2)
    plt.scatter(gridmet[:,i+29], yield_csv)
    plt.title('ACI^2 v.s Yield')
    plt.suptitle(str(ACI_list[i]), y =0.9)
    plt.show()
    

##plot training and testing R2
R2_test_sum = np.zeros((0))
for i in range(1,11):
    R2_test = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/data_4_23/score_test_'+str(i)+'.csv', delimiter = ',')
    R2_test_sum = np.concatenate((R2_test_sum, R2_test))

R2_train_sum = np.zeros((0))
for i in range(1,11):
    R2_train = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/data_4_23/score_train_'+str(i)+'.csv', delimiter = ',')
    R2_train_sum = np.concatenate((R2_train_sum, R2_train))

R2_total_sum = np.zeros((0))
for i in range(1,11):
    R2 = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/data_4_23/score_'+str(i)+'.csv', delimiter = ',')
    R2_total_sum = np.concatenate((R2_total_sum, R2))
    
R2 = np.column_stack((R2_train_sum, R2_test_sum, R2_total_sum))
labels = ['training set', 'testing set', 'total']
colors = ['yellow', 'goldenrod', 'darkorange']
fig, (ax0,ax1,ax2) = plt.subplots(1,3,gridspec_kw={'width_ratios': [1,2,2]}, figsize = (35,12))
plt.suptitle('Statiscal model (Lasso) performance', fontsize = 35)
box = ax0.boxplot(R2, patch_artist= True, labels = labels, boxprops={'linewidth' : 2}, whiskerprops={'linewidth' : 3},capprops={'linewidth' : 3}, medianprops={'color' : 'black', 'linewidth' : 3}, widths = 0.7,showfliers=False)
ax0.set_ylabel(r'$R^2$', fontsize=35)
ax0.tick_params(axis='y', which='major', labelsize=35)
ax0.tick_params(axis='x', which='major', labelsize=35, rotation = 60)
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set(linewidth = 2)
ax1 = plt.subplot(1,3,2)
ax1.plot(np.arange(1980,2019,1),yield_observed_state[0:39], label = 'Actual Yield',linewidth = 4, color = 'green')
t = np.arange(1980,2019)
tsplot(t,np.transpose(yield_gridmet_state[0:39]), color = 'darkorange')
plt.plot(np.arange(1980,2019,1),np.median(yield_gridmet_state[0:39],axis=1), label = 'GridMet-modeled median', color = 'darkorange', linestyle = 'solid')
plt.xticks(np.arange(1980,2021,4),np.arange(1980,2021,4), fontsize = 35, rotation = 60)
plt.yticks(fontsize = 35)
plt.ylim(0,1.6)
plt.xlim(1980,2021)
plt.ylabel('Yield ton/acre', fontsize = 35)
darkorange_patch = mpatches.Patch(color = 'darkorange',label = 'GridMet')
plt.legend(handles = [darkorange_patch, Line2D([0], [0], color='green', lw=4, label='Observed')],loc = 'upper left', fontsize = 35)
#plt.title('CA area-weighted almond yield', fontsize = 35, x = 1)
box_legend = np.zeros((1000,3))
box_legend[:,0] = np.random.normal(1.4,0.06,size = (1000))
box_legend[:,1] = box_legend[:,0]
box_legend[:,2] = box_legend[:,0]
tsplot(np.arange(1999,2002), box_legend,color = 'darkorange')
plt.text(x = 2002,y = 1.27, s='95% CI', fontsize = 28)
plt.text(x = 2002,y = 1.33, s='67% CI', fontsize = 28)
plt.text(x = 2002,y = 1.39, s='Median', fontsize = 28)
plt.plot(np.arange(1999,2002),np.median(box_legend,axis=0), color = 'darkorange', linestyle = 'solid', linewidth = 4)
ax2 = plt.subplot(1,3,3)
ax2.plot(np.arange(1980,2019), np.median(yield_all_hist_rcp45, axis=1) , color = 'black', linewidth =4)
tsplot(np.arange(1980,2019), np.transpose(yield_all_hist_rcp45) , color = 'grey')
ax2.plot(np.arange(1980,2019,1),yield_observed_state[0:39], label = 'Actual Yield',linewidth = 4, color = 'green')
plt.xticks(np.arange(1980,2021,4),np.arange(1980,2021,4), fontsize = 35, rotation = 60)
plt.yticks(fontsize = 35)
plt.ylim(0,1.6)
grey_patch = mpatches.Patch(color = 'grey',label = 'MACA')
plt.legend(handles = [grey_patch, Line2D([0], [0], color='green', lw=4, label='Observed')],loc = 'upper left', fontsize = 35)
box_legend = np.zeros((1000,3))
box_legend[:,0] = np.random.normal(1.4,0.06,size = (1000))
box_legend[:,1] = box_legend[:,0]
box_legend[:,2] = box_legend[:,0]
tsplot(np.arange(1999,2002), box_legend,color = 'grey')
plt.text(x = 2002,y = 1.27, s='95% CI', fontsize = 28)
plt.text(x = 2002,y = 1.33, s='67% CI', fontsize = 28)
plt.text(x = 2002,y = 1.39, s='Median', fontsize = 28)
plt.plot(np.arange(1999,2002),np.median(box_legend,axis=0), color = 'black', linestyle = 'solid', linewidth = 4)
plt.ylabel('Yield ton/acre', fontsize = 35)
plt.xlim(1980,2021)
plt.tight_layout()
plt.savefig('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/Almond_plots/plot_hnrg/almond-land-avg/All_ACI/lasso_performance.png', dpi = 200)

## remove 10 yr trend
ten_yr_trend_ca = np.zeros((1000))
for i in range(0,1000):
    ten_yr_trend_ca[i] = np.sum(coef_sum[i,aci_num*2:aci_num*2+16]*area_csv[38,1:]/np.sum(area_csv[38,1:]))
ten_yr_trend_ca_average_model = np.zeros((81,1000))
for i in range(0,81):
    ten_yr_trend_ca_average_model[i,:]= ten_yr_trend_ca
ten_yr_trend_ca_total = np.zeros((81,17000))
for i in range(0,81):
    for j in range(0,17):
        ten_yr_trend_ca_total[i, 1000*j:1000*(j+1)] = ten_yr_trend_ca


##plot time-series aci
aci_rcp45_hist_sum = np.zeros((17,656,aci_num))
aci_rcp85_hist_sum = np.zeros((17,656,aci_num))

for model in range(0,17):
    aci_rcp45 = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_6_19/to_2020/'+str(model_list[model])+'hist_rcp45_ACI.csv', delimiter = ',')[:,0:aci_num]
    aci_rcp85 = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_6_19/to_2020/'+str(model_list[model])+'hist_rcp85_ACI.csv', delimiter = ',')[:,0:aci_num]
    aci_rcp45_hist_sum[model,:,:] = aci_rcp45
    aci_rcp85_hist_sum[model,:,:] = aci_rcp85

aci_rcp45_future_sum = np.zeros((17,1264,aci_num))
aci_rcp85_future_sum = np.zeros((17,1264,aci_num))

for model in range(0,17):
    aci_rcp45 = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_6_19/to_2020/'+str(model_list[model])+'future_rcp45_ACI.csv', delimiter = ',')[:,0:aci_num]
    aci_rcp85 = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_6_19/to_2020/'+str(model_list[model])+'future_rcp85_ACI.csv', delimiter = ',')[:,0:aci_num]
    aci_rcp45_future_sum[model,:,:] = aci_rcp45
    aci_rcp85_future_sum[model,:,:] = aci_rcp85

aci_rcp45_hist_sum_ca = np.zeros((17,41,aci_num))
aci_rcp85_hist_sum_ca = np.zeros((17,41,aci_num))
aci_rcp45_future_sum_ca = np.zeros((17,79,aci_num))
aci_rcp85_future_sum_ca = np.zeros((17,79,aci_num))

area = area_csv[0:41,1:]
for model in range(0,17):
    for index in range(0,16):
        for year in range(0,41):
            aci_rcp45_hist_sum_ca[model, year, :] = aci_rcp45_hist_sum_ca[model, year, :] + (np.split(aci_rcp45_hist_sum[model], 16)[index][year])*area[year,index]/np.sum(area[-1])
            aci_rcp85_hist_sum_ca[model, year, :] = aci_rcp85_hist_sum_ca[model, year, :] + (np.split(aci_rcp85_hist_sum[model], 16)[index][year])*area[year,index]/np.sum(area[-1])

for model in range(0,17):
    for index in range(0,16):
        aci_rcp45_future_sum_ca[model, :, :] = aci_rcp45_future_sum_ca[model, :, :] + (np.split(aci_rcp45_future_sum[model], 16)[index])*area[-1,index]/np.sum(area[-1])
        aci_rcp85_future_sum_ca[model, :, :] = aci_rcp85_future_sum_ca[model, :, :] + (np.split(aci_rcp85_future_sum[model], 16)[index])*area[-1,index]/np.sum(area[-1])

aci_rcp_45_sum_ca = np.concatenate((aci_rcp45_hist_sum_ca, aci_rcp45_future_sum_ca), axis=1)
aci_rcp_85_sum_ca = np.concatenate((aci_rcp85_hist_sum_ca, aci_rcp85_future_sum_ca), axis=1)

plt.figure(figsize = (60,40))
for i in range(0,aci_num):
    plt.subplot(4,4,i+1)
    plt.plot(np.arange(1980,2100),np.median(aci_rcp_45_sum_ca, axis=0)[:,i], color = 'r', label = 'rcp 4.5')
    plt.plot(np.arange(1980,2100),np.median(aci_rcp_85_sum_ca, axis=0)[:,i], color = 'b', label = 'rcp 8.5')
    plt.title(str(ACI_list[i]), fontsize = 30)
    plt.legend(fontsize = 25)
plt.savefig('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/Almond_plots/plot_hnrg/almond-land-avg/Growth_stage_ACI_6_19/aci_time_series.png', dpi = 200)


## plot aci change map
aci_rcp45_hist_sum = np.zeros((17,624,aci_num))
aci_rcp85_hist_sum = np.zeros((17,624,aci_num))

for model in range(0,17):
    aci_rcp45 = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_5_18/'+str(model_list[model])+'hist_rcp45_ACI.csv', delimiter = ',')[:,0:aci_num]
    aci_rcp85 = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_5_18/'+str(model_list[model])+'hist_rcp85_ACI.csv', delimiter = ',')[:,0:aci_num]
    aci_rcp45_hist_sum[model,:,:] = aci_rcp45
    aci_rcp85_hist_sum[model,:,:] = aci_rcp85

aci_rcp45_future_sum = np.zeros((17,1296,aci_num))
aci_rcp85_future_sum = np.zeros((17,1296,aci_num))

for model in range(0,17):
    aci_rcp45 = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_5_18/'+str(model_list[model])+'future_rcp45_ACI.csv', delimiter = ',')[:,0:aci_num]
    aci_rcp85 = genfromtxt('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/lasso_Csv/MACA/ACI_csv_summary/ACI_5_18/'+str(model_list[model])+'future_rcp85_ACI.csv', delimiter = ',')[:,0:aci_num]
    aci_rcp45_future_sum[model,:,:] = aci_rcp45
    aci_rcp85_future_sum[model,:,:] = aci_rcp85

aci_rcp45_hist_sum_county = np.zeros((17,39,16,aci_num))
aci_rcp85_hist_sum_county = np.zeros((17,39,16,aci_num))
aci_rcp45_future_sum_county = np.zeros((17,81,16,aci_num))
aci_rcp85_future_sum_county = np.zeros((17,81,16,aci_num))

for model in range(0,17):
    for county in range(0,16):
        aci_rcp45_hist_sum_county[model, :,county, :] = np.split(aci_rcp45_hist_sum[model],16)[county]
        aci_rcp85_hist_sum_county[model, :,county, :] = np.split(aci_rcp85_hist_sum[model],16)[county]

for model in range(0,17):
    for county in range(0,16):
        aci_rcp45_future_sum_county[model, :,county, :] = np.split(aci_rcp45_future_sum[model],16)[county]
        aci_rcp85_future_sum_county[model, :,county, :]= np.split(aci_rcp85_future_sum[model],16)[county]

aci_rcp45_hist_sum_county_mean = np.mean(aci_rcp45_hist_sum_county, axis=0)
aci_rcp85_hist_sum_county_mean = np.mean(aci_rcp85_hist_sum_county, axis=0)
aci_rcp45_future_sum_county_mean = np.mean(aci_rcp45_future_sum_county, axis=0)
aci_rcp85_future_sum_county_mean = np.mean(aci_rcp85_future_sum_county, axis=0)

aci_rcp45_sum_county = np.row_stack((aci_rcp45_hist_sum_county_mean,aci_rcp45_future_sum_county_mean))
aci_rcp85_sum_county = np.row_stack((aci_rcp85_hist_sum_county_mean,aci_rcp85_future_sum_county_mean))

aci_rcp45_sum_county_2000_2020 = np.median(aci_rcp45_sum_county[21:41],axis=0)
aci_rcp85_sum_county_2000_2020 = np.median(aci_rcp85_sum_county[21:41],axis=0)
aci_rcp45_sum_county_2041_2060 = np.median(aci_rcp45_sum_county[61:81],axis=0)
aci_rcp85_sum_county_2041_2060 = np.median(aci_rcp85_sum_county[61:81],axis=0)
aci_rcp45_sum_county_2080_2099 = np.median(aci_rcp45_sum_county[100:120],axis=0)
aci_rcp85_sum_county_2080_2099 = np.median(aci_rcp85_sum_county[100:120],axis=0)

aci_rcp45_sum_county_2050_change_percent = 100*(aci_rcp45_sum_county_2041_2060-aci_rcp45_sum_county_2000_2020)/aci_rcp45_sum_county_2000_2020
aci_rcp85_sum_county_2050_change_percent = 100*(aci_rcp85_sum_county_2041_2060-aci_rcp85_sum_county_2000_2020)/aci_rcp85_sum_county_2000_2020
aci_rcp45_sum_county_2090_change_percent = 100*(aci_rcp45_sum_county_2080_2099-aci_rcp45_sum_county_2000_2020)/aci_rcp45_sum_county_2000_2020
aci_rcp85_sum_county_2090_change_percent = 100*(aci_rcp85_sum_county_2080_2099-aci_rcp85_sum_county_2000_2020)/aci_rcp85_sum_county_2000_2020

for aci in range(0,aci_num):
    locals()[str(aci)+'aci_shp_change_2090_rcp45'] = np.zeros((58))
    locals()[str(aci)+'aci_shp_change_2090_rcp45'][:] = np.nan
    locals()[str(aci)+'aci_shp_change_2090_rcp85'] = np.zeros((58))
    locals()[str(aci)+'aci_shp_change_2090_rcp85'][:] = np.nan
    for i in range(0,58):
        for index in range(0,16):
            if county_list[index] == ca.NAME[i]:
                locals()[str(aci)+'aci_shp_change_2090_rcp45'][i] = aci_rcp45_sum_county_2090_change_percent[index,aci]
                locals()[str(aci)+'aci_shp_change_2090_rcp85'][i] = aci_rcp85_sum_county_2090_change_percent[index,aci]

df_aci_shp_change_2090_rcp45 = pd.DataFrame({'NAME' : ca.NAME})
df_aci_shp_change_2090_rcp85 = pd.DataFrame({'NAME' : ca.NAME})
for aci in range(0,aci_num):
    df_aci_shp_change_2090_rcp45[str(ACI_list[aci])] = locals()[str(aci)+'aci_shp_change_2090_rcp45']
    df_aci_shp_change_2090_rcp85[str(ACI_list[aci])] = locals()[str(aci)+'aci_shp_change_2090_rcp85']




## plot tech coef map
county_list = ['Butte', 'Colusa', 'Fresno', 'Glenn', 'Kern', 'Kings', 'Madera', 'Merced', 'San Joaquin', 'Solano', 'Stanislaus', 'Sutter', 'Tehama', 'Tulare', 'Yolo', 'Yuba']

tech_coef = np.median(coef_sum[:,aci_num*2:aci_num*2+16],axis=0)
fixed_coef = np.median(coef_sum[:,aci_num*2+16:aci_num*2+32],axis=0)

tech_coef_sum = np.zeros((58))
tech_coef_sum[:] = np.nan
fixed_coef_sum = np.zeros((58))
fixed_coef_sum[:] = np.nan    
for i in range(0,58):
    for index in range(0,16):
        if county_list[index] == ca.NAME[i]:
            tech_coef_sum[i] = tech_coef[index]
            fixed_coef_sum[i] = fixed_coef[index]

df_tech_coef = pd.DataFrame({'NAME' : ca.NAME, 'tech_coef' : tech_coef_sum})
df_fixed_coef = pd.DataFrame({'NAME' : ca.NAME, 'fixed_coef' : fixed_coef_sum})
ca_tech_coef = ca.merge(df_tech_coef, on = 'NAME')
ca_fixed_coef = ca.merge(df_fixed_coef, on = 'NAME')

fig, axes = plt.subplots(1,2, figsize=(30,12))
ax1 = ca_tech_coef.plot(ax = axes[0], column = ca_tech_coef.tech_coef, edgecolor='black',missing_kwds={'color': 'white'}, legend = True, cmap = 'Purples', figsize = (15,15), vmin = 0.002, vmax = 0.02)
ax1.set_axis_off()
ax1.set_title('Coefficient of Tech Trend (ton/acre-year)', fontsize = 35)
ax2 = ca_fixed_coef.plot(ax = axes[1], column = ca_fixed_coef.fixed_coef, edgecolor='black',missing_kwds={'color': 'white'}, legend = True, cmap = 'Greens', figsize = (15,15), vmin = 0, vmax = 0.8)
ax2.set_axis_off()
ax2.set_title('Fixed Constant (ton/acre)', fontsize =35)
fig1 = ax1.figure
fig1.axes[2].tick_params(labelsize = 35)
fig1.axes[3].tick_params(labelsize = 35)
fig1.axes[2].set_yticks([0.002, 0.008,0.014,0.02])
fig1.axes[2].set_yticklabels(['0.002', '0.008', '0.014','0.020'])
fig1.axes[3].set_yticks([0, 0.2,0.4,0.6, 0.8])
fig1.axes[3].set_yticklabels(['0', '0.2', '0.4','0.6','0.8'])
plt.savefig('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/Almond_plots/plot_hnrg/almond-land-avg/Growth_stage_ACI_6_19/tech_fixed_coef.png', dpi = 200)


##corr matrix
cor_gm = pd.DataFrame(gridmet[:,13:26], columns = ACI_list) 
plt.figure(figsize=(20, 16))
heatmap = sns.heatmap(cor_gm.corr(), annot =True,cmap='BrBG', vmin = -1, vmax =1, annot_kws={'fontsize': 18})
sns.set(font_scale=3.5)
plt.tight_layout()
plt.savefig('C:/Users/Pancake/Box/UCDGlobalChange/shqwu/QE-presentation/aci_cor_matrix.png', dpi = 200)