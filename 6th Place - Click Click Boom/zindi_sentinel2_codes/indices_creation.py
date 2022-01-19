import pandas as pd
import numpy as np
import math

def get_band_ndvi_red(df,dates):

  for date in dates:
    b8 = df['B08_'+date]
    b4 = df['B04_'+date]
    df[date+'_NDVI_red']           = ((b8-b4)/(b8+b4))
  return df

def get_band_afri(df,dates):
  for date in dates:
    b8  = df['B08_'+date]
    b11 = df['B11_'+date]
    df[date+'_AFRI'] = b8 - 0.66*( b11 / (b8 + 0.66 * b11))
  return df

def get_band_evi2(df,dates):
  for date in dates:
    b8  = df['B08_'+date]
    b4  = df['B04_'+date]
    df[date+'_EVI2'] = 2.4*(b8-b4)/(b8+b4+1.0)
  return df

def get_band_ndmi(df,dates):
  for date in dates:
    b8  = df['B08_'+date]
    b11 = df['B11_'+date]
    df[date+'_NDMI'] = (b8-b11)/(b8+b11)
  return df


def get_band_ndvi(df,dates):
  for date in dates:
    b8 = df['B08_'+date]
    b12 = df['B12_'+date]
    df[date+'_NDVI'] = ((b12-b8)/(b12+b8))
  return df

def get_band_evi(df,dates):
  for date in dates:
    b8 = df['B08_'+date]
    b4 = df['B04_'+date]
    b2 = df['B02_'+date]
    df[date+'_EVI'] = 2.5 * (b8 - b4) / ((b8 + 6.0 * b4 - 7.5 * b2) + 1.0)
  return df

def get_band_bndvi(df,dates):
  for date in dates:
    b8 = df['B08_'+date]
    b2 = df['B02_'+date]
    df[date+'_BNDVI'] = (b8-b2)/(b8+b2)
  return df

def get_band_nli(df,dates):
  for date in dates:
    b8 = df['B08_'+date]
    b4 = df['B04_'+date]
    df[date+'_NLI'] = ((b8**2)-b4)/((b8**2)+b4)
  return df

def get_band_lci(df,dates):
  for date in dates:
    b8 = df['B08_'+date]
    b4 = df['B04_'+date]
    b5 = df['B05_'+date]
    df[date+'_LCI'] =(b8 - b5) / (b8 + b4)
  return df