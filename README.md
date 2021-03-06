## RSMAS InSAR Atmospheric Column Water Vapor over CONUS

![Generic badge](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Generic badge](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)
![Generic badge](https://img.shields.io/badge/Python3.0-<COLOR>.svg)

This repository is for high-resolution atmospheric column water vapor (CWV) retrieval based on Inteferograms SAR data (InSAR).

Microwaves-based satellites have difficulties in detecting vapor field over lands, due to surface properties.
Fortunately, interferograms of SAR, used to detecting surface displacements (e.g., earthquakes and volcanic activities), provide an unprecedented oppertunity to measure the vapor field with super fine resolutions (~few meters).

The delay signal affected by atmospheric components relfects the zenith total delay (dry delay + wet delay due to vapor, ZWD) between two acquisition times. Such signal over the targeted swath is provided by interferometric images and absolute values of ZWD at imagery pixels are then retrived with the help of GPS stations (our ground truth for on-site ZTD, ZWD and CWV).

<div align="center">
<img src="/doc/InSAR_GPS_views.png" width="350" height="400" img> 
</div>
  
  
  
InSAR-CWV observations show unpresedented fine resolutions for water vapor over lands, compared to our top numerical weather predictions. Here are some snapshots for InSAR-CWV aligned with HRRR CWV (simulated by the atmospheric model and assimuilated with radar observations). 

<div align="center">
<img src="/doc/InSAR_CWV_example.png" width="1000" height="600" img> 
</div>



<br>
Wei-Ming Tsai, University of Miami, RSMAS<br>
wxt108@rsmas.miami.edu
