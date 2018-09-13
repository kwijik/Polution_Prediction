# Air pollution prediction Haut-de-France

This project is about predicting pollution :foggy: in north region of France.  At the moment I used 3 models: Seq2seq, LSTM and SVR.


## Data
The data is text files (.txt) with following schema:
`Date`: observable date
`PM10`: level of pollution
`RR`: precipitations
`TN`: minimal temperature
`TX`: maximal temperature
`TM`: medium temperature
`PMERM`: pressure at sea level
`FFM`: average wind speed
`UM`: average relative humidity
`GLOT`: global radiation
`DXY`: average wind direction

LSTM_final.py, SVR_final.py and Seq2seq_final.py are final versions with optimized parametres.
