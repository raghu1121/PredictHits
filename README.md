# PredictHits
__A  model that predicts the number of hits per session, depending on the given parameters.__

The features available are :
- row_num: a number uniquely identifying each row.
- locale: the platform of the session.
- day_of_week: Mon-Fri, the day of the week of the session.
- hour_of_day: 00-23, the hour of the day of the session.
- agent_id: the device used for the session.
- entry_page: describes the landing page of the session.
- path_id_set: shows all the locations that were visited during the session.
- traffic_type: indicates the channel the user cane through eg. search engine, email, ...
- session_duration: the duration in seconds of the session.
- hits: the number of interactions with the Webpage during the session.
