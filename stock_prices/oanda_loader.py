# account token : 8979703ec9eb299b9d8fd6af46603346-83aaa75c48b2eb71ee5bab916eeb2e6d
import json
import calendar as cal
import oandapyV20 as api
import oandapyV20.endpoints.instruments as instruments
import datetime

print (datetime.datetime.now().time(), 'extraction begins ...\n')

oanda = api.API(environment="practice",
                access_token="8979703ec9eb299b9d8fd6af46603346-83aaa75c48b2eb71ee5bab916eeb2e6d")

params = {
  "granularity": "M1"
}

base_from = 'T11:00:00Z'
base_to = 'T23:59:00Z'
year = 2005
month = 1
day = 1
last_day_of_month = cal.monthrange(year, month)[1]
closing_prices = []

while True:

    if year == 2017 and month == 7:
        break

    m = str(month) if len(str(month)) > 1 else '0' + str(month)

    d = str(day) if len(str(day)) > 1 else '0' + str(day)

    extracted = str(year) + '-' + m + '-' + d
    date_from = extracted + base_from
    date_to = extracted + base_to

    print ('\nProcessing day: ', extracted)

    params['from'] = date_from
    params['to'] = date_to

    r = instruments.InstrumentsCandles(instrument="EUR_USD", params=params)
    oanda.request(r)
    response = json.loads(json.dumps(r.response))
    candles = response['candles']

    for c in candles:
        mid = c['mid']
        closing_prices.append(mid['c'])

    print ('Amount of prices: ', len(closing_prices))

    if day == last_day_of_month:
        day = 1
        month += 1
        if month == 13:
            year += 1
            month = 1
        last_day_of_month = cal.monthrange(year, month)[1]
    else:
        day += 1

    print ('Finished processing day: ', extracted)

file = open('EUR_USD_M1_2005-01-01_TO_2017-06-30.csv', 'w')

print ('Writing to file ...')

for prices in closing_prices:
    file.write(prices + '\n')

file.close()

print ('Done')

print (datetime.datetime.now().time(), 'extraction finishes ...')